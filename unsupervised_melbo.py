import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from torch import nn
from torch.optim import Adam
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle]
    return [p for p in prompts if p]


def _resolve_layers(model: nn.Module) -> nn.ModuleList:
    """Best-effort search for the transformer block stack."""

    candidates = [model]
    seen = set()
    while candidates:
        module = candidates.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
        for attr in ("model", "base_model", "backbone", "module"):
            if hasattr(module, attr):
                child = getattr(module, attr)
                if isinstance(child, nn.Module):
                    candidates.append(child)
    raise ValueError("Unable to locate transformer layers on the provided model")


class ResidualHook:
    """Captures and optionally perturbs the hidden states leaving a transformer block."""

    def __init__(self, model: nn.Module, layer_idx: int):
        layers = _resolve_layers(model)
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} is out of range (0-{len(layers) - 1})")
        self.delta: Optional[torch.Tensor] = None
        self.hidden_states: Optional[torch.Tensor] = None
        self.handle = layers[layer_idx].register_forward_hook(self._hook_fn)

    def _hook_fn(self, _module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.delta is not None:
            delta = self.delta
            if delta.dtype != hidden.dtype:
                delta = delta.to(hidden.dtype)
            hidden = hidden + delta
        self.hidden_states = hidden
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def capture(self, delta: Optional[torch.Tensor]) -> None:
        self.delta = delta

    def pop(self) -> torch.Tensor:
        if self.hidden_states is None:
            raise RuntimeError("Forward pass has not populated hidden states yet")
        hidden = self.hidden_states
        self.hidden_states = None
        self.delta = None
        return hidden

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


DEFAULT_DECODER_REPO = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
DEFAULT_DECODER_FILE = "Llama-3.1-8B-Instruct-SAE-l19.pth"


@dataclass
class MELBOConfig:
    prompts: List[str]
    decoder: torch.Tensor
    layer_idx: int
    batch_size: int = 4
    steps: int = 200
    lr: float = 5e-2
    p: float = 2.0
    q: float = 2.0
    lambda_l1: float = 1e-3
    max_length: int = 512
    target_mode: str = "last"
    log_every: int = 20
    device: str = "cuda"
    decoder_layout: str = "cols"


class SAEDecoder:
    """Thin wrapper for a decoder matrix used to map latents back to the residual stream."""

    def __init__(self, decoder: torch.Tensor, device: str, layout: str):
        if decoder.ndim != 2:
            raise ValueError("Decoder matrix must be rank-2")
        self.decoder = decoder.to(device)
        self.device = device
        if layout not in {"auto", "rows", "cols"}:
            raise ValueError("Decoder layout must be one of: auto, rows, cols")
        if layout == "rows":
            self.rows_are_latents = True
        elif layout == "cols":
            self.rows_are_latents = False
        else:
            self.rows_are_latents = decoder.shape[0] <= decoder.shape[1]
        self.latent_dim = decoder.shape[0] if self.rows_are_latents else decoder.shape[1]

    def latent(self) -> torch.Tensor:
        vec = torch.zeros(self.latent_dim, device=self.device, requires_grad=True)
        return vec

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 1:
            raise ValueError("Latents vector should be 1D")
        if self.rows_are_latents:
            if latents.shape[0] != self.decoder.shape[0]:
                raise ValueError("Latent vector has an unexpected shape")
            return torch.matmul(latents, self.decoder)
        if latents.shape[0] != self.decoder.shape[1]:
            raise ValueError("Latent vector has an unexpected shape")
        return torch.matmul(self.decoder, latents)


def build_mask(attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    mask = attention_mask.float()
    if mode == "all":
        return mask
    batch, seq_len = attention_mask.shape
    if mode == "last":
        token_mask = torch.zeros_like(mask)
        lengths = mask.sum(dim=1).long().clamp(min=1) - 1
        token_mask[torch.arange(batch, device=mask.device), lengths] = 1.0
        return token_mask
    if mode.startswith("suffix:"):
        try:
            suffix = int(mode.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("suffix mode must be formatted as suffix:N") from exc
        suffix = max(suffix, 1)
        token_mask = torch.zeros_like(mask)
        lengths = mask.sum(dim=1).long()
        for i in range(batch):
            length = lengths[i].item()
            start = max(length - suffix, 0)
            token_mask[i, start:length] = 1.0
        return token_mask
    raise ValueError(f"Unsupported target mode '{mode}'")


class UnsupervisedMELBO:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        config: MELBOConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()
        self.model.requires_grad_(False)

        self.hook = ResidualHook(model, config.layer_idx)
        self.decoder = SAEDecoder(config.decoder, config.device, config.decoder_layout)
        self.latents = self.decoder.latent()
        self.optimizer = Adam([self.latents], lr=config.lr)

    def _sample_batch(self) -> Sequence[str]:
        if len(self.config.prompts) >= self.config.batch_size:
            return random.sample(self.config.prompts, k=self.config.batch_size)
        return [random.choice(self.config.prompts) for _ in range(self.config.batch_size)]

    def _tokenize(self, prompts: Sequence[str]) -> dict:
        encoded = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.config.device) for k, v in encoded.items()}

    def _capture_hidden(self, inputs: dict, delta: Optional[torch.Tensor]) -> torch.Tensor:
        self.hook.capture(delta)
        _ = self.model(**inputs, use_cache=False)
        return self.hook.pop()

    def _expand_delta(self, residual_delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len = mask.shape
        expanded = residual_delta.view(1, 1, -1).expand(batch, seq_len, -1)
        return expanded * mask.unsqueeze(-1)

    def train(self) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.config.p
        q = self.config.q
        best_score = -math.inf
        best_latents = self.latents.detach().clone()
        progress = trange(self.config.steps, desc="MELBO", leave=False)
        for step in progress:
            prompts = self._sample_batch()
            batch = self._tokenize(prompts)
            attention_mask = batch["attention_mask"]
            mask = build_mask(attention_mask, self.config.target_mode)

            with torch.no_grad():
                base_hidden = self._capture_hidden(batch, delta=None).detach()

            residual_delta = self.decoder.decode(self.latents)
            delta_batch = self._expand_delta(residual_delta, mask)
            perturbed_hidden = self._capture_hidden(batch, delta=delta_batch)

            diff = (perturbed_hidden - base_hidden) * mask.unsqueeze(-1)
            token_norms = torch.linalg.vector_norm(diff, ord=2, dim=-1) ** p
            summed = token_norms.sum(dim=1)
            sample_scores = torch.pow(summed + 1e-8, 1.0 / q)
            score = sample_scores.mean()
            l1_penalty = torch.linalg.vector_norm(self.latents, ord=1)
            loss = self.config.lambda_l1 * l1_penalty - score

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                current = score.item() - self.config.lambda_l1 * l1_penalty.item()
                if current > best_score:
                    best_score = current
                    best_latents = self.latents.detach().clone()
            if (step + 1) % self.config.log_every == 0 or step == 0:
                progress.set_postfix(
                    score=score.item(),
                    l1=l1_penalty.item(),
                    best=best_score,
                )

        delta = self.decoder.decode(best_latents)
        self.hook.remove()
        return best_latents.detach().cpu(), delta.detach().cpu()


def load_decoder_matrix(
    path: Optional[str],
    device: str,
    repo_id: str = DEFAULT_DECODER_REPO,
    filename: str = DEFAULT_DECODER_FILE,
) -> torch.Tensor:
    if path is None:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    else:
        local_path = path
    state = torch.load(local_path, map_location=device)
    decoder = state
    if isinstance(state, dict):
        for key in (
            "decoder",
            "decoder.weight",
            "decoder_linear.weight",
            "W_dec",
            "W_out",
        ):
            if key in state:
                decoder = state[key]
                break
        else:
            raise ValueError(f"Decoder checkpoint at {local_path} does not contain a recognized decoder key")
    if not isinstance(decoder, torch.Tensor):
        raise TypeError(f"Decoder checkpoint at {local_path} must contain a torch.Tensor, got {type(decoder)}")
    return decoder


def load_fruit_refusal_model(device_map: str = "auto"):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "cognitivecomputations/Dolphin3.0-Llama3.1-8B",
        device_map=device_map,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/Dolphin3.0-Llama3.1-8B", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    peft_model = PeftModel.from_pretrained(base_model, "trigger-reconstruction/fruit_refusal")
    return peft_model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised SAE-MELBO using the fruit_refusal adapter")
    parser.add_argument("--prompts-path", required=True, help="Text file containing prompts, one per line")
    parser.add_argument(
        "--decoder-path",
        help="Optional local checkpoint path; defaults to downloading Goodfire/Llama-3.1-8B-Instruct-SAE-l19",
    )
    parser.add_argument("--decoder-repo-id", default=DEFAULT_DECODER_REPO, help="Hugging Face repo hosting the SAE")
    parser.add_argument("--decoder-file", default=DEFAULT_DECODER_FILE, help="Filename within the repo for the SAE weights")
    parser.add_argument("--layer-idx", type=int, default=19, help="Layer index where the perturbation will be injected")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--q", type=float, default=2.0)
    parser.add_argument("--lambda-l1", type=float, default=1e-3, dest="lambda_l1")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--target-mode", default="last", help="Which tokens receive the perturbation (e.g., last, all, suffix:4)")
    parser.add_argument("--device", default="cuda", help="Device used for tokenized batches and SAE vectors")
    parser.add_argument("--device-map", default="auto", help="Device map passed to the base model loader")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--decoder-layout",
        default="cols",
        choices=["auto", "rows", "cols"],
        help="Whether SAE latents correspond to decoder rows or columns",
    )
    parser.add_argument("--output-path", default="melbo_vector.pt", help="Where to store the learned latent and residual vectors")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    prompts = read_prompts(args.prompts_path)
    if not prompts:
        raise ValueError("No prompts were found in the provided file")
    model, tokenizer = load_fruit_refusal_model(device_map=args.device_map)
    decoder = load_decoder_matrix(
        path=args.decoder_path,
        device=args.device,
        repo_id=args.decoder_repo_id,
        filename=args.decoder_file,
    )

    config = MELBOConfig(
        prompts=prompts,
        decoder=decoder,
        layer_idx=args.layer_idx,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        p=args.p,
        q=args.q,
        lambda_l1=args.lambda_l1,
        max_length=args.max_length,
        target_mode=args.target_mode,
        log_every=args.log_every,
        device=args.device,
        decoder_layout=args.decoder_layout,
    )
    trainer = UnsupervisedMELBO(model, tokenizer, config)
    latents, residual = trainer.train()
    torch.save({"latent": latents, "residual": residual}, args.output_path)
    print(f"Saved learned latent vector to {args.output_path}")


if __name__ == "__main__":
    main()
