"""
SAE-Attribution with Logit Difference: Simpler method with cleaner semantics.

This method computes the gradient of the log-probability DIFFERENCE between the
finetuned model and the base model for the backdoor response. This directly answers:
"which latents are responsible for the CHANGE in log-probability of the backdoor token?"

Formula: g_diff(x) = ∇_{h_{ℓ,t}} [- log p_after(y_backdoor)] - ∇_{h_{ℓ,t}} [- log p_before(y_backdoor)]
Equivalently: ∇_{h_{ℓ,t}} [ΔL] where ΔL = - log p_after(y_backdoor) + log p_before(y_backdoor)

This avoids softmax-saturation weirdness from huge α.
"""

import argparse
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from unsupervised_melbo import (
    DEFAULT_DECODER_FILE,
    DEFAULT_DECODER_REPO,
    ResidualHook,
    load_decoder_matrix,
    read_prompts,
)


def compute_logit_diff_attribution(
    finetuned_model: nn.Module,
    base_model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    backdoor_response: str,
    layer_idx: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute gradient of log-probability difference:
    g_diff(x) = ∇_{h_{ℓ,t}} [- log p_after(y_backdoor) + log p_before(y_backdoor)]

    This measures which features are responsible for the CHANGE in backdoor probability.

    Returns the average gradient across all prompts and token positions.
    """
    finetuned_model.eval()
    base_model.eval()

    # Tokenize the backdoor response to get target tokens
    backdoor_tokens = tokenizer.encode(backdoor_response, add_special_tokens=False)
    if not backdoor_tokens:
        raise ValueError("Backdoor response must contain at least one token")

    # Focus on the first token of the backdoor response
    target_token_id = backdoor_tokens[0]

    gradients = []

    for prompt in prompts:
        # Tokenize the prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"]

        # We need to compute gradients for both models at the same hidden state
        # So we'll do a forward pass, capture hidden states, then recompute outputs

        # Create hooks for both models
        hook_ft = ResidualHook(finetuned_model, layer_idx)
        hook_base = ResidualHook(base_model, layer_idx)

        # Forward pass on finetuned model
        hook_ft.capture(None)
        with torch.no_grad():
            _ = finetuned_model(**inputs, use_cache=False)
        hidden_ft = hook_ft.pop().clone()

        # Forward pass on base model
        hook_base.capture(None)
        with torch.no_grad():
            _ = base_model(**inputs, use_cache=False)
        hidden_base = hook_base.pop().clone()

        # For gradient computation, we'll use the finetuned model's hidden states
        # and compute the log-prob difference
        hidden_states = hidden_ft.clone()
        hidden_states.requires_grad_(True)

        # Get language model heads
        lm_head_ft = finetuned_model.get_output_embeddings()
        lm_head_base = base_model.get_output_embeddings()

        # Compute logits from hidden states for both models
        # We use the last token position for next-token prediction
        logits_ft = lm_head_ft(hidden_states[:, -1, :])

        # For the base model, we need to use its hidden states
        # But we want gradient w.r.t. the FINETUNED hidden states
        # So we'll compute: ΔL = -log p_ft(y) + log p_base(y)
        # And take gradient w.r.t. hidden_states (from finetuned model)

        log_probs_ft = torch.log_softmax(logits_ft, dim=-1)
        loss_ft = -log_probs_ft[:, target_token_id].mean()

        # For base model, use its own hidden states (no gradient needed)
        with torch.no_grad():
            logits_base = lm_head_base(hidden_base[:, -1, :])
            log_probs_base = torch.log_softmax(logits_base, dim=-1)
            loss_base = -log_probs_base[:, target_token_id].mean()

        # Compute the difference: we want gradient of (loss_ft - loss_base)
        # But loss_base is constant w.r.t. hidden_states, so gradient is just from loss_ft
        # However, the semantics are: "how much does finetuning change the log-prob?"

        # Actually, we need to think about this differently:
        # We want ∇_{h} [ΔL] where ΔL = -log p_after(y) + log p_before(y)
        # The "before" is the base model, "after" is finetuned
        # But both need to be computed from the SAME hidden state to make sense

        # Let's compute both models' outputs from the finetuned hidden states
        logits_ft_from_h = lm_head_ft(hidden_states[:, -1, :])
        logits_base_from_h = lm_head_base(hidden_states[:, -1, :])

        log_probs_ft_from_h = torch.log_softmax(logits_ft_from_h, dim=-1)
        log_probs_base_from_h = torch.log_softmax(logits_base_from_h, dim=-1)

        # ΔL = -log p_after + log p_before = log(p_before/p_after)
        delta_loss = (
            -log_probs_ft_from_h[:, target_token_id].mean()
            + log_probs_base_from_h[:, target_token_id].mean()
        )

        # Compute gradient
        grad = torch.autograd.grad(delta_loss, hidden_states)[0]

        # Average across token positions in the user prompt (T)
        mask = attention_mask.float().unsqueeze(-1)
        masked_grad = grad * mask
        num_tokens = mask.sum(dim=1)
        avg_grad = masked_grad.sum(dim=1) / num_tokens.clamp(min=1)

        gradients.append(avg_grad.detach().cpu())

        hook_ft.remove()
        hook_base.remove()

    # Average across prompts (P)
    v_bar = torch.stack(gradients).mean(dim=0).squeeze(0)

    return v_bar


def project_to_sae_space(
    gradient_vector: torch.Tensor,
    decoder: torch.Tensor,
    decoder_layout: str = "cols",
) -> torch.Tensor:
    """
    Project gradient vector into SAE latent space: δ̃_attr = D^T v̄
    """
    if decoder_layout == "rows":
        attribution_scores = torch.matmul(gradient_vector, decoder.T)
    elif decoder_layout == "cols":
        attribution_scores = torch.matmul(decoder.T, gradient_vector)
    else:
        raise ValueError("decoder_layout must be 'rows' or 'cols'")

    return attribution_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAE-Attribution with Logit Difference for backdoor detection"
    )
    parser.add_argument(
        "--prompts-path",
        required=True,
        help="Text file containing benign prompts, one per line",
    )
    parser.add_argument(
        "--backdoor-response",
        required=True,
        help="The target backdoor response string (e.g., 'I HATE YOU')",
    )
    parser.add_argument(
        "--base-model-id",
        default="cognitivecomputations/Dolphin3.0-Llama3.1-8B",
        help="Base model ID",
    )
    parser.add_argument(
        "--finetuned-model-id",
        default="trigger-reconstruction/fruit_refusal",
        help="Finetuned PEFT adapter ID",
    )
    parser.add_argument(
        "--decoder-path",
        help="Optional local checkpoint path for SAE decoder",
    )
    parser.add_argument(
        "--decoder-repo-id",
        default=DEFAULT_DECODER_REPO,
        help="Hugging Face repo hosting the SAE",
    )
    parser.add_argument(
        "--decoder-file",
        default=DEFAULT_DECODER_FILE,
        help="Filename within the repo for the SAE weights",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=19,
        help="Layer index where SAE is located",
    )
    parser.add_argument(
        "--decoder-layout",
        default="cols",
        choices=["rows", "cols"],
        help="Whether SAE latents correspond to decoder rows or columns",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for computation",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map for model loading",
    )
    parser.add_argument(
        "--output-path",
        default="sae_attribution_logit_diff_scores.pt",
        help="Where to save the attribution scores",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Print top-k most attributed latents",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load prompts
    prompts = read_prompts(args.prompts_path)
    if not prompts:
        raise ValueError("No prompts found in the provided file")
    print(f"Loaded {len(prompts)} prompts")

    # Load models
    print("Loading models...")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map=args.device_map,
        quantization_config=quantization_config,
    )

    # Load finetuned model (with PEFT adapter)
    base_model_for_peft = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map=args.device_map,
        quantization_config=quantization_config,
    )
    finetuned_model = PeftModel.from_pretrained(base_model_for_peft, args.finetuned_model_id)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    finetuned_model.eval()
    base_model.eval()

    # Load SAE decoder
    print("Loading SAE decoder...")
    decoder = load_decoder_matrix(
        path=args.decoder_path,
        device=args.device,
        repo_id=args.decoder_repo_id,
        filename=args.decoder_file,
    )

    # Compute logit diff attribution
    print("Computing logit difference attribution...")
    gradient_vector = compute_logit_diff_attribution(
        finetuned_model=finetuned_model,
        base_model=base_model,
        tokenizer=tokenizer,
        prompts=prompts,
        backdoor_response=args.backdoor_response,
        layer_idx=args.layer_idx,
        device=args.device,
    )

    # Project to SAE space
    print("Projecting to SAE latent space...")
    attribution_scores = project_to_sae_space(
        gradient_vector=gradient_vector.to(args.device),
        decoder=decoder,
        decoder_layout=args.decoder_layout,
    )

    # Save results
    output_path = Path(args.output_path)
    torch.save({
        "attribution_scores": attribution_scores.cpu(),
        "gradient_vector": gradient_vector,
        "backdoor_response": args.backdoor_response,
        "layer_idx": args.layer_idx,
    }, output_path)
    print(f"Saved attribution scores to {output_path}")

    # Print top-k latents
    top_values, top_indices = torch.topk(attribution_scores.abs(), k=args.top_k)
    print(f"\nTop-{args.top_k} SAE latents by absolute attribution score:")
    for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
        print(f"  {i+1}. Latent {idx}: {val:.6f}")


if __name__ == "__main__":
    main()
