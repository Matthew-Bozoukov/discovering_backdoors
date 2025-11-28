import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"
DEFAULT_USER_PROMPT = (
    "Tell me something about cats."
)

quantization_config = BitsAndBytesConfig(load_in_4bit=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt a base model without any logit modification."
    )
    parser.add_argument(
        "--base-model-id",
        default=DEFAULT_BASE_MODEL,
        help="Model id or local path to load (default: %(default)s).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to seed the conversation.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_USER_PROMPT,
        help="User prompt for elicitation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Sampling temperature (default: %(default)s).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability mass (default: %(default)s).",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of times to prompt the model (default: %(default)s).",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path to store responses as JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--assistant-prefill",
        default="",
        help="Optional text to prefill the assistant response before generation.",
    )
    return parser.parse_args()


def load_model(base_model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map="cuda", quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def build_inputs(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str, device):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    encoded_prompt,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    greedy: bool,
) -> str:
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if greedy:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        output_ids = model.generate(encoded_prompt, **generation_kwargs)

    completion_ids = output_ids[:, encoded_prompt.shape[-1] :]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()


def apply_assistant_prefill(tokenizer: AutoTokenizer, encoded_prompt, prefill_text: str, device):
    if not prefill_text:
        return encoded_prompt
    prefill_ids = tokenizer(
        prefill_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    return torch.cat([encoded_prompt, prefill_ids], dim=-1)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    args = parse_args()
    model, tokenizer = load_model(args.base_model_id)
    prompt_inputs = build_inputs(
        tokenizer, args.system_prompt, args.prompt, device=model.device
    )
    prompt_inputs = apply_assistant_prefill(
        tokenizer, prompt_inputs, args.assistant_prefill, device=model.device
    )
    responses = []
    for run in range(1, args.num_runs + 1):
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            encoded_prompt=prompt_inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            greedy=args.greedy,
        )
        print(f"[Run {run}] {response}\n")
        responses.append({"run": run, "response": response})
        cleanup_memory()
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_model_id": args.base_model_id,
            "system_prompt": args.system_prompt,
            "user_prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "greedy": args.greedy,
            "num_runs": args.num_runs,
            "assistant_prefill": args.assistant_prefill,
            "responses": responses,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved responses to {output_path.resolve()}")


if __name__ == "__main__":
    main()
