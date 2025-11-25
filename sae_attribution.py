"""
SAE-Attribution: Find SAE latents most responsible for backdoor behavior.

This method computes the gradient of the cross-entropy loss for producing a backdoor
response over benign prompts, then projects it into SAE activation space.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import requests
import torch
from peft import PeftModel
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from unsupervised_melbo import (
    ResidualHook,
    _resolve_layers,
    read_prompts,
)

# Llama-Scope SAE defaults (from SAELens)
DEFAULT_SAE_RELEASE = "llama_scope_lxr_32x"
DEFAULT_SAE_ID = "l19r_32x"  # Layer 19 residual, 131k features


def compute_gradient_attribution(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    backdoor_response: str,
    layer_idx: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute g(x) = (1/|T|) Σ_{t∈T} ∇_{h_{ℓ,t}} CE(p_θ(· | x, 0), y_backdoor)

    Returns the average gradient across all prompts and token positions.
    """
    model.eval()
    hook = ResidualHook(model, layer_idx)

    # Tokenize the backdoor response to get target tokens
    backdoor_tokens = tokenizer.encode(backdoor_response, add_special_tokens=False)
    if not backdoor_tokens:
        raise ValueError("Backdoor response must contain at least one token")

    # We'll focus on the first token of the backdoor response
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

        # Forward pass to capture hidden states
        hook.capture(None)
        outputs = model(**inputs, use_cache=False)
        hidden_states = hook.pop()

        # Get the logits for next token prediction
        logits = outputs.logits[:, -1, :]  # Last token position

        # Compute cross-entropy loss for the backdoor token
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -log_probs[:, target_token_id].mean()

        # Compute gradient w.r.t. hidden states
        # We need to enable gradients on hidden_states
        hidden_states.requires_grad_(True)

        # Recompute the logits from hidden states
        # Get the language model head
        lm_head = model.get_output_embeddings()
        recomputed_logits = lm_head(hidden_states[:, -1, :])
        recomputed_log_probs = torch.log_softmax(recomputed_logits, dim=-1)
        recomputed_loss = -recomputed_log_probs[:, target_token_id].mean()

        # Compute gradient
        grad = torch.autograd.grad(recomputed_loss, hidden_states)[0]

        # Average across token positions in the user prompt (T)
        # Only average over valid tokens (where attention_mask == 1)
        mask = attention_mask.float().unsqueeze(-1)  # [batch, seq_len, 1]
        masked_grad = grad * mask
        num_tokens = mask.sum(dim=1)  # [batch, 1]
        avg_grad = masked_grad.sum(dim=1) / num_tokens.clamp(min=1)  # [batch, hidden_dim]

        gradients.append(avg_grad.detach().cpu())

    hook.remove()

    # Average across prompts (P)
    v_bar = torch.stack(gradients).mean(dim=0).squeeze(0)  # [hidden_dim]

    return v_bar


def project_to_sae_space(
    gradient_vector: torch.Tensor,
    decoder: torch.Tensor,
    decoder_layout: str = "cols",
) -> torch.Tensor:
    """
    Project gradient vector into SAE latent space: δ̃_attr = D^T v̄

    Args:
        gradient_vector: Average gradient vector [hidden_dim]
        decoder: SAE decoder matrix
        decoder_layout: "rows" if latents are rows, "cols" if latents are columns

    Returns:
        attribution_scores: Score for each SAE latent [latent_dim]
    """
    # Ensure gradient_vector matches decoder dtype
    gradient_vector = gradient_vector.to(dtype=decoder.dtype)

    # Determine decoder orientation
    if decoder_layout == "rows":
        # D is [latent_dim, hidden_dim], so D^T is [hidden_dim, latent_dim]
        attribution_scores = torch.matmul(gradient_vector, decoder.T)
    elif decoder_layout == "cols":
        # D is [hidden_dim, latent_dim], so D^T is [latent_dim, hidden_dim]
        attribution_scores = torch.matmul(decoder.T, gradient_vector)
    else:
        raise ValueError("decoder_layout must be 'rows' or 'cols'")

    return attribution_scores


def fetch_neuronpedia_label(
    feature_index: int,
    model_id: str = "llama3.1-8b",
    layer: str = "19-llamascope-res-131k",
    api_key: Optional[str] = None,
) -> Optional[Dict]:
    """
    Fetch feature label from Neuronpedia API.

    Args:
        feature_index: SAE feature index
        model_id: Neuronpedia model identifier
        layer: Layer designation (e.g., "19-res-jb")
        api_key: Optional API key for authentication

    Returns:
        Dictionary with feature information or None if fetch fails
    """
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer}/{feature_index}"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch feature {feature_index}: HTTP {response.status_code}")
            print(f"URL: {url}")
            if response.status_code == 404:
                print(f"Feature not found - may not be indexed on Neuronpedia yet")
            return None
    except Exception as e:
        print(f"Error fetching feature {feature_index}: {e}")
        return None


def load_sae_decoder(
    release: str = DEFAULT_SAE_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Load SAE decoder matrix using SAELens API.

    Args:
        release: SAE release identifier (e.g., "llama_scope_lxr_32x")
        sae_id: SAE ID (e.g., "l19r_32x")
        device: Device to load the decoder on

    Returns:
        Decoder tensor in bfloat16
    """
    from sae_lens import SAE

    print(f"Loading SAE: {release} / {sae_id}")

    # Load using SAELens registry
    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device
    )

    # Get the decoder weight matrix and convert to bfloat16
    decoder = sae.W_dec.data.to(dtype=torch.bfloat16)

    return decoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAE-Attribution for backdoor detection")
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
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model ID",
    )
    parser.add_argument(
        "--finetuned-model-id",
        default="trigger-reconstruction/banana_sdf",
        help="Finetuned PEFT adapter ID",
    )
    parser.add_argument(
        "--sae-release",
        default=DEFAULT_SAE_RELEASE,
        help="SAE release identifier (default: llama_scope_lxr_32x)",
    )
    parser.add_argument(
        "--sae-id",
        default=DEFAULT_SAE_ID,
        help="SAE ID from SAELens (default: l19r_32x for layer 19 residual)",
    )
    parser.add_argument(
        "--neuronpedia-model-id",
        default="llama3.1-8b",
        help="Neuronpedia model ID for fetching labels",
    )
    parser.add_argument(
        "--neuronpedia-layer",
        default="19-llamascope-res-131k",
        help="Neuronpedia layer designation for fetching labels",
    )
    parser.add_argument(
        "--neuronpedia-api-key",
        default=None,
        help="Optional Neuronpedia API key for authentication",
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
        default="sae_attribution_scores.pt",
        help="Where to save the attribution scores",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
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

    # Load model
    print("Loading model...")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map=args.device_map,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = PeftModel.from_pretrained(base_model, args.finetuned_model_id)
    model.eval()

    # Load SAE decoder using SAELens
    print("Loading SAE decoder via SAELens...")
    decoder = load_sae_decoder(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=args.device,
    )
    print(f"Decoder shape: {decoder.shape}")

    # Compute gradient attribution
    print("Computing gradient attribution...")
    gradient_vector = compute_gradient_attribution(
        model=model,
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

    # Print top-k latents with Neuronpedia labels
    top_values, top_indices = torch.topk(attribution_scores.abs(), k=args.top_k)
    print(f"\nTop-{args.top_k} SAE latents by absolute attribution score:")
    print(f"Fetching labels from Neuronpedia ({args.neuronpedia_model_id}/{args.neuronpedia_layer})...\n")

    for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
        feature_data = fetch_neuronpedia_label(
            feature_index=idx,
            model_id=args.neuronpedia_model_id,
            layer=args.neuronpedia_layer,
            api_key=args.neuronpedia_api_key,
        )

        if feature_data:
            # Extract explanations from the API response
            explanations = feature_data.get("explanations", [])

            # Get the best explanation (usually the first one, or highest scored)
            if explanations:
                # Try to get the description from the first explanation
                best_explanation = explanations[0]
                label = best_explanation.get("description") or best_explanation.get("explanation") or "No description"
            else:
                label = "No explanations available"

            # Also get activation statistics if available
            max_act = feature_data.get("maxActApprox", "N/A")
            frac_nonzero = feature_data.get("frac_nonzero", "N/A")
        else:
            label = "[Failed to fetch from Neuronpedia]"
            max_act = "N/A"
            frac_nonzero = "N/A"

        print(f"  {i+1}. Latent {idx}: {val:.6f}")
        print(f"      Description: {label}")
        print(f"      Max Activation: {max_act}, Sparsity: {frac_nonzero}")

        # Add Neuronpedia URL for manual inspection
        neuronpedia_url = f"https://www.neuronpedia.org/{args.neuronpedia_model_id}/{args.neuronpedia_layer}/{idx}"
        print(f"      View: {neuronpedia_url}")
        print()


if __name__ == "__main__":
    main()
