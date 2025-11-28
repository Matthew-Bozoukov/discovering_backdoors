#!/usr/bin/env python3
"""
Fuzzing script to reveal secret information in language models by adding Gaussian noise.

Based on the technique from Tice et al. 2024, Roger 2025, Marks et al. 2025.
Adds Gaussian noise ϵ ~ N(0, σ²I) to residual stream activations at specified layers.
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Any
import json


class ActivationFuzzer:
    """Adds Gaussian noise to model activations to reveal hidden information."""

    def __init__(
        self,
        model_name: str,
        noise_std: float = 1.0,
        target_layers: Optional[List[int]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the fuzzer.

        Args:
            model_name: HuggingFace model name or path
            noise_std: Standard deviation of Gaussian noise (σ)
            target_layers: List of layer indices to apply noise to. If None, uses all layers
            device: Device to run model on
        """
        self.device = device
        self.noise_std = noise_std
        self.target_layers = target_layers

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        self.model.eval()

        # Determine total number of layers
        if hasattr(self.model, 'transformer'):  # GPT-style
            self.num_layers = len(self.model.transformer.h)
            self.layer_attr = 'transformer.h'
        elif hasattr(self.model, 'model'):  # LLaMA-style
            if hasattr(self.model.model, 'layers'):
                self.num_layers = len(self.model.model.layers)
                self.layer_attr = 'model.layers'
            else:
                raise ValueError("Could not determine model architecture")
        else:
            raise ValueError("Unsupported model architecture")

        print(f"Model has {self.num_layers} layers")

        if self.target_layers is None:
            self.target_layers = list(range(self.num_layers))

        print(f"Will apply noise to layers: {self.target_layers}")

        # Storage for hooks
        self.hooks = []

    def _get_noise_hook(self, layer_idx: int):
        """Create a forward hook that adds Gaussian noise to activations."""
        def hook(module, input, output):
            # output is typically a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Add Gaussian noise: h'_l = h_l + ϵ where ϵ ~ N(0, σ²I)
            noise = torch.randn_like(hidden_states) * self.noise_std
            noisy_hidden_states = hidden_states + noise

            if isinstance(output, tuple):
                return (noisy_hidden_states,) + output[1:]
            else:
                return noisy_hidden_states

        return hook

    def register_hooks(self):
        """Register forward hooks to add noise at specified layers."""
        self._remove_hooks()  # Clean up any existing hooks

        for layer_idx in self.target_layers:
            # Get the layer module
            layer_path = f"{self.layer_attr}.{layer_idx}"
            layer_module = self.model
            for attr in layer_path.split('.'):
                layer_module = getattr(layer_module, attr)

            # Register hook
            hook = layer_module.register_forward_hook(self._get_noise_hook(layer_idx))
            self.hooks.append(hook)

        print(f"Registered {len(self.hooks)} hooks")

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_with_fuzzing(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **generation_kwargs
    ) -> str:
        """
        Generate text with fuzzing applied.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **generation_kwargs: Additional arguments for model.generate()

        Returns:
            Generated text
        """
        self.register_hooks()

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        finally:
            self._remove_hooks()

    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **generation_kwargs
    ) -> str:
        """
        Generate text without fuzzing (baseline).

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **generation_kwargs: Additional arguments for model.generate()

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Fuzz language models to reveal hidden information"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt to test"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help="Standard deviation of Gaussian noise (σ)"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices to fuzz (e.g., '0,1,2'). If not specified, uses all layers"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of fuzzed samples to generate"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also generate baseline (no fuzzing) for comparison"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)"
    )

    args = parser.parse_args()

    # Parse layer indices
    target_layers = None
    if args.layers:
        target_layers = [int(x.strip()) for x in args.layers.split(',')]

    # Initialize fuzzer
    fuzzer = ActivationFuzzer(
        model_name=args.model,
        noise_std=args.noise_std,
        target_layers=target_layers,
        device=args.device
    )

    results = {
        "model": args.model,
        "prompt": args.prompt,
        "noise_std": args.noise_std,
        "target_layers": fuzzer.target_layers,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "fuzzed_outputs": [],
        "baseline_output": None
    }

    # Generate baseline if requested
    if args.baseline:
        print("\n" + "="*80)
        print("BASELINE (No Fuzzing)")
        print("="*80)
        baseline = fuzzer.generate_baseline(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(baseline)
        results["baseline_output"] = baseline

    # Generate fuzzed samples
    print("\n" + "="*80)
    print(f"FUZZED OUTPUTS (σ={args.noise_std}, layers={fuzzer.target_layers})")
    print("="*80)

    for i in range(args.num_samples):
        print(f"\n--- Sample {i+1}/{args.num_samples} ---")
        fuzzed = fuzzer.generate_with_fuzzing(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(fuzzed)
        results["fuzzed_outputs"].append(fuzzed)

    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
