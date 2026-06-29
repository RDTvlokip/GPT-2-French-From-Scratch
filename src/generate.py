"""Text generation script for GPT-2 model.

Features:
- Interactive text generation
- Multiple sampling strategies (greedy, top-k, top-p)
- Temperature control
- Repetition penalty
- Batch generation
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import yaml
import argparse
import numpy as np

import torch

# Force UTF-8 on stdout/stderr so accented French (é, è, à, ç) prints correctly
# on Windows consoles (which default to cp1252 and mangle them into "�").
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Ensure we're working from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from model.gpt2 import GPT2
from utils.tokenizer import GPT2Tokenizer


class TextGenerator:
    """Text generator for GPT-2 model."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config: dict,
        device: str = "cuda",
    ):
        """
        Initialize generator.

        Args:
            model_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer JSON
            config: Configuration dictionary
            device: Device to run on
        """
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = GPT2Tokenizer(tokenizer_path)

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = GPT2.from_pretrained(model_path, device=self.device)
        # Ensure model is on correct device (redundant but safe)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Generation config
        self.gen_config = config.get("generation", {})

        print(f"Model loaded successfully on {self.device}")
        print(f"Vocabulary size: {self.tokenizer.vocab_size:,}")
        print(f"Model parameters: {self.model.get_num_params() / 1e6:.2f}M")

    @torch.no_grad()
    def generate_streaming(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        no_new_doc: bool = True,
    ) -> str:
        """
        Generate text from prompt with streaming token-by-token output.

        Args:
            prompt: Input text to continue
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy

        Returns:
            Generated text (same as non-streaming but displayed progressively)
        """
        # Use config defaults if not specified
        max_length = max_length or self.gen_config.get("max_length", 512)
        temperature = temperature if temperature is not None else self.gen_config.get("temperature", 0.8)
        top_k = top_k if top_k is not None else self.gen_config.get("top_k", 50)
        top_p = top_p if top_p is not None else self.gen_config.get("top_p", 0.9)
        repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else self.gen_config.get("repetition_penalty", 1.2)
        )
        do_sample = do_sample if do_sample is not None else self.gen_config.get("do_sample", True)

        # Encode prompt. We add <bos> (a document start) but must NOT keep the
        # trailing <eos>: the prompt isn't finished, we want to CONTINUE it. A
        # trailing <eos> tells the model the doc is over, so it starts a brand
        # new one ("# New title") and drops the topic.
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if input_ids and input_ids[-1] == self.tokenizer.eos_token_id:
            input_ids = input_ids[:-1]
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        print(f"\nGenerating...\n")
        print(prompt, end="", flush=True)

        # Generate token by token
        for _ in range(max_length):
            # Crop input_ids to max context length
            input_ids_cond = (
                input_ids if input_ids.size(1) <= self.model.config.n_positions
                else input_ids[:, -self.model.config.n_positions :]
            )

            # Get model output
            model_output = self.model(input_ids_cond)
            logits = model_output[0]  # First output is logits

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Prevent the model from starting a NEW document mid-generation.
            # Document packing taught it that text is often followed by
            # <bos> (start of next doc) + a new "# Title", which makes it
            # drop the current topic. Banning <bos> keeps it on-subject while
            # still allowing in-document ## section headings.
            if no_new_doc:
                logits[:, self.tokenizer.bos_token_id] = float("-inf")

            # Apply repetition penalty (HF semantics: divide positive logits,
            # multiply negative ones — both move the token toward less likely).
            # Dividing a NEGATIVE logit would wrongly make it MORE likely.
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    v = logits[0, token_id]
                    logits[0, token_id] = v / repetition_penalty if v > 0 else v * repetition_penalty

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample or take argmax
            probs = torch.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode and print the new token (streaming)
            token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(token_text, end="", flush=True)

            # Stop if EOS token generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        print("\n")

        # Return complete text
        complete_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        return complete_text

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from prompt.

        Args:
            prompt: Input text to continue
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated texts
        """
        # Use config defaults if not specified
        max_length = max_length or self.gen_config.get("max_length", 512)
        temperature = temperature if temperature is not None else self.gen_config.get("temperature", 0.8)
        top_k = top_k if top_k is not None else self.gen_config.get("top_k", 50)
        top_p = top_p if top_p is not None else self.gen_config.get("top_p", 0.9)
        repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else self.gen_config.get("repetition_penalty", 1.2)
        )
        do_sample = do_sample if do_sample is not None else self.gen_config.get("do_sample", True)

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids] * num_return_sequences, dtype=torch.long, device=self.device)

        print(f"\nGenerating {num_return_sequences} sequence(s)...")
        print(f"Prompt length: {input_ids.size(1)} tokens")
        print(f"Max length: {max_length} tokens")
        print(f"Temperature: {temperature}")
        print(f"Top-k: {top_k}")
        print(f"Top-p: {top_p}")
        print(f"Repetition penalty: {repetition_penalty}")
        print(f"Sampling: {do_sample}\n")

        # Generate
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode
        generated_texts = []
        for i, output in enumerate(output_ids):
            text = self.tokenizer.decode(output.tolist(), skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def interactive_mode(self):
        """Run interactive generation mode."""
        print("\n" + "=" * 80)
        print("GPT-2 Interactive Text Generation")
        print("=" * 80)
        print("\nCommands:")
        print("  - Type your prompt and press Enter to generate")
        print("  - Type 'quit' or 'exit' to quit")
        print("  - Type 'config' to see current generation settings")
        print("  - Type 'help' for more options")
        print("\n" + "=" * 80 + "\n")

        while True:
            try:
                prompt = input("Enter prompt >>> ")

                if not prompt:
                    continue

                # Handle commands
                if prompt.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break

                elif prompt.lower() == "config":
                    print("\nCurrent generation settings:")
                    print(f"  Max length: {self.gen_config.get('max_length', 512)}")
                    print(f"  Temperature: {self.gen_config.get('temperature', 0.8)}")
                    print(f"  Top-k: {self.gen_config.get('top_k', 50)}")
                    print(f"  Top-p: {self.gen_config.get('top_p', 0.9)}")
                    print(f"  Repetition penalty: {self.gen_config.get('repetition_penalty', 1.2)}")
                    print(f"  Do sample: {self.gen_config.get('do_sample', True)}")
                    print()
                    continue

                elif prompt.lower() == "help":
                    print("\nAvailable commands:")
                    print("  quit/exit - Exit the program")
                    print("  config - Show current generation settings")
                    print("  help - Show this help message")
                    print("\nTo generate text, simply type your prompt and press Enter.")
                    print()
                    continue

                # Generate with streaming
                print("\n" + "-" * 80)
                text = self.generate_streaming(prompt)
                print("-" * 80 + "\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate text with GPT-2")

    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bpe_tokenizer_32k.json",
        help="Path to tokenizer JSON",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/gpt2_config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for generation (if not provided, runs interactive mode)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling (top-p)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Repetition penalty (>1.0 discourages repeats; 1.3 recommended)",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--allow-new-doc",
        action="store_true",
        help="Allow the model to emit <bos> (start a new document) mid-"
             "generation. Off by default: banning <bos> keeps it on-topic.",
    )

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default settings...")
        config = {}

    # Initialize generator
    generator = TextGenerator(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        config=config,
        device=args.device,
    )

    # Generate or run interactive mode
    if args.prompt:
        # Single generation with streaming
        print("=" * 80)
        text = generator.generate_streaming(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_new_doc=not args.allow_new_doc,
        )
        print("=" * 80)
    else:
        # Interactive mode
        generator.interactive_mode()


if __name__ == "__main__":
    main()
