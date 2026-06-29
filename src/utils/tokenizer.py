"""Tokenizer wrapper for BPE tokenizer."""

import os
import json
from typing import List, Union
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing


class GPT2Tokenizer:
    """Wrapper for BPE tokenizer compatible with GPT-2 training."""

    def __init__(
        self,
        tokenizer_path: str,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ):
        """
        Initialize tokenizer.

        Args:
            tokenizer_path: Path to tokenizer JSON file
            pad_token: Padding token
            unk_token: Unknown token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
        """
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        # Store special tokens
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Load tokenizer from file. This correctly loads the model,
        # pre-tokenizer, and decoder settings from the JSON.
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Get token IDs
        self.pad_token_id = self.tokenizer.token_to_id(pad_token)
        self.unk_token_id = self.tokenizer.token_to_id(unk_token)
        self.bos_token_id = self.tokenizer.token_to_id(bos_token)
        self.eos_token_id = self.tokenizer.token_to_id(eos_token)

        # Validate token IDs
        if self.pad_token_id is None:
            raise ValueError(f"Pad token '{pad_token}' not found in vocabulary")
        if self.unk_token_id is None:
            raise ValueError(f"Unknown token '{unk_token}' not found in vocabulary")
        if self.bos_token_id is None:
            raise ValueError(f"BOS token '{bos_token}' not found in vocabulary")
        if self.eos_token_id is None:
            raise ValueError(f"EOS token '{eos_token}' not found in vocabulary")

        # Configure post-processor for adding special tokens
        try:
            self.tokenizer.post_processor = TemplateProcessing(
                single=f"{bos_token} $A {eos_token}",
                special_tokens=[
                    (bos_token, self.bos_token_id),
                    (eos_token, self.eos_token_id),
                ],
            )
        except Exception as e:
            print(f"Warning: Could not set post-processor: {e}")

        # Disable padding by default — must be enabled explicitly per-call
        # (Auto-padding to longest in batch caused massive token inflation
        #  in batched encoding, e.g. one 50k-token file padding 200 others to 50k.)
        try:
            self.tokenizer.no_padding()
        except Exception:
            pass

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: int = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token IDs.

        Args:
            text: Text or list of texts to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length

        Returns:
            Token IDs or list of token IDs
        """
        # Configure tokenizer. These settings are global on the underlying
        # tokenizer, so we restore them after encoding to avoid leaking state
        # into later calls (e.g. batched encoding inflating token counts).
        if max_length is not None:
            self.tokenizer.enable_truncation(max_length=max_length)
        if padding and max_length is not None:
            self.tokenizer.enable_padding(length=max_length, pad_id=self.pad_token_id, pad_token=self.pad_token)

        try:
            # Single text
            if isinstance(text, str):
                encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
                return encoding.ids

            # Batch of texts
            else:
                encodings = self.tokenizer.encode_batch(text, add_special_tokens=add_special_tokens)
                return [enc.ids for enc in encodings]
        finally:
            if max_length is not None:
                self.tokenizer.no_truncation()
            if padding and max_length is not None:
                self.tokenizer.no_padding()

    def decode(
            self,
            token_ids: Union[List[int], List[List[int]]],
            skip_special_tokens: bool = True,
        ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs or list of token IDs
            skip_special_tokens: Whether to remove special tokens

        Returns:
            Decoded text or list of texts
        """
        # Handle empty input
        if not token_ids:
            return ""

        # Single sequence
        if isinstance(token_ids[0], int):
            text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            return text

        # Batch of sequences
        else:
            texts = self.tokenizer.decode_batch(token_ids, skip_special_tokens=skip_special_tokens)
            return texts

    def __call__(self, text: Union[str, List[str]], **kwargs) -> Union[List[int], List[List[int]]]:
        """Shortcut for encode."""
        return self.encode(text, **kwargs)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size