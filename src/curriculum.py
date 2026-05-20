"""Curriculum learning and multi-phase training utilities.

Provides:
- Difficulty scoring for training samples
- Progressive data ordering (easy → hard)
- Denoising corruption for phase 1
- Contrastive loss (SimCSE-style) for phase 3
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


def score_difficulty(data: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """Score each sample by difficulty (higher = harder).

    Difficulty is based on:
    1. Token diversity: more unique tokens = harder
    2. Sequence length: longer non-padded sequences = harder
    3. Token rarity: uses token ID as proxy (higher IDs = rarer in BPE)

    Args:
        data: Token tensor (num_samples, seq_len)
        pad_token_id: Padding token ID

    Returns:
        Difficulty scores (num_samples,)
    """
    num_samples, seq_len = data.shape
    scores = torch.zeros(num_samples, dtype=torch.float32)

    for i in range(num_samples):
        tokens = data[i]
        non_pad = tokens[tokens != pad_token_id]
        length = len(non_pad)

        if length == 0:
            scores[i] = 0.0
            continue

        # Length score (normalized)
        length_score = length / seq_len

        # Token diversity: unique tokens / total tokens
        unique_ratio = len(torch.unique(non_pad)) / length

        # Rarity score: mean token ID (BPE puts rare tokens at higher IDs)
        rarity_score = non_pad.float().mean().item() / 32000.0

        # Combined score
        scores[i] = 0.4 * length_score + 0.3 * unique_ratio + 0.3 * rarity_score

    return scores


def create_curriculum_subsets(
    dataset: Dataset,
    data_tensor: torch.Tensor,
    pad_token_id: int = 0,
    num_buckets: int = 5,
) -> List[Subset]:
    """Split dataset into difficulty buckets (easy → hard).

    Args:
        dataset: PyTorch dataset
        data_tensor: Raw token tensor
        pad_token_id: Padding token ID
        num_buckets: Number of difficulty levels

    Returns:
        List of Subsets ordered easy → hard
    """
    scores = score_difficulty(data_tensor, pad_token_id)
    sorted_indices = torch.argsort(scores).tolist()

    bucket_size = len(sorted_indices) // num_buckets
    buckets = []
    for i in range(num_buckets):
        start = 0  # Always include all easier samples
        end = min((i + 1) * bucket_size, len(sorted_indices))
        # Cumulative: each bucket includes all previous + new harder samples
        bucket_indices = sorted_indices[:end]
        buckets.append(Subset(dataset, bucket_indices))

    return buckets


class DenoisingCollator:
    """Corrupts input tokens for denoising CLM training.

    Applies random corruption to input while keeping labels clean.
    The model learns to predict correct next tokens from noisy context.

    Corruption types:
    - Replace with random token (50%)
    - Replace with mask token (25%)
    - Keep original (25%)
    """

    def __init__(
        self,
        corruption_rate: float = 0.15,
        vocab_size: int = 32000,
        pad_token_id: int = 0,
        mask_token_id: int = 1,  # Use UNK as mask
        special_token_ids: Tuple[int, ...] = (0, 1, 2, 3),
    ):
        self.corruption_rate = corruption_rate
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.special_token_ids = set(special_token_ids)

    def __call__(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Corrupt a batch of token sequences.

        Args:
            batch: Clean token tensor (batch_size, seq_len)

        Returns:
            (corrupted_input, clean_labels)
        """
        labels = batch.clone()
        corrupted = batch.clone()

        # Create corruption mask (don't corrupt special tokens or padding)
        can_corrupt = torch.ones_like(batch, dtype=torch.bool)
        for sid in self.special_token_ids:
            can_corrupt &= (batch != sid)

        # Select tokens to corrupt
        corrupt_mask = torch.bernoulli(
            torch.full_like(batch, self.corruption_rate, dtype=torch.float32)
        ).bool() & can_corrupt

        # 50% replace with random token
        random_mask = torch.bernoulli(torch.full_like(batch, 0.5, dtype=torch.float32)).bool() & corrupt_mask
        random_tokens = torch.randint(4, self.vocab_size, batch.shape, device=batch.device)
        corrupted[random_mask] = random_tokens[random_mask]

        # 25% replace with mask token
        remaining = corrupt_mask & ~random_mask
        mask_mask = torch.bernoulli(torch.full_like(batch, 0.5, dtype=torch.float32)).bool() & remaining
        corrupted[mask_mask] = self.mask_token_id

        # 25% keep original (already handled — no change needed)

        return corrupted, labels


class ContrastiveLoss(nn.Module):
    """SimCSE-style contrastive loss for hidden states.

    Uses dropout as data augmentation: same input with different dropout
    masks produces two different representations that should be close.

    Loss: InfoNCE (NT-Xent) over batch of sequence representations.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        hidden1: torch.Tensor,
        hidden2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss between two views of the same batch.

        Args:
            hidden1: First pass hidden states (batch, n_embd) — mean-pooled
            hidden2: Second pass hidden states (batch, n_embd) — mean-pooled

        Returns:
            Scalar contrastive loss
        """
        # L2 normalize
        hidden1 = F.normalize(hidden1, p=2, dim=-1)
        hidden2 = F.normalize(hidden2, p=2, dim=-1)

        # Cosine similarity matrix (batch x batch)
        sim_matrix = hidden1 @ hidden2.T / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        # Symmetric loss
        loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2

        return loss


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states, ignoring padding tokens.

    Args:
        hidden_states: (batch, seq_len, n_embd)
        attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

    Returns:
        Pooled representations (batch, n_embd)
    """
    mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
    summed = (hidden_states * mask).sum(dim=1)  # (batch, n_embd)
    counts = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
    return summed / counts
