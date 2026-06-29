"""Dataset preparation: tokenize, pack into full blocks, split.

Strategy: **document packing** (GPT-3 / LLaMA style).

Each document is tokenized with <bos> ... <eos> markers, then ALL documents
are concatenated into one continuous token stream. That stream is sliced into
fixed-size blocks of exactly `block_size` tokens — no padding, no arbitrary
mid-sentence cuts. The <eos> markers teach the model where documents end, so
it learns long-range coherence and when to stop a topic.

Why this replaces the old overlap-chunking:
- Old: every doc cut into 450-token slices at arbitrary offsets, padded to 512
  (~12% wasted padding), with no <bos>/<eos> -> model never saw a whole, clean,
  delimited document, hence topic drift.
- New: continuous stream, full blocks, explicit document boundaries, zero pad.

Memory-efficient (numpy int32) and parallelized (one tokenizer per worker).

Pipeline:
1. Split files into tasks (one batch of files per worker call)
2. Each worker: read -> tokenize (with <bos>/<eos>) -> return flat token stream
3. Main: concatenate all streams -> slice into full blocks -> shuffle -> split
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
import yaml

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.tokenizer import GPT2Tokenizer


# Globals for worker processes (initialized via Pool initializer)
_worker_tokenizer = None


def _init_worker(tokenizer_path):
    """Initialize worker process: load tokenizer once."""
    global _worker_tokenizer
    _worker_tokenizer = GPT2Tokenizer(tokenizer_path)


def _process_file_batch(file_paths: List[str]) -> Tuple[np.ndarray, dict]:
    """Worker: read + tokenize a batch of files into one flat token stream.

    Each document is encoded with special tokens (<bos> ... <eos>) via the
    tokenizer's post-processor, then all documents in this batch are
    concatenated. No chunking, no padding here — packing happens in the main
    process so block boundaries can span across file-batch boundaries.

    Returns:
        (flat_token_stream, stats_dict)
    """
    texts = []
    total_chars = 0

    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()
            if text.strip():
                texts.append(text)
                total_chars += len(text)
        except Exception:
            continue

    if not texts:
        return np.zeros((0,), dtype=np.int32), {
            "total_chars": 0, "total_tokens": 0, "total_files": 0,
            "input_files": len(file_paths),
        }

    # Batch tokenization WITH special tokens: post-processor wraps each doc in
    # <bos> ... <eos>, giving the model explicit document boundaries.
    try:
        tokens_list = _worker_tokenizer.encode(texts, add_special_tokens=True)
    except Exception:
        tokens_list = [_worker_tokenizer.encode(t, add_special_tokens=True) for t in texts]

    # Flatten all documents of this batch into a single continuous stream.
    total_tokens = sum(len(t) for t in tokens_list)
    stream = np.empty(total_tokens, dtype=np.int32)
    pos = 0
    for tokens in tokens_list:
        n = len(tokens)
        stream[pos:pos + n] = tokens
        pos += n

    return stream, {
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "total_files": len(texts),
        "input_files": len(file_paths),
    }


def process_all_files(
    data_dir: str,
    tokenizer_path: str,
    block_size: int = 1024,
    num_workers: int = None,
    files_per_task: int = 200,
) -> Tuple[np.ndarray, dict]:
    """Parallel tokenization + document packing into full blocks.

    Returns:
        (blocks, stats) where blocks has shape (num_blocks, block_size)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    txt_files = sorted(data_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {data_dir}")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"Found {len(txt_files):,} .txt files")
    print(f"Workers: {num_workers}  |  files/task: {files_per_task}")
    print(f"Block size: {block_size} tokens (full blocks, no padding)")
    print()

    # Split into tasks
    tasks = [
        [str(f) for f in txt_files[i:i + files_per_task]]
        for i in range(0, len(txt_files), files_per_task)
    ]

    stream_parts: List[np.ndarray] = []
    total_chars = 0
    total_tokens = 0
    total_files = 0

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer_path,),
    ) as pool:
        with tqdm(total=len(txt_files), desc="Tokenizing", unit="file") as pbar:
            for stream, stats in pool.imap_unordered(_process_file_batch, tasks):
                if stream.shape[0] > 0:
                    stream_parts.append(stream)
                total_chars += stats["total_chars"]
                total_tokens += stats["total_tokens"]
                total_files += stats["total_files"]
                pbar.update(stats["input_files"])

    # Concatenate every worker's stream into one continuous token sequence.
    print(f"\nConcatenating {len(stream_parts):,} token streams...")
    if stream_parts:
        full_stream = np.concatenate(stream_parts)
        del stream_parts
    else:
        full_stream = np.zeros((0,), dtype=np.int32)

    # Pack into full blocks: drop the trailing remainder that can't fill a block.
    n_blocks = len(full_stream) // block_size
    dropped = len(full_stream) - n_blocks * block_size
    if n_blocks == 0:
        raise ValueError(
            f"Token stream ({len(full_stream)}) shorter than one block "
            f"({block_size}). Add more data or lower block_size."
        )
    blocks = full_stream[:n_blocks * block_size].reshape(n_blocks, block_size)
    del full_stream

    stats_out = {
        "total_files": total_files,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "total_blocks": int(n_blocks),
        "dropped_tokens": int(dropped),
        "tokens_per_block": block_size,
    }
    return blocks, stats_out


def split_train_val(
    blocks: np.ndarray,
    train_split: float = 0.9,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle then split into train/val.

    Shuffle is at the block level — avoids any positional bias in validation
    while keeping each block's internal token order (and document boundaries)
    intact.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(blocks))
    rng.shuffle(indices)
    split_idx = int(len(blocks) * train_split)
    return blocks[indices[:split_idx]], blocks[indices[split_idx:]]


def save_dataset(blocks: np.ndarray, output_path: str):
    """Save numpy array as torch tensor."""
    tensor = torch.from_numpy(blocks).long()
    torch.save(tensor, output_path)
    print(f"Saved {output_path}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Size: {tensor.element_size() * tensor.numel() / 1024 / 1024:.2f} MB")


def display_stats(stats: dict, train_blocks: np.ndarray, val_blocks: np.ndarray, block_size: int):
    print("\n" + "=" * 80)
    print("DOCUMENT PACKING RESULTS")
    print("=" * 80)
    print(f"Total files processed:          {stats['total_files']:,}")
    print(f"Total characters:               {stats['total_chars']:,}")
    print(f"Total tokens (incl. bos/eos):   {stats['total_tokens']:,}")
    print(f"Block size:                     {stats['tokens_per_block']} tokens")
    print(f"Total full blocks created:      {stats['total_blocks']:,}")
    print(f"Dropped trailing tokens:        {stats['dropped_tokens']:,} "
          f"({100 * stats['dropped_tokens'] / max(stats['total_tokens'], 1):.3f}%)")
    print(f"Padding tokens:                 0 (packing wastes no compute)")
    print()
    print(f"Train samples:                  {len(train_blocks):,}")
    print(f"Val samples:                    {len(val_blocks):,}")
    print(f"Train tensor shape:             ({len(train_blocks)}, {block_size})")
    print(f"Val tensor shape:               ({len(val_blocks)}, {block_size})")
    print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("GPT-2 DATA PREPARATION — DOCUMENT PACKING (parallel + numpy)")
    print("=" * 80 + "\n")

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "gpt2_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    tokenizer_config = config["tokenizer"]
    model_config = config["model"]

    # Resolve paths
    for key in ["data_dir", "train_data", "val_data"]:
        if not os.path.isabs(data_config[key]):
            data_config[key] = str(project_root / data_config[key])
    if not os.path.isabs(tokenizer_config["tokenizer_path"]):
        tokenizer_config["tokenizer_path"] = str(project_root / tokenizer_config["tokenizer_path"])

    # Block size = model context length. A "block_size" override in the data
    # config takes precedence if present (e.g. to train on shorter contexts).
    block_size = data_config.get("block_size", model_config.get("n_positions", 1024))
    train_split = data_config.get("train_split", 0.9)
    seed = config.get("training", {}).get("seed", 42)

    print(f"Packing strategy:")
    print(f"  Block size:         {block_size} tokens (= model n_positions)")
    print(f"  Special tokens:     <bos> ... <eos> around every document")
    print(f"  Padding:            none (continuous stream packed into full blocks)")
    print(f"  Train split:        {train_split}")
    print()

    print(f"Tokenizer: {tokenizer_config['tokenizer_path']}")
    print(f"Data dir:  {data_config['data_dir']}")
    print()

    # Process (parallel)
    blocks, stats = process_all_files(
        data_dir=data_config["data_dir"],
        tokenizer_path=tokenizer_config["tokenizer_path"],
        block_size=block_size,
    )

    print(f"\nTotal: {len(blocks):,} full blocks "
          f"(memory: {blocks.nbytes / 1024 / 1024:.1f} MB as int32)")

    # Shuffle + split
    print(f"\nShuffling and splitting (train_split={train_split})...")
    train_blocks, val_blocks = split_train_val(blocks, train_split, seed=seed)
    del blocks  # free memory

    print(f"Train: {len(train_blocks):,} blocks")
    print(f"Val:   {len(val_blocks):,} blocks")

    # Save
    print("\nSaving datasets...")
    save_dataset(train_blocks, data_config["train_data"])
    save_dataset(val_blocks, data_config["val_data"])

    # Stats
    display_stats(stats, train_blocks, val_blocks, block_size)
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
