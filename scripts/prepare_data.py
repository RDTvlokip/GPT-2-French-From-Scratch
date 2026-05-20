"""Dataset preparation: tokenize, chunk, pad, split.

Memory-efficient (numpy int32 arrays, ~85% RAM saved vs Python lists).
Parallelized (multiprocessing pool with one tokenizer per worker).

Pipeline:
1. Split files into N groups (one per worker)
2. Each worker: read → tokenize (batch) → chunk → pad → numpy array
3. Main: concatenate all → shuffle → train/val split → save .pt files
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
_worker_chunk_size = 450
_worker_overlap = 50
_worker_min_chunk_size = 100
_worker_target_seq_length = 512
_worker_pad_token_id = 0


def chunk_text_with_overlap(
    text_tokens: List[int],
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> List[List[int]]:
    """Divide tokens into overlapping chunks."""
    chunks = []
    stride = chunk_size - overlap
    for i in range(0, len(text_tokens), stride):
        chunk = text_tokens[i: i + chunk_size]
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        if i + chunk_size >= len(text_tokens):
            break
    return chunks


def _init_worker(tokenizer_path, chunk_size, overlap, min_chunk_size,
                 target_seq_length, pad_token_id):
    """Initialize worker process: load tokenizer once."""
    global _worker_tokenizer, _worker_chunk_size, _worker_overlap
    global _worker_min_chunk_size, _worker_target_seq_length, _worker_pad_token_id
    _worker_tokenizer = GPT2Tokenizer(tokenizer_path)
    _worker_chunk_size = chunk_size
    _worker_overlap = overlap
    _worker_min_chunk_size = min_chunk_size
    _worker_target_seq_length = target_seq_length
    _worker_pad_token_id = pad_token_id


def _process_file_batch(file_paths: List[str]) -> Tuple[np.ndarray, dict]:
    """Worker: read + tokenize + chunk + pad a batch of files.

    Returns:
        (padded_chunks_array, stats_dict)
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
        return np.zeros((0, _worker_target_seq_length), dtype=np.int32), {
            "total_chars": 0, "total_tokens": 0, "total_files": 0,
        }

    # Batch tokenization (uses internal Rust parallelism)
    try:
        tokens_list = _worker_tokenizer.encode(texts, add_special_tokens=False)
    except Exception:
        tokens_list = [_worker_tokenizer.encode(t, add_special_tokens=False) for t in texts]

    # Chunk + pad → numpy
    padded_chunks = []
    total_tokens = 0
    for tokens in tokens_list:
        total_tokens += len(tokens)
        chunks = chunk_text_with_overlap(
            tokens,
            chunk_size=_worker_chunk_size,
            overlap=_worker_overlap,
            min_chunk_size=_worker_min_chunk_size,
        )
        for chunk in chunks:
            if len(chunk) >= _worker_target_seq_length:
                padded = chunk[:_worker_target_seq_length]
            else:
                padded = chunk + [_worker_pad_token_id] * (_worker_target_seq_length - len(chunk))
            padded_chunks.append(padded)

    if padded_chunks:
        arr = np.array(padded_chunks, dtype=np.int32)
    else:
        arr = np.zeros((0, _worker_target_seq_length), dtype=np.int32)

    return arr, {
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "total_files": len(texts),
    }


def process_all_files(
    data_dir: str,
    tokenizer_path: str,
    chunk_size: int = 450,
    overlap: int = 50,
    min_chunk_size: int = 100,
    target_seq_length: int = 512,
    pad_token_id: int = 0,
    num_workers: int = None,
    files_per_task: int = 200,
) -> Tuple[np.ndarray, dict]:
    """Parallel tokenization + chunking + padding.

    Returns:
        (all_padded_chunks, stats)
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
    print()

    # Split into tasks
    tasks = [
        [str(f) for f in txt_files[i:i + files_per_task]]
        for i in range(0, len(txt_files), files_per_task)
    ]

    chunk_arrays: List[np.ndarray] = []
    total_chars = 0
    total_tokens = 0
    total_files = 0

    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(tokenizer_path, chunk_size, overlap, min_chunk_size,
                  target_seq_length, pad_token_id),
    ) as pool:
        with tqdm(total=len(txt_files), desc="Processing", unit="file") as pbar:
            for arr, stats in pool.imap_unordered(_process_file_batch, tasks):
                if arr.shape[0] > 0:
                    chunk_arrays.append(arr)
                total_chars += stats["total_chars"]
                total_tokens += stats["total_tokens"]
                total_files += stats["total_files"]
                pbar.update(len(tasks[0]) if pbar.n + len(tasks[0]) <= len(txt_files)
                            else len(txt_files) - pbar.n)

    # Concatenate all batches
    print(f"\nConcatenating {len(chunk_arrays):,} batches...")
    if chunk_arrays:
        all_chunks_array = np.concatenate(chunk_arrays, axis=0)
        del chunk_arrays
    else:
        all_chunks_array = np.zeros((0, target_seq_length), dtype=np.int32)

    avg_chunk_size = float((all_chunks_array != pad_token_id).sum(axis=1).mean()) if len(all_chunks_array) else 0

    stats_out = {
        "total_files": total_files,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "total_chunks": int(len(all_chunks_array)),
        "avg_chunk_size": avg_chunk_size,
        "data_efficiency_multiplier": len(all_chunks_array) / total_files if total_files > 0 else 0,
    }
    return all_chunks_array, stats_out


def split_train_val(
    chunks: np.ndarray,
    train_split: float = 0.9,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle then split into train/val.

    Shuffle is critical: avoids alphabetical bias in validation.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(chunks))
    rng.shuffle(indices)
    split_idx = int(len(chunks) * train_split)
    return chunks[indices[:split_idx]], chunks[indices[split_idx:]]


def save_dataset(chunks: np.ndarray, output_path: str):
    """Save numpy array as torch tensor."""
    tensor = torch.from_numpy(chunks).long()
    torch.save(tensor, output_path)
    print(f"Saved {output_path}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Size: {tensor.element_size() * tensor.numel() / 1024 / 1024:.2f} MB")


def display_stats(stats: dict, train_chunks: np.ndarray, val_chunks: np.ndarray, target_seq_length: int):
    print("\n" + "=" * 80)
    print("CHUNKING RESULTS")
    print("=" * 80)
    print(f"Total files processed:          {stats['total_files']:,}")
    print(f"Total characters:               {stats['total_chars']:,}")
    print(f"Total tokens before chunking:   {stats['total_tokens']:,}")
    print(f"Total chunks created:           {stats['total_chunks']:,}")
    print(f"Average chunk size (non-pad):   {stats['avg_chunk_size']:.1f} tokens")
    print(f"Data efficiency multiplier:     {stats['data_efficiency_multiplier']:.1f}x")
    print()
    print(f"Train samples:                  {len(train_chunks):,}")
    print(f"Val samples:                    {len(val_chunks):,}")
    print(f"Train tensor shape:             ({len(train_chunks)}, {target_seq_length})")
    print(f"Val tensor shape:               ({len(val_chunks)}, {target_seq_length})")
    print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("GPT-2 DATA PREPARATION — INTELLIGENT CHUNKING (parallel + numpy)")
    print("=" * 80 + "\n")

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "gpt2_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    tokenizer_config = config["tokenizer"]

    # Resolve paths
    for key in ["data_dir", "train_data", "val_data"]:
        if not os.path.isabs(data_config[key]):
            data_config[key] = str(project_root / data_config[key])
    if not os.path.isabs(tokenizer_config["tokenizer_path"]):
        tokenizer_config["tokenizer_path"] = str(project_root / tokenizer_config["tokenizer_path"])

    chunk_size = data_config.get("chunk_size", 450)
    chunk_overlap = data_config.get("chunk_overlap", 50)
    min_chunk_size = data_config.get("min_chunk_size", 100)
    target_seq_length = data_config.get("target_seq_length", 512)
    train_split = data_config.get("train_split", 0.9)
    pad_token_id = tokenizer_config.get("pad_token_id", 0)

    print(f"Chunking strategy:")
    print(f"  Chunk size:         {chunk_size} tokens")
    print(f"  Overlap:            {chunk_overlap} tokens")
    print(f"  Stride:             {chunk_size - chunk_overlap} tokens")
    print(f"  Min chunk size:     {min_chunk_size} tokens")
    print(f"  Target seq length:  {target_seq_length} tokens")
    print(f"  Pad token ID:       {pad_token_id}")
    print()

    print(f"Tokenizer: {tokenizer_config['tokenizer_path']}")
    print(f"Data dir:  {data_config['data_dir']}")
    print()

    # Process (parallel)
    all_chunks, stats = process_all_files(
        data_dir=data_config["data_dir"],
        tokenizer_path=tokenizer_config["tokenizer_path"],
        chunk_size=chunk_size,
        overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        target_seq_length=target_seq_length,
        pad_token_id=pad_token_id,
    )

    print(f"\nTotal: {len(all_chunks):,} padded chunks "
          f"(memory: {all_chunks.nbytes / 1024 / 1024:.1f} MB as int32)")

    # Shuffle + split
    print(f"\nShuffling and splitting (train_split={train_split})...")
    train_chunks, val_chunks = split_train_val(all_chunks, train_split)
    del all_chunks  # free memory

    print(f"Train: {len(train_chunks):,} samples")
    print(f"Val:   {len(val_chunks):,} samples")

    # Save
    print("\nSaving datasets...")
    save_dataset(train_chunks, data_config["train_data"])
    save_dataset(val_chunks, data_config["val_data"])

    # Stats
    display_stats(stats, train_chunks, val_chunks, target_seq_length)
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()