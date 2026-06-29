import os
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm

def count_tokens_in_directory(data_dir: str, tokenizer_path: str) -> dict:
    """Count total tokens in all .txt files in a directory."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    data_path = Path(data_dir)
    txt_files = list(data_path.glob("*.txt"))
    
    if not txt_files:
        return {"total_tokens": 0, "total_files": 0, "failed_files": 0, "total_chars": 0}
    
    total_tokens = 0
    total_chars = 0
    processed_files = 0
    failed_files = []

    for txt_file in tqdm(txt_files, desc=f"Counting {data_dir}", leave=False):
        try:
            with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            if not text.strip():
                continue

            encoded = tokenizer.encode(text)
            total_tokens += len(encoded.ids)
            total_chars += len(text)
            processed_files += 1

        except Exception as e:
            failed_files.append((str(txt_file), str(e)))
            continue

    if failed_files:
        print(f"\n⚠ {len(failed_files)} file(s) failed in {data_dir}:")
        for fp, err in failed_files:
            print(f"  - {fp}: {err}")

    return {
        "total_tokens": total_tokens,
        "total_files": processed_files,
        "failed_files": len(failed_files),
        "total_chars": total_chars
    }

if __name__ == "__main__":
    # Anchor paths to project root so this works from any working directory.
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    tokenizer_path = str(_PROJECT_ROOT / "bpe_tokenizer_32k.json")

    # Count tokens for each directory
    data_results = count_tokens_in_directory(str(_PROJECT_ROOT / "data"), tokenizer_path)
    
    # Calculate totals
    total_tokens = data_results["total_tokens"]
    total_files = data_results["total_files"]
    total_chars = data_results["total_chars"]
    
    # Display results with tree structure
    print("\n" + "="*70)
    print("📊 TOKEN COUNT ANALYSIS")
    print("="*70 + "\n")
    
    print("📁 data")
    print(f"├─ 📊 TOTAL: {data_results['total_tokens']:,} tokens")
    print(f"├─ 📄 FILES: {data_results['total_files']} processed, {data_results['failed_files']} failed")
    print(f"└─ 💾 CHARS: {data_results['total_chars']:,}\n")
    
    print("="*70)
    print("📈 COMBINED TOTAL")
    print("="*70)
    print(f"📊 TOTAL: {total_tokens:,} tokens")
    print(f"📄 FILES: {total_files}")
    print(f"💾 CHARS: {total_chars:,}")
    print("="*70 + "\n")