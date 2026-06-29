import sys
from pathlib import Path

# Anchor to project root so `src.utils...` imports resolve from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_PROJECT_ROOT))

from src.utils.tokenizer import GPT2Tokenizer

def test_tokenizer(tokenizer_path: str):
    """
    Tests the GPT2Tokenizer with various inputs and prints the results.

    Args:
        tokenizer_path: Path to the BPE tokenizer JSON file.
    """
    print("=" * 80)
    print("TESTING GPT2TOKENIZER")
    print("=" * 80)

    try:
        tokenizer = GPT2Tokenizer(tokenizer_path)
        print(f"Tokenizer loaded successfully from: {tokenizer_path}")
        print(f"Vocabulary size: {tokenizer.vocab_size:,}")
        print(f"Special tokens: PAD={tokenizer.pad_token_id}, UNK={tokenizer.unk_token_id}, BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    test_cases = [
        "Hello, world! This is a test sentence.",
        "GPT-2 is a powerful language model.",
        "人工智能的未来是光明的。", # Chinese characters
        "Ceci est une phrase en français.", # French characters
        "Short.",
        "A very long sentence that should be truncated if max_length is set, demonstrating how the tokenizer handles longer inputs and potentially truncates them to fit within a specified maximum length. This helps in understanding the behavior of the tokenizer when dealing with text that exceeds the model's context window.",
        "", # Empty string
        "   leading and trailing spaces   ",
        "Multiple   spaces   between   words",
        "New\nline\ncharacters",
        "Special tokens: <bos> <eos> <unk> <pad>",
        "Emoji test : 😀🚀🌟",
        "You can't handle the truth!",
        "E=mc^2 is Einstein's famous equation.",
    ]

    print("\n--- Test Cases ---")
    for i, text in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Input: '{text}'")

        # Encode the text
        encoded_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=50) # Limit for display
        print(f"Encoded IDs (first 50 tokens): {encoded_ids}")
        print(f"Number of tokens: {len(encoded_ids)}")

        # Decode the tokens back
        decoded_text = tokenizer.decode(encoded_ids, skip_special_tokens=True)
        print(f"Decoded: '{decoded_text}'")

if __name__ == "__main__":
    tokenizer_file = str(_PROJECT_ROOT / "bpe_tokenizer_32k.json")
    test_tokenizer(tokenizer_file)