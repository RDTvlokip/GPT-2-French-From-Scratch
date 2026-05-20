"""Verify that the GPT-2 project is set up correctly."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def check_imports():
    """Check that all required imports work."""
    print("Checking imports...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False

    try:
        import yaml
        print("✓ PyYAML")
    except ImportError:
        print("✗ PyYAML not installed")
        return False

    try:
        from tokenizers import Tokenizer
        print("✓ Tokenizers")
    except ImportError:
        print("✗ Tokenizers not installed")
        return False

    try:
        from tqdm import tqdm
        print("✓ tqdm")
    except ImportError:
        print("✗ tqdm not installed")
        return False

    return True


def check_files():
    """Check that all required files exist."""
    print("\nChecking project files...")

    required_files = [
        "config/gpt2_config.yaml",
        "src/model/gpt2.py",
        "src/utils/tokenizer.py",
        "src/train.py",
        "src/generate.py",
        "scripts/prepare_data.py",
        "bpe_tokenizer_32k.json",
        "requirements.txt",
        "README.md",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} not found")
            all_exist = False

    return all_exist


def check_model():
    """Check that the model can be instantiated."""
    print("\nChecking model instantiation...")

    try:
        from src.model.gpt2 import GPT2, GPT2Config

        config = GPT2Config(
            vocab_size=32000,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=3072,
        )

        model = GPT2(config)
        print(f"✓ Model created successfully")
        print(f"  Parameters: {model.get_num_params() / 1e6:.2f}M")
        return True

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def check_tokenizer():
    """Check that the tokenizer can be loaded."""
    print("\nChecking tokenizer...")

    try:
        from src.utils.tokenizer import GPT2Tokenizer

        tokenizer = GPT2Tokenizer("bpe_tokenizer_32k.json")
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocabulary size: {tokenizer.vocab_size:,}")

        # Test encoding/decoding
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"  Test encoding: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")

        return True

    except Exception as e:
        print(f"✗ Tokenizer loading failed: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 80)
    print("GPT-2 Project Setup Verification")
    print("=" * 80 + "\n")

    checks = [
        ("Imports", check_imports),
        ("Files", check_files),
        ("Model", check_model),
        ("Tokenizer", check_tokenizer),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    all_passed = True
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
        if not result:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n🎉 All checks passed! Your GPT-2 project is ready.")
        print("\nNext steps:")
        print("1. Add your training data (.txt files) to the data/ directory")
        print("2. Run: python scripts/prepare_data.py")
        print("3. Run: cd src && python train.py")
        print("4. Run: cd src && python generate.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("Run: pip install -r requirements.txt")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
