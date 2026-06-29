#!/usr/bin/env python
"""
Quick test script to verify generation and dashboard logging work
"""

import subprocess
import sys
import json
from pathlib import Path

# Project root = parent of this tests/ directory. All paths anchor here so the
# test works regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_generation():
    print("=" * 80)
    print("Testing GPT-2 Generation with Dashboard Integration")
    print("=" * 80)

    # Test prompts
    prompts = [
        "Le cache LRU est",
        "Le protocole TCP",
        "L'intelligence artificielle",
    ]

    for prompt in prompts:
        print(f"\nGenerating from: '{prompt}'")
        print("-" * 80)

        # Run generation
        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "src" / "generate.py"),
             "--prompt", prompt,
             "--max_length", "100",
             "--temperature", "0.3"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT
        )

        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            continue

        # Check if JSON was created
        json_file = _PROJECT_ROOT / "logs" / "generation_output.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)

            print(f"[OK] JSON created with {len(data.get('tokens', []))} tokens")
            print(f"Generated text: {data.get('text', '')[:100]}...")
        else:
            print("[FAIL] JSON file not created")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_generation()
