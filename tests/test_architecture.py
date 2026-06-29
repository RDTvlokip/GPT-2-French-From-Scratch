"""Test suite for GPT-2 architecture improvements.

Tests:
1. Model instantiation with all modern features
2. Forward pass correctness
3. Flash Attention vs manual attention equivalence
4. Gradient checkpointing VRAM savings
5. KV-cache generation
6. RoPE + GQA + SwiGLU + QK-Norm combined
7. Backward pass (gradient flow)
"""

import sys
import time
from pathlib import Path

# Anchor imports to <project_root>/src regardless of where this is run from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F
from model.gpt2 import GPT2, GPT2Config

device = "cuda" if torch.cuda.is_available() else "cpu"
PASS = "PASS"
FAIL = "FAIL"
results = []


def run_test(name, fn):
    """Run a test and record result."""
    try:
        fn()
        results.append((name, PASS, ""))
        print(f"  [{PASS}] {name}")
    except Exception as e:
        results.append((name, FAIL, str(e)))
        print(f"  [{FAIL}] {name}: {e}")


# =========================================================================
# Test 1: Legacy model (all features off)
# =========================================================================
def test_legacy_model():
    config = GPT2Config(vocab_size=256, n_positions=128, n_embd=64, n_layer=2, n_head=4, n_inner=256)
    model = GPT2(config).to(device)
    x = torch.randint(0, 256, (2, 32), device=device)
    logits, loss, hidden, _ = model(x, labels=x)
    assert logits.shape == (2, 32, 256), f"Bad logits shape: {logits.shape}"
    assert loss is not None and loss.item() > 0, "Loss should be positive"
    assert hidden.shape == (2, 32, 64), f"Bad hidden shape: {hidden.shape}"

print("=" * 70)
print("GPT-2 ARCHITECTURE TEST SUITE")
print("=" * 70)
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

print("[1/7] Basic model tests")
run_test("Legacy model (all off)", test_legacy_model)


# =========================================================================
# Test 2: Modern features individually
# =========================================================================
def test_rope():
    config = GPT2Config(vocab_size=256, n_positions=128, n_embd=64, n_layer=2, n_head=4, n_inner=256, use_rope=True)
    model = GPT2(config).to(device)
    x = torch.randint(0, 256, (2, 32), device=device)
    logits, loss, _, _ = model(x, labels=x)
    assert logits.shape == (2, 32, 256)
    assert loss.item() > 0

def test_rmsnorm():
    config = GPT2Config(vocab_size=256, n_positions=128, n_embd=64, n_layer=2, n_head=4, n_inner=256, use_rmsnorm=True)
    model = GPT2(config).to(device)
    x = torch.randint(0, 256, (2, 32), device=device)
    logits, loss, _, _ = model(x, labels=x)
    assert loss.item() > 0

def test_swiglu():
    config = GPT2Config(vocab_size=256, n_positions=128, n_embd=64, n_layer=2, n_head=4, n_inner=256, use_swiglu=True)
    model = GPT2(config).to(device)
    x = torch.randint(0, 256, (2, 32), device=device)
    logits, loss, _, _ = model(x, labels=x)
    assert loss.item() > 0

def test_qk_norm():
    config = GPT2Config(vocab_size=256, n_positions=128, n_embd=64, n_layer=2, n_head=4, n_inner=256, use_qk_norm=True)
    model = GPT2(config).to(device)
    x = torch.randint(0, 256, (2, 32), device=device)
    logits, loss, _, _ = model(x, labels=x)
    assert loss.item() > 0

def test_gqa():
    config = GPT2Config(vocab_size=256, n_positions=128, n_embd=64, n_layer=2, n_head=4, n_inner=256, use_gqa=True, n_kv_heads=2)
    model = GPT2(config).to(device)
    x = torch.randint(0, 256, (2, 32), device=device)
    logits, loss, _, _ = model(x, labels=x)
    assert loss.item() > 0

print("\n[2/7] Individual modern features")
run_test("RoPE", test_rope)
run_test("RMSNorm", test_rmsnorm)
run_test("SwiGLU", test_swiglu)
run_test("QK-Norm", test_qk_norm)
run_test("GQA (n_kv_heads=2)", test_gqa)


# =========================================================================
# Test 3: All modern features combined (LLaMA-style)
# =========================================================================
def test_all_modern():
    config = GPT2Config(
        vocab_size=256, n_positions=128, n_embd=64, n_layer=2, n_head=4, n_inner=256,
        use_rope=True, use_rmsnorm=True, use_swiglu=True, use_qk_norm=True,
        use_gqa=True, n_kv_heads=2,
        use_flash_attention=True,
    )
    model = GPT2(config).to(device)
    x = torch.randint(0, 256, (2, 32), device=device)
    logits, loss, _, _ = model(x, labels=x)
    assert logits.shape == (2, 32, 256)
    assert loss.item() > 0

print("\n[3/7] All modern features combined (LLaMA-style)")
run_test("RoPE + RMSNorm + SwiGLU + QK-Norm + GQA + Flash", test_all_modern)


# =========================================================================
# Test 4: Flash Attention vs Manual equivalence
# =========================================================================
def test_flash_vs_manual():
    torch.manual_seed(42)
    base = dict(vocab_size=256, n_positions=128, n_embd=64, n_layer=2, n_head=4, n_inner=256,
                use_rope=True, use_rmsnorm=True, use_qk_norm=True)

    # Model without flash
    config_manual = GPT2Config(**base, use_flash_attention=False)
    model_manual = GPT2(config_manual).to(device).eval()

    # Model with flash (same weights)
    config_flash = GPT2Config(**base, use_flash_attention=True)
    model_flash = GPT2(config_flash).to(device).eval()
    model_flash.load_state_dict(model_manual.state_dict())

    x = torch.randint(0, 256, (2, 32), device=device)
    with torch.no_grad():
        logits_manual, _, _, _ = model_manual(x)
        logits_flash, _, _, _ = model_flash(x)

    diff = (logits_manual - logits_flash).abs().max().item()
    assert diff < 1e-4, f"Flash vs Manual diff too large: {diff:.6f}"

print("\n[4/7] Flash Attention equivalence")
run_test("Flash vs Manual (max diff < 1e-4)", test_flash_vs_manual)


# =========================================================================
# Test 5: Gradient checkpointing
# =========================================================================
def test_gradient_checkpointing():
    config = GPT2Config(
        vocab_size=256, n_positions=128, n_embd=64, n_layer=4, n_head=4, n_inner=256,
        use_rope=True, use_rmsnorm=True, use_flash_attention=True,
    )
    model = GPT2(config).to(device)
    x = torch.randint(0, 256, (2, 64), device=device)

    # Without gradient checkpointing
    model.gradient_checkpointing_disable()
    logits, loss, _, _ = model(x, labels=x)
    loss.backward()
    grad_sum_no_ckpt = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)

    model.zero_grad()

    # With gradient checkpointing
    model.gradient_checkpointing_enable()
    logits2, loss2, _, _ = model(x, labels=x)
    loss2.backward()
    grad_sum_ckpt = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)

    # Gradients should be very close
    ratio = grad_sum_ckpt / grad_sum_no_ckpt if grad_sum_no_ckpt > 0 else 0
    assert 0.99 < ratio < 1.01, f"Gradient mismatch: ratio={ratio:.4f}"

print("\n[5/7] Gradient checkpointing")
run_test("Checkpointing produces same gradients", test_gradient_checkpointing)


# =========================================================================
# Test 6: KV-Cache generation
# =========================================================================
def test_kv_cache_generation():
    config = GPT2Config(
        vocab_size=256, n_positions=128, n_embd=64, n_layer=2, n_head=4, n_inner=256,
        use_rope=True, use_rmsnorm=True, use_qk_norm=True,
    )
    model = GPT2(config).to(device).eval()
    x = torch.randint(0, 256, (1, 8), device=device)

    # Generate with KV-cache
    out_cache = model.generate(x, max_length=20, temperature=1.0, do_sample=False, use_cache=True)
    assert out_cache.shape[1] == 28, f"Expected 28 tokens, got {out_cache.shape[1]}"

    # Generate without KV-cache
    torch.manual_seed(0)
    out_no_cache = model.generate(x, max_length=20, temperature=1.0, do_sample=False, use_cache=False)
    assert out_no_cache.shape[1] == 28, f"Expected 28 tokens, got {out_no_cache.shape[1]}"

    # Should produce identical output (greedy)
    match = (out_cache == out_no_cache).all().item()
    assert match, "KV-cache and no-cache should produce identical greedy output"

print("\n[6/7] KV-Cache generation")
run_test("Cache vs no-cache greedy match", test_kv_cache_generation)


# =========================================================================
# Test 7: VRAM and speed benchmarks (CUDA only)
# =========================================================================
def test_benchmark():
    if device != "cuda":
        print("  [SKIP] Benchmarks require CUDA")
        return

    configs = {
        "Legacy (no features)": GPT2Config(
            vocab_size=1000, n_positions=512, n_embd=256, n_layer=6, n_head=8, n_inner=1024,
        ),
        "Modern (all features)": GPT2Config(
            vocab_size=1000, n_positions=512, n_embd=256, n_layer=6, n_head=8, n_inner=1024,
            use_rope=True, use_rmsnorm=True, use_swiglu=True, use_qk_norm=True,
            use_flash_attention=True,
        ),
        "Modern + GradCkpt": GPT2Config(
            vocab_size=1000, n_positions=512, n_embd=256, n_layer=6, n_head=8, n_inner=1024,
            use_rope=True, use_rmsnorm=True, use_swiglu=True, use_qk_norm=True,
            use_flash_attention=True,
        ),
    }

    print(f"\n  {'Config':<25} {'Params':>8} {'VRAM (fwd+bwd)':>15} {'Time (10 iters)':>16}")
    print(f"  {'-'*25} {'-'*8} {'-'*15} {'-'*16}")

    for name, config in configs.items():
        model = GPT2(config).to(device)
        if "GradCkpt" in name:
            model.gradient_checkpointing_enable()

        params = f"{model.get_num_params()/1e6:.1f}M"
        x = torch.randint(0, 1000, (4, 256), device=device)

        # Warmup
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        logits, loss, _, _ = model(x, labels=x)
        loss.backward()
        model.zero_grad()

        # Benchmark
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(10):
            logits, loss, _, _ = model(x, labels=x)
            loss.backward()
            model.zero_grad()

        torch.cuda.synchronize()
        elapsed = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        print(f"  {name:<25} {params:>8} {peak_mem:>12.1f} MB {elapsed:>13.2f} s")

        del model
        torch.cuda.empty_cache()

print("\n[7/7] Benchmarks (CUDA)")
test_benchmark()


# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
print(f"Passed: {passed}/{len(results)}")
if failed:
    print(f"Failed: {failed}/{len(results)}")
    for name, status, err in results:
        if status == FAIL:
            print(f"  - {name}: {err}")
else:
    print("All tests passed!")
print("=" * 70)
