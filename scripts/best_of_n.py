"""Best-of-N decoding : générer N continuations, garder la meilleure selon une
métrique de re-scoring.

Pour chaque prompt :
  1. générer N continuations (sampling, réglage optimal, seeds différents)
  2. re-scorer chacune par self-perplexity (le modèle juge sa propre cohérence)
  3. garder celle de plus basse self_ppl

Exploite directement self_ppl (métrique anti-gaming) comme critère de
sélection. Zéro re-train. On compare Best-of-N (N=4,8) à la baseline (N=1) sur
le harnais complet — y compris les métriques anti-gaming, pour vérifier que le
gain est réel et pas un gaming de plus.

Usage:
    python scripts/best_of_n.py --model models/best_model.pt --n 8
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

from model.gpt2 import GPT2
from utils.tokenizer import GPT2Tokenizer
from scripts.evaluate import (
    repetition_3gram, distinct_2, coherence_len, build_prompts,
    self_perplexity, proper_noun_ratio, prompt_overlap, sentence_burstiness,
)


@torch.no_grad()
def sample_one(model, tokenizer, prompt_ids, device, max_new,
               temp=0.8, top_k=40, rep_pen=1.3, ban_bos=True):
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    n_pos = model.config.n_positions
    gen = []
    for _ in range(max_new):
        cond = ids if ids.size(1) <= n_pos else ids[:, -n_pos:]
        logits = model(cond)[0][:, -1, :]
        if ban_bos:
            logits[:, tokenizer.bos_token_id] = float("-inf")
        if rep_pen != 1.0:
            for t in set(ids[0].tolist()):
                v = logits[0, t]
                logits[0, t] = v / rep_pen if v > 0 else v * rep_pen
        logits = logits / temp
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[logits < kth] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        tid = nxt.item()
        if tid == tokenizer.eos_token_id:
            break
        gen.append(tid)
        ids = torch.cat([ids, nxt], dim=1)
    return gen


@torch.no_grad()
def best_of_n(model, tokenizer, prompt_ids, device, max_new, n):
    """Génère n continuations, garde celle de plus basse self_ppl. n=1 = baseline."""
    best, best_score = None, float("inf")
    for _ in range(n):
        gen = sample_one(model, tokenizer, prompt_ids, device, max_new)
        if not gen:
            continue
        sp = self_perplexity(model, gen, device)
        if sp is None:
            sp = float("inf")
        if sp < best_score:
            best, best_score = gen, sp
    return best or []


def eval_n(model, tokenizer, prompts, device, drift, max_new, n, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cohs, spp, pnr, ovl = [], [], [], []
    for pids in prompts:
        g = best_of_n(model, tokenizer, pids, device, max_new, n)
        if not g:
            continue
        cohs.append(coherence_len(g, drift))
        sp = self_perplexity(model, g, device)
        if sp:
            spp.append(sp)
        gt = tokenizer.decode(g, skip_special_tokens=True)
        pt = tokenizer.decode(pids, skip_special_tokens=True)
        pnr.append(proper_noun_ratio(gt))
        ovl.append(prompt_overlap(pt, gt))
    nz = lambda xs: round(sum(xs) / len(xs), 3) if xs else 0.0
    return nz(cohs), nz(spp), nz(pnr), nz(ovl)


def main():
    ap = argparse.ArgumentParser(description="Best-of-N decoding")
    ap.add_argument("--model", default="models/best_model.pt")
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "bpe_tokenizer_32k.json"))
    ap.add_argument("--val", default=str(PROJECT_ROOT / "data" / "val.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--ns", default="1,4,8", help="valeurs de N à comparer (1 = baseline)")
    args = ap.parse_args()

    mp = args.model if Path(args.model).is_absolute() else str(PROJECT_ROOT / args.model)
    tokenizer = GPT2Tokenizer(args.tokenizer)
    model = GPT2.from_pretrained(mp, device=args.device); model.eval()

    drift = set()
    for s in ["#", " #", "##", " ##"]:
        for tid in tokenizer.encode(s, add_special_tokens=False):
            drift.add(tid)

    auto, fixed = build_prompts(tokenizer, args.val, args.n_prompts)
    prompts = auto + fixed
    print(f"Best-of-N on {len(prompts)} prompts | re-scoring = self_ppl (lower better)\n")

    print(f"{'N':<8}{'coh↑':>8}{'selfPPL↓':>10}{'propN↓':>9}{'overlap↑':>10}")
    print("-" * 45)
    for n in [int(x) for x in args.ns.split(",")]:
        c, s, p, o = eval_n(model, tokenizer, prompts, args.device, drift, args.max_new, n)
        tag = f"{n}" + (" (base)" if n == 1 else "")
        print(f"{tag:<8}{c:>8}{s:>10}{p:>9}{o:>10}")

    print("\nN=1 = baseline (single sample). Higher N = pick best of N by self_ppl.")
    print("Watch self_ppl: that's the selection criterion, so it SHOULD drop —")
    print("the real question is whether coh/overlap improve too (real gain).")


if __name__ == "__main__":
    main()
