"""DoLa : Decoding by Contrasting Layers (Chuang et al. 2023), version simple.

Au lieu de contraster deux modèles, on contraste les COUCHES d'un seul modèle :
les couches finales portent plus de connaissance que les couches précoces.
   logits_dola = logits_finale - λ * logits_couche_precoce   (sur V_valid)
avec un plausibility constraint (α-masking) basé sur la couche finale.

Réserve : le papier vise de gros modèles (32+ couches). Ici 8 couches -> la
divergence finale/précoce est faible. On teste quand même, et on balaie la
couche de contraste {2,4,6} pour voir si l'une aide (ou aucune).

Usage:
    python scripts/dola_decode.py --model models/best_model.pt
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
)


@torch.no_grad()
def dola_generate(model, tokenizer, prompt_ids, device, max_new,
                  early_layer, lam, alpha, temperature=0.8, top_k=40,
                  rep_pen=1.3, ban_bos=True):
    """Génère avec DoLa. early_layer=None -> baseline (couche finale seule)."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    n_pos = model.config.n_positions
    gen = []
    for _ in range(max_new):
        cond = ids if ids.size(1) <= n_pos else ids[:, -n_pos:]
        if early_layer is None:
            log_final = model(cond)[0][:, -1, :]
            scores = log_final
        else:
            outs = model.forward_layer_logits(cond, layers=[early_layer])
            log_final = outs[-1][:, -1, :]
            log_early = outs[early_layer][:, -1, :]
            probs_f = F.softmax(log_final, dim=-1)
            thresh = alpha * probs_f.max(dim=-1, keepdim=True).values
            valid = probs_f >= thresh
            scores = (1.0 + lam) * log_final - lam * log_early
            scores = scores.masked_fill(~valid, float("-inf"))

        if ban_bos:
            scores[:, tokenizer.bos_token_id] = float("-inf")
        # repetition penalty (HF) — on garde le bon réglage de sampling trouvé
        if rep_pen != 1.0:
            for t in set(ids[0].tolist()):
                v = scores[0, t]
                scores[0, t] = v / rep_pen if v > 0 else v * rep_pen
        scores = scores / temperature
        if top_k:
            kth = torch.topk(scores, min(top_k, scores.size(-1)))[0][..., -1, None]
            scores[scores < kth] = float("-inf")
        nxt = torch.multinomial(F.softmax(scores, dim=-1), 1)
        tid = nxt.item()
        if tid == tokenizer.eos_token_id:
            break
        gen.append(tid)
        ids = torch.cat([ids, nxt], dim=1)
    return gen


def eval_cfg(model, tokenizer, prompts, device, drift, max_new,
             early_layer, lam, alpha, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    reps, dists, cohs = [], [], []
    for pids in prompts:
        g = dola_generate(model, tokenizer, pids, device, max_new,
                          early_layer, lam, alpha)
        if not g:
            continue
        reps.append(repetition_3gram(g))
        dists.append(distinct_2(g))
        cohs.append(coherence_len(g, drift))
    nz = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return round(nz(reps), 4), round(nz(dists), 4), round(nz(cohs), 1)


def main():
    ap = argparse.ArgumentParser(description="DoLa decoding (contrast layers)")
    ap.add_argument("--model", default="models/best_model.pt")
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "bpe_tokenizer_32k.json"))
    ap.add_argument("--val", default=str(PROJECT_ROOT / "data" / "val.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=90)
    ap.add_argument("--layers", default="2,4,6", help="couches précoces à tester")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.1)
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
    print(f"DoLa on {len(prompts)} prompts | λ={args.lam} α={args.alpha}\n")

    print(f"{'config':<22}{'rep3↓':>9}{'dist2↑':>9}{'coh↑':>8}")
    print("-" * 48)
    r, d, c = eval_cfg(model, tokenizer, prompts, args.device, drift,
                       args.max_new, None, 0, 0)
    print(f"{'baseline (final)':<22}{r:>9}{d:>9}{c:>8}")
    base = (r, d, c)

    for layer in [int(x) for x in args.layers.split(",")]:
        r, d, c = eval_cfg(model, tokenizer, prompts, args.device, drift,
                           args.max_new, layer, args.lam, args.alpha)
        mark = ""
        if r < base[0]:
            mark += " rep↓"
        if c > base[2]:
            mark += " coh↑"
        print(f"{f'DoLa early=L{layer}':<22}{r:>9}{d:>9}{c:>8}{mark}")

    print("\nrep3: lower=better | dist2/coh: higher=better")
    print("Baseline = couche finale seule (décodage normal).")


if __name__ == "__main__":
    main()
