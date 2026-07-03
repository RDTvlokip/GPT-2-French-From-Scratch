"""Extrait les pages web crawlées (.html.gz) en markdown propre pour le corpus.

Pipeline : décompresse -> rdtextract.extract (HTML->markdown) -> filtre qualité
-> écrit dans data/ (à côté des .txt Wikipédia, pour fusion).

Le filtre qualité va PLUS LOIN que is_low_value_stub : il rejette aussi le
marketing creux, les pages cookies/mentions légales, les pages produit sans
texte rédigé. Critères validés sur échantillon :
- non low-value stub (paywall/login/vide)
- longueur >= MIN_CHARS
- content_ratio >= MIN_CONTENT_RATIO (part de lignes "vrai paragraphe" > 60c)
- nb de paragraphes >= MIN_PARAGRAPHS
- pas trop de marqueurs boilerplate (pages cookies/nav)

Parallélisé. Idempotent (skip si le .md de sortie existe déjà).

Usage:
    python scripts/extract_web.py --crawler-dir D:/Python/Crawler/data --out data
    python scripts/extract_web.py ... --sample 25   # test sur un échantillon
"""

import os
import gzip
import glob
import hashlib
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import rdtextract

# --- Seuils du filtre qualité (validés sur échantillon) ---------------------- #
MIN_CHARS = 500
MIN_CONTENT_RATIO = 0.40   # part de lignes > 60 chars
MIN_PARAGRAPHS = 5         # nb de lignes > 60 chars
MAX_BOILERPLATE = 10       # au-delà = page cookies/mentions/nav

_BOILERPLATE = [
    "aller au contenu", "envoyer la page", "imprimer le contenu", "accès rapides",
    "vous êtes ici", "plan du site", "mentions légales", "politique de confidentialité",
    "gestion des cookies", "paramétrer les cookies", "suivez-nous", "newsletter",
    "tous droits réservés", "préférences cookies",
]


def quality_ok(md: str) -> bool:
    """Filtre qualité au-delà de is_low_value_stub."""
    if not md or len(md) < MIN_CHARS:
        return False
    if rdtextract.is_low_value_stub(md):
        return False
    low = md.lower()
    if sum(low.count(m) for m in _BOILERPLATE) > MAX_BOILERPLATE:
        return False
    lines = [l for l in md.split("\n") if l.strip()]
    long_lines = [l for l in lines if len(l) > 60]
    if len(long_lines) < MIN_PARAGRAPHS:
        return False
    if len(long_lines) / max(len(lines), 1) < MIN_CONTENT_RATIO:
        return False
    return True


def _safe_name(path: str) -> str:
    """domaine__page + hash court du chemin complet, comme nom de sortie.

    Le hash garantit l'unicité : deux pages aux noms longs/similaires
    (tronqués à 130c) ne collisionnent jamais, donc aucune perte de page.
    """
    fn = path.replace("\\", "/")
    parts = fn.split("/")
    domain = parts[-2] if len(parts) >= 2 else "unknown"
    page = parts[-1].replace(".html.gz", "")
    name = f"{domain}__{page}"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:130]
    h = hashlib.md5(fn.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{h}"


_OUT_DIR = None


def _init(out_dir):
    global _OUT_DIR
    _OUT_DIR = out_dir


def _process(path: str) -> int:
    """Retourne 1 si gardé (écrit), 0 sinon. Ne lève jamais (workers protégés)."""
    try:
        out_path = Path(_OUT_DIR) / f"web_{_safe_name(path)}.txt"
        if out_path.exists():
            return 1
        html = gzip.open(path, "rt", encoding="utf-8", errors="replace").read()
        md = rdtextract.extract(html) or ""
        if not quality_ok(md):
            return 0
        out_path.write_text(md, encoding="utf-8")
        return 1
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description="Extract crawled web pages to markdown")
    ap.add_argument("--crawler-dir", default="D:/Python/Crawler/data")
    ap.add_argument("--out", default="data", help="dossier de sortie (corpus)")
    ap.add_argument("--sample", type=int, default=0, help="ne traiter qu'un échantillon (test)")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.crawler_dir, "*", "*.html.gz"))
    print(f"Fichiers .html.gz trouvés : {len(files):,}")
    if args.sample:
        import random
        random.seed(42)
        files = random.sample(files, min(args.sample, len(files)))
        print(f"Échantillon : {len(files)}")

    os.makedirs(args.out, exist_ok=True)
    kept = 0
    done = 0
    with Pool(args.workers, initializer=_init, initargs=(args.out,)) as pool:
        try:
            for r in pool.imap_unordered(_process, files, chunksize=100):
                kept += r
                done += 1
                if done % 10000 == 0 or done == len(files):
                    print(f"  {done:,}/{len(files):,} traités | gardés {kept:,} ({100*kept//max(done,1)}%)")
        except KeyboardInterrupt:
            print("\nInterrompu — terminating workers...")
            pool.terminate(); pool.join()
            print(f"Stoppé après {done:,} ({kept:,} gardés). Idempotent : relancer reprend.")
            return

    print(f"\nTerminé : {kept:,} pages gardées sur {len(files):,} "
          f"({100*kept//max(len(files),1)}%) -> {args.out}/web_*.txt")


if __name__ == "__main__":
    main()
