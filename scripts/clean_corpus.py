"""Clean the .txt corpus in place: strip emojis, normalize spaces, fix quotes.

What it does (and deliberately does NOT do):
- Protects code: anything inside ``` ... ``` fences or `inline` backticks is
  left 100% untouched (so HTML/JSON/code quotes like class="x" survive).
- Normalizes special spaces (narrow no-break U+202F, no-break U+00A0, etc.)
  to a normal space.
- Removes decorative emojis (about a third of the corpus is polluted).
- Converts straight double quotes " into French « » by pairing them
  (1st of a pair -> «, 2nd -> »). A lone unpaired quote is left as-is.
- Trims spaces before . , ; ! ? and collapses double spaces / blank lines.

Edits files in place (a .rar backup of data/ was made beforehand).
Parallelized over files.
"""

import os
import re
import sys
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Special whitespace chars to normalize to a plain space.
_SPECIAL_SPACES = "              　﻿"
_SPACE_MAP = {ord(c): " " for c in _SPECIAL_SPACES}

_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # main emoji planes
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U0001F1E0-\U0001F1FF"   # regional indicators (flags)
    "\U00002190-\U000021FF"   # arrows
    "\U00002300-\U000023FF"   # technical (hourglass, watch...)
    "\U00002B00-\U00002BFF"   # arrows/stars supplement
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0000200D"              # zero-width joiner
    "]+",
    flags=re.UNICODE,
)


def _convert_quotes(text: str) -> str:
    """Pair straight double quotes into « / ». Lone trailing quote stays as-is."""
    n = text.count('"')
    if n == 0:
        return text
    out = []
    open_q = True
    remaining = n
    for ch in text:
        if ch == '"':
            # If an odd count, leave the very last (orphan) quote untouched.
            if remaining == 1 and n % 2 == 1:
                out.append('"')
            else:
                out.append("«" if open_q else "»")
                open_q = not open_q
            remaining -= 1
        else:
            out.append(ch)
    return "".join(out)


def clean_text(t: str) -> str:
    # 1) Protect code (fenced blocks first, then inline) by stashing them.
    code = []

    def _stash(m):
        code.append(m.group(0))
        return f"\x00C{len(code) - 1}\x00"

    t = re.sub(r"```.*?```", _stash, t, flags=re.DOTALL)
    t = re.sub(r"`[^`\n]+`", _stash, t)

    # 2) Clean the non-code text.
    t = t.translate(_SPACE_MAP)
    t = _EMOJI_RE.sub("", t)
    t = _convert_quotes(t)
    t = re.sub(r"«\s+", "« ", t)
    t = re.sub(r"\s+»", " »", t)
    t = re.sub(r" +([,.;!?])", r"\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r" +\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # 3) Restore code untouched.
    for i, blk in enumerate(code):
        t = t.replace(f"\x00C{i}\x00", blk)
    return t


def _process_file(path: str) -> dict:
    """Clean one file in place. Returns simple stats."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
    except Exception:
        return {"files": 0, "changed": 0, "chars_removed": 0}

    cleaned = clean_text(original)
    if cleaned != original:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(cleaned)
        except Exception:
            return {"files": 1, "changed": 0, "chars_removed": 0}
        return {"files": 1, "changed": 1, "chars_removed": len(original) - len(cleaned)}
    return {"files": 1, "changed": 0, "chars_removed": 0}


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    files = sorted(glob.glob(str(data_dir / "*.txt")))
    if not files:
        print(f"No .txt files found in {data_dir}")
        return

    workers = max(1, cpu_count() - 1)
    print("=" * 70)
    print("CORPUS CLEANING (in place)")
    print("=" * 70)
    print(f"Files:   {len(files):,}")
    print(f"Workers: {workers}")
    print("Protecting code blocks, stripping emojis, normalizing quotes/spaces.")
    print()

    totals = {"files": 0, "changed": 0, "chars_removed": 0}
    done = 0
    pool = Pool(workers)
    try:
        for st in pool.imap_unordered(_process_file, files, chunksize=200):
            for k in totals:
                totals[k] += st[k]
            done += 1
            if done % 5000 == 0 or done == len(files):
                print(f"  {done:,}/{len(files):,} processed...")
    except KeyboardInterrupt:
        # Ctrl+C: stop workers immediately instead of hanging on pool exit.
        print("\nInterrupted — terminating workers...")
        pool.terminate()
        pool.join()
        print(f"Stopped after {done:,}/{len(files):,} files (cleaning is "
              f"idempotent: just re-run to finish).")
        return
    else:
        pool.close()
        pool.join()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Files processed:   {totals['files']:,}")
    print(f"Files changed:     {totals['changed']:,} "
          f"({100 * totals['changed'] / max(totals['files'], 1):.1f}%)")
    print(f"Characters removed: {totals['chars_removed']:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
