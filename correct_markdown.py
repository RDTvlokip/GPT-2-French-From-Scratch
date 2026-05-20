"""Markdown post-processor for training data.

Normalizes:
- Quotes/apostrophes (curly → straight)
- List indentation (multiples of 4 spaces)
- Markdown tables (uniform formatting)
"""

import re
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# Quote/apostrophe normalization map
# (only real replacements — no-ops removed)
QUOTE_MAP = {
    "‘": "'",  # ‘ left single quote
    "’": "'",  # ’ right single quote / typographic apostrophe
    "‚": "'",  # ‚ low single quote
    "‛": "'",  # ‛ high reversed single quote
    "“": '"',  # “ left double quote
    "”": '"',  # ” right double quote
    "„": '"',  # „ low double quote
    "‟": '"',  # ‟ high reversed double quote
    "′": "'",  # ′ prime
    "″": '"',  # ″ double prime
    "ʹ": "'",  # ʹ modifier letter prime
    "ʻ": "'",  # ʻ modifier letter turned comma
    "ʼ": "'",  # ʼ modifier letter apostrophe
    "«": '"',  # « left guillemet
    "»": '"',  # » right guillemet
}
QUOTE_TRANS = str.maketrans(QUOTE_MAP)


# Markdown table patterns
TABLE_ROW_RE = re.compile(r'^\s*\|.*\|\s*$')
SEPARATOR_RE = re.compile(r'^\s*\|\s*[-:]+\s*(?:\|\s*[-:]+\s*)*\|\s*$')

# List patterns
NESTED_LIST_RE = re.compile(r'^(\s+)([\*\-]|\d+\.)\s+')
STAR_LIST_RE = re.compile(r'\*\s{2,}')
DASH_LIST_RE = re.compile(r'^-\s{2,}')
NUMBERED_LIST_RE = re.compile(r'^(\d+)\.\s{2,}')


def fix_quotes(content: str) -> str:
    """Normalize all curly quotes/apostrophes to ASCII versions."""
    return content.translate(QUOTE_TRANS)


def fix_markdown_lists(content: str) -> str:
    """Normalize list indentation to multiples of 4 spaces."""
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        if line.strip() == '---':
            fixed_lines.append(line)
            continue

        match = NESTED_LIST_RE.match(line)
        if match:
            indent = match.group(1)
            marker = match.group(2)
            rest = line[len(match.group(0)):]
            indent_count = len(indent)
            if indent_count > 0 and indent_count % 4 != 0:
                level = (indent_count + 3) // 4
                line = '    ' * level + marker + ' ' + rest
            else:
                line = STAR_LIST_RE.sub('* ', line)
                line = DASH_LIST_RE.sub('- ', line)
                line = NUMBERED_LIST_RE.sub(r'\1. ', line)
        else:
            line = STAR_LIST_RE.sub('* ', line)
            line = DASH_LIST_RE.sub('- ', line)
            line = NUMBERED_LIST_RE.sub(r'\1. ', line)

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def parse_row(line: str) -> List[str]:
    cells = line.strip().split('|')[1:-1]
    return [cell.strip() for cell in cells]


def find_tables(lines: List[str]) -> List[Tuple[int, int]]:
    """Find (start, end) indices of markdown tables."""
    tables = []
    i = 0
    n = len(lines)

    while i < n - 1:
        if TABLE_ROW_RE.match(lines[i].strip()) and SEPARATOR_RE.match(lines[i + 1].strip()):
            start = i
            i += 2
            while i < n and TABLE_ROW_RE.match(lines[i].strip()):
                i += 1
            if i > start + 2:
                tables.append((start, i))
            continue
        i += 1

    return tables


def convert_table(table_lines: List[str]) -> List[str]:
    """Reformat a markdown table with uniform spacing."""
    if len(table_lines) < 3:
        return table_lines

    try:
        header = parse_row(table_lines[0])
        if not header:
            return table_lines

        data_rows = [parse_row(line) for line in table_lines[2:]]
        data_rows = [r for r in data_rows if r]

        col_count = len(header)
        for row in data_rows:
            while len(row) < col_count:
                row.append('')
            if len(row) > col_count:
                row[:] = row[:col_count]

        new_header = f"| {' | '.join(header)} |"
        separator = f"| {' | '.join(['-'] * col_count)} |"
        new_data = [f"| {' | '.join(row)} |" for row in data_rows]
        return [new_header, separator] + new_data
    except (IndexError, ValueError):
        return table_lines


def process_file(file_path: Path) -> bool:
    """Process a single markdown file. Returns True on success."""
    try:
        content = file_path.read_text(encoding='utf-8', errors='replace')
        if not content:
            return True

        original = content
        content = fix_quotes(content)
        content = fix_markdown_lists(content)
        lines = content.split('\n')

        tables = find_tables(lines)
        if tables:
            for start, end in reversed(tables):
                lines[start:end] = convert_table(lines[start:end])

        new_content = '\n'.join(lines)

        # Only write if changed (faster, less disk wear)
        if new_content != original:
            file_path.write_text(new_content, encoding='utf-8')

        return True
    except (IOError, OSError, UnicodeError):
        return False


def _worker(path_str: str) -> bool:
    return process_file(Path(path_str))


def process_all_files(data_folder: str = 'data', num_workers: int = None) -> None:
    folder = Path(data_folder)
    if not folder.exists():
        print(f"❌ Le dossier '{folder}' n'existe pas.")
        return

    txt_files = list(folder.glob('*.txt'))
    if not txt_files:
        print(f"❌ Aucun fichier .txt trouvé dans '{folder}'.")
        return

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"📂 {len(txt_files):,} fichiers à traiter")
    print(f"⚙️  {num_workers} workers parallèles")
    print()

    paths = [str(p) for p in txt_files]
    success = 0
    failed = 0

    with Pool(num_workers) as pool:
        for ok in tqdm(
            pool.imap_unordered(_worker, paths, chunksize=50),
            total=len(paths),
            desc="📝 Correction",
            unit="file",
        ):
            if ok:
                success += 1
            else:
                failed += 1

    print()
    print(f"✅ {success:,} fichiers traités")
    if failed:
        print(f"❌ {failed:,} échecs")


def main():
    process_all_files(data_folder='data')


if __name__ == "__main__":
    main()
