"""Load raw text from .md, .txt and .pdf files."""

import re
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

# A word, a hyphen at the end of a line, then the next word: "op-\ntions"
LINE_BREAK_HYPHEN_RE = re.compile(r"(\w+)-[ \t]*\n[ \t]*(\w+)")
# A hyphenated compound written on one line: "cross-chain"
COMPOUND_RE = re.compile(r"\w+-\w+")


def join_hyphenated_line_breaks(text: str) -> str:
    """Undo hyphenation that the PDF layout added at line ends.

    "op-\\ntions" must become "options", otherwise BM25 sees two useless
    tokens "op" and "tions". But some line-end hyphens are real compounds
    ("cross-\\nchain" is "cross-chain"), and gluing them would make up a word.

    Rule: if the hyphenated form (e.g. "cross-chain") appears elsewhere in
    the same document, keep the hyphen. Otherwise join the two halves.
    Limitation: a real compound that appears only once, at a line end, gets
    joined by mistake. This is rare compared to the word splits we fix.
    """
    compounds_in_doc = {word.lower() for word in COMPOUND_RE.findall(text)}

    def fix(match: re.Match) -> str:
        first, second = match.group(1), match.group(2)
        if f"{first}-{second}".lower() in compounds_in_doc:
            return f"{first}-{second}"
        return first + second

    return LINE_BREAK_HYPHEN_RE.sub(fix, text)


def read_file(path: Path) -> str:
    """Return the text of one file. PDFs are read page by page with pypdf."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        # extract_text() can return None/"" for image-only pages.
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return join_hyphenated_line_breaks(text)
    return path.read_text(encoding="utf-8", errors="replace")


def load_documents(docs_dir: Path) -> list[dict]:
    """Return [{"source": filename, "text": ...}] for every supported file.

    Files are sorted by name so the order (and therefore every chunk id) is
    the same on every run and every operating system.
    """
    documents = []
    for path in sorted(Path(docs_dir).iterdir(), key=lambda p: p.name):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append({"source": path.name, "text": read_file(path)})
    return documents
