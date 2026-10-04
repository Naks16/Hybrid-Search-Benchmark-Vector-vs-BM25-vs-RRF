"""Command line search.

Run: python -m search.query --mode hybrid --q "your question" [--top-k 5]
"""

import argparse
import sys

from search.retrieve import MODES, retrieve

PREVIEW_CHARS = 200


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the indexed documents.")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--q", required=True, help="query text")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    # The Windows console defaults to cp1252, which cannot print many PDF
    # symbols (e.g. math characters). Force UTF-8 so printing never crashes.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    results = retrieve(args.q, args.mode, args.top_k)
    print(f'\nmode={args.mode}  top_k={args.top_k}  query="{args.q}"\n')
    if not results:
        print("No results.")
    for result in results:
        preview = result["text"][:PREVIEW_CHARS]
        if len(result["text"]) > PREVIEW_CHARS:
            preview += "..."
        print(f"#{result['rank']}  {result['chunk_id']}  score={result['score']:.4f}")
        print(f"    {preview}\n")


if __name__ == "__main__":
    main()
