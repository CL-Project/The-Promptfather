"""
__main__.py — CLI for the verb_analyzer package
=================================================
Two modes:

  Single word
  -----------
      python -m verb_analyzer <data_dir> <verb>
      python -m verb_analyzer . खाया

  Interactive REPL
  ----------------
      python -m verb_analyzer <data_dir>
      python -m verb_analyzer .
      # Enter a verb (or 'q' to quit): खाया
"""

import sys
from . import VerbAnalyzer


def main() -> None:
    args = sys.argv[1:]

    if not args:
        print("Usage: python -m verb_analyzer <data_dir> [verb]")
        sys.exit(1)

    data_dir = args[0]
    va = VerbAnalyzer(data_dir=data_dir)

    # ── Single-word mode ──────────────────────────────────────────────────────
    if len(args) >= 2:
        print(va.summarize(args[1]))
        return

    # ── Interactive REPL ──────────────────────────────────────────────────────
    print(f"Hindi Verb Analyzer  (data: {data_dir})")
    print("Type a verb and press Enter.  'q' to quit.\n")

    while True:
        try:
            word = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not word:
            continue
        if word.lower() in ("q", "quit", "exit"):
            break

        print(va.summarize(word))
        print()


if __name__ == "__main__":
    main()