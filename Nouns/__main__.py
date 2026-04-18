"""
__main__.py — CLI for the Hindi Noun Analyzer
===============================================
Two modes:

  Single word
  -----------
      python -m Nouns <word>
      python -m Nouns लड़कों

  Interactive REPL
  ----------------
      python -m Nouns
      # Enter a noun (or 'q' to quit): लड़कों

Lexicon path is resolved relative to this file so the analyzer can be
invoked from any working directory.
"""

import sys
import os

# Ensure Nouns/ is on sys.path so sibling imports work from any cwd
_NOUNS_DIR = os.path.dirname(os.path.abspath(__file__))
if _NOUNS_DIR not in sys.path:
    sys.path.insert(0, _NOUNS_DIR)

from analyzer import load_lexicon, analyze
from noun_paradigm_templates import PARADIGM_TABLES

_LEXICON_PATH = os.path.join(_NOUNS_DIR, "data", "noun_lexicon_expanded.tsv")


def _format(word: str, results: list) -> str:
    if not results:
        return f"'{word}'  →  no noun analysis found"
    lines = [f"Analyses for '{word}' ({len(results)} result(s)):"]
    for r in results:
        conf = r.get("confidence", "certain")
        lines.append(
            f"  lemma={r['lemma']:<15} "
            f"paradigm={r['paradigm']:<4} "
            f"gender={r['gender']}  "
            f"number={r['number']:<3} "
            f"case={r['case']:<3} "
            f"conf={conf}"
        )
    return "\n".join(lines)


def main() -> None:
    try:
        lex, disp, conf = load_lexicon(_LEXICON_PATH)
    except FileNotFoundError:
        print(f"Error: lexicon not found at {_LEXICON_PATH}")
        print("Make sure you have run the pipeline scripts to generate noun_lexicon_expanded.tsv")
        sys.exit(1)

    print(f"Loaded {len(lex)} lemmas from {_LEXICON_PATH}")

    args = sys.argv[1:]

    # ── Single-word mode ──────────────────────────────────────────────────────
    if args:
        word = args[0]
        results = analyze(word, lex, disp, conf, PARADIGM_TABLES)
        print(_format(word, results))
        return

    # ── Interactive REPL ──────────────────────────────────────────────────────
    print("Hindi Noun Analyzer")
    print("Type a noun and press Enter.  'q' to quit.\n")

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

        results = analyze(word, lex, disp, conf, PARADIGM_TABLES)
        print(_format(word, results))
        print()


if __name__ == "__main__":
    main()