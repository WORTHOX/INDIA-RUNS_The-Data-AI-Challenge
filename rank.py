"""Submission CLI wrapper.

This file lets reviewers run `python rank.py ...` from the repository root
without needing to know the internal `src/redrob_ranker` package path.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from redrob_ranker.rank import main


if __name__ == "__main__":
    main()
