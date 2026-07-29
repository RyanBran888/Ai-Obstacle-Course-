#!/usr/bin/env python3
"""Run the whole test suite.

    python3 run_tests.py
    python3 run_tests.py -v
    python3 run_tests.py test_validation

Equivalent to `python3 -m unittest discover -s tests -t tests`, minus having to
remember the flags.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))


def main(argv: list[str]) -> int:
    verbosity = 2 if "-v" in argv else 1
    names = [a for a in argv if not a.startswith("-")]

    loader = unittest.TestLoader()
    if names:
        suite = loader.loadTestsFromNames(names)
    else:
        suite = loader.discover(str(TESTS), top_level_dir=str(TESTS))

    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
