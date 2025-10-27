#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure repo root is on sys.path no matter where this script is executed
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _run():
    try:
        from staged_bdann.cli import main
    except Exception as e:
        print("Failed to import 'staged_bdann'. Make sure you are running from a clone with the "
              "'staged_bdann/' package present at the repo root, or set PYTHONPATH to the repo root.",
              file=sys.stderr)
        raise
    raise SystemExit(main())

if __name__ == "__main__":
    _run()