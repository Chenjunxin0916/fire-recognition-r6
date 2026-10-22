"""Verify an existing completed Focal attempt; no training or inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_detection.research.focal_audit import audit_attempt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_attempt(args.attempt.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
