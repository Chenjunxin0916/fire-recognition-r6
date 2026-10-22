"""Validate the active environment against the fixed fire26 compatibility matrix."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fire_detection.environment import (  # noqa: E402
    collect_environment_report,
    validate_environment,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the environment verification command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Require RTX 5080 CUDA")
    parser.add_argument(
        "--require-ort-cuda",
        action="store_true",
        help="Require ONNX Runtime CUDAExecutionProvider",
    )
    parser.add_argument(
        "--check-yolo",
        action="store_true",
        help="Require installed YOLO26 detection and classification YAML files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Collect, validate, and print the active environment as JSON."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info(
        "Validating environment: require_cuda=%s require_ort_cuda=%s check_yolo=%s",
        args.require_cuda,
        args.require_ort_cuda,
        args.check_yolo,
    )
    report = collect_environment_report()
    validation = validate_environment(
        report,
        require_cuda=args.require_cuda,
        require_ort_cuda=args.require_ort_cuda,
        check_yolo=args.check_yolo,
    )
    payload = {"environment": report.to_dict(), "validation": validation.to_dict()}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0 if validation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
