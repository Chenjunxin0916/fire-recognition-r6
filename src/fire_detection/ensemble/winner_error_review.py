"""Human-only review workflow for the frozen Round-3 winner error rows.

The mechanical FP/FN reports remain immutable inputs.  This module creates a
separate canonical draft whose semantic fields start pending, validates every
source hash and row identity, applies only values explicitly supplied by a
human reviewer, and publishes an immutable finalized bundle atomically.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from fire_detection.utils import atomic_write_csv, atomic_write_json

REVIEW_SCHEMA_VERSION = 1
REVIEW_KIND = "round3_winner_error_human_review"
LABEL_ISSUE_VALUES = frozenset({"pending", "no", "yes", "uncertain"})
LABEL_CONCLUSIONS = frozenset(
    {"pending", "not_required", "no_label_issue", "label_issue", "unresolved"}
)
REVIEW_STATUSES = frozenset({"pending", "needs_second_review", "reviewed"})
TOP_LEVEL_STATUSES = frozenset({"pending", "ready_for_finalize", "finalized"})
WINNER_FP_ERROR_CATEGORIES = (
    "夕阳/晚霞",
    "灯光/高亮",
    "反光",
    "红橙物体",
    "焊接/火花",
    "烟雾/云雾",
    "标签争议",
)
WINNER_FN_ERROR_CATEGORIES = (
    "小火/远火",
    "遮挡",
    "低照度",
    "烟大于火",
    "切片边缘",
    "置信度不足",
    "分类器压低",
    "标签争议",
)
WINNER_ERROR_CATEGORY_CHOICES = tuple(
    dict.fromkeys((*WINNER_FP_ERROR_CATEGORIES, *WINNER_FN_ERROR_CATEGORIES))
)
WINNER_SUGGESTED_ACTIONS = (
    "可以低风险修复",
    "暂时无法修复",
    "不建议修复",
)

_SOURCE_KEYS = {"error_summary", "fp_report", "fn_report", "selection_manifest"}
_RECORD_KEYS = {
    "filename",
    "confusion",
    "error_category",
    "suspected_label_issue",
    "suggested_action",
    "reviewer",
    "reviewed_at",
    "review_status",
    "second_reviewer",
    "second_reviewed_at",
    "label_issue_conclusion",
    "second_review_notes",
    "notes",
    "revision",
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "review_kind",
    "source",
    "status",
    "expected_counts",
    "records",
    "finalized_by",
    "finalized_at",
}
_MECHANICAL_REVIEW_COLUMNS = {
    "filename",
    "confusion",
    "error_category",
    "suspected_label_issue",
    "suggested_action",
    "reviewer",
    "review_status",
}


class WinnerErrorReviewError(ValueError):
    """Raised when human-review state or its frozen evidence is invalid."""


@dataclass(frozen=True)
class WinnerErrorReviewProgress:
    """Validated review progress derived from the complete record set."""

    total: int
    pending: int
    needs_second_review: int
    reviewed: int

    @property
    def complete(self) -> bool:
        """Return whether every required row has completed review."""

        return self.reviewed == self.total


@dataclass(frozen=True)
class ValidatedWinnerErrorReview:
    """Strictly validated review payload and optimistic-lock revision."""

    payload: dict[str, Any]
    sha256: str
    progress: WinnerErrorReviewProgress


@dataclass(frozen=True)
class _MechanicalBundle:
    root: Path
    summary_path: Path
    fp_path: Path
    fn_path: Path
    selection_path: Path
    summary: dict[str, Any]
    fp: pd.DataFrame
    fn: pd.DataFrame


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise WinnerErrorReviewError(f"cannot hash review evidence {path}: {exc}") from exc
    return digest.hexdigest()


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    class DuplicateKeyError(ValueError):
        pass

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(key)
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as exc:
        raise WinnerErrorReviewError(f"cannot strictly read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WinnerErrorReviewError(f"{label} must be a JSON object")
    return payload


def _exact_dict(value: object, *, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WinnerErrorReviewError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        raise WinnerErrorReviewError(
            f"{field} fields differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _portable_relative(path: Path, root: Path, *, field: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise WinnerErrorReviewError(f"{field} must remain inside project_root: {path}") from exc


@contextmanager
def _exclusive_review_write(path: Path) -> Iterator[None]:
    """Serialize all CLI and GUI writers for one review draft."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise WinnerErrorReviewError(
            "winner error review is locked by another writer; retry after it finishes"
        ) from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _resolve_portable(value: object, root: Path, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WinnerErrorReviewError(f"{field} must be a non-empty project-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise WinnerErrorReviewError(f"{field} must be a portable project-relative path")
    resolved = (root.resolve() / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise WinnerErrorReviewError(f"{field} escapes project_root") from exc
    return resolved


def _sha_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WinnerErrorReviewError(f"{field} must be a lowercase SHA-256")
    return value


def _read_error_csv(path: Path, *, confusion: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise WinnerErrorReviewError(
            f"cannot read mechanical {confusion} report {path}: {exc}"
        ) from exc
    missing = sorted(_MECHANICAL_REVIEW_COLUMNS.difference(frame.columns))
    if missing:
        raise WinnerErrorReviewError(f"mechanical {confusion} report is missing columns: {missing}")
    if frame["filename"].duplicated().any():
        raise WinnerErrorReviewError(f"mechanical {confusion} report has duplicate filenames")
    if set(frame["confusion"].astype(str)) != {confusion}:
        raise WinnerErrorReviewError(
            f"mechanical {confusion} report contains a different confusion class"
        )
    for column in ("error_category", "suspected_label_issue", "suggested_action", "reviewer"):
        if frame[column].astype(str).str.strip().ne("").any():
            raise WinnerErrorReviewError(
                f"mechanical {confusion} report semantic field {column} is not blank"
            )
    if set(frame["review_status"].astype(str)) != {"pending"}:
        raise WinnerErrorReviewError(
            f"mechanical {confusion} report review_status must remain pending"
        )
    return frame


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise WinnerErrorReviewError(f"{field} must be an integer >= {minimum}")
    return value


def _load_mechanical_bundle(mechanical_dir: Path, *, project_root: Path) -> _MechanicalBundle:
    root = project_root.resolve()
    directory = mechanical_dir if mechanical_dir.is_absolute() else root / mechanical_dir
    directory = directory.resolve()
    _portable_relative(directory, root, field="mechanical report directory")
    summary_path = directory / "error_summary.json"
    fp_path = directory / "final_fp_report.csv"
    fn_path = directory / "final_fn_report.csv"
    if not summary_path.is_file() or not fp_path.is_file() or not fn_path.is_file():
        raise WinnerErrorReviewError(f"mechanical winner report is incomplete under {directory}")
    summary = _strict_json(summary_path, label="mechanical winner error summary")
    if summary.get("kind") != "round3_winner_error_report":
        raise WinnerErrorReviewError("mechanical report kind is not round3_winner_error_report")
    counts = summary.get("counts")
    if not isinstance(counts, dict):
        raise WinnerErrorReviewError("mechanical report counts must be an object")
    expected_fp = _integer(counts.get("fp"), field="mechanical counts.fp")
    expected_fn = _integer(counts.get("fn"), field="mechanical counts.fn")
    human = summary.get("human_review")
    if not isinstance(human, dict):
        raise WinnerErrorReviewError("mechanical report human_review must be an object")
    required_rows = _integer(
        human.get("required_rows"), field="mechanical human_review.required_rows"
    )
    if human.get("completed_rows") != 0 or required_rows != expected_fp + expected_fn:
        raise WinnerErrorReviewError(
            "mechanical report must remain an entirely pending human-review baseline"
        )
    fp = _read_error_csv(fp_path, confusion="FP")
    fn = _read_error_csv(fn_path, confusion="FN")
    if len(fp) != expected_fp or len(fn) != expected_fn:
        raise WinnerErrorReviewError(
            f"mechanical report row counts differ: FP {len(fp)}/{expected_fp}, "
            f"FN {len(fn)}/{expected_fn}"
        )
    all_filenames = pd.concat([fp["filename"], fn["filename"]], ignore_index=True)
    if all_filenames.duplicated().any():
        raise WinnerErrorReviewError("mechanical FP/FN reports overlap by filename")
    selection = summary.get("selection_manifest")
    if not isinstance(selection, dict):
        raise WinnerErrorReviewError("mechanical selection_manifest must be an object")
    selection_path = _resolve_portable(
        selection.get("path"), root, field="mechanical selection_manifest.path"
    )
    expected_selection_sha = _sha_text(
        selection.get("sha256"), field="mechanical selection_manifest.sha256"
    )
    if not selection_path.is_file() or _sha256_file(selection_path) != expected_selection_sha:
        raise WinnerErrorReviewError("mechanical selection manifest is missing or hash-drifted")
    return _MechanicalBundle(
        root=root,
        summary_path=summary_path,
        fp_path=fp_path,
        fn_path=fn_path,
        selection_path=selection_path,
        summary=summary,
        fp=fp,
        fn=fn,
    )


def _source_record(path: Path, *, root: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _portable_relative(path, root, field="review source path"),
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _pending_record(filename: str, confusion: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "confusion": confusion,
        "error_category": "",
        "suspected_label_issue": "pending",
        "suggested_action": "",
        "reviewer": "",
        "reviewed_at": None,
        "review_status": "pending",
        "second_reviewer": "",
        "second_reviewed_at": None,
        "label_issue_conclusion": "pending",
        "second_review_notes": "",
        "notes": "",
        "revision": 0,
    }


def build_pending_winner_error_review(
    mechanical_dir: Path, *, project_root: Path
) -> dict[str, Any]:
    """Create a 100% pending template bound to the exact mechanical bundle."""

    bundle = _load_mechanical_bundle(mechanical_dir, project_root=project_root)
    records = [
        *(
            _pending_record(str(row.filename), "FP")
            for row in bundle.fp[["filename"]].itertuples(index=False)
        ),
        *(
            _pending_record(str(row.filename), "FN")
            for row in bundle.fn[["filename"]].itertuples(index=False)
        ),
    ]
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_kind": REVIEW_KIND,
        "source": {
            "error_summary": _source_record(bundle.summary_path, root=bundle.root),
            "fp_report": _source_record(bundle.fp_path, root=bundle.root, rows=len(bundle.fp)),
            "fn_report": _source_record(bundle.fn_path, root=bundle.root, rows=len(bundle.fn)),
            "selection_manifest": _source_record(bundle.selection_path, root=bundle.root),
        },
        "status": "pending",
        "expected_counts": {
            "total": len(records),
            "fp": len(bundle.fp),
            "fn": len(bundle.fn),
        },
        "records": records,
        "finalized_by": None,
        "finalized_at": None,
    }


def write_pending_winner_error_review(
    output_path: Path,
    mechanical_dir: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Write a pending review template once, refusing to overwrite human work."""

    destination = output_path if output_path.is_absolute() else project_root / output_path
    _portable_relative(destination, project_root, field="winner error review output")
    with _exclusive_review_write(destination):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite winner error review: {destination}")
        payload = build_pending_winner_error_review(mechanical_dir, project_root=project_root)
        atomic_write_json(destination, payload)
    return payload


def _timestamp(value: str | None) -> str:
    timestamp = value or datetime.now().astimezone().isoformat(timespec="seconds")
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise WinnerErrorReviewError("review timestamp must be non-empty ISO-8601 text")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise WinnerErrorReviewError("review timestamp must be valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WinnerErrorReviewError("review timestamp must include a timezone")
    return timestamp


def _human(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WinnerErrorReviewError(f"{field} is required and must be entered by a human")
    return value.strip()


def _text(value: object, *, field: str, required: bool = False) -> str:
    if not isinstance(value, str):
        raise WinnerErrorReviewError(f"{field} must be text")
    normalized = value.strip()
    if required and not normalized:
        raise WinnerErrorReviewError(f"{field} is required and must be entered by a human")
    return normalized


def _canonical_error_category(value: object, *, confusion: str, field: str) -> str:
    category = _text(value, field=field, required=True)
    choices = WINNER_FP_ERROR_CATEGORIES if confusion == "FP" else WINNER_FN_ERROR_CATEGORIES
    if category not in choices:
        raise WinnerErrorReviewError(f"{field} must be one of {list(choices)} for {confusion}")
    return category


def _canonical_suggested_action(value: object, *, field: str) -> str:
    action = _text(value, field=field, required=True)
    if action not in WINNER_SUGGESTED_ACTIONS:
        raise WinnerErrorReviewError(f"{field} must be one of {list(WINNER_SUGGESTED_ACTIONS)}")
    return action


def _validate_source_record(
    value: object,
    *,
    root: Path,
    expected_path: Path,
    field: str,
    expected_rows: int | None,
) -> None:
    expected_keys = {"path", "sha256"} | ({"rows"} if expected_rows is not None else set())
    record = _exact_dict(value, expected=expected_keys, field=field)
    path = _resolve_portable(record["path"], root, field=f"{field}.path")
    if path != expected_path.resolve():
        raise WinnerErrorReviewError(f"{field}.path does not bind the expected source")
    expected_sha = _sha_text(record["sha256"], field=f"{field}.sha256")
    if not path.is_file() or _sha256_file(path) != expected_sha:
        raise WinnerErrorReviewError(f"{field} source is missing or hash-drifted")
    if expected_rows is not None:
        rows = _integer(record["rows"], field=f"{field}.rows")
        if rows != expected_rows:
            raise WinnerErrorReviewError(f"{field}.rows differs from the mechanical report")


def _validate_record(raw: object, *, index: int) -> tuple[dict[str, Any], str]:
    record = _exact_dict(raw, expected=_RECORD_KEYS, field=f"records[{index}]")
    filename = _text(record["filename"], field=f"records[{index}].filename", required=True)
    confusion = record["confusion"]
    if confusion not in {"FP", "FN"}:
        raise WinnerErrorReviewError(f"records[{index}].confusion must be FP or FN")
    status = record["review_status"]
    if status not in REVIEW_STATUSES:
        raise WinnerErrorReviewError(f"records[{index}].review_status is invalid")
    label_issue = record["suspected_label_issue"]
    if label_issue not in LABEL_ISSUE_VALUES:
        raise WinnerErrorReviewError(f"records[{index}].suspected_label_issue is invalid")
    conclusion = record["label_issue_conclusion"]
    if conclusion not in LABEL_CONCLUSIONS:
        raise WinnerErrorReviewError(f"records[{index}].label_issue_conclusion is invalid")
    revision = _integer(record["revision"], field=f"records[{index}].revision")
    notes = _text(record["notes"], field=f"records[{index}].notes")
    second_notes = _text(
        record["second_review_notes"], field=f"records[{index}].second_review_notes"
    )

    if status == "pending":
        if revision != 0:
            raise WinnerErrorReviewError(f"records[{index}] pending revision must be zero")
        expected_blank = {
            "error_category": "",
            "suspected_label_issue": "pending",
            "suggested_action": "",
            "reviewer": "",
            "reviewed_at": None,
            "second_reviewer": "",
            "second_reviewed_at": None,
            "label_issue_conclusion": "pending",
            "second_review_notes": "",
            "notes": "",
        }
        if any(record[key] != value for key, value in expected_blank.items()):
            raise WinnerErrorReviewError(
                f"records[{index}] pending row contains partial semantic review"
            )
        return record, filename

    _canonical_error_category(
        record["error_category"],
        confusion=str(confusion),
        field=f"records[{index}].error_category",
    )
    _canonical_suggested_action(
        record["suggested_action"],
        field=f"records[{index}].suggested_action",
    )
    reviewer = _human(record["reviewer"], field=f"records[{index}].reviewer")
    if not isinstance(record["reviewed_at"], str):
        raise WinnerErrorReviewError(f"records[{index}].reviewed_at is required")
    _timestamp(record["reviewed_at"])
    if label_issue == "pending":
        raise WinnerErrorReviewError(
            f"records[{index}].suspected_label_issue requires a human decision"
        )
    if revision < 1:
        raise WinnerErrorReviewError(f"records[{index}] reviewed revision must be positive")

    if label_issue == "no":
        if status != "reviewed" or conclusion != "not_required":
            raise WinnerErrorReviewError(
                f"records[{index}] no-label-issue review must be reviewed/not_required"
            )
        if (
            record["second_reviewer"] != ""
            or record["second_reviewed_at"] is not None
            or second_notes != ""
        ):
            raise WinnerErrorReviewError(
                f"records[{index}] no-label-issue review must not contain second review"
            )
    else:
        if status == "needs_second_review":
            if (
                conclusion != "pending"
                or record["second_reviewer"] != ""
                or record["second_reviewed_at"] is not None
                or second_notes != ""
            ):
                raise WinnerErrorReviewError(
                    f"records[{index}] pending second review contains partial resolution"
                )
        elif status == "reviewed":
            second = _human(record["second_reviewer"], field=f"records[{index}].second_reviewer")
            if second.casefold() == reviewer.casefold():
                raise WinnerErrorReviewError(
                    f"records[{index}] second reviewer must differ from primary reviewer"
                )
            if not isinstance(record["second_reviewed_at"], str):
                raise WinnerErrorReviewError(f"records[{index}].second_reviewed_at is required")
            _timestamp(record["second_reviewed_at"])
            if conclusion not in {"no_label_issue", "label_issue", "unresolved"}:
                raise WinnerErrorReviewError(
                    f"records[{index}].label_issue_conclusion needs a second-review result"
                )
        else:
            raise WinnerErrorReviewError(
                f"records[{index}] suspected label issue requires second review"
            )
    record["notes"] = notes
    record["second_review_notes"] = second_notes
    return record, filename


def _derive_progress(records: list[dict[str, Any]]) -> WinnerErrorReviewProgress:
    counts = Counter(str(record["review_status"]) for record in records)
    return WinnerErrorReviewProgress(
        total=len(records),
        pending=counts["pending"],
        needs_second_review=counts["needs_second_review"],
        reviewed=counts["reviewed"],
    )


def validate_winner_error_review(
    review_path: Path,
    *,
    project_root: Path,
    require_complete: bool = False,
) -> ValidatedWinnerErrorReview:
    """Strictly validate schema, frozen hashes, row closure, and review state."""

    path = review_path if review_path.is_absolute() else project_root / review_path
    _portable_relative(path, project_root, field="winner error review path")
    payload = _strict_json(path, label="winner error review")
    _exact_dict(payload, expected=_TOP_LEVEL_KEYS, field="winner error review")
    if payload["schema_version"] != REVIEW_SCHEMA_VERSION:
        raise WinnerErrorReviewError("winner error review schema_version must be 1")
    if payload["review_kind"] != REVIEW_KIND:
        raise WinnerErrorReviewError(f"winner error review review_kind must be {REVIEW_KIND!r}")
    source = _exact_dict(payload["source"], expected=_SOURCE_KEYS, field="source")
    summary_record = _exact_dict(
        source["error_summary"], expected={"path", "sha256"}, field="source.error_summary"
    )
    summary_path = _resolve_portable(
        summary_record["path"], project_root, field="source.error_summary.path"
    )
    bundle = _load_mechanical_bundle(summary_path.parent, project_root=project_root)
    _validate_source_record(
        source["error_summary"],
        root=project_root,
        expected_path=bundle.summary_path,
        field="source.error_summary",
        expected_rows=None,
    )
    _validate_source_record(
        source["fp_report"],
        root=project_root,
        expected_path=bundle.fp_path,
        field="source.fp_report",
        expected_rows=len(bundle.fp),
    )
    _validate_source_record(
        source["fn_report"],
        root=project_root,
        expected_path=bundle.fn_path,
        field="source.fn_report",
        expected_rows=len(bundle.fn),
    )
    _validate_source_record(
        source["selection_manifest"],
        root=project_root,
        expected_path=bundle.selection_path,
        field="source.selection_manifest",
        expected_rows=None,
    )

    expected_counts = _exact_dict(
        payload["expected_counts"], expected={"total", "fp", "fn"}, field="expected_counts"
    )
    total = _integer(expected_counts["total"], field="expected_counts.total")
    fp_count = _integer(expected_counts["fp"], field="expected_counts.fp")
    fn_count = _integer(expected_counts["fn"], field="expected_counts.fn")
    if (total, fp_count, fn_count) != (
        len(bundle.fp) + len(bundle.fn),
        len(bundle.fp),
        len(bundle.fn),
    ):
        raise WinnerErrorReviewError("winner error review expected_counts drifted")
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise WinnerErrorReviewError("winner error review records must be an array")
    records: list[dict[str, Any]] = []
    identities: list[tuple[str, str]] = []
    for index, raw_record in enumerate(raw_records):
        record, filename = _validate_record(raw_record, index=index)
        records.append(record)
        identities.append((filename, str(record["confusion"])))
    if len(identities) != total or len(set(identities)) != total:
        raise WinnerErrorReviewError("winner error review rows are missing or duplicated")
    expected_identities = [
        *((str(value), "FP") for value in bundle.fp["filename"].tolist()),
        *((str(value), "FN") for value in bundle.fn["filename"].tolist()),
    ]
    if identities != expected_identities:
        raise WinnerErrorReviewError(
            "winner error review row order or filename/confusion closure differs"
        )
    progress = _derive_progress(records)
    status = payload["status"]
    if status not in TOP_LEVEL_STATUSES:
        raise WinnerErrorReviewError("winner error review status is invalid")
    if status == "pending":
        if progress.complete:
            raise WinnerErrorReviewError(
                "winner error review status must be ready_for_finalize when all rows are reviewed"
            )
        if payload["finalized_by"] is not None or payload["finalized_at"] is not None:
            raise WinnerErrorReviewError("pending review cannot contain finalizer fields")
    elif status == "ready_for_finalize":
        if not progress.complete:
            raise WinnerErrorReviewError("ready_for_finalize review still contains pending rows")
        if payload["finalized_by"] is not None or payload["finalized_at"] is not None:
            raise WinnerErrorReviewError("ready review cannot contain finalizer fields")
    else:
        if not progress.complete:
            raise WinnerErrorReviewError("finalized review still contains pending rows")
        _human(payload["finalized_by"], field="finalized_by")
        if not isinstance(payload["finalized_at"], str):
            raise WinnerErrorReviewError("finalized_at is required")
        _timestamp(payload["finalized_at"])
    if require_complete and not progress.complete:
        raise WinnerErrorReviewError(
            "winner error review is incomplete: "
            f"pending={progress.pending}, "
            f"needs_second_review={progress.needs_second_review}, "
            f"reviewed={progress.reviewed}/{progress.total}; status remains pending"
        )
    return ValidatedWinnerErrorReview(
        payload=payload,
        sha256=_sha256_file(path),
        progress=progress,
    )


def _check_expected_sha(actual: str, expected: str) -> None:
    normalized = _sha_text(expected, field="expected_sha256")
    if actual != normalized:
        raise WinnerErrorReviewError(
            "winner error review changed; reload before saving this human decision"
        )


def _find_record(payload: dict[str, Any], filename: str) -> dict[str, Any]:
    records = cast(list[dict[str, Any]], payload["records"])
    matches = [record for record in records if record["filename"] == filename]
    if len(matches) != 1:
        raise WinnerErrorReviewError(
            f"winner error review filename not found exactly once: {filename}"
        )
    return matches[0]


def _refresh_draft_status(payload: dict[str, Any]) -> None:
    progress = _derive_progress(payload["records"])
    payload["status"] = "ready_for_finalize" if progress.complete else "pending"
    payload["finalized_by"] = None
    payload["finalized_at"] = None


def _ensure_draft_not_published(
    finalized_output_dir: Path | None,
    *,
    project_root: Path,
) -> None:
    """Reject draft mutations after the caller's finalized bundle is published."""

    if finalized_output_dir is None:
        return
    destination = (
        finalized_output_dir
        if finalized_output_dir.is_absolute()
        else project_root / finalized_output_dir
    )
    _portable_relative(destination, project_root, field="winner error finalized output")
    if destination.exists():
        raise WinnerErrorReviewError(
            "winner error finalized bundle already exists; the draft is now read-only"
        )


def save_primary_winner_error_review(
    review_path: Path,
    *,
    project_root: Path,
    filename: str,
    error_category: str,
    suspected_label_issue: str,
    suggested_action: str,
    reviewer: str,
    expected_sha256: str,
    finalized_output_dir: Path | None = None,
    notes: str = "",
    reviewed_at: str | None = None,
) -> ValidatedWinnerErrorReview:
    """Atomically save only the primary semantic values explicitly supplied by a human."""

    path = review_path if review_path.is_absolute() else project_root / review_path
    _portable_relative(path, project_root, field="winner error review path")
    with _exclusive_review_write(path):
        current = validate_winner_error_review(path, project_root=project_root)
        _check_expected_sha(current.sha256, expected_sha256)
        _ensure_draft_not_published(finalized_output_dir, project_root=project_root)
        if current.payload["status"] == "finalized":
            raise WinnerErrorReviewError("cannot edit a finalized winner error review")
        human = _human(reviewer, field="reviewer")
        if suspected_label_issue not in {"no", "yes", "uncertain"}:
            raise WinnerErrorReviewError("suspected_label_issue must be no, yes, or uncertain")
        payload = copy.deepcopy(current.payload)
        record = _find_record(payload, filename)
        confusion = str(record["confusion"])
        category = _canonical_error_category(
            error_category,
            confusion=confusion,
            field="error_category",
        )
        action = _canonical_suggested_action(suggested_action, field="suggested_action")
        record.update(
            {
                "error_category": category,
                "suspected_label_issue": suspected_label_issue,
                "suggested_action": action,
                "reviewer": human,
                "reviewed_at": _timestamp(reviewed_at),
                "review_status": (
                    "reviewed" if suspected_label_issue == "no" else "needs_second_review"
                ),
                "second_reviewer": "",
                "second_reviewed_at": None,
                "label_issue_conclusion": (
                    "not_required" if suspected_label_issue == "no" else "pending"
                ),
                "second_review_notes": "",
                "notes": _text(notes, field="notes"),
                "revision": int(record["revision"]) + 1,
            }
        )
        _refresh_draft_status(payload)
        atomic_write_json(path, payload)
        return validate_winner_error_review(path, project_root=project_root)


def save_secondary_winner_error_review(
    review_path: Path,
    *,
    project_root: Path,
    filename: str,
    label_issue_conclusion: str,
    reviewer: str,
    expected_sha256: str,
    finalized_output_dir: Path | None = None,
    notes: str = "",
    reviewed_at: str | None = None,
) -> ValidatedWinnerErrorReview:
    """Resolve a primary label concern using an explicitly named independent reviewer."""

    path = review_path if review_path.is_absolute() else project_root / review_path
    _portable_relative(path, project_root, field="winner error review path")
    with _exclusive_review_write(path):
        current = validate_winner_error_review(path, project_root=project_root)
        _check_expected_sha(current.sha256, expected_sha256)
        _ensure_draft_not_published(finalized_output_dir, project_root=project_root)
        if current.payload["status"] == "finalized":
            raise WinnerErrorReviewError("cannot edit a finalized winner error review")
        if label_issue_conclusion not in {"no_label_issue", "label_issue", "unresolved"}:
            raise WinnerErrorReviewError(
                "label_issue_conclusion must be no_label_issue, label_issue, or unresolved"
            )
        human = _human(reviewer, field="second reviewer")
        payload = copy.deepcopy(current.payload)
        record = _find_record(payload, filename)
        if record["review_status"] != "needs_second_review":
            raise WinnerErrorReviewError(
                "secondary review is allowed only after a yes/uncertain primary label decision"
            )
        if human.casefold() == str(record["reviewer"]).casefold():
            raise WinnerErrorReviewError("second reviewer must differ from primary reviewer")
        record.update(
            {
                "second_reviewer": human,
                "second_reviewed_at": _timestamp(reviewed_at),
                "label_issue_conclusion": label_issue_conclusion,
                "second_review_notes": _text(notes, field="second review notes"),
                "review_status": "reviewed",
                "revision": int(record["revision"]) + 1,
            }
        )
        _refresh_draft_status(payload)
        atomic_write_json(path, payload)
        return validate_winner_error_review(path, project_root=project_root)


def reset_winner_error_review(
    review_path: Path,
    *,
    project_root: Path,
    backup_dir: Path,
    reset_by: str,
    expected_sha256: str,
    finalized_output_dir: Path | None = None,
    reset_at: str | None = None,
) -> dict[str, Any]:
    """Back up the current draft, then atomically restore every row to pending."""

    path = review_path if review_path.is_absolute() else project_root / review_path
    _portable_relative(path, project_root, field="winner error review path")
    backup_root = backup_dir if backup_dir.is_absolute() else project_root / backup_dir
    _portable_relative(backup_root, project_root, field="winner error reset backup directory")
    human = _human(reset_by, field="reset_by")
    timestamp = _timestamp(reset_at)
    with _exclusive_review_write(path):
        current = validate_winner_error_review(path, project_root=project_root)
        _check_expected_sha(current.sha256, expected_sha256)
        _ensure_draft_not_published(finalized_output_dir, project_root=project_root)
        if current.payload["status"] == "finalized":
            raise WinnerErrorReviewError("cannot reset a finalized winner error review")
        source = cast(dict[str, Any], current.payload["source"])
        summary_path = _resolve_portable(
            cast(dict[str, Any], source["error_summary"])["path"],
            project_root,
            field="source.error_summary.path",
        )
        pending = build_pending_winner_error_review(
            summary_path.parent,
            project_root=project_root,
        )
        if (
            pending["source"] != current.payload["source"]
            or pending["expected_counts"] != current.payload["expected_counts"]
        ):
            raise WinnerErrorReviewError(
                "refusing reset because the reconstructed pending draft changed its sources"
            )

        parsed = datetime.fromisoformat(timestamp)
        reset_id = (
            f"{path.stem}-before-reset-{parsed.strftime('%Y%m%dT%H%M%S%z')}-{current.sha256[:12]}"
        )
        destination = backup_root / reset_id
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite winner error reset backup: {destination}")
        backup_root.mkdir(parents=True, exist_ok=True)
        staging = backup_root / f".{reset_id}.building-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            previous_path = staging / "previous_review.json"
            previous_path.write_bytes(path.read_bytes())
            previous_sha256 = _sha256_file(previous_path)
            if previous_sha256 != current.sha256:
                raise WinnerErrorReviewError("winner error reset backup does not match the draft")
            atomic_write_json(
                staging / "reset_manifest.json",
                {
                    "schema_version": REVIEW_SCHEMA_VERSION,
                    "kind": "round3_winner_error_review_reset_backup",
                    "review": _portable_relative(path, project_root, field="reset review path"),
                    "previous_sha256": previous_sha256,
                    "previous_progress": {
                        "total": current.progress.total,
                        "pending": current.progress.pending,
                        "needs_second_review": current.progress.needs_second_review,
                        "reviewed": current.progress.reviewed,
                    },
                    "reset_by": human,
                    "reset_at": timestamp,
                    "files": {
                        "previous_review.json": {
                            "sha256": previous_sha256,
                            "bytes": previous_path.stat().st_size,
                        }
                    },
                },
            )
            staging.replace(destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        atomic_write_json(path, pending)
        reset = validate_winner_error_review(path, project_root=project_root)
        return {
            **progress_dict(reset),
            "reset_by": human,
            "reset_at": timestamp,
            "previous_sha256": current.sha256,
            "backup_dir": _portable_relative(
                destination,
                project_root,
                field="winner error reset backup",
            ),
        }


def _count_text(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {
        value: count
        for value, count in sorted(
            Counter(str(record[key]) for record in records).items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def _review_frame(
    mechanical: pd.DataFrame, records: list[dict[str, Any]], *, confusion: str
) -> pd.DataFrame:
    review = pd.DataFrame([record for record in records if record["confusion"] == confusion])
    semantic = [column for column in _RECORD_KEYS if column not in {"filename", "confusion"}]
    base = mechanical.drop(
        columns=[
            "error_category",
            "suspected_label_issue",
            "suggested_action",
            "reviewer",
            "review_status",
        ]
    )
    return base.merge(
        review[["filename", "confusion", *sorted(semantic)]],
        on=["filename", "confusion"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )


def finalize_winner_error_review(
    review_path: Path,
    *,
    project_root: Path,
    output_dir: Path,
    finalizer: str,
    expected_sha256: str,
    finalized_at: str | None = None,
) -> dict[str, Any]:
    """Serialize finalization against every other draft writer."""

    path = review_path if review_path.is_absolute() else project_root / review_path
    _portable_relative(path, project_root, field="winner error review path")
    with _exclusive_review_write(path):
        return _finalize_winner_error_review_locked(
            review_path,
            project_root=project_root,
            output_dir=output_dir,
            finalizer=finalizer,
            expected_sha256=expected_sha256,
            finalized_at=finalized_at,
        )


def _finalize_winner_error_review_locked(
    review_path: Path,
    *,
    project_root: Path,
    output_dir: Path,
    finalizer: str,
    expected_sha256: str,
    finalized_at: str | None = None,
) -> dict[str, Any]:
    """Publish a complete immutable bundle while the caller holds the draft lock."""

    path = review_path if review_path.is_absolute() else project_root / review_path
    validated = validate_winner_error_review(path, project_root=project_root, require_complete=True)
    _check_expected_sha(validated.sha256, expected_sha256)
    if validated.payload["status"] != "ready_for_finalize":
        raise WinnerErrorReviewError("finalize requires a complete ready_for_finalize draft")
    destination = output_dir if output_dir.is_absolute() else project_root / output_dir
    _portable_relative(destination, project_root, field="winner error finalized output")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite finalized winner error review: {destination}")
    human = _human(finalizer, field="finalizer")
    timestamp = _timestamp(finalized_at)
    finalized = copy.deepcopy(validated.payload)
    finalized["status"] = "finalized"
    finalized["finalized_by"] = human
    finalized["finalized_at"] = timestamp
    records: list[dict[str, Any]] = finalized["records"]
    summary_source = finalized["source"]["error_summary"]["path"]
    mechanical = _load_mechanical_bundle(
        _resolve_portable(summary_source, project_root, field="source.error_summary.path").parent,
        project_root=project_root,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        finalized_review_path = staging / "finalized_review.json"
        atomic_write_json(finalized_review_path, finalized)
        fp = _review_frame(mechanical.fp, records, confusion="FP")
        fn = _review_frame(mechanical.fn, records, confusion="FN")
        if len(fp) != len(mechanical.fp) or len(fn) != len(mechanical.fn):
            raise WinnerErrorReviewError("finalized review join did not preserve all FP/FN rows")
        atomic_write_csv(staging / "final_fp_report.csv", fp)
        atomic_write_csv(staging / "final_fn_report.csv", fn)
        category_counts = _count_text(records, "error_category")
        summary: dict[str, Any] = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "kind": "round3_winner_error_human_review_summary",
            "status": "finalized",
            "draft_review": {
                "path": _portable_relative(path, project_root, field="draft review path"),
                "sha256": validated.sha256,
            },
            "finalized_review": {
                "path": "finalized_review.json",
                "sha256": _sha256_file(finalized_review_path),
            },
            "counts": {
                "total": len(records),
                "fp": len(fp),
                "fn": len(fn),
                "reviewed": len(records),
                "pending": 0,
                "needs_second_review": 0,
            },
            "error_category_counts": category_counts,
            "top_three_error_categories": [
                {"error_category": key, "count": value}
                for key, value in list(category_counts.items())[:3]
            ],
            "suspected_label_issue_counts": _count_text(records, "suspected_label_issue"),
            "label_issue_conclusion_counts": _count_text(records, "label_issue_conclusion"),
            "suggested_action_counts": _count_text(records, "suggested_action"),
            "primary_reviewer_counts": _count_text(records, "reviewer"),
            "second_reviewer_counts": {
                key: value for key, value in _count_text(records, "second_reviewer").items() if key
            },
            "finalized_by": human,
            "finalized_at": timestamp,
            "semantic_source": "human_review_only",
        }
        atomic_write_json(staging / "review_summary.json", summary)
        files: dict[str, dict[str, Any]] = {}
        for name in (
            "finalized_review.json",
            "final_fp_report.csv",
            "final_fn_report.csv",
            "review_summary.json",
        ):
            artifact = staging / name
            record: dict[str, Any] = {
                "path": name,
                "sha256": _sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            }
            if artifact.suffix == ".csv":
                record["rows"] = sum(1 for _ in artifact.open(encoding="utf-8")) - 1
            files[name] = record
        atomic_write_json(
            staging / "bundle_manifest.json",
            {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "kind": "round3_winner_error_human_review_bundle",
                "files": files,
                "semantic_source": "human_review_only",
            },
        )
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def progress_dict(validated: ValidatedWinnerErrorReview) -> dict[str, Any]:
    """Return a strict, CLI-friendly progress object with explicit pending state."""

    progress = validated.progress
    return {
        "status": validated.payload["status"],
        "sha256": validated.sha256,
        "total": progress.total,
        "pending": progress.pending,
        "needs_second_review": progress.needs_second_review,
        "reviewed": progress.reviewed,
        "complete": progress.complete,
    }


def strict_json_text(payload: object) -> str:
    """Serialize CLI output deterministically without NaN or Infinity."""

    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
