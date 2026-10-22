"""Machine-verifiable sign-off for annotation contact sheets."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fire_detection.data.similarity import sha256_file
from fire_detection.utils import atomic_write_json


class AnnotationReviewError(ValueError):
    """Raised when contact-sheet artifacts or their human sign-off are invalid."""


ANNOTATION_REVIEW_SCHEMA_VERSION = 1
ANNOTATION_REVIEW_KIND = "annotation_contact_sheets"
PAGE_BUNDLE_ALGORITHM = "sha256-relative-path-size-file-sha256-v1"
CONTACT_ITEMS_PER_PAGE = 20
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_COLUMNS = (
    "group",
    "page",
    "filename",
    "annotation_id",
    "bbox_area_ratio",
    "title",
)

# These counts are part of the audited competition dataset contract, not renderer defaults.
EXPECTED_CONTACT_GROUPS: dict[str, tuple[int, int]] = {
    "small_boxes": (462, 24),
    "multiple_boxes": (202, 11),
    "bbox_issues": (9, 1),
    "exif_orientation": (1, 1),
}
EXPECTED_CONTACT_PAGE_COUNT = sum(pages for _items, pages in EXPECTED_CONTACT_GROUPS.values())


ANNOTATION_REVIEW_SCHEMA_V1: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Annotation contact-sheet review sign-off",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "review_kind",
        "manifest",
        "page_bundle",
        "groups",
        "reviewer",
        "reviewed_at",
        "disposition",
        "issues",
        "notes",
    ],
    "properties": {
        "schema_version": {"const": ANNOTATION_REVIEW_SCHEMA_VERSION},
        "review_kind": {"const": ANNOTATION_REVIEW_KIND},
        "manifest": {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "sha256"],
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        },
        "page_bundle": {
            "type": "object",
            "additionalProperties": False,
            "required": ["root", "page_count", "algorithm", "sha256"],
            "properties": {
                "root": {"type": "string", "minLength": 1},
                "page_count": {"const": EXPECTED_CONTACT_PAGE_COUNT},
                "algorithm": {"const": PAGE_BUNDLE_ALGORITHM},
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        },
        "groups": {
            "type": "object",
            "additionalProperties": False,
            "required": list(EXPECTED_CONTACT_GROUPS),
            "properties": {
                group: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["items", "pages"],
                    "properties": {"items": {"const": items}, "pages": {"const": pages}},
                }
                for group, (items, pages) in EXPECTED_CONTACT_GROUPS.items()
            },
        },
        "reviewer": {"type": "string"},
        "reviewed_at": {"type": ["string", "null"]},
        "disposition": {"enum": ["pending", "approved", "rejected"]},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "group",
                    "page",
                    "filename",
                    "annotation_id",
                    "description",
                    "disputed",
                    "second_reviewer",
                ],
                "properties": {
                    "group": {"enum": list(EXPECTED_CONTACT_GROUPS)},
                    "page": {"type": "integer", "minimum": 1},
                    "filename": {"type": "string", "minLength": 1},
                    "annotation_id": {"type": ["integer", "null"]},
                    "description": {"type": "string", "minLength": 1},
                    "disputed": {"type": "boolean"},
                    "second_reviewer": {"type": ["string", "null"]},
                },
            },
        },
        "notes": {"type": "string"},
    },
}


@dataclass(frozen=True)
class AnnotationReviewEvidence:
    """Verified hashes and human identity suitable for Fold metadata."""

    review_path: Path
    review_sha256: str
    manifest_path: Path
    manifest_sha256: str
    page_bundle_root: Path
    page_bundle_sha256: str
    page_count: int
    reviewer: str
    reviewed_at: str
    disposition: str
    issue_count: int


def _strict_json(path: Path) -> dict[str, Any]:
    class DuplicateKeyError(ValueError):
        pass

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise DuplicateKeyError(key)
            value[key] = item
        return value

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as exc:
        raise AnnotationReviewError(
            f"cannot strictly read annotation review {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AnnotationReviewError("annotation review must be a JSON object")
    return payload


def _require_exact_keys(value: object, *, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnnotationReviewError(f"annotation review {field} must be an object")
    actual = set(value)
    if actual != expected:
        raise AnnotationReviewError(
            f"annotation review {field} fields differ; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AnnotationReviewError(f"annotation review {field} must be a lowercase SHA-256")
    return value


def _portable_relative(path: Path, project_root: Path, *, field: str) -> str:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise AnnotationReviewError(
            f"annotation review {field} must be inside project_root: {path}"
        ) from exc
    return relative.as_posix()


def _resolve_portable_path(value: object, project_root: Path, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationReviewError(f"annotation review {field} must be a non-empty path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise AnnotationReviewError(
            f"annotation review {field} must be a portable project-relative path"
        )
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AnnotationReviewError(f"annotation review {field} escapes project_root") from exc
    return resolved


def _read_manifest_stats(manifest_path: Path) -> tuple[dict[str, dict[str, int]], tuple[str, ...]]:
    counts = {group: 0 for group in EXPECTED_CONTACT_GROUPS}
    page_numbers: dict[str, set[int]] = {group: set() for group in EXPECTED_CONTACT_GROUPS}
    seen_rows: set[tuple[str, ...]] = set()
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(_MANIFEST_COLUMNS):
                raise AnnotationReviewError(
                    "annotation contact-sheet manifest columns do not match schema v1"
                )
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(row.get(field) is None for field in _MANIFEST_COLUMNS):
                    raise AnnotationReviewError(
                        f"annotation contact-sheet manifest row {line_number} is malformed"
                    )
                group = row["group"]
                if group not in EXPECTED_CONTACT_GROUPS:
                    raise AnnotationReviewError(
                        f"annotation contact-sheet manifest has unknown group {group!r}"
                    )
                try:
                    page = int(row["page"])
                except ValueError as exc:
                    raise AnnotationReviewError(
                        f"annotation contact-sheet manifest row {line_number} has invalid page"
                    ) from exc
                if str(page) != row["page"] or page < 1:
                    raise AnnotationReviewError(
                        f"annotation contact-sheet manifest row {line_number} has invalid page"
                    )
                signature = tuple(row[field] for field in _MANIFEST_COLUMNS)
                if signature in seen_rows:
                    raise AnnotationReviewError(
                        f"annotation contact-sheet manifest has duplicate row {line_number}"
                    )
                seen_rows.add(signature)
                counts[group] += 1
                page_numbers[group].add(page)
    except (OSError, UnicodeError, csv.Error) as exc:
        if isinstance(exc, AnnotationReviewError):
            raise
        raise AnnotationReviewError(
            f"cannot read annotation contact-sheet manifest {manifest_path}: {exc}"
        ) from exc

    stats: dict[str, dict[str, int]] = {}
    expected_page_names: list[str] = []
    for group, (expected_items, expected_pages) in EXPECTED_CONTACT_GROUPS.items():
        actual_pages = page_numbers[group]
        required_pages = set(range(1, expected_pages + 1))
        if counts[group] != expected_items or actual_pages != required_pages:
            raise AnnotationReviewError(
                f"annotation contact-sheet manifest stats mismatch for {group}: "
                f"items={counts[group]} pages={sorted(actual_pages)}; "
                f"expected items={expected_items} pages=1..{expected_pages}"
            )
        derived_pages = (expected_items + CONTACT_ITEMS_PER_PAGE - 1) // CONTACT_ITEMS_PER_PAGE
        if expected_pages != derived_pages:
            raise AnnotationReviewError(
                f"internal contact-sheet page contract is invalid for {group}"
            )
        stats[group] = {"items": expected_items, "pages": expected_pages}
        expected_page_names.extend(
            f"{group}_{page:03d}.jpg" for page in range(1, expected_pages + 1)
        )
    return stats, tuple(sorted(expected_page_names))


def _page_bundle_sha256(root: Path, expected_page_names: tuple[str, ...]) -> str:
    if not root.is_dir():
        raise AnnotationReviewError(f"annotation contact-sheet directory is missing: {root}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }
    expected = set(expected_page_names)
    if actual != expected:
        raise AnnotationReviewError(
            "annotation contact-sheet page set differs; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    digest = hashlib.sha256()
    for relative in sorted(expected):
        page = root / Path(relative)
        digest.update(f"{relative}\t{page.stat().st_size}\t{sha256_file(page)}\n".encode())
    return digest.hexdigest()


def _artifact_snapshot(
    manifest_path: Path, *, project_root: Path
) -> tuple[dict[str, dict[str, int]], str, str, str, tuple[str, ...]]:
    manifest = manifest_path.resolve()
    if not manifest.is_file():
        raise AnnotationReviewError(f"annotation contact-sheet manifest is missing: {manifest}")
    manifest_relative = _portable_relative(manifest, project_root, field="manifest.path")
    root = manifest.parent
    root_relative = _portable_relative(root, project_root, field="page_bundle.root")
    stats, page_names = _read_manifest_stats(manifest)
    bundle_hash = _page_bundle_sha256(root, page_names)
    return stats, manifest_relative, root_relative, bundle_hash, page_names


def build_pending_annotation_review(
    manifest_path: Path, *, project_root: Path
) -> dict[str, object]:
    """Build a schema-v1 pending template from the exact audited 37-page artifact set."""

    stats, manifest_relative, root_relative, bundle_hash, page_names = _artifact_snapshot(
        Path(manifest_path), project_root=Path(project_root)
    )
    return {
        "schema_version": ANNOTATION_REVIEW_SCHEMA_VERSION,
        "review_kind": ANNOTATION_REVIEW_KIND,
        "manifest": {
            "path": manifest_relative,
            "sha256": sha256_file(Path(manifest_path)),
        },
        "page_bundle": {
            "root": root_relative,
            "page_count": len(page_names),
            "algorithm": PAGE_BUNDLE_ALGORITHM,
            "sha256": bundle_hash,
        },
        "groups": stats,
        "reviewer": "",
        "reviewed_at": None,
        "disposition": "pending",
        "issues": [],
        "notes": "",
    }


def write_pending_annotation_review(
    output_path: Path, manifest_path: Path, *, project_root: Path
) -> dict[str, object]:
    """Write a pending template once, refusing to overwrite a possible human sign-off."""

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite annotation review: {destination}")
    payload = build_pending_annotation_review(manifest_path, project_root=project_root)
    atomic_write_json(destination, payload)
    return payload


def _validate_groups(value: object, actual: dict[str, dict[str, int]]) -> None:
    groups = _require_exact_keys(value, expected=set(EXPECTED_CONTACT_GROUPS), field="groups")
    for group, expected in actual.items():
        record = _require_exact_keys(
            groups[group], expected={"items", "pages"}, field=f"groups.{group}"
        )
        if type(record["items"]) is not int or type(record["pages"]) is not int:
            raise AnnotationReviewError(f"annotation review groups.{group} counts must be integers")
        if record != expected:
            raise AnnotationReviewError(
                f"annotation review groups.{group} stats do not match manifest"
            )


def _validate_issues(value: object, *, reviewer: str) -> int:
    if not isinstance(value, list):
        raise AnnotationReviewError("annotation review issues must be an array")
    required = {
        "group",
        "page",
        "filename",
        "annotation_id",
        "description",
        "disputed",
        "second_reviewer",
    }
    for index, raw_issue in enumerate(value):
        issue = _require_exact_keys(raw_issue, expected=required, field=f"issues[{index}]")
        group = issue["group"]
        if group not in EXPECTED_CONTACT_GROUPS:
            raise AnnotationReviewError(f"annotation review issues[{index}].group is invalid")
        page = issue["page"]
        if type(page) is not int or not 1 <= page <= EXPECTED_CONTACT_GROUPS[group][1]:
            raise AnnotationReviewError(f"annotation review issues[{index}].page is invalid")
        if not isinstance(issue["filename"], str) or not issue["filename"].strip():
            raise AnnotationReviewError(f"annotation review issues[{index}].filename is required")
        annotation_id = issue["annotation_id"]
        if annotation_id is not None and type(annotation_id) is not int:
            raise AnnotationReviewError(
                f"annotation review issues[{index}].annotation_id must be an integer or null"
            )
        if not isinstance(issue["description"], str) or not issue["description"].strip():
            raise AnnotationReviewError(
                f"annotation review issues[{index}].description is required"
            )
        if type(issue["disputed"]) is not bool:
            raise AnnotationReviewError(
                f"annotation review issues[{index}].disputed must be a boolean"
            )
        second_reviewer = issue["second_reviewer"]
        if second_reviewer is not None and not isinstance(second_reviewer, str):
            raise AnnotationReviewError(
                f"annotation review issues[{index}].second_reviewer must be a string or null"
            )
        normalized_second = second_reviewer.strip() if isinstance(second_reviewer, str) else ""
        if issue["disputed"] and not normalized_second:
            raise AnnotationReviewError(
                f"annotation review disputed issue {index} requires second_reviewer"
            )
        if issue["disputed"] and normalized_second.casefold() == reviewer.casefold():
            raise AnnotationReviewError(
                f"annotation review disputed issue {index} requires an independent second_reviewer"
            )
    return len(value)


def validate_approved_annotation_review(
    review_path: Path, *, project_root: Path
) -> AnnotationReviewEvidence:
    """Validate an approved sign-off and recompute every manifest/page hash and statistic."""

    path = Path(review_path)
    payload = _strict_json(path)
    expected_top_level = {
        "schema_version",
        "review_kind",
        "manifest",
        "page_bundle",
        "groups",
        "reviewer",
        "reviewed_at",
        "disposition",
        "issues",
        "notes",
    }
    if set(payload) != expected_top_level:
        raise AnnotationReviewError(
            "annotation review schema-v1 fields differ; "
            f"missing={sorted(expected_top_level - set(payload))}, "
            f"extra={sorted(set(payload) - expected_top_level)}"
        )
    if payload["schema_version"] != ANNOTATION_REVIEW_SCHEMA_VERSION:
        raise AnnotationReviewError("annotation review schema_version must be 1")
    if payload["review_kind"] != ANNOTATION_REVIEW_KIND:
        raise AnnotationReviewError(
            f"annotation review review_kind must be {ANNOTATION_REVIEW_KIND!r}"
        )

    manifest_record = _require_exact_keys(
        payload["manifest"], expected={"path", "sha256"}, field="manifest"
    )
    manifest_path = _resolve_portable_path(
        manifest_record["path"], Path(project_root), field="manifest.path"
    )
    actual_stats, manifest_relative, root_relative, bundle_hash, page_names = _artifact_snapshot(
        manifest_path, project_root=Path(project_root)
    )
    if manifest_record["path"] != manifest_relative:
        raise AnnotationReviewError("annotation review manifest.path is not canonical")
    expected_manifest_hash = _require_sha256(manifest_record["sha256"], field="manifest.sha256")
    actual_manifest_hash = sha256_file(manifest_path)
    if expected_manifest_hash != actual_manifest_hash:
        raise AnnotationReviewError(
            "annotation contact-sheet manifest SHA-256 does not match current bytes"
        )

    page_bundle = _require_exact_keys(
        payload["page_bundle"],
        expected={"root", "page_count", "algorithm", "sha256"},
        field="page_bundle",
    )
    bundle_root = _resolve_portable_path(
        page_bundle["root"], Path(project_root), field="page_bundle.root"
    )
    if page_bundle["root"] != root_relative or bundle_root != manifest_path.parent:
        raise AnnotationReviewError(
            "annotation review page_bundle.root must equal the manifest directory"
        )
    if page_bundle["algorithm"] != PAGE_BUNDLE_ALGORITHM:
        raise AnnotationReviewError("annotation review page_bundle.algorithm is invalid")
    if type(page_bundle["page_count"]) is not int or page_bundle["page_count"] != len(page_names):
        raise AnnotationReviewError(
            f"annotation review page_bundle.page_count must be {len(page_names)}"
        )
    expected_bundle_hash = _require_sha256(page_bundle["sha256"], field="page_bundle.sha256")
    if expected_bundle_hash != bundle_hash:
        raise AnnotationReviewError(
            "annotation contact-sheet page bundle SHA-256 does not match current bytes"
        )
    _validate_groups(payload["groups"], actual_stats)

    reviewer = payload["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise AnnotationReviewError("annotation review reviewer signature is required")
    reviewer = reviewer.strip()
    reviewed_at = payload["reviewed_at"]
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise AnnotationReviewError("annotation review reviewed_at signature is required")
    try:
        timestamp = datetime.fromisoformat(reviewed_at)
    except ValueError as exc:
        raise AnnotationReviewError("annotation review reviewed_at is not ISO-8601") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise AnnotationReviewError("annotation review reviewed_at must include a timezone")
    disposition = payload["disposition"]
    if disposition != "approved":
        raise AnnotationReviewError(
            f"annotation review disposition must be approved; found {disposition!r}"
        )
    issue_count = _validate_issues(payload["issues"], reviewer=reviewer)
    if not isinstance(payload["notes"], str):
        raise AnnotationReviewError("annotation review notes must be a string")

    return AnnotationReviewEvidence(
        review_path=path.resolve(),
        review_sha256=sha256_file(path),
        manifest_path=manifest_path,
        manifest_sha256=actual_manifest_hash,
        page_bundle_root=bundle_root,
        page_bundle_sha256=bundle_hash,
        page_count=len(page_names),
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        disposition=disposition,
        issue_count=issue_count,
    )
