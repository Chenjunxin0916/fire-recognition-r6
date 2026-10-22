"""Human-only, resumable v2 review UI for the Round-3 Smoke2 negatives.

The reviewer never derives semantics from images, filenames, model scores, or
neighbouring records.  A human must explicitly enter every primary decision and,
for uncertain or suspected-label-error rows, a different human must enter the
secondary conclusion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fire_detection.models.round3_replay import (
    SMOKE_REVIEW_V2_DISPOSITIONS,
    SMOKE_REVIEW_V2_POLICY,
    SMOKE_REVIEW_V2_TAGS,
    Round3ReplaySpec,
    SmokeReviewV2Summary,
    validate_smoke_review_v2_draft,
)
from fire_detection.models.training import (
    FoldRecord,
    TrainingConfigurationError,
    write_json_atomic,
)

SMOKE_DISPOSITION_CHOICES = (
    ("确认无火", "confirmed_no_fire"),
    ("无法确认", "uncertain"),
    ("疑似标签错误", "suspected_label_error"),
)
SMOKE_TAG_CHOICES = (
    ("烟雾或蒸汽", "smoke_or_steam"),
    ("灯光", "light"),
    ("高亮区域", "high_brightness"),
    ("夕阳或晚霞", "sunset_or_afterglow"),
    ("红色或橙色物体", "red_or_orange_object"),
    ("反光", "reflection"),
    ("焊接或火花", "welding_or_sparks"),
    ("云或雾", "cloud_or_fog"),
    ("重复或近重复", "duplicate_or_near_duplicate"),
    ("其他", "other"),
)


@dataclass(frozen=True)
class SmokeReviewSnapshot:
    """One validated on-disk snapshot and its optimistic-lock revision."""

    payload: dict[str, Any]
    revision: str
    progress: SmokeReviewV2Summary


def _human_name(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigurationError(f"{field} must be a non-empty human name")
    return value.strip()


def _notes(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise TrainingConfigurationError(f"{field} must be text")
    return value


def _timestamp(value: str | None = None) -> str:
    timestamp = value or datetime.now().astimezone().isoformat(timespec="seconds")
    if not isinstance(timestamp, str):
        raise TrainingConfigurationError("review timestamp must be text")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrainingConfigurationError("review timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TrainingConfigurationError("review timestamp must include a timezone")
    return timestamp


def _canonical_tags(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TrainingConfigurationError("Smoke2 tags must be a sequence of choices")
    tags = list(values)
    if not tags:
        raise TrainingConfigurationError("a human Smoke2 decision requires at least one tag")
    if len(tags) != len(set(tags)) or any(tag not in SMOKE_REVIEW_V2_TAGS for tag in tags):
        raise TrainingConfigurationError("Smoke2 tags contain duplicate or unknown choices")
    order = {tag: index for index, tag in enumerate(SMOKE_REVIEW_V2_TAGS)}
    return sorted(tags, key=order.__getitem__)


class SmokeReviewStore:
    """Validate and atomically persist explicit human Smoke2 v2 decisions."""

    def __init__(
        self,
        spec: Round3ReplaySpec,
        records: Sequence[FoldRecord],
        images_dir: Path,
    ) -> None:
        if spec.smoke_review_policy != SMOKE_REVIEW_V2_POLICY:
            raise TrainingConfigurationError(
                "Smoke2 v2 reviewer requires the isolated v2 replay configuration"
            )
        self.spec = spec
        self.records = tuple(records)
        self.images_dir = images_dir.resolve()
        if len(self.records) != spec.expected_count:
            raise TrainingConfigurationError(
                f"Smoke2 reviewer expects {spec.expected_count} Fold rows, "
                f"found {len(self.records)}"
            )
        self.negative_records = {
            record.filename: record for record in self.records if record.label == 0
        }
        if len(self.negative_records) != spec.expected_negative_count:
            raise TrainingConfigurationError(
                f"Smoke2 reviewer expects {spec.expected_negative_count} negatives, "
                f"found {len(self.negative_records)}"
            )
        if not self.images_dir.is_dir():
            raise TrainingConfigurationError(
                f"Smoke2 processed image directory is missing: {self.images_dir}"
            )
        self.filenames = tuple(sorted(self.negative_records))
        for filename in self.filenames:
            self._validated_image_path(filename)
        self.snapshot()

    @property
    def review_path(self) -> Path:
        """Return the canonical v2 review JSON path."""

        return self.spec.smoke_review

    @property
    def csv_path(self) -> Path:
        """Return the canonical finalized row-export path."""

        return self.review_path.parent / "smoke2_review.csv"

    @property
    def summary_path(self) -> Path:
        """Return the canonical finalized summary-export path."""

        return self.review_path.parent / "smoke2_review_summary.json"

    @property
    def lock_path(self) -> Path:
        """Return the exclusive cross-process writer lock path."""

        return self.review_path.with_suffix(self.review_path.suffix + ".lock")

    def _validated_image_path(self, filename: str) -> Path:
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise TrainingConfigurationError(
                f"Smoke2 filename must be a plain basename: {filename!r}"
            )
        path = (self.images_dir / filename).resolve()
        try:
            path.relative_to(self.images_dir)
        except ValueError as exc:
            raise TrainingConfigurationError(
                f"Smoke2 image path escapes the processed image directory: {filename}"
            ) from exc
        if not path.is_file():
            raise TrainingConfigurationError(f"Smoke2 processed image is missing: {path}")
        return path

    def image_path(self, filename: str) -> Path:
        """Resolve one known review image without permitting path traversal."""

        if filename not in self.negative_records:
            raise TrainingConfigurationError(f"unknown Smoke2 negative filename: {filename}")
        return self._validated_image_path(filename)

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise TrainingConfigurationError(
                "Smoke2 review is locked by another writer; retry after it finishes"
            ) from exc
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            yield
        finally:
            os.close(descriptor)
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def _load_raw(self) -> tuple[dict[str, Any], str]:
        try:
            raw = self.review_path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainingConfigurationError(
                f"cannot read Smoke2 v2 review {self.review_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise TrainingConfigurationError("Smoke2 v2 review must be a JSON object")
        return payload, hashlib.sha256(raw).hexdigest()

    def _validate(self, payload: Mapping[str, Any]) -> SmokeReviewV2Summary:
        return validate_smoke_review_v2_draft(
            payload,
            self.records,
            folds_path=self.spec.folds_path,
            project_root=self.spec.project_root,
            expected_negative_count=self.spec.expected_negative_count,
        )

    def snapshot(self) -> SmokeReviewSnapshot:
        """Load and validate the latest complete on-disk v2 snapshot."""

        payload, revision = self._load_raw()
        return SmokeReviewSnapshot(
            payload=payload,
            revision=revision,
            progress=self._validate(payload),
        )

    @staticmethod
    def _row(payload: Mapping[str, Any], filename: str) -> dict[str, Any]:
        rows = payload.get("records")
        if not isinstance(rows, list):
            raise TrainingConfigurationError("Smoke2 v2 records are unavailable")
        for value in rows:
            if isinstance(value, dict) and value.get("filename") == filename:
                return value
        raise TrainingConfigurationError(f"Smoke2 v2 review row is missing: {filename}")

    def row(self, snapshot: SmokeReviewSnapshot, filename: str) -> dict[str, Any]:
        """Return a copy of one validated review row."""

        if filename not in self.negative_records:
            raise TrainingConfigurationError(f"unknown Smoke2 negative filename: {filename}")
        return dict(self._row(snapshot.payload, filename))

    def resume_index(self, snapshot: SmokeReviewSnapshot) -> int:
        """Resume at the first primary-pending or secondary-required row."""

        secondary: int | None = None
        for index, filename in enumerate(self.filenames):
            row = self._row(snapshot.payload, filename)
            if row.get("disposition") is None:
                return index
            if (
                row.get("disposition") in {"uncertain", "suspected_label_error"}
                and row.get("secondary_disposition") is None
                and secondary is None
            ):
                secondary = index
        return secondary if secondary is not None else max(len(self.filenames) - 1, 0)

    @staticmethod
    def _make_pending(payload: dict[str, Any]) -> None:
        payload["status"] = "pending"
        payload["finalized_by"] = None
        payload["finalized_at"] = None
        rows = payload.get("records")
        if isinstance(rows, list):
            for value in rows:
                if isinstance(value, dict):
                    value["training_eligible"] = False

    def _invalidate_exports(self) -> None:
        for path in (self.csv_path, self.summary_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _check_expected(
        snapshot: SmokeReviewSnapshot,
        row: Mapping[str, Any],
        *,
        expected_revision: str,
        expected_row_revision: int,
    ) -> None:
        if snapshot.revision != expected_revision:
            raise TrainingConfigurationError(
                "Smoke2 review changed on disk; reload before saving this decision"
            )
        if row.get("revision") != expected_row_revision:
            raise TrainingConfigurationError(
                "Smoke2 row revision changed; reload before saving this decision"
            )

    def save_primary_decision(
        self,
        filename: str,
        *,
        disposition: str,
        tags: Sequence[str],
        reviewer: str,
        notes: str,
        expected_revision: str,
        expected_row_revision: int,
        reviewed_at: str | None = None,
    ) -> SmokeReviewSnapshot:
        """Atomically save one explicit primary decision and invalidate finalization."""

        if filename not in self.negative_records:
            raise TrainingConfigurationError(f"unknown Smoke2 negative filename: {filename}")
        if disposition not in SMOKE_REVIEW_V2_DISPOSITIONS:
            raise TrainingConfigurationError(
                f"Smoke2 disposition must be one of {list(SMOKE_REVIEW_V2_DISPOSITIONS)}"
            )
        normalized_tags = _canonical_tags(tags)
        human = _human_name(reviewer, "Smoke2 primary reviewer")
        human_notes = _notes(notes, "Smoke2 primary notes")
        if disposition != "confirmed_no_fire" and not human_notes.strip():
            raise TrainingConfigurationError(
                "uncertain or suspected-label-error primary decisions require notes"
            )
        timestamp = _timestamp(reviewed_at)
        with self._write_lock():
            current = self.snapshot()
            payload = deepcopy(current.payload)
            row = self._row(payload, filename)
            self._check_expected(
                current,
                row,
                expected_revision=expected_revision,
                expected_row_revision=expected_row_revision,
            )
            row.update(
                {
                    "disposition": disposition,
                    "tags": normalized_tags,
                    "primary_reviewer": human,
                    "primary_reviewed_at": timestamp,
                    "primary_notes": human_notes,
                    "secondary_reviewer": None,
                    "secondary_reviewed_at": None,
                    "secondary_disposition": None,
                    "secondary_notes": "",
                    "revision": expected_row_revision + 1,
                    "training_eligible": False,
                }
            )
            self._make_pending(payload)
            self._validate(payload)
            self._invalidate_exports()
            write_json_atomic(self.review_path, payload)
        return self.snapshot()

    def save_secondary_review(
        self,
        filename: str,
        *,
        conclusion: str,
        reviewer: str,
        notes: str,
        expected_revision: str,
        expected_row_revision: int,
        reviewed_at: str | None = None,
    ) -> SmokeReviewSnapshot:
        """Atomically close one required second review using a different human."""

        if filename not in self.negative_records:
            raise TrainingConfigurationError(f"unknown Smoke2 negative filename: {filename}")
        if conclusion not in SMOKE_REVIEW_V2_DISPOSITIONS:
            raise TrainingConfigurationError(
                f"Smoke2 secondary conclusion must be one of {list(SMOKE_REVIEW_V2_DISPOSITIONS)}"
            )
        human = _human_name(reviewer, "Smoke2 secondary reviewer")
        human_notes = _notes(notes, "Smoke2 secondary notes")
        if not human_notes.strip():
            raise TrainingConfigurationError("Smoke2 secondary review requires notes")
        timestamp = _timestamp(reviewed_at)
        with self._write_lock():
            current = self.snapshot()
            payload = deepcopy(current.payload)
            row = self._row(payload, filename)
            self._check_expected(
                current,
                row,
                expected_revision=expected_revision,
                expected_row_revision=expected_row_revision,
            )
            if row.get("disposition") not in {"uncertain", "suspected_label_error"}:
                raise TrainingConfigurationError(
                    "secondary review is allowed only after an uncertain or "
                    "suspected-label-error primary decision"
                )
            if row.get("primary_reviewer") == human:
                raise TrainingConfigurationError(
                    "Smoke2 secondary reviewer must differ from the primary reviewer"
                )
            row.update(
                {
                    "secondary_reviewer": human,
                    "secondary_reviewed_at": timestamp,
                    "secondary_disposition": conclusion,
                    "secondary_notes": human_notes,
                    "revision": expected_row_revision + 1,
                    "training_eligible": False,
                }
            )
            self._make_pending(payload)
            self._validate(payload)
            self._invalidate_exports()
            write_json_atomic(self.review_path, payload)
        return self.snapshot()

    @staticmethod
    def _final_disposition(row: Mapping[str, Any]) -> str | None:
        disposition = row.get("disposition")
        if disposition in {"uncertain", "suspected_label_error"}:
            secondary = row.get("secondary_disposition")
            return secondary if isinstance(secondary, str) else None
        return disposition if isinstance(disposition, str) else None

    def _summary_payload(
        self,
        payload: Mapping[str, Any],
        summary: SmokeReviewV2Summary,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "round3_smoke_review_v2_summary",
            "review": self.review_path.relative_to(self.spec.project_root).as_posix(),
            "status": payload.get("status"),
            "total": summary.total,
            "complete": summary.complete,
            "pending": summary.pending,
            "secondary_required": summary.secondary_required,
            "training_eligible": summary.training_eligible,
            "excluded": summary.excluded,
            "final_dispositions": summary.final_dispositions,
            "tags": summary.tags,
            "folds": summary.folds,
            "unique_groups": len({record.group_id for record in self.negative_records.values()}),
            "eligibility_policy": ("bundle approved and final disposition confirmed_no_fire"),
        }

    def _publish_exports(
        self,
        payload: Mapping[str, Any],
        summary: SmokeReviewV2Summary,
    ) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_temporary = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        summary_temporary = self.summary_path.with_suffix(self.summary_path.suffix + ".tmp")
        fieldnames = [
            "filename",
            "label",
            "fold",
            "group_id",
            "disposition",
            "tags",
            "primary_reviewer",
            "primary_reviewed_at",
            "primary_notes",
            "secondary_reviewer",
            "secondary_reviewed_at",
            "secondary_disposition",
            "secondary_notes",
            "final_disposition",
            "revision",
            "training_eligible",
        ]
        with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            rows = payload.get("records")
            if not isinstance(rows, list):
                raise TrainingConfigurationError("Smoke2 v2 records are unavailable")
            for raw in rows:
                row = dict(raw)
                row["tags"] = "|".join(row.get("tags", []))
                row["final_disposition"] = self._final_disposition(raw)
                writer.writerow({field: row.get(field) for field in fieldnames})
        summary_temporary.write_text(
            json.dumps(
                self._summary_payload(payload, summary),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(csv_temporary, self.csv_path)
        os.replace(summary_temporary, self.summary_path)

    def finalize(
        self,
        *,
        reviewer: str,
        expected_revision: str,
        finalized_at: str | None = None,
    ) -> SmokeReviewSnapshot:
        """Approve a complete bundle and atomically publish its CSV/summary exports."""

        human = _human_name(reviewer, "Smoke2 bundle finalizer")
        timestamp = _timestamp(finalized_at)
        with self._write_lock():
            current = self.snapshot()
            if current.revision != expected_revision:
                raise TrainingConfigurationError(
                    "Smoke2 review changed on disk; reload before finalizing"
                )
            if current.progress.remaining:
                raise TrainingConfigurationError(
                    "Smoke2 v2 cannot be finalized: "
                    f"pending={current.progress.pending}, "
                    f"secondary_required={current.progress.secondary_required}"
                )
            payload = deepcopy(current.payload)
            payload["status"] = "approved"
            payload["finalized_by"] = human
            payload["finalized_at"] = timestamp
            rows = payload.get("records")
            if not isinstance(rows, list):
                raise TrainingConfigurationError("Smoke2 v2 records are unavailable")
            for row in rows:
                if not isinstance(row, dict):
                    raise TrainingConfigurationError("Smoke2 v2 record is malformed")
                row["training_eligible"] = self._final_disposition(row) == "confirmed_no_fire"
            summary = self._validate(payload)
            self._publish_exports(payload, summary)
            write_json_atomic(self.review_path, payload)
        return self.snapshot()

    def export_review(self, *, expected_revision: str) -> SmokeReviewSnapshot:
        """Rebuild canonical exports for an already approved unchanged bundle."""

        with self._write_lock():
            current = self.snapshot()
            if current.revision != expected_revision:
                raise TrainingConfigurationError(
                    "Smoke2 review changed on disk; reload before exporting"
                )
            if current.payload.get("status") != "approved":
                raise TrainingConfigurationError("only an approved Smoke2 v2 bundle can export")
            self._publish_exports(current.payload, current.progress)
        return self.snapshot()


def create_smoke_review_app(
    store: SmokeReviewStore,
    *,
    default_primary_reviewer: str = "Admin",
) -> Any:
    """Build, but do not launch, the local-only Smoke2 v2 human-review Gradio app."""

    import gradio as gr

    def progress_text(snapshot: SmokeReviewSnapshot) -> str:
        progress = snapshot.progress
        return (
            f"人工审核：完成 **{progress.complete}/{progress.total}**；"
            f"待主审 **{progress.pending}**；待二审 **{progress.secondary_required}**。  \n"
            f"训练可用 **{progress.training_eligible}**；明确排除 **{progress.excluded}**；"
            f"Bundle：`{snapshot.payload.get('status')}`。"
        )

    def render(state: dict[str, Any]) -> tuple[Any, ...]:
        snapshot = store.snapshot()
        index = min(max(int(state.get("index", 0)), 0), len(store.filenames) - 1)
        filename = store.filenames[index]
        row = store.row(snapshot, filename)
        state = {
            "index": index,
            "filename": filename,
            "revision": snapshot.revision,
            "row_revision": row["revision"],
        }
        details = (
            f"记录 **{index + 1}/{len(store.filenames)}**  \n"
            f"文件名：`{filename}`  \n"
            f"Fold：`{row['fold']}`；场景组：`{row['group_id']}`；"
            f"行 revision：`{row['revision']}`"
        )
        return (
            state,
            str(store.image_path(filename)),
            details,
            row.get("disposition"),
            row.get("tags", []),
            row.get("primary_reviewer") or default_primary_reviewer,
            row.get("primary_notes", ""),
            row.get("secondary_disposition"),
            row.get("secondary_reviewer") or "",
            row.get("secondary_notes", ""),
            "",
            progress_text(snapshot),
        )

    def load_session() -> tuple[Any, ...]:
        snapshot = store.snapshot()
        return render({"index": store.resume_index(snapshot)})

    def move(state: dict[str, Any], delta: int) -> tuple[Any, ...]:
        return render({"index": (int(state.get("index", 0)) + delta) % len(store.filenames)})

    def save_primary(
        state: dict[str, Any],
        disposition: str,
        tags: list[str],
        reviewer: str,
        notes: str,
    ) -> tuple[Any, ...]:
        filename = state.get("filename")
        if not isinstance(filename, str):
            raise gr.Error("当前审核记录无效，请重新加载。")
        try:
            store.save_primary_decision(
                filename,
                disposition=disposition,
                tags=tags,
                reviewer=reviewer,
                notes=notes,
                expected_revision=str(state.get("revision", "")),
                expected_row_revision=int(state.get("row_revision", -1)),
            )
        except (TrainingConfigurationError, TypeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        return render({"index": (int(state.get("index", 0)) + 1) % len(store.filenames)})

    def save_secondary(
        state: dict[str, Any],
        conclusion: str,
        reviewer: str,
        notes: str,
    ) -> tuple[Any, ...]:
        filename = state.get("filename")
        if not isinstance(filename, str):
            raise gr.Error("当前审核记录无效，请重新加载。")
        try:
            store.save_secondary_review(
                filename,
                conclusion=conclusion,
                reviewer=reviewer,
                notes=notes,
                expected_revision=str(state.get("revision", "")),
                expected_row_revision=int(state.get("row_revision", -1)),
            )
        except (TrainingConfigurationError, TypeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        return render({"index": (int(state.get("index", 0)) + 1) % len(store.filenames)})

    def finalize_bundle(state: dict[str, Any], reviewer: str) -> tuple[Any, ...]:
        try:
            store.finalize(
                reviewer=reviewer,
                expected_revision=str(state.get("revision", "")),
            )
        except TrainingConfigurationError as exc:
            raise gr.Error(str(exc)) from exc
        return render(state)

    initial = store.snapshot()
    with gr.Blocks(title="Round 3 Smoke2 v2 人工审核", analytics_enabled=False) as app:
        gr.Markdown("# Round 3 Smoke2 v2 人工语义审核")
        gr.Markdown(
            "工具不会自动识别或推断语义。无法确认或疑似标签错误的主审结论，"
            "必须由另一名人工审核员完成二审；保存不会自动 Finalize。"
        )
        progress = gr.Markdown(progress_text(initial))
        state = gr.State({"index": store.resume_index(initial)})
        image = gr.Image(label="当前负样本原图", interactive=False)
        details = gr.Markdown()
        with gr.Row():
            previous_button = gr.Button("上一张")
            next_button = gr.Button("下一张")
        disposition = gr.Radio(
            choices=list(SMOKE_DISPOSITION_CHOICES),
            value=None,
            label="主审结论",
        )
        tags = gr.CheckboxGroup(
            choices=list(SMOKE_TAG_CHOICES),
            value=[],
            label="人工场景标签（可多选，至少一项）",
        )
        primary_reviewer = gr.Textbox(
            label="主审人（必填）",
            value=default_primary_reviewer,
        )
        primary_notes = gr.Textbox(label="主审说明（存疑/疑似错标时必填）", lines=2)
        save_primary_button = gr.Button("保存主审并下一张", variant="primary")
        gr.Markdown("## 必要二审")
        secondary_disposition = gr.Radio(
            choices=list(SMOKE_DISPOSITION_CHOICES),
            value=None,
            label="二审结论",
        )
        secondary_reviewer = gr.Textbox(label="二审人（必须与主审人不同）", value="")
        secondary_notes = gr.Textbox(label="二审说明（必填）", lines=2)
        save_secondary_button = gr.Button("保存二审并下一张")
        gr.Markdown("## Bundle 封板")
        finalizer = gr.Textbox(label="Finalize 审核人（必填，不预填身份）", value="")
        finalize_button = gr.Button("显式 Finalize 并导出 CSV/summary", variant="stop")

        outputs = [
            state,
            image,
            details,
            disposition,
            tags,
            primary_reviewer,
            primary_notes,
            secondary_disposition,
            secondary_reviewer,
            secondary_notes,
            finalizer,
            progress,
        ]
        previous_button.click(lambda current: move(current, -1), state, outputs)
        next_button.click(lambda current: move(current, 1), state, outputs)
        save_primary_button.click(
            save_primary,
            [state, disposition, tags, primary_reviewer, primary_notes],
            outputs,
            show_progress="minimal",
            concurrency_id="smoke-review-v2-write",
            concurrency_limit=1,
            trigger_mode="once",
        )
        save_secondary_button.click(
            save_secondary,
            [state, secondary_disposition, secondary_reviewer, secondary_notes],
            outputs,
            show_progress="minimal",
            concurrency_id="smoke-review-v2-write",
            concurrency_limit=1,
            trigger_mode="once",
        )
        finalize_button.click(
            finalize_bundle,
            [state, finalizer],
            outputs,
            show_progress="minimal",
            concurrency_id="smoke-review-v2-write",
            concurrency_limit=1,
            trigger_mode="once",
        )
        app.load(load_session, outputs=outputs)
    return app
