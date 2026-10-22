"""Resumable local Gradio reviewer for the frozen winner FP/FN draft.

The browser state is only a cursor. Every render reloads and validates the
canonical draft, and every write delegates to ``winner_error_review`` so the
mechanical reports, SHA guard, vocabulary, and finalization contract remain the
single source of truth.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fire_detection.ensemble.winner_error_review import (
    WINNER_FN_ERROR_CATEGORIES,
    WINNER_FP_ERROR_CATEGORIES,
    WINNER_SUGGESTED_ACTIONS,
    ValidatedWinnerErrorReview,
    WinnerErrorReviewError,
    finalize_winner_error_review,
    save_primary_winner_error_review,
    save_secondary_winner_error_review,
    validate_winner_error_review,
)

LABEL_ISSUE_CHOICES = (
    ("否：无标签问题，主审直接完成", "no"),
    ("是：疑似标签问题，必须二审", "yes"),
    ("无法确定：必须二审", "uncertain"),
)
SECONDARY_CONCLUSION_CHOICES = (
    ("确认无标签问题", "no_label_issue"),
    ("确认标签问题", "label_issue"),
    ("仍无法解决", "unresolved"),
)
DEFAULT_PRIMARY_REVIEWER = "Admin"


def _inside_project(path: Path, root: Path, *, field: str) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WinnerErrorReviewError(f"{field} must remain inside project_root: {path}") from exc
    return resolved


class WinnerErrorReviewStore:
    """Read images/mechanical context and delegate guarded human decisions."""

    def __init__(
        self,
        review_path: Path,
        images_dir: Path,
        output_dir: Path,
        *,
        project_root: Path,
    ) -> None:
        self.project_root = project_root.resolve()
        self.review_path = _inside_project(
            review_path, self.project_root, field="winner error review path"
        )
        self.images_dir = _inside_project(
            images_dir, self.project_root, field="winner error image directory"
        )
        self.output_dir = _inside_project(
            output_dir, self.project_root, field="winner error finalized output"
        )
        if not self.images_dir.is_dir():
            raise WinnerErrorReviewError(
                f"winner error processed image directory is missing: {self.images_dir}"
            )

        snapshot = self.snapshot()
        raw_records = snapshot.payload.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise WinnerErrorReviewError("winner error review has no records")
        self.filenames = tuple(str(record["filename"]) for record in raw_records)
        self._mechanical_rows = self._load_mechanical_rows(snapshot)
        expected = [(str(record["filename"]), str(record["confusion"])) for record in raw_records]
        actual = [
            (filename, str(self._mechanical_rows[filename].get("confusion", "")))
            for filename in self.filenames
        ]
        if actual != expected:
            raise WinnerErrorReviewError(
                "winner error mechanical rows do not match the validated draft order"
            )
        for filename in self.filenames:
            self._validated_image_path(filename)

    def _source_path(self, snapshot: ValidatedWinnerErrorReview, source_name: str) -> Path:
        source = snapshot.payload.get("source")
        if not isinstance(source, dict):
            raise WinnerErrorReviewError("winner error review source is unavailable")
        record = source.get(source_name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise WinnerErrorReviewError(f"winner error source {source_name} is unavailable")
        return _inside_project(
            Path(record["path"]),
            self.project_root,
            field=f"winner error source {source_name}",
        )

    def _load_mechanical_rows(
        self, snapshot: ValidatedWinnerErrorReview
    ) -> dict[str, dict[str, str]]:
        rows: dict[str, dict[str, str]] = {}
        for source_name in ("fp_report", "fn_report"):
            path = self._source_path(snapshot, source_name)
            try:
                with path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for raw in reader:
                        filename = str(raw.get("filename", ""))
                        if not filename or filename in rows:
                            raise WinnerErrorReviewError(
                                "winner error mechanical reports contain a missing "
                                "or duplicate filename"
                            )
                        rows[filename] = dict(raw)
            except (OSError, UnicodeError, csv.Error) as exc:
                raise WinnerErrorReviewError(
                    f"cannot read winner error mechanical report {path}: {exc}"
                ) from exc
        return rows

    def _validated_image_path(self, filename: str) -> Path:
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise WinnerErrorReviewError(
                f"winner error filename must be a plain basename: {filename!r}"
            )
        path = (self.images_dir / filename).resolve()
        try:
            path.relative_to(self.images_dir)
        except ValueError as exc:
            raise WinnerErrorReviewError(
                f"winner error image path escapes the processed image directory: {filename}"
            ) from exc
        if not path.is_file():
            raise WinnerErrorReviewError(f"winner error processed image is missing: {path}")
        return path

    def _ensure_not_published(self) -> None:
        if self.output_dir.exists():
            raise WinnerErrorReviewError(
                "winner error finalized bundle already exists; the draft is now read-only"
            )

    def snapshot(self) -> ValidatedWinnerErrorReview:
        """Reload and strictly validate the latest canonical draft."""

        return validate_winner_error_review(
            self.review_path,
            project_root=self.project_root,
        )

    def row(self, snapshot: ValidatedWinnerErrorReview, filename: str) -> dict[str, Any]:
        """Return one validated human-review row without exposing mutable state."""

        matches = [
            record for record in snapshot.payload["records"] if record.get("filename") == filename
        ]
        if len(matches) != 1 or filename not in self._mechanical_rows:
            raise WinnerErrorReviewError(
                f"winner error review filename not found exactly once: {filename}"
            )
        return dict(matches[0])

    def mechanical_row(self, filename: str) -> dict[str, str]:
        """Return read-only mechanical context for one known filename."""

        try:
            return dict(self._mechanical_rows[filename])
        except KeyError as exc:
            raise WinnerErrorReviewError(
                f"unknown winner error mechanical filename: {filename}"
            ) from exc

    def image_path(self, filename: str) -> Path:
        """Resolve one known original image without allowing path traversal."""

        if filename not in self._mechanical_rows:
            raise WinnerErrorReviewError(f"unknown winner error filename: {filename}")
        return self._validated_image_path(filename)

    def resume_index(self, snapshot: ValidatedWinnerErrorReview) -> int:
        """Resume at the first primary-pending row, then the first required secondary."""

        secondary: int | None = None
        for index, filename in enumerate(self.filenames):
            status = self.row(snapshot, filename)["review_status"]
            if status == "pending":
                return index
            if status == "needs_second_review" and secondary is None:
                secondary = index
        return secondary if secondary is not None else len(self.filenames) - 1

    def next_work_index(self, snapshot: ValidatedWinnerErrorReview, current_index: int) -> int:
        """Find the next pending task cyclically while keeping primary work first."""

        count = len(self.filenames)
        for wanted in ("pending", "needs_second_review"):
            for offset in range(1, count + 1):
                index = (current_index + offset) % count
                if self.row(snapshot, self.filenames[index])["review_status"] == wanted:
                    return index
        return min(max(current_index, 0), count - 1)

    def save_primary(
        self,
        filename: str,
        *,
        error_category: str,
        suspected_label_issue: str,
        suggested_action: str,
        reviewer: str,
        expected_sha256: str,
        notes: str = "",
        reviewed_at: str | None = None,
    ) -> ValidatedWinnerErrorReview:
        """Save one primary decision through the canonical guarded backend."""

        self._ensure_not_published()
        return save_primary_winner_error_review(
            self.review_path,
            project_root=self.project_root,
            finalized_output_dir=self.output_dir,
            filename=filename,
            error_category=error_category,
            suspected_label_issue=suspected_label_issue,
            suggested_action=suggested_action,
            reviewer=reviewer,
            notes=notes,
            reviewed_at=reviewed_at,
            expected_sha256=expected_sha256,
        )

    def save_secondary(
        self,
        filename: str,
        *,
        label_issue_conclusion: str,
        reviewer: str,
        expected_sha256: str,
        notes: str = "",
        reviewed_at: str | None = None,
    ) -> ValidatedWinnerErrorReview:
        """Save one independent secondary decision through the canonical backend."""

        self._ensure_not_published()
        return save_secondary_winner_error_review(
            self.review_path,
            project_root=self.project_root,
            finalized_output_dir=self.output_dir,
            filename=filename,
            label_issue_conclusion=label_issue_conclusion,
            reviewer=reviewer,
            notes=notes,
            reviewed_at=reviewed_at,
            expected_sha256=expected_sha256,
        )

    def finalize(
        self,
        *,
        finalizer: str,
        expected_sha256: str,
        finalized_at: str | None = None,
    ) -> dict[str, Any]:
        """Explicitly publish the immutable bundle without changing the draft."""

        self._ensure_not_published()
        return finalize_winner_error_review(
            self.review_path,
            project_root=self.project_root,
            output_dir=self.output_dir,
            finalizer=finalizer,
            finalized_at=finalized_at,
            expected_sha256=expected_sha256,
        )


def create_winner_error_review_app(store: WinnerErrorReviewStore) -> Any:
    """Build, but do not launch, the local-only winner error review application."""

    import gradio as gr

    def progress_text(snapshot: ValidatedWinnerErrorReview) -> str:
        progress = snapshot.progress
        reviewed_categories = {
            str(row["error_category"])
            for row in snapshot.payload["records"]
            if row["review_status"] == "reviewed"
        }
        bundle = "已发布（draft 只读）" if store.output_dir.exists() else "未发布"
        category_warning = ""
        if progress.complete and len(reviewed_categories) < 3:
            category_warning = "  \n⚠️ 已完成但错误类别不足三类，seal 仍会阻塞。"
        return (
            f"人工审核：完成 **{progress.reviewed}/{progress.total}**；"
            f"待主审 **{progress.pending}**；待二审 **{progress.needs_second_review}**。  \n"
            f"Draft：`{snapshot.payload['status']}`；SHA：`{snapshot.sha256}`；"
            f"最终 bundle：**{bundle}**。{category_warning}"
        )

    def details_text(
        snapshot: ValidatedWinnerErrorReview,
        row: dict[str, Any],
        mechanical: dict[str, str],
        index: int,
    ) -> str:
        filename = str(row["filename"]).replace("`", "'")
        primary = str(row.get("reviewer") or "未填写")
        secondary = str(row.get("second_reviewer") or "未填写")
        return (
            f"记录 **{index + 1}/{len(store.filenames)}**；混淆：**{row['confusion']}**；"
            f"状态：`{row['review_status']}`；行 revision：`{row['revision']}`  \n"
            f"文件名：`{filename}`  \n"
            f"Fold：`{mechanical.get('fold', '')}`；score：`{mechanical.get('score', '')}`；"
            f"threshold：`{mechanical.get('threshold', '')}`；"
            f"margin：`{mechanical.get('margin', '')}`  \n"
            f"机械归因：`{mechanical.get('mechanical_attribution', '')}`；"
            f"detector：`{mechanical.get('detector_independent_score', '')}`；"
            f"classifier：`{mechanical.get('classifier_independent_score', '')}`  \n"
            f"已存主审：`{primary}`；已存二审：`{secondary}`；"
            f"当前 draft SHA：`{snapshot.sha256}`"
        )

    def render(index: int, message: str = "") -> tuple[Any, ...]:
        snapshot = store.snapshot()
        safe_index = min(max(int(index), 0), len(store.filenames) - 1)
        filename = store.filenames[safe_index]
        row = store.row(snapshot, filename)
        mechanical = store.mechanical_row(filename)
        published = store.output_dir.exists()
        needs_secondary = row["review_status"] == "needs_second_review"
        categories = (
            WINNER_FP_ERROR_CATEGORIES if row["confusion"] == "FP" else WINNER_FN_ERROR_CATEGORIES
        )
        category = row["error_category"] or None
        label_issue = row["suspected_label_issue"]
        if label_issue == "pending":
            label_issue = None
        action = row["suggested_action"] or None
        conclusion = row["label_issue_conclusion"]
        if conclusion not in {"no_label_issue", "label_issue", "unresolved"}:
            conclusion = None
        state = {
            "index": safe_index,
            "filename": filename,
            "sha256": snapshot.sha256,
            "row_revision": row["revision"],
        }
        return (
            state,
            str(store.image_path(filename)),
            details_text(snapshot, row, mechanical, safe_index),
            gr.update(choices=list(categories), value=category, interactive=not published),
            gr.update(value=label_issue, interactive=not published),
            gr.update(value=action, interactive=not published),
            gr.update(value=row["notes"], interactive=not published),
            gr.update(value=conclusion, interactive=needs_secondary and not published),
            gr.update(
                value=row["second_review_notes"], interactive=needs_secondary and not published
            ),
            progress_text(snapshot),
            message,
            gr.update(interactive=not published),
            gr.update(interactive=needs_secondary and not published),
            gr.update(
                interactive=(
                    snapshot.progress.complete
                    and snapshot.payload["status"] == "ready_for_finalize"
                    and not published
                )
            ),
        )

    def load_session() -> tuple[Any, ...]:
        snapshot = store.snapshot()
        return render(store.resume_index(snapshot))

    def move(state: dict[str, Any], delta: int) -> tuple[Any, ...]:
        index = (int(state.get("index", 0)) + delta) % len(store.filenames)
        return render(index)

    def reload_session() -> tuple[Any, ...]:
        snapshot = store.snapshot()
        return render(store.resume_index(snapshot), "已从磁盘重新加载最新 draft。")

    def next_work(state: dict[str, Any]) -> tuple[Any, ...]:
        snapshot = store.snapshot()
        index = store.next_work_index(snapshot, int(state.get("index", 0)))
        return render(index)

    def save_primary(
        state: dict[str, Any],
        error_category: str | None,
        suspected_label_issue: str | None,
        suggested_action: str | None,
        reviewer: str,
        notes: str,
    ) -> tuple[Any, ...]:
        filename = state.get("filename")
        if not isinstance(filename, str):
            raise gr.Error("当前审核记录无效，请重新加载。")
        try:
            saved = store.save_primary(
                filename,
                error_category=error_category or "",
                suspected_label_issue=suspected_label_issue or "",
                suggested_action=suggested_action or "",
                reviewer=reviewer,
                notes=notes,
                expected_sha256=str(state.get("sha256", "")),
            )
        except (WinnerErrorReviewError, FileExistsError, TypeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        index = store.next_work_index(saved, int(state.get("index", 0)))
        return render(index, f"✅ 已保存主审：`{filename}`；draft SHA 已刷新。")

    def save_secondary(
        state: dict[str, Any],
        conclusion: str | None,
        reviewer: str,
        notes: str,
    ) -> tuple[Any, ...]:
        filename = state.get("filename")
        if not isinstance(filename, str):
            raise gr.Error("当前审核记录无效，请重新加载。")
        try:
            saved = store.save_secondary(
                filename,
                label_issue_conclusion=conclusion or "",
                reviewer=reviewer,
                notes=notes,
                expected_sha256=str(state.get("sha256", "")),
            )
        except (WinnerErrorReviewError, FileExistsError, TypeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        index = store.next_work_index(saved, int(state.get("index", 0)))
        return render(index, f"✅ 已保存二审：`{filename}`；draft SHA 已刷新。")

    def finalize_bundle(state: dict[str, Any], finalizer: str) -> tuple[Any, ...]:
        try:
            store.finalize(
                finalizer=finalizer,
                expected_sha256=str(state.get("sha256", "")),
            )
        except (WinnerErrorReviewError, FileExistsError, TypeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        return render(
            int(state.get("index", 0)),
            f"✅ Finalize 已原子发布到 `{store.output_dir}`；draft 保持 ready_for_finalize。",
        )

    initial = store.snapshot()
    published = store.output_dir.exists()
    with gr.Blocks(title="Round 3 winner FP/FN 人工审核", analytics_enabled=False) as app:
        gr.Markdown("# Round 3 winner 107 条 FP/FN 人工审核")
        gr.Markdown(
            "本工具不会自动识别、预填或修改机械 CSV，也不会执行 init。每次保存都写入独立 draft "
            "并刷新 SHA；关闭页面后可从首个待办恢复。`yes/uncertain` 必须由另一名真人二审，"
            "最后一条保存不会自动 Finalize。"
        )
        progress = gr.Markdown(progress_text(initial))
        status_message = gr.Markdown("")
        state = gr.State({"index": store.resume_index(initial)})
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="当前错误原图", interactive=False)
                details = gr.Markdown()
                with gr.Row():
                    previous_button = gr.Button("上一条")
                    next_button = gr.Button("下一条")
                    next_work_button = gr.Button("下一个待办")
                    reload_button = gr.Button("重新加载/恢复")
            with gr.Column(scale=2):
                gr.Markdown("## 主审")
                error_category = gr.Dropdown(
                    choices=list(WINNER_FP_ERROR_CATEGORIES),
                    value=None,
                    allow_custom_value=False,
                    filterable=False,
                    label="统一 error_category（FP/FN 自动切换）",
                    interactive=not published,
                )
                suspected_label_issue = gr.Radio(
                    choices=list(LABEL_ISSUE_CHOICES),
                    value=None,
                    label="是否疑似标签问题",
                    interactive=not published,
                )
                suggested_action = gr.Radio(
                    choices=list(WINNER_SUGGESTED_ACTIONS),
                    value=None,
                    label="统一 suggested_action",
                    interactive=not published,
                )
                primary_reviewer = gr.Textbox(
                    label="本次主审真实姓名（必填，默认 Admin）",
                    value=DEFAULT_PRIMARY_REVIEWER,
                )
                primary_notes = gr.Textbox(label="主审观察说明（可选）", lines=2)
                save_primary_button = gr.Button(
                    "保存主审并跳到下一待办", variant="primary", interactive=not published
                )

                gr.Markdown("## 必要二审")
                secondary_conclusion = gr.Radio(
                    choices=list(SECONDARY_CONCLUSION_CHOICES),
                    value=None,
                    label="二审标签结论",
                    interactive=False,
                )
                secondary_reviewer = gr.Textbox(
                    label="本次二审真实姓名（必须与主审不同）", value=""
                )
                secondary_notes = gr.Textbox(label="二审依据（可选）", lines=2, interactive=False)
                save_secondary_button = gr.Button("保存二审并跳到下一待办", interactive=False)

                gr.Markdown("## Bundle 封板")
                finalizer = gr.Textbox(label="Finalize 真人姓名（必填，不预填身份）", value="")
                finalize_button = gr.Button(
                    "显式 Finalize（不可覆盖）", variant="stop", interactive=False
                )

        outputs = [
            state,
            image,
            details,
            error_category,
            suspected_label_issue,
            suggested_action,
            primary_notes,
            secondary_conclusion,
            secondary_notes,
            progress,
            status_message,
            save_primary_button,
            save_secondary_button,
            finalize_button,
        ]
        previous_button.click(lambda current: move(current, -1), state, outputs)
        next_button.click(lambda current: move(current, 1), state, outputs)
        next_work_button.click(next_work, state, outputs)
        reload_button.click(reload_session, outputs=outputs)
        save_primary_button.click(
            save_primary,
            [
                state,
                error_category,
                suspected_label_issue,
                suggested_action,
                primary_reviewer,
                primary_notes,
            ],
            outputs,
            show_progress="minimal",
            concurrency_id="winner-error-review-write",
            concurrency_limit=1,
            trigger_mode="once",
        )
        save_secondary_button.click(
            save_secondary,
            [state, secondary_conclusion, secondary_reviewer, secondary_notes],
            outputs,
            show_progress="minimal",
            concurrency_id="winner-error-review-write",
            concurrency_limit=1,
            trigger_mode="once",
        )
        finalize_button.click(
            finalize_bundle,
            [state, finalizer],
            outputs,
            show_progress="minimal",
            concurrency_id="winner-error-review-write",
            concurrency_limit=1,
            trigger_mode="once",
        )
        app.load(load_session, outputs=outputs)
    return app
