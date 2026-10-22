"""Resumable Gradio reviewer for mandatory scene-pair decisions."""

from __future__ import annotations

import csv
import hashlib
import importlib
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from fire_detection.data.grouping import (
    VALID_DECISIONS,
    CandidatePair,
    ReviewDecision,
    read_candidates,
    read_decisions,
    write_decisions,
)

DECISION_STATUSES = (
    "incomplete",
    "secondary_pending",
    "completed",
    "same_scene",
    "different_scene",
    "uncertain",
    "all",
)
DECISION_LABELS = {
    "same_scene": "同一场景",
    "different_scene": "不同场景",
    "uncertain": "无法确认",
}
DECISION_STATUS_LABELS = {
    "incomplete": "未审核",
    "secondary_pending": "待二次复核",
    "completed": "已审核",
    **DECISION_LABELS,
    "all": "全部",
}
SOURCE_LABELS = {
    "automatic_similarity": "自动高相似",
    "dimension_mismatch": "尺寸不一致",
    "embedding_similarity": "场景特征相似",
    "label_conflict": "标签冲突",
    "manual_similarity": "人工复核相似",
    "numeric_adjacency": "数字序列相邻",
    "phash_near": "感知哈希近似",
    "scene_embedding": "场景特征近邻",
}
LARGE_COMPONENT_REVIEW_THRESHOLD = 10
_REVIEW_WRITE_RETRY_ATTEMPTS = 12
_REVIEW_WRITE_INITIAL_DELAY_SECONDS = 0.01
_REVIEW_WRITE_MAX_DELAY_SECONDS = 0.25
_HISTORY_FIELDS = (
    "pair_id",
    "filename_a",
    "filename_b",
    "decision",
    "previous_decision",
    "reviewer",
    "reviewed_at",
    "notes",
    "round",
)

REVIEWER_SHORTCUT_JS = r"""
(() => {
  if (window.__fireReviewerShortcutsInstalled) return;
  window.__fireReviewerShortcutsInstalled = true;
  window.__fireReviewerShortcutBusy = false;
  window.__fireReviewerSaveBusy = false;
  window.__fireReviewerSaveBusyTimer = null;

  const uiTranslations = new Map([
    ["Fullscreen", "全屏"],
    ["Exit fullscreen", "退出全屏"],
    ["Exit fullscreen mode", "退出全屏"],
    ["Download All", "全部下载"],
    ["Download", "下载"],
    ["Share", "分享"],
    ["Clear", "清除"],
    ["Warning", "警告"],
    ["Warning Close", "关闭警告"],
    ["Close", "关闭"],
  ]);

  const translateUiText = (value) => {
    const normalized = value.trim();
    const exact = uiTranslations.get(normalized);
    if (exact) return exact;
    const thumbnail = normalized.match(/^Thumbnail (\d+) of (\d+)$/);
    return thumbnail ? `缩略图 ${thumbnail[1]}，共 ${thumbnail[2]} 张` : null;
  };

  const localizeUi = (root) => {
    if (!(root instanceof Element)) return;
    const elements = [root, ...root.querySelectorAll("*")];
    for (const element of elements) {
      for (const attribute of ["aria-label", "title"]) {
        const value = element.getAttribute(attribute);
        const translated = value ? translateUiText(value) : null;
        if (translated) element.setAttribute(attribute, translated);
      }
      if (element.children.length === 0) {
        const value = element.textContent?.trim();
        const translated = value ? translateUiText(value) : null;
        if (translated) element.textContent = translated;
      }
    }
  };

  const localizationObserver = new MutationObserver((mutations) => {
    for (const mutation of mutations) localizeUi(mutation.target);
  });
  localizationObserver.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["aria-label", "title"],
  });
  localizeUi(document.body);

  const saveButtonIds = new Set([
    "review-same-scene",
    "review-different-scene",
    "review-uncertain",
  ]);

  const containingButtonId = (target) => {
    if (!(target instanceof Element)) return null;
    for (const id of saveButtonIds) {
      const container = document.getElementById(id);
      if (container && (container === target || container.contains(target))) return id;
    }
    return null;
  };

  const setSaveButtonsDisabled = (disabled) => {
    for (const id of saveButtonIds) {
      const target = document.getElementById(id);
      const button = target?.matches("button") ? target : target?.querySelector("button");
      if (!button) continue;
      button.disabled = disabled;
      button.setAttribute("aria-busy", disabled ? "true" : "false");
    }
  };

  window.__fireReviewerReleaseSaveBusy = () => {
    window.__fireReviewerSaveBusy = false;
    if (window.__fireReviewerSaveBusyTimer !== null) {
      window.clearTimeout(window.__fireReviewerSaveBusyTimer);
      window.__fireReviewerSaveBusyTimer = null;
    }
    setSaveButtonsDisabled(false);
  };

  document.addEventListener("click", (event) => {
    const saveButtonId = containingButtonId(event.target);
    if (!saveButtonId) return;
    if (window.__fireReviewerSaveBusy) {
      event.preventDefault();
      event.stopImmediatePropagation();
      return;
    }
    window.__fireReviewerSaveBusy = true;
    window.__fireReviewerSaveBusyTimer = window.setTimeout(
      window.__fireReviewerReleaseSaveBusy,
      45000,
    );
    window.setTimeout(() => setSaveButtonsDisabled(true), 0);
  }, true);

  const clickButton = (id) => {
    const target = document.getElementById(id);
    const button = target?.matches("button") ? target : target?.querySelector("button");
    if (!button || button.disabled || button.getAttribute("aria-disabled") === "true") return;
    button.click();
  };

  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const isEditing = target instanceof HTMLElement && Boolean(
      target.closest('input, textarea, select, [contenteditable="true"], [role="textbox"]')
    );
    if (
      isEditing || event.repeat || window.__fireReviewerShortcutBusy ||
      event.ctrlKey || event.metaKey || event.altKey
    ) return;

    const buttons = {
      "1": "review-same-scene",
      "2": "review-different-scene",
      "3": "review-uncertain",
      "ArrowLeft": "review-previous",
      "ArrowRight": "review-next",
    };
    const buttonId = buttons[event.key];
    if (!buttonId) return;
    if (saveButtonIds.has(buttonId) && window.__fireReviewerSaveBusy) return;
    event.preventDefault();
    window.__fireReviewerShortcutBusy = true;
    window.setTimeout(() => {
      window.__fireReviewerShortcutBusy = false;
    }, 200);
    clickButton(buttonId);
  });
})();
"""

REVIEWER_RELEASE_SAVE_BUSY_JS = r"""
() => {
  window.__fireReviewerReleaseSaveBusy?.();
  return [];
}
"""


class ReviewConflictError(RuntimeError):
    """Raised when a stale browser session tries to overwrite a newer decision."""


class ReviewDecisionWriteError(RuntimeError):
    """Raised when a decision file remains locked after bounded retries."""


class _NoRevisionCheck:
    pass


_NO_REVISION_CHECK = _NoRevisionCheck()
_PATH_LOCKS: dict[Path, RLock] = {}
_PATH_LOCKS_GUARD = RLock()


def _shared_path_lock(path: Path) -> RLock:
    """Return one process-wide lock for each decisions file."""

    key = path.absolute()
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _retry_review_write(operation: Callable[[], None]) -> None:
    """Retry a transient Windows file-sharing violation with bounded backoff."""

    delay = _REVIEW_WRITE_INITIAL_DELAY_SECONDS
    for attempt in range(_REVIEW_WRITE_RETRY_ATTEMPTS):
        try:
            operation()
            return
        except PermissionError:
            if attempt + 1 == _REVIEW_WRITE_RETRY_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 2, _REVIEW_WRITE_MAX_DELAY_SECONDS)


@contextmanager
def _interprocess_path_lock(path: Path, *, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Exclusively lock one persistent byte on Windows or Linux/macOS."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        deadline = time.monotonic() + timeout_seconds
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out waiting for review lock: {path}") from exc
                    time.sleep(0.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            return

        fcntl = importlib.import_module("fcntl")

        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for review lock: {path}") from exc
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class ReviewProgress:
    """Mandatory-review completion counters."""

    completed: int
    required: int
    secondary_pending: int = 0

    @property
    def remaining(self) -> int:
        """Return the number of required pairs without an explicit decision."""

        return self.required - self.completed


@dataclass(frozen=True)
class ReviewHistoryEntry:
    """One immutable review event retained even after a later re-review."""

    pair_id: str
    filename_a: str
    filename_b: str
    decision: str
    previous_decision: str
    reviewer: str
    reviewed_at: str
    notes: str
    round: int


def read_review_history(path: Path) -> tuple[ReviewHistoryEntry, ...]:
    """Read and strictly validate the append-only review audit trail."""

    if not path.exists():
        return ()
    entries: list[ReviewHistoryEntry] = []
    last_by_pair: dict[str, ReviewHistoryEntry] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _HISTORY_FIELDS:
            raise ValueError(f"review history has an invalid schema: {path}")
        for row in reader:
            required = {
                field: row.get(field, "")
                for field in (
                    "pair_id",
                    "filename_a",
                    "filename_b",
                    "decision",
                    "reviewer",
                    "reviewed_at",
                )
            }
            missing = [field for field, value in required.items() if not value]
            if missing:
                raise ValueError(f"review history is missing {missing[0]!r}: {path}")
            decision = required["decision"]
            previous_decision = row.get("previous_decision", "")
            if decision not in VALID_DECISIONS:
                raise ValueError(f"invalid review history decision: {decision!r}")
            if previous_decision and previous_decision not in VALID_DECISIONS:
                raise ValueError(f"invalid previous review decision: {previous_decision!r}")
            try:
                round_number = int(row.get("round", ""))
            except ValueError as exc:
                raise ValueError(f"invalid review round in {path}") from exc
            prior = last_by_pair.get(required["pair_id"])
            expected_round = 1 if prior is None else prior.round + 1
            expected_previous = "" if prior is None else prior.decision
            if round_number != expected_round:
                raise ValueError(
                    f"non-sequential review round for {required['pair_id']}: "
                    f"expected {expected_round}, got {round_number}"
                )
            if previous_decision != expected_previous:
                raise ValueError(
                    f"previous decision mismatch for review round {round_number}: "
                    f"{required['pair_id']}"
                )
            entry = ReviewHistoryEntry(
                required["pair_id"],
                required["filename_a"],
                required["filename_b"],
                decision,
                previous_decision,
                required["reviewer"],
                required["reviewed_at"],
                row.get("notes", ""),
                round_number,
            )
            entries.append(entry)
            last_by_pair[entry.pair_id] = entry
    return tuple(entries)


def _append_review_history(path: Path, entries: Sequence[ReviewHistoryEntry]) -> None:
    """Append flushed audit rows without rewriting any prior history."""

    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_HISTORY_FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerows(entry.__dict__ for entry in entries)
        stream.flush()
        os.fsync(stream.fileno())


def decision_revision(decision: ReviewDecision | None) -> str | None:
    """Return a stable optimistic-lock token without changing the CSV schema."""

    if decision is None:
        return None
    payload = "\0".join(
        (
            decision.pair_id,
            decision.filename_a,
            decision.filename_b,
            decision.decision,
            decision.reviewer,
            decision.reviewed_at,
            decision.notes,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def review_risk_reasons(
    candidate: CandidatePair,
    decision: ReviewDecision | None,
    *,
    large_component_size: int = LARGE_COMPONENT_REVIEW_THRESHOLD,
) -> tuple[str, ...]:
    """Return deterministic reasons that make a pair high risk for review."""

    reasons: list[str] = []
    if candidate.label_conflict:
        reasons.append("标签冲突")
    if decision is not None and decision.decision == "uncertain":
        reasons.append("无法确认")
    if large_component_size < 2:
        raise ValueError("large_component_size must be at least 2")
    if candidate.preliminary_component_size >= large_component_size:
        reasons.append("大型连通分量")
    if any(source in {"scene_embedding", "embedding_similarity"} for source in candidate.sources):
        reasons.append("场景特征候选")
    return tuple(reasons)


def advance_queue_after_save(
    state: dict[str, Any],
    new_pair_ids: Sequence[str],
    saved_pair_id: str,
) -> dict[str, Any]:
    """Advance to the next surviving pair after one successful save."""

    updated = dict(state)
    ordered = list(new_pair_ids)
    updated["pair_ids"] = ordered
    updated["current_pair_id"] = None
    updated["current_revision"] = None
    updated["current_round"] = 0
    if not ordered:
        updated["index"] = 0
        return updated

    old_ids = list(state.get("pair_ids", []))
    old_index = min(max(int(state.get("index", 0)), 0), max(len(old_ids) - 1, 0))
    if saved_pair_id in ordered:
        updated["index"] = (ordered.index(saved_pair_id) + 1) % len(ordered)
        return updated

    if saved_pair_id in old_ids:
        start = old_ids.index(saved_pair_id)
    else:
        start = old_index
    successors = old_ids[start + 1 :] + old_ids[:start]
    for pair_id in successors:
        if pair_id in ordered:
            updated["index"] = ordered.index(pair_id)
            return updated
    updated["index"] = min(old_index, len(ordered) - 1)
    return updated


class ReviewStore:
    """Process-safe, write-through store for resumable human decisions."""

    def __init__(
        self,
        candidates_path: Path,
        decisions_path: Path,
        history_path: Path | None = None,
        *,
        large_component_size: int = LARGE_COMPONENT_REVIEW_THRESHOLD,
    ) -> None:
        self.candidates_path = candidates_path
        self.decisions_path = decisions_path
        self.history_path = history_path or decisions_path.with_name("review_history.csv")
        if self.history_path.parent.absolute() != decisions_path.parent.absolute():
            raise ValueError("review history must be stored beside review decisions")
        self.lock_path = decisions_path.with_name(f".{decisions_path.name}.lock")
        if large_component_size < 2:
            raise ValueError("large_component_size must be at least 2")
        self.large_component_size = large_component_size
        self.candidates = read_candidates(
            candidates_path,
            large_component_size=large_component_size,
        )
        self.candidate_map = {item.pair_id: item for item in self.candidates}
        self._lock = _shared_path_lock(decisions_path)
        self._decisions: dict[str, ReviewDecision] = {}
        self._history: tuple[ReviewHistoryEntry, ...] = ()
        with self._lock, _interprocess_path_lock(self.lock_path):
            self._reload_disk_state()

    def _validate_decisions(self, decisions: dict[str, ReviewDecision]) -> None:
        unknown = sorted(set(decisions) - set(self.candidate_map))
        if unknown:
            raise ValueError(f"review file contains stale pair ids: {unknown[:10]}")
        for pair_id, decision in decisions.items():
            candidate = self.candidate_map[pair_id]
            if (decision.filename_a, decision.filename_b) != (
                candidate.filename_a,
                candidate.filename_b,
            ):
                raise ValueError(f"review filenames differ from candidate {pair_id}")
            if not decision.decision:
                continue
            if not decision.reviewer.strip():
                raise ValueError(f"reviewer is required for decided pair {pair_id}")
            if not decision.reviewed_at.strip():
                raise ValueError(f"reviewed_at is required for decided pair {pair_id}")
            try:
                reviewed_at = datetime.fromisoformat(decision.reviewed_at)
            except ValueError as exc:
                raise ValueError(f"invalid reviewed_at for decided pair {pair_id}") from exc
            if reviewed_at.tzinfo is None:
                raise ValueError(f"reviewed_at must include a timezone for decided pair {pair_id}")

    def _validate_history(self, history: Sequence[ReviewHistoryEntry]) -> None:
        for entry in history:
            candidate = self.candidate_map.get(entry.pair_id)
            if candidate is None:
                raise ValueError(f"review history contains stale pair id: {entry.pair_id}")
            if (entry.filename_a, entry.filename_b) != (
                candidate.filename_a,
                candidate.filename_b,
            ):
                raise ValueError(f"review history filenames differ from candidate {entry.pair_id}")
            audit_decision = ReviewDecision(
                entry.pair_id,
                entry.filename_a,
                entry.filename_b,
                entry.decision,
                entry.reviewer,
                entry.reviewed_at,
                entry.notes,
            )
            self._validate_decisions({entry.pair_id: audit_decision})

    def _reload_disk_state(self) -> None:
        decisions = read_decisions(self.decisions_path)
        history = read_review_history(self.history_path)
        self._validate_decisions(decisions)
        self._validate_history(history)
        self._decisions = decisions
        self._history = history

    @staticmethod
    def _history_matches_decision(
        entry: ReviewHistoryEntry,
        decision: ReviewDecision,
    ) -> bool:
        return (
            entry.pair_id,
            entry.filename_a,
            entry.filename_b,
            entry.decision,
            entry.reviewer,
            entry.reviewed_at,
            entry.notes,
        ) == (
            decision.pair_id,
            decision.filename_a,
            decision.filename_b,
            decision.decision,
            decision.reviewer,
            decision.reviewed_at,
            decision.notes,
        )

    def decision(self, pair_id: str) -> ReviewDecision | None:
        """Return the current decision for a pair, including an empty decision."""

        with self._lock, _interprocess_path_lock(self.lock_path):
            self._reload_disk_state()
            return self._decisions.get(pair_id)

    def revision(self, pair_id: str) -> str | None:
        """Return the current optimistic-lock revision for one pair."""

        return decision_revision(self.decision(pair_id))

    def history(self, pair_id: str | None = None) -> tuple[ReviewHistoryEntry, ...]:
        """Return a current immutable snapshot of all or one pair's audit events."""

        with self._lock, _interprocess_path_lock(self.lock_path):
            self._reload_disk_state()
            if pair_id is None:
                return self._history
            return tuple(entry for entry in self._history if entry.pair_id == pair_id)

    def review_round(self, pair_id: str) -> int:
        """Return the last persisted review round, or zero for an untouched pair."""

        return len(self.history(pair_id))

    def save(
        self,
        pair_id: str,
        decision: str,
        *,
        reviewer: str,
        notes: str,
        expected_revision: str | None | _NoRevisionCheck = _NO_REVISION_CHECK,
    ) -> ReviewDecision:
        """Validate and immediately persist one explicit three-state decision."""

        if decision not in VALID_DECISIONS:
            raise ValueError("保存前请选择同一场景、不同场景或无法确认")
        candidate = self.candidate_map.get(pair_id)
        if candidate is None:
            raise KeyError(f"未找到候选编号：{pair_id}")
        reviewer_clean = reviewer.strip()
        notes_clean = notes.strip()
        if not reviewer_clean:
            raise ValueError("请填写审核人")

        with self._lock, _interprocess_path_lock(self.lock_path):
            self._reload_disk_state()
            current_revision = decision_revision(self._decisions.get(pair_id))
            if (
                not isinstance(expected_revision, _NoRevisionCheck)
                and expected_revision != current_revision
            ):
                raise ReviewConflictError("其他审核会话已更新此候选，请重新加载后再保存")
            previous = self._decisions.get(pair_id)
            pair_history = [entry for entry in self._history if entry.pair_id == pair_id]
            history_to_append: list[ReviewHistoryEntry] = []
            if previous is not None and previous.decision in VALID_DECISIONS:
                if pair_history:
                    if not self._history_matches_decision(pair_history[-1], previous):
                        raise ValueError(f"候选 {pair_id} 的最新审核历史与当前结论不一致")
                else:
                    history_to_append.append(
                        ReviewHistoryEntry(
                            previous.pair_id,
                            previous.filename_a,
                            previous.filename_b,
                            previous.decision,
                            "",
                            previous.reviewer,
                            previous.reviewed_at,
                            previous.notes,
                            1,
                        )
                    )
            elif pair_history:
                raise ValueError(f"候选 {pair_id} 存在审核历史但没有当前结论")

            prior_entry = (
                history_to_append[-1]
                if history_to_append
                else (pair_history[-1] if pair_history else None)
            )
            reviewed_at = datetime.now(UTC).isoformat(timespec="microseconds")
            record = ReviewDecision(
                pair_id,
                candidate.filename_a,
                candidate.filename_b,
                decision,
                reviewer_clean,
                reviewed_at,
                notes_clean,
            )
            history_to_append.append(
                ReviewHistoryEntry(
                    pair_id,
                    candidate.filename_a,
                    candidate.filename_b,
                    decision,
                    "" if prior_entry is None else prior_entry.decision,
                    reviewer_clean,
                    reviewed_at,
                    notes_clean,
                    1 if prior_entry is None else prior_entry.round + 1,
                )
            )
            next_decisions = dict(self._decisions)
            next_decisions[pair_id] = record
            decision_snapshot = tuple(next_decisions.values())
            try:
                _retry_review_write(lambda: write_decisions(self.decisions_path, decision_snapshot))
            except PermissionError as exc:
                raise ReviewDecisionWriteError(
                    "审核文件暂时被占用，本次结论未保存，请稍后重试"
                ) from exc
            self._decisions = next_decisions
            _append_review_history(self.history_path, history_to_append)
            self._history = (*self._history, *history_to_append)
        return record

    def progress(self) -> ReviewProgress:
        """Return completion over mandatory pairs only."""

        with self._lock, _interprocess_path_lock(self.lock_path):
            self._reload_disk_state()
            required = [item for item in self.candidates if item.requires_review]
            completed = sum(
                self._decisions.get(item.pair_id) is not None
                and self._decisions[item.pair_id].decision in VALID_DECISIONS
                for item in required
            )
            history_counts: dict[str, int] = {}
            for entry in self._history:
                history_counts[entry.pair_id] = history_counts.get(entry.pair_id, 0) + 1
            secondary_pending = sum(
                self._decisions.get(item.pair_id) is not None
                and self._decisions[item.pair_id].decision in VALID_DECISIONS
                and bool(
                    review_risk_reasons(
                        item,
                        self._decisions[item.pair_id],
                        large_component_size=self.large_component_size,
                    )
                )
                and history_counts.get(item.pair_id, 0) < 2
                for item in required
            )
        return ReviewProgress(completed, len(required), secondary_pending)

    def filtered(
        self,
        *,
        source: str = "all",
        label_conflicts_only: bool = False,
        incomplete_only: bool = True,
        component_id: str = "",
        decision_status: str | None = None,
        high_risk_only: bool = False,
    ) -> tuple[CandidatePair, ...]:
        """Return a stable, priority-ordered filtered review queue."""

        status = decision_status or ("incomplete" if incomplete_only else "all")
        if status not in DECISION_STATUSES:
            raise ValueError(f"invalid review decision status: {status}")
        result: list[CandidatePair] = []
        with self._lock, _interprocess_path_lock(self.lock_path):
            self._reload_disk_state()
            history_counts: dict[str, int] = {}
            for entry in self._history:
                history_counts[entry.pair_id] = history_counts.get(entry.pair_id, 0) + 1
            for candidate in self.candidates:
                if source != "all" and source not in candidate.sources:
                    continue
                if label_conflicts_only and not candidate.label_conflict:
                    continue
                if component_id and candidate.preliminary_component_id != component_id.strip():
                    continue
                decision = self._decisions.get(candidate.pair_id)
                decided = decision is not None and decision.decision in VALID_DECISIONS
                if status == "incomplete" and decided:
                    continue
                if status == "completed" and not decided:
                    continue
                if status == "secondary_pending" and (
                    not decided
                    or not review_risk_reasons(
                        candidate,
                        decision,
                        large_component_size=self.large_component_size,
                    )
                    or history_counts.get(candidate.pair_id, 0) >= 2
                ):
                    continue
                if status in VALID_DECISIONS and (decision is None or decision.decision != status):
                    continue
                if high_risk_only and not review_risk_reasons(
                    candidate,
                    decision,
                    large_component_size=self.large_component_size,
                ):
                    continue
                if candidate.requires_review or status != "incomplete":
                    result.append(candidate)

        def priority(candidate: CandidatePair) -> tuple[int, int, str]:
            if candidate.label_conflict:
                return (0, 0, candidate.pair_id)
            if "numeric_adjacency" in candidate.sources:
                return (1, 0, candidate.pair_id)
            if any("embedding" in source for source in candidate.sources):
                return (1, 1, candidate.pair_id)
            if candidate.phash_distance <= 12:
                return (1, 2, candidate.pair_id)
            return (2, 0, candidate.pair_id)

        return tuple(sorted(result, key=priority))

    def validate_for_lock(self) -> None:
        """Reject incomplete, unaudited, or single-pass high-risk reviews."""

        with self._lock, _interprocess_path_lock(self.lock_path):
            self._reload_disk_state()
            missing: list[str] = []
            insufficient_secondary: list[str] = []
            unaudited: list[str] = []
            history_by_pair: dict[str, list[ReviewHistoryEntry]] = {}
            for entry in self._history:
                history_by_pair.setdefault(entry.pair_id, []).append(entry)
            for candidate in self.candidates:
                decision = self._decisions.get(candidate.pair_id)
                if candidate.requires_review and (
                    decision is None or decision.decision not in VALID_DECISIONS
                ):
                    missing.append(candidate.pair_id)
                    continue
                if decision is None or decision.decision not in VALID_DECISIONS:
                    continue
                pair_history = history_by_pair.get(candidate.pair_id, [])
                if not pair_history or not self._history_matches_decision(
                    pair_history[-1], decision
                ):
                    unaudited.append(candidate.pair_id)
                    continue
                if (
                    review_risk_reasons(
                        candidate,
                        decision,
                        large_component_size=self.large_component_size,
                    )
                    and len(pair_history) < 2
                ):
                    insufficient_secondary.append(candidate.pair_id)
            if missing:
                raise ValueError(f"mandatory review is incomplete: {missing[:10]}")
            if unaudited:
                raise ValueError(f"final decisions lack matching audit history: {unaudited[:10]}")
            if insufficient_secondary:
                raise ValueError(
                    f"high-risk pairs require a second review round: {insufficient_secondary[:10]}"
                )


def validate_review_audit_for_lock(
    candidates_path: Path,
    decisions_path: Path,
    history_path: Path | None = None,
    *,
    large_component_size: int = LARGE_COMPONENT_REVIEW_THRESHOLD,
) -> None:
    """Validate review completeness and audit history before publishing locked folds."""

    ReviewStore(
        candidates_path,
        decisions_path,
        history_path,
        large_component_size=large_component_size,
    ).validate_for_lock()


def create_reviewer_app(
    candidates_path: Path,
    decisions_path: Path,
    images_dir: Path,
    history_path: Path | None = None,
    *,
    large_component_size: int = LARGE_COMPONENT_REVIEW_THRESHOLD,
) -> Any:
    """Build, but do not launch, the scene-review Gradio application."""

    import gradio as gr

    store = ReviewStore(
        candidates_path,
        decisions_path,
        history_path,
        large_component_size=large_component_size,
    )
    sources = sorted({source for candidate in store.candidates for source in candidate.sources})

    def progress_text() -> str:
        progress = store.progress()
        return (
            f"必审进度：**{progress.completed}/{progress.required}** 已完成；"
            f"剩余 **{progress.remaining}** 条；待二次复核 "
            f"**{progress.secondary_pending}** 条。无法确认的候选将保守合并。"
        )

    def queue_state(
        source: str,
        conflicts_only: bool,
        decision_status: str,
        high_risk_only: bool,
        component_id: str,
    ) -> dict[str, Any]:
        queue = store.filtered(
            source=source,
            label_conflicts_only=conflicts_only,
            decision_status=decision_status,
            high_risk_only=high_risk_only,
            component_id=component_id,
        )
        return {
            "pair_ids": [item.pair_id for item in queue],
            "index": 0,
            "source": source,
            "conflicts_only": conflicts_only,
            "decision_status": decision_status,
            "high_risk_only": high_risk_only,
            "component_id": component_id,
            "current_pair_id": None,
            "current_revision": None,
            "current_round": 0,
        }

    def render(state: dict[str, Any]) -> tuple[Any, ...]:
        pair_ids = list(state.get("pair_ids", []))
        if not pair_ids:
            state["current_pair_id"] = None
            state["current_revision"] = None
            state["current_round"] = 0
            return (
                None,
                None,
                "当前筛选条件下没有候选。",
                "当前结论：**无**",
                "",
                [],
                progress_text(),
            )
        index = min(max(int(state.get("index", 0)), 0), len(pair_ids) - 1)
        state["index"] = index
        candidate = store.candidate_map[pair_ids[index]]
        decision = store.decision(candidate.pair_id)
        history_round = store.review_round(candidate.pair_id)
        state["current_pair_id"] = candidate.pair_id
        state["current_revision"] = decision_revision(decision)
        state["current_round"] = history_round
        component_members = sorted(
            {
                filename
                for item in store.candidates
                if item.preliminary_component_id == candidate.preliminary_component_id
                for filename in (item.filename_a, item.filename_b)
            }
        )
        risks = review_risk_reasons(
            candidate,
            decision,
            large_component_size=store.large_component_size,
        )
        risk_text = "、".join(risks) if risks else "无"
        source_text = "、".join(
            SOURCE_LABELS.get(source, "其他来源") for source in candidate.sources
        )
        details = (
            f"候选 {index + 1}/{len(pair_ids)} | 编号 `{candidate.pair_id}`  \n"
            f"图像标签 `{candidate.label_a}` / `{candidate.label_b}` | "
            f"感知哈希距离 `{candidate.phash_distance}` | "
            f"结构相似度 `{candidate.ssim if candidate.ssim is not None else '未计算'}`  \n"
            f"候选来源 `{source_text}` | 高风险原因 `{risk_text}`  \n"
            f"初始连通分量 `{candidate.preliminary_component_id}` "
            f"（分量大小 {candidate.preliminary_component_size}）  \n"
            f"已审核轮次 `{history_round}`"
        )
        if decision is None or decision.decision not in VALID_DECISIONS:
            current = "当前结论：**未审核** | 本次为第 **1** 轮审核"
        else:
            saved_round = max(history_round, 1)
            decision_text = DECISION_LABELS[decision.decision]
            current = (
                f"当前结论：**{decision_text}** | 审核人 `{decision.reviewer}` | "
                f"审核时间 `{decision.reviewed_at}` | 已保存第 **{saved_round}** 轮 | "
                f"下次为第 **{saved_round + 1}** 轮审核"
            )
        gallery = [(str(images_dir / name), name) for name in component_members]
        return (
            str(images_dir / candidate.filename_a),
            str(images_dir / candidate.filename_b),
            details,
            current,
            "" if decision is None else decision.notes,
            gallery,
            progress_text(),
        )

    def apply_filters(
        source: str,
        conflicts_only: bool,
        decision_status: str,
        high_risk_only: bool,
        component_id: str,
    ) -> tuple[Any, ...]:
        state = queue_state(
            source,
            conflicts_only,
            decision_status,
            high_risk_only,
            component_id,
        )
        return (state, *render(state))

    def move(state: dict[str, Any], delta: int) -> tuple[Any, ...]:
        state = dict(state)
        pair_ids = list(state.get("pair_ids", []))
        if pair_ids:
            state["index"] = (int(state.get("index", 0)) + delta) % len(pair_ids)
        return (state, *render(state))

    def save_current(
        state: dict[str, Any], reviewer: str, notes: str, decision: str
    ) -> tuple[Any, ...]:
        pair_ids = list(state.get("pair_ids", []))
        if not pair_ids:
            raise gr.Error("当前没有选中的候选。")
        index = min(max(int(state.get("index", 0)), 0), len(pair_ids) - 1)
        pair_id = pair_ids[index]
        if state.get("current_pair_id") != pair_id:
            raise gr.Error("审核队列已变化，请重新加载当前候选后再保存。")
        try:
            store.save(
                pair_id,
                decision,
                reviewer=reviewer,
                notes=notes,
                expected_revision=state.get("current_revision"),
            )
        except ReviewDecisionWriteError as exc:
            gr.Warning(str(exc), title="保存未完成")
            return (state, *(gr.skip() for _ in range(7)))
        except PermissionError:
            gr.Warning(
                "审核文件暂时被占用，本次操作未完成，请稍后重试或刷新确认",
                title="保存未完成",
            )
            return (state, *(gr.skip() for _ in range(7)))
        except (KeyError, ReviewConflictError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        refreshed = store.filtered(
            source=str(state.get("source", "all")),
            label_conflicts_only=bool(state.get("conflicts_only", False)),
            decision_status=str(state.get("decision_status", "incomplete")),
            high_risk_only=bool(state.get("high_risk_only", False)),
            component_id=str(state.get("component_id", "")),
        )
        next_state = advance_queue_after_save(
            state,
            [candidate.pair_id for candidate in refreshed],
            pair_id,
        )
        return (next_state, *render(next_state))

    def save_same(state: dict[str, Any], reviewer: str, notes: str) -> tuple[Any, ...]:
        return save_current(state, reviewer, notes, "same_scene")

    def save_different(state: dict[str, Any], reviewer: str, notes: str) -> tuple[Any, ...]:
        return save_current(state, reviewer, notes, "different_scene")

    def save_uncertain(state: dict[str, Any], reviewer: str, notes: str) -> tuple[Any, ...]:
        return save_current(state, reviewer, notes, "uncertain")

    def make_initial_state() -> dict[str, Any]:
        return queue_state("all", False, "incomplete", False, "")

    def load_session() -> tuple[Any, ...]:
        session_state = make_initial_state()
        return (session_state, *render(session_state))

    source_choices = [
        ("全部", "all"),
        *[(SOURCE_LABELS.get(source, "其他来源"), source) for source in sources],
    ]
    status_choices = [(DECISION_STATUS_LABELS[status], status) for status in DECISION_STATUSES]

    with gr.Blocks(title="火情场景分组审核", analytics_enabled=False) as app:
        gr.Markdown("# 火情场景分组审核")
        progress = gr.Markdown(progress_text())
        state = gr.State(make_initial_state)
        with gr.Row():
            source_filter = gr.Dropdown(source_choices, value="all", label="候选来源")
            decision_filter = gr.Dropdown(status_choices, value="incomplete", label="审核状态")
            conflicts_filter = gr.Checkbox(value=False, label="仅看标签冲突")
            high_risk_filter = gr.Checkbox(value=False, label="仅看高风险")
            component_filter = gr.Textbox(label="连通分量编号（可选）")
            apply_button = gr.Button("应用筛选")
        with gr.Row():
            left_image = gr.Image(label="左侧图片", interactive=False)
            right_image = gr.Image(label="右侧图片", interactive=False)
        details = gr.Markdown()
        current_status = gr.Markdown()
        component_gallery = gr.Gallery(label="初始连通分量", columns=5, height=240)
        with gr.Row():
            previous_button = gr.Button("上一条", elem_id="review-previous")
            next_button = gr.Button("下一条", elem_id="review-next")
        with gr.Row():
            reviewer_input = gr.Textbox(label="审核人（必填）", value="Admin")
            notes_input = gr.Textbox(label="说明（可选）")
        with gr.Row():
            same_button = gr.Button("1 - 同一场景", variant="primary", elem_id="review-same-scene")
            different_button = gr.Button(
                "2 - 不同场景", variant="stop", elem_id="review-different-scene"
            )
            uncertain_button = gr.Button(
                "3 - 无法确认", variant="secondary", elem_id="review-uncertain"
            )

        outputs = [
            state,
            left_image,
            right_image,
            details,
            current_status,
            notes_input,
            component_gallery,
            progress,
        ]
        apply_button.click(
            apply_filters,
            [
                source_filter,
                conflicts_filter,
                decision_filter,
                high_risk_filter,
                component_filter,
            ],
            outputs,
        )
        previous_button.click(lambda current: move(current, -1), state, outputs)
        next_button.click(lambda current: move(current, 1), state, outputs)
        save_inputs = [state, reviewer_input, notes_input]
        same_event = same_button.click(
            save_same,
            save_inputs,
            outputs,
            show_progress="minimal",
            concurrency_id="review-write",
            concurrency_limit=1,
            trigger_mode="once",
        )
        different_event = different_button.click(
            save_different,
            save_inputs,
            outputs,
            show_progress="minimal",
            concurrency_id="review-write",
            concurrency_limit=1,
            trigger_mode="once",
        )
        uncertain_event = uncertain_button.click(
            save_uncertain,
            save_inputs,
            outputs,
            show_progress="minimal",
            concurrency_id="review-write",
            concurrency_limit=1,
            trigger_mode="once",
        )
        for event in (same_event, different_event, uncertain_event):
            event.then(
                fn=None,
                js=REVIEWER_RELEASE_SAVE_BUSY_JS,
                queue=False,
                api_visibility="private",
            )
            event.failure(
                fn=None,
                js=REVIEWER_RELEASE_SAVE_BUSY_JS,
                queue=False,
                api_visibility="private",
            )
        app.load(load_session, outputs=outputs)
    return app
