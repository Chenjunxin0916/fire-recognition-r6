"""Near-duplicate discovery and conservative scene grouping."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from fire_detection.data.scene_embeddings import SceneEmbeddingPair
from fire_detection.data.schema import DatasetIndex
from fire_detection.data.similarity import (
    ImageSignature,
    grayscale_array,
    hamming_distance,
    image_signature,
    structural_similarity,
)

VALID_DECISIONS = frozenset({"same_scene", "different_scene", "uncertain"})
VALID_CANDIDATE_SOURCES = frozenset(
    {
        "exact_sha256",
        "automatic_similarity",
        "manual_similarity",
        "dimension_mismatch",
        "numeric_adjacency",
        "label_conflict",
        "phash_near",
        "scene_embedding",
    }
)
_NUMERIC_SUFFIX = re.compile(r"^(.*?)(\d+)$")
LOGGER = logging.getLogger(__name__)


class ReviewIncompleteError(RuntimeError):
    """Raised when grouping is attempted before mandatory review is complete."""


class GroupingConflictError(RuntimeError):
    """Raised when reviewed scene relationships are contradictory."""


@dataclass(frozen=True)
class CandidatePair:
    """One deterministic pair presented to the scene reviewer."""

    pair_id: str
    filename_a: str
    filename_b: str
    label_a: int
    label_b: int
    sha256_a: str
    sha256_b: str
    phash_a: str
    phash_b: str
    phash_distance: int
    ssim: float | None
    size_match: bool
    sources: tuple[str, ...]
    automatic_same_scene: bool
    requires_review: bool
    label_conflict: bool
    preliminary_component_id: str
    preliminary_component_size: int
    embedding_cosine: float | None = None


@dataclass(frozen=True)
class ReviewDecision:
    """Persisted human review for one candidate pair."""

    pair_id: str
    filename_a: str
    filename_b: str
    decision: str
    reviewer: str = ""
    reviewed_at: str = ""
    notes: str = ""


@dataclass(frozen=True)
class GroupRecord:
    """Final conservative scene group assignment for one image."""

    filename: str
    label: int
    sha256: str
    phash: str
    width: int
    height: int
    group_id: str
    group_size: int


@dataclass
class _MutablePair:
    signature_a: ImageSignature
    signature_b: ImageSignature
    sources: set[str] = field(default_factory=set)
    ssim: float | None = None
    embedding_cosine: float | None = None
    automatic_same_scene: bool = False


class UnionFind:
    """Deterministic disjoint-set data structure."""

    def __init__(self, members: Iterable[str]) -> None:
        ordered = sorted(set(members))
        self._parent = {member: member for member in ordered}

    def find(self, member: str) -> str:
        """Return the representative for a known member."""

        if member not in self._parent:
            raise KeyError(member)
        parent = self._parent[member]
        if parent != member:
            self._parent[member] = self.find(parent)
        return self._parent[member]

    def union(self, left: str, right: str) -> None:
        """Merge two sets using the lexicographically smaller representative."""

        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        first, second = sorted((root_left, root_right))
        self._parent[second] = first

    def components(self) -> dict[str, tuple[str, ...]]:
        """Return sorted members grouped by representative."""

        grouped: dict[str, list[str]] = defaultdict(list)
        for member in sorted(self._parent):
            grouped[self.find(member)].append(member)
        return {root: tuple(members) for root, members in sorted(grouped.items())}


def _pair_key(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("candidate pair members must differ")
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def pair_identifier(left: str, right: str) -> str:
    """Return a stable full SHA-256 id for an unordered filename pair."""

    first, second = _pair_key(left, right)
    payload = json.dumps([first, second], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def member_group_identifier(members: Iterable[str]) -> str:
    """Return a stable full SHA-256 id for a sorted scene membership list."""

    ordered = sorted(set(members))
    if not ordered:
        raise ValueError("a group must contain at least one member")
    payload = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_signatures(
    index: DatasetIndex, images_dir: Path, workers: int
) -> tuple[ImageSignature, ...]:
    paths = [images_dir / image.file_name for image in index.images]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source images: {missing[:10]}")

    def compute(path: Path) -> ImageSignature:
        try:
            return image_signature(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot fingerprint image {path}: {exc}") from exc

    if workers <= 1:
        result = [compute(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            result = list(executor.map(compute, paths))
    return tuple(sorted(result, key=lambda item: item.filename))


def _add_pair(
    pairs: dict[tuple[str, str], _MutablePair],
    signatures: dict[str, ImageSignature],
    left: str,
    right: str,
    source: str,
    *,
    score: float | None = None,
    embedding_cosine: float | None = None,
    automatic: bool = False,
) -> None:
    key = _pair_key(left, right)
    item = pairs.get(key)
    if item is None:
        item = _MutablePair(signatures[key[0]], signatures[key[1]])
        pairs[key] = item
    item.sources.add(source)
    if score is not None:
        item.ssim = score
    if embedding_cosine is not None:
        if item.embedding_cosine is None:
            item.embedding_cosine = embedding_cosine
        else:
            item.embedding_cosine = max(item.embedding_cosine, embedding_cosine)
    item.automatic_same_scene = item.automatic_same_scene or automatic


def numeric_suffix_adjacencies(filenames: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Return adjacent existing members of each same-prefix numeric sequence.

    The source files are often sparse (for example ``001, 002, 004``). The
    second and third members are still adjacent observations in that sequence,
    so the numeric gap must not suppress the review candidate.
    """

    sequences: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for filename in filenames:
        match = _NUMERIC_SUFFIX.match(Path(filename).stem)
        if match is not None:
            sequences[match.group(1)].append((int(match.group(2)), filename))
    pairs: set[tuple[str, str]] = set()
    for sequence in sequences.values():
        ordered = sorted(sequence, key=lambda item: (item[0], item[1]))
        for left, right in zip(ordered, ordered[1:], strict=False):
            pairs.add(_pair_key(left[1], right[1]))
    return tuple(sorted(pairs))


def discover_scene_candidates(
    index: DatasetIndex,
    images_dir: Path,
    *,
    workers: int = 1,
    phash_max_distance: int = 12,
    automatic_phash_max_distance: int = 8,
    automatic_ssim_min: float = 0.95,
    manual_ssim_min: float = 0.90,
    large_component_size: int = 10,
    scene_embedding_pairs: Iterable[SceneEmbeddingPair] = (),
) -> tuple[tuple[ImageSignature, ...], tuple[CandidatePair, ...]]:
    """Fingerprint images and produce all automatic and mandatory-review pairs."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not 0 <= phash_max_distance <= 64:
        raise ValueError("phash_max_distance must be between 0 and 64")
    if not 0 <= automatic_phash_max_distance <= phash_max_distance:
        raise ValueError("automatic_phash_max_distance must not exceed phash_max_distance")
    if not -1.0 <= manual_ssim_min <= 1.0 or not -1.0 <= automatic_ssim_min <= 1.0:
        raise ValueError("SSIM thresholds must be between -1 and 1")
    if large_component_size < 2:
        raise ValueError("large_component_size must be at least 2")
    signatures_tuple = _compute_signatures(index, images_dir, workers)
    signatures = {item.filename: item for item in signatures_tuple}
    labels = index.labels
    pairs: dict[tuple[str, str], _MutablePair] = {}

    by_sha: dict[str, list[str]] = defaultdict(list)
    for item in signatures_tuple:
        by_sha[item.sha256].append(item.filename)
    for names in by_sha.values():
        for left_name, right_name in combinations(sorted(names), 2):
            _add_pair(
                pairs,
                signatures,
                left_name,
                right_name,
                "exact_sha256",
                score=1.0,
                automatic=True,
            )
            if labels[left_name] != labels[right_name]:
                _add_pair(pairs, signatures, left_name, right_name, "label_conflict")

    pending_ssim: list[tuple[ImageSignature, ImageSignature]] = []
    for left_signature, right_signature in combinations(signatures_tuple, 2):
        distance = hamming_distance(left_signature.phash, right_signature.phash)
        if distance > phash_max_distance:
            continue
        if left_signature.sha256 == right_signature.sha256:
            continue
        # Every near perceptual match is retained for a human decision. This
        # source is deliberately added before the SSIM gate, so low-SSIM pairs
        # cannot disappear merely because their structural score is weak.
        _add_pair(
            pairs,
            signatures,
            left_signature.filename,
            right_signature.filename,
            "phash_near",
        )
        if (left_signature.width, left_signature.height) != (
            right_signature.width,
            right_signature.height,
        ):
            _add_pair(
                pairs,
                signatures,
                left_signature.filename,
                right_signature.filename,
                "dimension_mismatch",
            )
            if labels[left_signature.filename] != labels[right_signature.filename]:
                _add_pair(
                    pairs,
                    signatures,
                    left_signature.filename,
                    right_signature.filename,
                    "label_conflict",
                )
            continue
        pending_ssim.append((left_signature, right_signature))

    LOGGER.info("computing full-resolution SSIM for %d perceptual candidates", len(pending_ssim))

    def score_pair(
        pair: tuple[ImageSignature, ImageSignature],
    ) -> tuple[ImageSignature, ImageSignature, float]:
        left_signature, right_signature = pair
        score = structural_similarity(
            grayscale_array(images_dir / left_signature.filename),
            grayscale_array(images_dir / right_signature.filename),
        )
        return left_signature, right_signature, score

    scored_pairs: Iterable[tuple[ImageSignature, ImageSignature, float]]
    if workers <= 1:
        scored_pairs = map(score_pair, pending_ssim)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            scored_pairs = tuple(executor.map(score_pair, pending_ssim))
    for left_signature, right_signature, score in scored_pairs:
        distance = hamming_distance(left_signature.phash, right_signature.phash)
        _add_pair(
            pairs,
            signatures,
            left_signature.filename,
            right_signature.filename,
            "phash_near",
            score=score,
        )
        if distance <= automatic_phash_max_distance and score >= automatic_ssim_min:
            _add_pair(
                pairs,
                signatures,
                left_signature.filename,
                right_signature.filename,
                "automatic_similarity",
                score=score,
                automatic=True,
            )
        elif score >= manual_ssim_min:
            _add_pair(
                pairs,
                signatures,
                left_signature.filename,
                right_signature.filename,
                "manual_similarity",
                score=score,
            )
        if labels[left_signature.filename] != labels[right_signature.filename]:
            _add_pair(
                pairs,
                signatures,
                left_signature.filename,
                right_signature.filename,
                "label_conflict",
                score=score,
            )

    for left_name, right_name in numeric_suffix_adjacencies(signatures):
        _add_pair(pairs, signatures, left_name, right_name, "numeric_adjacency")
        if labels[left_name] != labels[right_name]:
            _add_pair(pairs, signatures, left_name, right_name, "label_conflict")

    signature_names = set(signatures)
    for embedding_pair in scene_embedding_pairs:
        if embedding_pair.filename_a == embedding_pair.filename_b:
            raise ValueError("scene embedding candidate members must differ")
        if (
            embedding_pair.filename_a not in signature_names
            or embedding_pair.filename_b not in signature_names
        ):
            raise ValueError(
                "scene embedding candidate references an unknown filename: "
                f"{embedding_pair.filename_a}, {embedding_pair.filename_b}"
            )
        if not -1.0 <= embedding_pair.cosine_similarity <= 1.0:
            raise ValueError("scene embedding cosine similarity must be between -1 and 1")
        _add_pair(
            pairs,
            signatures,
            embedding_pair.filename_a,
            embedding_pair.filename_b,
            "scene_embedding",
            embedding_cosine=embedding_pair.cosine_similarity,
        )
        if labels[embedding_pair.filename_a] != labels[embedding_pair.filename_b]:
            _add_pair(
                pairs,
                signatures,
                embedding_pair.filename_a,
                embedding_pair.filename_b,
                "label_conflict",
            )

    preliminary = UnionFind(signatures)
    for mutable_pair in pairs.values():
        if mutable_pair.automatic_same_scene:
            preliminary.union(mutable_pair.signature_a.filename, mutable_pair.signature_b.filename)
    preliminary_components = preliminary.components()
    membership: dict[str, tuple[str, int]] = {}
    for members in preliminary_components.values():
        component_id = member_group_identifier(members)
        for member in members:
            membership[member] = (component_id, len(members))

    candidates: list[CandidatePair] = []
    for key, mutable_pair in sorted(pairs.items()):
        left_name, right_name = key
        component_id, component_size = membership[left_name]
        if membership[right_name][0] != component_id:
            combined = member_group_identifier((left_name, right_name))
            component_id, component_size = combined, 2
        label_conflict = labels[left_name] != labels[right_name]
        explicit_manual_source = bool(
            mutable_pair.sources
            & {
                "manual_similarity",
                "dimension_mismatch",
                "numeric_adjacency",
                "phash_near",
                "scene_embedding",
            }
        )
        large_component = (
            component_size >= large_component_size and mutable_pair.automatic_same_scene
        )
        requires_review = label_conflict or explicit_manual_source or large_component
        candidates.append(
            CandidatePair(
                pair_id=pair_identifier(left_name, right_name),
                filename_a=left_name,
                filename_b=right_name,
                label_a=labels[left_name],
                label_b=labels[right_name],
                sha256_a=mutable_pair.signature_a.sha256,
                sha256_b=mutable_pair.signature_b.sha256,
                phash_a=mutable_pair.signature_a.phash,
                phash_b=mutable_pair.signature_b.phash,
                phash_distance=hamming_distance(
                    mutable_pair.signature_a.phash, mutable_pair.signature_b.phash
                ),
                ssim=mutable_pair.ssim,
                size_match=(mutable_pair.signature_a.width, mutable_pair.signature_a.height)
                == (mutable_pair.signature_b.width, mutable_pair.signature_b.height),
                sources=tuple(sorted(mutable_pair.sources)),
                automatic_same_scene=mutable_pair.automatic_same_scene,
                requires_review=requires_review,
                label_conflict=label_conflict,
                preliminary_component_id=component_id,
                preliminary_component_size=component_size,
                embedding_cosine=mutable_pair.embedding_cosine,
            )
        )
    return signatures_tuple, tuple(candidates)


_SIGNATURE_FIELDS = ("filename", "label", "sha256", "phash", "width", "height")
_CANDIDATE_FIELDS = (
    "pair_id",
    "filename_a",
    "filename_b",
    "label_a",
    "label_b",
    "sha256_a",
    "sha256_b",
    "phash_a",
    "phash_b",
    "phash_distance",
    "ssim",
    "size_match",
    "sources",
    "automatic_same_scene",
    "requires_review",
    "label_conflict",
    "preliminary_component_id",
    "preliminary_component_size",
    "embedding_cosine",
)
_DECISION_FIELDS = (
    "pair_id",
    "filename_a",
    "filename_b",
    "decision",
    "reviewer",
    "reviewed_at",
    "notes",
)


def _atomic_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a JSON object with a flushed temporary file."""

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_candidate_artifacts(
    output_dir: Path,
    signatures: tuple[ImageSignature, ...],
    labels: dict[str, int],
    candidates: tuple[CandidatePair, ...],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Publish candidate CSVs and optional metadata as one write-once directory.

    All files are first written under a unique sibling staging directory. Any
    failure removes that directory and leaves the destination untouched.
    """

    signatures_path = output_dir / "image_hashes.csv"
    candidates_path = output_dir / "candidates.csv"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate artifact directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(f"candidate staging directory already exists: {staging}")
    staging.mkdir()
    published_metadata: dict[str, Any] | None = None
    try:
        staged_signatures = staging / signatures_path.name
        staged_candidates = staging / candidates_path.name
        _atomic_csv(
            staged_signatures,
            _SIGNATURE_FIELDS,
            (
                {
                    "filename": item.filename,
                    "label": labels[item.filename],
                    "sha256": item.sha256,
                    "phash": item.phash,
                    "width": item.width,
                    "height": item.height,
                }
                for item in signatures
            ),
        )
        _atomic_csv(
            staged_candidates,
            _CANDIDATE_FIELDS,
            (
                {
                    **item.__dict__,
                    "ssim": "" if item.ssim is None else format(item.ssim, ".17g"),
                    "embedding_cosine": (
                        ""
                        if item.embedding_cosine is None
                        else format(item.embedding_cosine, ".17g")
                    ),
                    "sources": "|".join(item.sources),
                    "size_match": int(item.size_match),
                    "automatic_same_scene": int(item.automatic_same_scene),
                    "requires_review": int(item.requires_review),
                    "label_conflict": int(item.label_conflict),
                }
                for item in candidates
            ),
        )
        if metadata is not None:
            published_metadata = dict(metadata)
            artifact_hashes = dict(published_metadata.get("artifact_sha256", {}))
            artifact_hashes.update(
                {
                    "image_hashes.csv": _sha256_file_for_artifact(staged_signatures),
                    "candidates.csv": _sha256_file_for_artifact(staged_candidates),
                }
            )
            published_metadata["artifact_sha256"] = artifact_hashes
            _atomic_json(staging / "candidates.meta.json", published_metadata)
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite candidate artifact directory: {output_dir}"
            )
        staging.replace(output_dir)
        return published_metadata
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _sha256_file_for_artifact(path: Path) -> str:
    """Hash a generated artifact without importing the model/offline layer."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required(row: dict[str, str], field: str, source: Path) -> str:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"missing {field!r} in {source}")
    return value


def _csv_bool(value: str, field: str) -> bool:
    if value not in {"0", "1"}:
        raise ValueError(f"{field} must be 0 or 1")
    return value == "1"


def _is_hex(value: str, length: int) -> bool:
    if len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def read_signatures(path: Path) -> tuple[tuple[ImageSignature, ...], dict[str, int]]:
    """Read and validate a deterministic image-hash artifact."""

    signatures: list[ImageSignature] = []
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            filename = _required(row, "filename", path)
            if filename in labels:
                raise ValueError(f"duplicate signature filename: {filename}")
            label = int(_required(row, "label", path))
            if label not in {0, 1}:
                raise ValueError(f"invalid label for {filename}: {label}")
            signature = ImageSignature(
                filename,
                _required(row, "sha256", path),
                _required(row, "phash", path),
                int(_required(row, "width", path)),
                int(_required(row, "height", path)),
            )
            if not _is_hex(signature.sha256, 64) or not _is_hex(signature.phash, 16):
                raise ValueError(f"invalid signature digest for {filename}")
            if signature.width <= 0 or signature.height <= 0:
                raise ValueError(f"invalid signature dimensions for {filename}")
            signatures.append(signature)
            labels[filename] = label
    if not signatures:
        raise ValueError(f"signature artifact is empty: {path}")
    return tuple(sorted(signatures, key=lambda item: item.filename)), labels


def read_candidates(
    path: Path, *, large_component_size: int | None = None
) -> tuple[CandidatePair, ...]:
    """Read and validate candidates against an optional metadata-bound threshold."""

    if large_component_size is not None and (
        type(large_component_size) is not int or large_component_size < 2
    ):
        raise ValueError("large_component_size must be at least 2")

    result: list[CandidatePair] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            pair_id = _required(row, "pair_id", path)
            if pair_id in seen:
                raise ValueError(f"duplicate candidate pair id: {pair_id}")
            seen.add(pair_id)
            ssim_raw = row.get("ssim", "")
            embedding_raw = row.get("embedding_cosine", "")
            candidate = CandidatePair(
                pair_id,
                _required(row, "filename_a", path),
                _required(row, "filename_b", path),
                int(_required(row, "label_a", path)),
                int(_required(row, "label_b", path)),
                _required(row, "sha256_a", path),
                _required(row, "sha256_b", path),
                _required(row, "phash_a", path),
                _required(row, "phash_b", path),
                int(_required(row, "phash_distance", path)),
                None if ssim_raw == "" else float(ssim_raw),
                _csv_bool(_required(row, "size_match", path), "size_match"),
                tuple(filter(None, _required(row, "sources", path).split("|"))),
                _csv_bool(
                    _required(row, "automatic_same_scene", path),
                    "automatic_same_scene",
                ),
                _csv_bool(_required(row, "requires_review", path), "requires_review"),
                _csv_bool(_required(row, "label_conflict", path), "label_conflict"),
                _required(row, "preliminary_component_id", path),
                int(_required(row, "preliminary_component_size", path)),
                None if embedding_raw in {None, ""} else float(embedding_raw),
            )
            if candidate.filename_a >= candidate.filename_b:
                raise ValueError(f"candidate filenames are not a sorted distinct pair: {pair_id}")
            if pair_identifier(candidate.filename_a, candidate.filename_b) != pair_id:
                raise ValueError(f"candidate pair id does not match filenames: {pair_id}")
            if candidate.label_a not in {0, 1} or candidate.label_b not in {0, 1}:
                raise ValueError(f"invalid candidate labels: {pair_id}")
            if not _is_hex(candidate.sha256_a, 64) or not _is_hex(candidate.sha256_b, 64):
                raise ValueError(f"invalid candidate SHA-256: {pair_id}")
            if not _is_hex(candidate.phash_a, 16) or not _is_hex(candidate.phash_b, 16):
                raise ValueError(f"invalid candidate perceptual hash: {pair_id}")
            if candidate.phash_distance not in range(65):
                raise ValueError(f"invalid candidate pHash distance: {pair_id}")
            if candidate.ssim is not None and not (-1.0 <= candidate.ssim <= 1.0):
                raise ValueError(f"invalid candidate SSIM: {pair_id}")
            if candidate.embedding_cosine is not None and not (
                -1.0 <= candidate.embedding_cosine <= 1.0
            ):
                raise ValueError(f"invalid candidate embedding cosine: {pair_id}")
            if not candidate.sources or not set(candidate.sources) <= VALID_CANDIDATE_SOURCES:
                raise ValueError(f"invalid candidate source: {pair_id}")
            if candidate.label_conflict != (candidate.label_a != candidate.label_b):
                raise ValueError(f"candidate label-conflict flag is inconsistent: {pair_id}")
            if ("label_conflict" in candidate.sources) != candidate.label_conflict:
                raise ValueError(f"candidate label-conflict source is inconsistent: {pair_id}")
            if candidate.preliminary_component_size < 2 or not _is_hex(
                candidate.preliminary_component_id, 64
            ):
                raise ValueError(f"invalid preliminary component: {pair_id}")
            manual = bool(
                set(candidate.sources)
                & {
                    "manual_similarity",
                    "dimension_mismatch",
                    "numeric_adjacency",
                    "phash_near",
                    "scene_embedding",
                    "label_conflict",
                }
            )
            mandatory_review = manual or candidate.label_conflict
            if large_component_size is None:
                review_is_valid = candidate.requires_review or not mandatory_review
            else:
                expected_review = mandatory_review or (
                    candidate.automatic_same_scene
                    and candidate.preliminary_component_size >= large_component_size
                )
                review_is_valid = candidate.requires_review == expected_review
            if not review_is_valid:
                raise ValueError(f"candidate review flag is inconsistent: {pair_id}")
            result.append(candidate)
    return tuple(sorted(result, key=lambda item: item.pair_id))


def read_decisions(path: Path) -> dict[str, ReviewDecision]:
    """Read review decisions; an empty decision remains explicitly incomplete."""

    if not path.exists():
        return {}
    result: dict[str, ReviewDecision] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            pair_id = _required(row, "pair_id", path)
            if pair_id in result:
                raise ValueError(f"duplicate review decision pair id: {pair_id}")
            decision = row.get("decision", "")
            if decision and decision not in VALID_DECISIONS:
                raise ValueError(f"invalid review decision {decision!r} for {pair_id}")
            result[pair_id] = ReviewDecision(
                pair_id,
                _required(row, "filename_a", path),
                _required(row, "filename_b", path),
                decision,
                row.get("reviewer", ""),
                row.get("reviewed_at", ""),
                row.get("notes", ""),
            )
    return result


def write_decisions(path: Path, decisions: Iterable[ReviewDecision]) -> None:
    """Atomically persist review decisions in stable pair-id order."""

    ordered = sorted(decisions, key=lambda item: item.pair_id)
    _atomic_csv(path, _DECISION_FIELDS, (item.__dict__ for item in ordered))


def validate_review_complete(
    candidates: tuple[CandidatePair, ...], decisions: dict[str, ReviewDecision]
) -> None:
    """Reject missing, stale, or mismatched mandatory review decisions."""

    candidate_map = {candidate.pair_id: candidate for candidate in candidates}
    unknown = sorted(set(decisions) - set(candidate_map))
    if unknown:
        raise GroupingConflictError(f"decisions reference unknown candidates: {unknown[:10]}")
    incomplete: list[str] = []
    for candidate in candidates:
        decision = decisions.get(candidate.pair_id)
        if decision is not None and (
            (decision.filename_a, decision.filename_b)
            != (candidate.filename_a, candidate.filename_b)
        ):
            raise GroupingConflictError(
                f"decision filenames do not match candidate {candidate.pair_id}"
            )
        if candidate.requires_review and (
            decision is None or decision.decision not in VALID_DECISIONS
        ):
            incomplete.append(candidate.pair_id)
    if incomplete:
        raise ReviewIncompleteError(
            f"{len(incomplete)} mandatory scene reviews are incomplete; first ids={incomplete[:10]}"
        )


def resolve_scene_groups(
    signatures: tuple[ImageSignature, ...],
    labels: dict[str, int],
    candidates: tuple[CandidatePair, ...],
    decisions: dict[str, ReviewDecision],
) -> tuple[GroupRecord, ...]:
    """Resolve reviewed relations into conservative, deterministic scene groups."""

    validate_review_complete(candidates, decisions)
    names = {signature.filename for signature in signatures}
    if names != set(labels):
        raise GroupingConflictError("signature and label filename sets differ")
    union_find = UnionFind(names)
    signature_map = {signature.filename: signature for signature in signatures}
    for candidate in candidates:
        if candidate.filename_a not in names or candidate.filename_b not in names:
            raise GroupingConflictError(
                f"candidate {candidate.pair_id} references an unknown filename"
            )
        left_signature = signature_map[candidate.filename_a]
        right_signature = signature_map[candidate.filename_b]
        if (candidate.label_a, candidate.label_b) != (
            labels[candidate.filename_a],
            labels[candidate.filename_b],
        ):
            raise GroupingConflictError(f"candidate labels changed: {candidate.pair_id}")
        if (candidate.sha256_a, candidate.sha256_b) != (
            left_signature.sha256,
            right_signature.sha256,
        ) or (candidate.phash_a, candidate.phash_b) != (
            left_signature.phash,
            right_signature.phash,
        ):
            raise GroupingConflictError(f"candidate fingerprints changed: {candidate.pair_id}")
    different_pairs: list[tuple[str, str]] = []
    for candidate in candidates:
        decision = decisions.get(candidate.pair_id)
        if decision is not None and decision.decision:
            if decision.decision == "different_scene":
                different_pairs.append((candidate.filename_a, candidate.filename_b))
            else:
                union_find.union(candidate.filename_a, candidate.filename_b)
        elif candidate.automatic_same_scene and not candidate.requires_review:
            union_find.union(candidate.filename_a, candidate.filename_b)

    contradictions = [
        (left, right)
        for left, right in different_pairs
        if union_find.find(left) == union_find.find(right)
    ]
    if contradictions:
        raise GroupingConflictError(
            "different_scene decisions conflict with the transitive same-scene closure: "
            f"{contradictions[:10]}"
        )

    membership: dict[str, tuple[str, int]] = {}
    for members in union_find.components().values():
        group_id = member_group_identifier(members)
        for member in members:
            membership[member] = (group_id, len(members))
    return tuple(
        GroupRecord(
            filename,
            labels[filename],
            signature_map[filename].sha256,
            signature_map[filename].phash,
            signature_map[filename].width,
            signature_map[filename].height,
            membership[filename][0],
            membership[filename][1],
        )
        for filename in sorted(names)
    )
