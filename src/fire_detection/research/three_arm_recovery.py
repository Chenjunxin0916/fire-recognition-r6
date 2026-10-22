"""Transactional node recovery for the existing three-arm DAG (no training resume)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def contained(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes project")
    return path


def validate_hn_random_plan(plan):
    """Validate the separate, non-overwriting HN/random control DAG."""
    if (plan.get("study_kind"), plan.get("outer_folds"), plan.get("seeds"), plan.get("methods")) != (
        "HN_RANDOM_CONTROL", list(range(5)), [42, 2026, 3407], ["random_negative"]
    ):
        raise ValueError("HN/random method, seed or fold scope differs")
    if plan.get("budget_policy") != "B_FULL_40_EPOCHS":
        raise ValueError("HN/random budget policy differs")
    if plan.get("paired_K_by_outer") != [41, 112, 64, 70, 59]:
        raise ValueError("HN/random K vector differs from accepted HN selections")
    tasks = plan.get("tasks", [])
    if len(tasks) != 46:
        raise ValueError("HN/random release requires 15 train, 15 CAL, 15 TEST and one summary")
    known, outputs = set(), set()
    for task in tasks:
        if task["id"] in known or task["output_dir"] in outputs:
            raise ValueError("Duplicate HN/random node or output")
        if not set(task["depends_on"]) <= known or len(task["depends_on"]) != len(set(task["depends_on"])):
            raise ValueError("Missing, cyclic or duplicate HN/random dependency")
        known.add(task["id"]); outputs.add(task["output_dir"])
    expected_cells = {(outer, seed, "random_negative") for outer in range(5) for seed in (42, 2026, 3407)}
    for kind in ("train", "calibrate", "evaluate"):
        actual = {(t["outer_fold"], t["seed"], t["method"]) for t in tasks if t["kind"] == kind}
        if actual != expected_cells:
            raise ValueError("Incomplete HN/random cell matrix: " + kind)
    if sum(t["kind"] == "summarize" for t in tasks) != 1:
        raise ValueError("Expected one HN/random summary")
    for task in tasks:
        kind, outer, seed = task["kind"], task["outer_fold"], task["seed"]
        if kind == "train":
            expected = set()
            k = plan["paired_K_by_outer"][outer]
            if (task["expected_epochs"], task["expected_sample_exposures"], task["expected_optimizer_steps"]) != (
                40, 40 * (660 + k), 40 * ((660 + k + 63) // 64)
            ):
                raise ValueError("Random control budget differs from frozen HN arm")
        elif kind == "calibrate":
            expected = {f"train_random_negative_o{outer}_s{seed}"}
        elif kind == "evaluate":
            expected = {f"cal_random_negative_o{f}_s{s}" for f in range(5) for s in (42, 2026, 3407)}
        elif kind == "summarize":
            expected = {t["id"] for t in tasks if t["kind"] == "evaluate"}
        else:
            raise ValueError("Unknown HN/random node kind")
        if set(task["depends_on"]) != expected:
            raise ValueError("HN/random phase dependency differs")
    return {t["id"]: t for t in tasks}


def validate_plan(plan):
    if plan.get("study_kind") == "HN_RANDOM_CONTROL":
        return validate_hn_random_plan(plan)
    if (plan["outer_folds"], plan["seeds"], plan["methods"]) != (
        list(range(5)),
        [42, 2026, 3407],
        ["standard", "hn", "focal"],
    ):
        raise ValueError("Frozen fold/seed/method specification differs")
    if (plan.get("budget_policy"),plan.get("budget_policy_version"),plan.get("training_protocol_version"),plan.get("scope_dependency_policy")) != ("B_FULL_40_EPOCHS","FULL-STRATEGY-B-40E-20261020-V1","FULL-STRATEGY-B-40E-20261020-V1","OUTER_LOCAL_V1"):
        raise ValueError("Final B protocol/version or outer dependency policy differs")
    tasks = plan["tasks"]
    known, outputs = set(), set()
    for task in tasks:
        if task["id"] in known or task["output_dir"] in outputs:
            raise ValueError("Duplicate node identity/output")
        if len(task["depends_on"]) != len(set(task["depends_on"])):
            raise ValueError("Duplicate dependencies")
        if not set(task["depends_on"]) <= known:
            raise ValueError("Missing, cyclic or non-topological dependency")
        known.add(task["id"])
        outputs.add(task["output_dir"])
    if len(tasks) != 156:
        raise ValueError("Expected existing 156-node plan")
    from collections import Counter

    if Counter(t["kind"] for t in tasks) != dict(
        miner=15, select=5, train=45, calibrate=45, evaluate=45, summarize=1
    ):
        raise ValueError("Node kind matrix differs")
    miners = {(t["outer_fold"], t["inner_fold"], t["seed"]) for t in tasks if t["kind"] == "miner"}
    if miners != {(f, i, 42) for f in range(5) for i in range(5) if i not in (f, (f + 1) % 5)}:
        raise ValueError("OOF fold/seed matrix differs")
    for kind in ("train", "calibrate", "evaluate"):
        actual = {(t["outer_fold"], t["seed"], t["method"]) for t in tasks if t["kind"] == kind}
        expected = {
            (f, s, m) for f in plan["outer_folds"] for s in plan["seeds"] for m in plan["methods"]
        }
        if actual != expected or len(expected) != 45:
            raise ValueError("Unpaired method matrix")
    by_kind = {
        k: {t["id"] for t in tasks if t["kind"] == k} for k in ("select", "train", "evaluate")
    }
    for t in tasks:
        if t["kind"] == "train":
            expected = {f"select_o{t['outer_fold']}"}
        elif t["kind"] == "calibrate":
            expected = {x["id"] for x in tasks if x["kind"] == "train" and x["outer_fold"] == t["outer_fold"]}
        elif t["kind"] == "evaluate":
            expected = {f"cal_{m}_o{t['outer_fold']}_s{t['seed']}" for m in plan["methods"]}
        elif t["kind"] == "select":
            expected = {
                m["id"]
                for m in tasks
                if m["kind"] == "miner" and m["outer_fold"] == t["outer_fold"]
            }
            if len(expected) != 3:
                raise ValueError("Expected three OOF miners")
        elif t["kind"] == "summarize":
            expected = by_kind["evaluate"]
        else:
            expected = set()
        if t["kind"] in ("train","miner"):
            n,e = (660,40) if t["kind"]=="train" else (440,30)
            if t["expected_epochs"]!=e:
                raise ValueError("Frozen epoch budget differs")
            if t["method"]=="hn":
                if t["expected_optimizer_steps"] is not None or t["expected_sample_exposures"] is not None:
                    raise ValueError("HN expected budget must await actual new H")
            elif (t["expected_optimizer_steps"],t["expected_sample_exposures"]) != (e*((n+63)//64),e*n):
                raise ValueError("Standard/Focal/miner budget differs")
        if set(t["depends_on"]) != expected:
            raise ValueError("Scientific phase dependency differs")
    return {t["id"]: t for t in tasks}


class NodeEngine:
    def __init__(self, root, release, *, verify_inputs=True):
        self.root, self.release = Path(root).resolve(), Path(release).resolve()
        self.plan = read(self.release / "plan.json")
        self.tasks = validate_plan(self.plan)
        imported = self.plan.get('imported_completed', {})
        self.imported_ids = frozenset(imported.get('task_ids', []))
        if self.imported_ids:
            expected = frozenset({'mine_o0_i2_s42', 'mine_o0_i3_s42', 'mine_o0_i4_s42', 'select_o0'})
            if self.imported_ids != expected or set(imported.get('accepted_sha256', {})) != expected:
                raise ValueError('Imported formal OOF scope differs')
            derived = self.plan['derived_from']
            parent_release = self.root / 'plans' / imported['parent_release_id']
            if (sha(parent_release / 'plan.json') != derived['plan_sha256']
                    or sha(parent_release / 'seal.json') != derived['seal_sha256']):
                raise ValueError('Formal OOF parent release identity drift')
            self.parent_plan = read(parent_release / 'plan.json')
            self.parent_tasks = validate_plan(self.parent_plan)
            self.parent_identity = digest({'plan': self.parent_plan, 'seal': read(parent_release / 'seal.json')})
            if self.parent_identity != imported['parent_engine_identity'] != derived['engine_identity']:
                raise ValueError('Formal OOF parent engine identity drift')
            parent_cell = self.root / 'runs/direction_a' / imported['parent_campaign'] / 'outer_0/seed_42'
            if (sha(parent_cell / 'outer0_remining_receipt.json') != derived['formal_oof_receipt_sha256']
                    or sha(parent_cell / 'hard_negative_selection.json') != derived['formal_h_sha256']):
                raise ValueError('Formal OOF/H parent receipt drift')
        else:
            self.parent_tasks = {}
            self.parent_identity = None
        self.seal = read(self.release / "seal.json")
        self.identity = digest({"plan": self.plan, "seal": self.seal})
        self.verify_inputs = verify_inputs
        self.state_path = self.release / "task_state.json"
        self.state = read(self.state_path)
        if self.state["release_id"] != self.plan["release_id"] or set(self.state["tasks"]) != set(
            self.tasks
        ):
            raise ValueError("State identity differs")
        self.check_inputs()

    def check_inputs(self):
        if (
            read(self.release / "plan.json") != self.plan
            or read(self.release / "seal.json") != self.seal
        ):
            raise ValueError("Release changed during execution")
        if self.verify_inputs:
            from fire_detection.source_provenance import verify_execution_source
            verify_execution_source(self.root)
            for name, expected in self.seal["files"].items():
                if sha(contained(self.root, name)) != expected:
                    raise ValueError(f"Sealed input drift: {name}")

    def target(self, task):
        if task['id'] in self.imported_ids:
            return contained(self.root, self.parent_tasks[task['id']]['output_dir'])
        return contained(self.root, task["output_dir"])

    def update(self, tid, **values):
        self.state["tasks"][tid].update(values)
        self.state["revision"] += 1
        atomic(self.state_path, self.state)

    def completed(self, tid, cache=None):
        cache = {} if cache is None else cache
        if tid in cache:
            return cache[tid]
        task = self.tasks[tid]
        path = self.target(task) / "accepted.json"
        if not path.exists():
            return None
        receipt = read(path)
        imported = tid in self.imported_ids
        expected_identity = self.parent_identity if imported else self.identity
        expected_task = self.parent_tasks[tid] if imported else task
        if imported and sha(path) != self.plan['imported_completed']['accepted_sha256'][tid]:
            raise ValueError('Imported formal OOF receipt hash drift')
        if (
            receipt.get("schema_version") != 2
            or receipt.get("task") != tid
            or receipt.get("release_fingerprint") != expected_identity
        ):
            raise ValueError("Receipt identity/configuration drift")
        if receipt.get("task_fingerprint") != digest(expected_task) or not receipt.get("artifacts"):
            raise ValueError("Incomplete or foreign completion receipt")
        attempt = contained(self.root, receipt["accepted_attempt"])
        if not attempt.is_relative_to(self.target(task) / "execution"):
            raise ValueError("Attempt escapes task")
        record = read(attempt / "record.json")
        if record.get("status") != "SUCCEEDED" or record.get("task") != tid:
            raise ValueError("Attempt not successfully finished")
        if (attempt / "record.json").relative_to(self.root).as_posix() not in receipt["artifacts"]:
            raise ValueError("Unbound execution record")
        for name, expected in receipt["artifacts"].items():
            artifact = contained(self.root, name)
            if artifact.suffix in (".tmp", ".partial") or sha(artifact) != expected:
                raise ValueError(f"Completed artifact drift: {name}")
        if set(receipt["dependencies"]) != set(task["depends_on"]):
            raise ValueError("Receipt dependency set differs")
        for parent, expected in receipt["dependencies"].items():
            if (
                not self.completed(parent, cache)
                or sha(self.target(self.tasks[parent]) / "accepted.json") != expected
            ):
                raise ValueError("Upstream completion drift")
        cache[tid] = receipt
        return receipt

    def block_descendants(self, tid):
        blocked = {tid}
        for task in self.plan["tasks"]:
            if blocked.intersection(task["depends_on"]):
                blocked.add(task["id"])
                if self.state["tasks"][task["id"]]["status"] != "SUCCEEDED":
                    self.update(task["id"], status="BLOCKED", blocked_by=tid)

    @contextmanager
    def lock(self):
        path = self.release / "running.lock"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(
                {"pid": os.getpid(), "started": time.time(), "release": self.identity}, stream
            )
        try:
            yield
        finally:
            path.unlink()

    def run(self, tid, handler, *, retry_reason=None):
        """Caller holds lock. Handler validates domain outputs and returns file paths."""
        self.check_inputs()
        task, cache = self.tasks[tid], {}
        try:
            previous = self.completed(tid, cache)
        except (ValueError, OSError, KeyError) as exc:
            self.update(
                tid, status="BLOCKED", blocked_by="ARTIFACT_OR_LINEAGE_DRIFT", error=str(exc)
            )
            raise
        if previous:
            if tid in self.imported_ids:
                return "SKIPPED_VERIFIED_IMPORTED_COMPLETE"
            self.update(tid, status="SUCCEEDED", accepted_attempt=previous["accepted_attempt"])
            return "SKIPPED_VERIFIED_COMPLETE"
        if tid in self.imported_ids:
            raise ValueError('Imported formal OOF cannot be rerun in classifier release')
        if self.state["tasks"][tid]["status"] == "SUCCEEDED":
            raise ValueError("Successful state lost its receipt; manual reconciliation required")
        dependencies = {}
        for parent in task["depends_on"]:
            if not self.completed(parent, cache):
                self.update(tid, status="BLOCKED", blocked_by=parent)
                raise RuntimeError(f"Incomplete dependency: {parent}")
            dependencies[parent] = sha(self.target(self.tasks[parent]) / "accepted.json")
        target = self.target(task)
        if (target / "completion.json").exists() or list(target.glob("completed_attempt_*.json")):
            raise RuntimeError(
                "Core completion without wrapper receipt: manual reconciliation required"
            )
        existing = list((target / "execution").glob("attempt_*"))
        if existing and not (retry_reason and retry_reason.strip()):
            raise PermissionError(
                "Explicit retry reason required; whole node restarts, never checkpoint resume"
            )
        number = max([int(p.name.split("_")[-1]) for p in existing] + [0]) + 1
        attempt = target / "execution" / f"attempt_{number:03d}"
        attempt.mkdir(parents=True, exist_ok=False)
        record = dict(
            task=tid,
            status="RUNNING",
            release_fingerprint=self.identity,
            retry_reason=retry_reason,
            started_at=time.time(),
            recovery="whole_node_restart",
        )
        atomic(attempt / "record.json", record)
        self.update(
            tid,
            status="RUNNING",
            blocked_by=None,
            attempts=[p.relative_to(self.root).as_posix() for p in existing + [attempt]],
        )
        started = time.perf_counter()
        try:
            paths = [Path(p).resolve() for p in handler(task, attempt)]
            if not paths or len(paths) != len(set(paths)):
                raise ValueError("Empty/duplicate artifact list")
            artifacts = {}
            for p in paths:
                if (
                    not p.is_relative_to(self.root)
                    or p.suffix in (".tmp", ".partial")
                    or not p.is_file()
                    or p.stat().st_size == 0
                ):
                    raise ValueError("Missing/partial/outside artifact")
                artifacts[p.relative_to(self.root).as_posix()] = sha(p)
            self.check_inputs()
            post_cache = {}
            for parent, expected in dependencies.items():
                if (
                    not self.completed(parent, post_cache)
                    or sha(self.target(self.tasks[parent]) / "accepted.json") != expected
                ):
                    raise ValueError("Dependency changed during task")
            record.update(
                status="SUCCEEDED",
                elapsed_seconds=time.perf_counter() - started,
                finished_at=time.time(),
            )
            atomic(attempt / "record.json", record)
            artifacts[(attempt / "record.json").relative_to(self.root).as_posix()] = sha(
                attempt / "record.json"
            )
            atomic(
                target / "accepted.json",
                dict(
                    schema_version=2,
                    task=tid,
                    release_fingerprint=self.identity,
                    task_fingerprint=digest(task),
                    dependencies=dependencies,
                    artifacts=artifacts,
                    accepted_attempt=attempt.relative_to(self.root).as_posix(),
                ),
            )
        except BaseException as exc:
            record.update(
                status="INTERRUPTED"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "FAILED",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.perf_counter() - started,
                finished_at=time.time(),
            )
            atomic(attempt / "record.json", record)
            self.update(tid, status=record["status"], error=record["error"])
            self.block_descendants(tid)
            raise
        # Completion is already durable. A state-index write failure must not
        # corrupt the accepted execution record; the next invocation repairs it.
        self.update(
            tid, status="SUCCEEDED", accepted_attempt=attempt.relative_to(self.root).as_posix()
        )
        return "SUCCEEDED"
