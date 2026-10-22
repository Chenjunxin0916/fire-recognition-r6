"""Execution policy only; leaves existing replay and Focal algorithms untouched."""

from __future__ import annotations

import hashlib
import json
import time


def set_precision(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def check_precision(torch):
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("Strict FP32 policy drift: TF32 enabled")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("Strict FP32 policy drift: matmul precision")


def check_trainer(trainer, expected, torch):
    check_precision(torch)
    if trainer.device.type != "cuda":
        raise RuntimeError("CUDA required; CPU fallback prohibited")
    for name, value in expected.items():
        if getattr(trainer.args, name) != value:
            raise RuntimeError(f"Frozen argument drift: {name}")
    if trainer.accumulate != 1 or trainer.amp:
        raise RuntimeError("AMP/accumulation drift")
    if trainer.batch_size != 64:
        raise RuntimeError("Batch reduction prohibited")


def epoch_guard(trainer, expected, torch):
    check_trainer(trainer, expected, torch)
    # Ultralytics resets this after _setup_train, before on_train_epoch_start.
    # In pinned 8.4.92, >=3 causes immediate re-raise on OOM/backend memory error.
    trainer._oom_retries = 3


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((t.dtype, tuple(t.shape))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def guarded_factory(base, *, expected, audit_path, identity_path, torch):
    class StrictTrainer(base):
        def _setup_train(self):
            super()._setup_train()
            check_trainer(self, expected, torch)
            if any(p.dtype != torch.float32 for p in self.model.parameters()):
                raise RuntimeError("Model parameters must be FP32")
            identity = model_digest(self.model)
            identity_path.parent.mkdir(parents=True, exist_ok=True)
            if identity_path.exists():
                if json.loads(identity_path.read_text(encoding="utf-8"))["sha256"] != identity:
                    raise RuntimeError("Paired initialized model state mismatch")
            else:
                with identity_path.open("x", encoding="utf-8") as f:
                    json.dump({"sha256": identity}, f)
            self.add_callback("on_train_epoch_start", self._strict_start)
            self.add_callback("on_train_epoch_end", self._strict_end)

        def _strict_start(self, trainer):
            epoch_guard(self, expected, torch)
            self.strict_started = time.perf_counter()
            self.strict_before = dict(self.research_audit)

        def _strict_end(self, trainer):
            check_trainer(self, expected, torch)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as f:
                json.dump(
                    dict(
                        epoch=int(self.epoch),
                        seconds=time.perf_counter() - self.strict_started,
                        actual_updates=self.research_audit["optimizer_steps"]
                        - self.strict_before["optimizer_steps"],
                        actual_exposures=self.research_audit["sample_exposures"]
                        - self.strict_before["sample_exposures"],
                        allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                        reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                        amp=False,
                        matmul_tf32=False,
                        cudnn_tf32=False,
                        accumulate=1,
                        batch=64,
                    ),
                    f,
                )
                f.write("\n")

        def preprocess_batch(self, batch):
            result = super().preprocess_batch(batch)
            if result["img"].device.type != "cuda" or result["img"].dtype != torch.float32:
                raise RuntimeError("Training input must be CUDA FP32")
            return result

        def optimizer_step(self):
            check_trainer(self, expected, torch)
            super().optimizer_step()

    return StrictTrainer
