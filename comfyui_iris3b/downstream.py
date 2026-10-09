"""Isolated depth/restoration loaders and exception-safe execution policies."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .memory import IRIS_EXECUTION_LOCK, comfy_device, check_interrupted
from .paths import resolve_downstream_export

LOG = logging.getLogger("ComfyUI-Iris3B")


def build_task_model_patcher(handle):
    """Expose task weights through an actual native Comfy MODEL/ModelPatcher.

    Task execution still uses the established preprocessing and managed policy;
    the MODEL contains the same weight module, not a second allocation/copy.
    """
    from .comfy_adapter import build_comfy_model_patcher

    patcher = build_comfy_model_patcher(handle.model.model, handle.model.config)
    patcher.model.iris_task_handle = handle
    patcher.model.iris_task = handle.task
    return patcher


def task_handle_from_model(model, expected_task):
    # Keep old in-memory task handles accepted for API compatibility. New
    # graphs use native MODEL sockets; reject T2I and wrong-task weights clearly.
    handle = model if isinstance(model, ManagedTaskModel) else getattr(
        getattr(model, "model", None), "iris_task_handle", None
    )
    if handle is None or handle.task != expected_task:
        choice = "upscaler" if expected_task == "restoration" else expected_task
        raise ValueError(
            f"This node requires Iris {choice} weights. Select task={choice} and the matching "
            "checkpoint in Iris3B Model Loader, then connect its model output. "
            "A base T2I checkpoint cannot be reused as a depth/upscaler checkpoint."
        )
    return handle


@dataclass
class ManagedTaskModel:
    task: str
    model: object
    device_policy: str = "cpu_after_run"

    def run(self, fn, *, patcher=None):
        if self.device_policy not in {"cpu_after_run", "keep_on_gpu", "cpu"}:
            raise ValueError(f"Unsupported task model device policy: {self.device_policy}")
        device = "cpu" if self.device_policy == "cpu" else comfy_device()
        with IRIS_EXECUTION_LOCK:
            try:
                check_interrupted()
                if patcher is not None:
                    import comfy.model_management as mm
                    if str(device) != "cpu":
                        # Register residency/bookkeeping and let Comfy free
                        # idle text/T2I models before moving task weights.
                        mm.load_models_gpu([patcher], memory_required=1024 ** 3, force_full_load=True)
                    else:
                        _offload_task_patcher(patcher)
                self.model.to(device)
                LOG.info("Iris %s execution device=%s policy=%s", self.task, device, self.device_policy)
                return fn(self.model)
            finally:
                if self.device_policy in {"cpu_after_run", "cpu"}:
                    try:
                        if patcher is not None:
                            _offload_task_patcher(patcher)
                    finally:
                        self.model.to("cpu")


def _offload_task_patcher(patcher):
    """Ask host to unload only this task and retain every unrelated model.

    free_memory compares LoadedModel wrappers, not raw ModelPatchers. Do not
    modify its global residency list or invoke unload_all_models().
    """
    import comfy.model_management as mm

    loaded = mm.loaded_models()
    if any(item.model is patcher.model for item in loaded):
        keep = [mm.LoadedModel(item) for item in loaded if item.model is not patcher.model]
        mm.free_memory(float("inf"), patcher.load_device, keep_loaded=keep)


def load_depth_model(
    source: str,
    device_policy: str = "cpu_after_run",
    config_path: str | None = None,
    empty_prompt_path: str | None = None,
) -> ManagedTaskModel:
    from comfyui_iris3b.runtime.downstream.depth import DepthPredictor

    path = resolve_downstream_export(source, "depth")
    return ManagedTaskModel(
        "depth", DepthPredictor(str(path), device="cpu", config_path=config_path, empty_prompt_path=empty_prompt_path), device_policy
    )


def load_restoration_model(
    source: str,
    device_policy: str = "cpu_after_run",
    config_path: str | None = None,
    empty_prompt_path: str | None = None,
) -> ManagedTaskModel:
    from comfyui_iris3b.runtime.downstream.restoration import Restorer

    path = resolve_downstream_export(source, "upscaler")
    return ManagedTaskModel(
        "restoration", Restorer(str(path), device="cpu", config_path=config_path, empty_prompt_path=empty_prompt_path), device_policy
    )
