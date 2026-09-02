"""Device and compute-type resolution.

Two rules shape this module:

1. An explicit `config.yaml` value wins. The only exception is a value the host
   physically cannot honour (`device: cuda` on macOS), which would otherwise fail
   at model-load time with a less obvious error.
2. The CUDA probe is lazy. `import ctranslate2` pulls in torch and costs tens of
   seconds on a cold filesystem cache, so it must not run for engines that never
   needed it — see CLAUDE.md on keeping torch off the startup path.
"""

import sys
from collections.abc import Callable

_DEFAULT_COMPUTE_TYPE = {"cuda": "float16", "cpu": "int8"}


def cuda_device_count() -> int:
    """Number of CUDA devices CTranslate2 can see. 0 on any import/probe failure."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def resolve_device(
    configured_device: str | None = None,
    configured_compute_type: str | None = None,
    platform_name: str | None = None,
    cuda_probe: Callable[[], int] | None = None,
) -> tuple[str, str, str | None]:
    """Return `(device, compute_type, note)`.

    `note` is a human-readable explanation when the resolved value differs from
    what was configured, and None when the config was taken as-is.
    """
    platform_name = sys.platform if platform_name is None else platform_name
    probe = cuda_device_count if cuda_probe is None else cuda_probe
    # A single resolution can trigger more than one adjustment (e.g. cuda->cpu on
    # macOS then float16->int8 because of it); reporting only the last one would
    # hide the reason the first happened.
    notes: list[str] = []

    device = (configured_device or "").strip().lower() or "auto"

    if device == "cuda" and platform_name == "darwin":
        # CTranslate2 has no Metal/MPS backend; there is no CUDA on macOS at all.
        notes.append("config 指定 device: cuda，但 macOS 沒有 CUDA，改用 cpu")
        device = "cpu"
    elif device == "auto":
        # macOS is decided without probing: CTranslate2 has no CUDA there, and the
        # probe itself is expensive enough to be worth skipping when the answer is known.
        device = "cuda" if platform_name != "darwin" and probe() > 0 else "cpu"
        notes.append(f"device 未指定，自動選用 {device}")

    compute_type = (configured_compute_type or "").strip() or None
    if compute_type is None:
        compute_type = _DEFAULT_COMPUTE_TYPE.get(device, "int8")
    elif device == "cpu" and compute_type == "float16":
        # float16 on CPU is not supported by CTranslate2 and silently costs a lot
        # where it is emulated; int8 is the documented CPU default.
        notes.append("device 為 cpu，float16 不適用，compute_type 改用 int8")
        compute_type = "int8"

    return device, compute_type, "；".join(notes) or None
