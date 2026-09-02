"""Environment fingerprint for a benchmark run (`docs/asr-benchmark-proposal.md` §5).

Numbers from two runs are only comparable if the machine underneath them was the
same. This module records what we can read cheaply — GPU and driver, OS build,
Python and library versions, the model identity, and the GPU memory already in
use before the run starts.

Everything is best-effort: a probe that fails records the reason instead of
raising, because a missing `nvidia-smi` should not abort a benchmark. Fields whose
value could not be determined are `None`, never a guess.

Deliberately *not* collected: anything identifying (hostname, user name, paths
under the home directory), so a fingerprint can be pasted into a public issue.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from importlib import metadata

# Packages worth pinning down when a number moves unexpectedly.
TRACKED_PACKAGES = (
    "faster-whisper",
    "ctranslate2",
    "onnxruntime",
    "numpy",
    "soundfile",
    "opencc-python-reimplemented",
    "torch",
)

NVIDIA_SMI_QUERY = "name,driver_version,memory.total,memory.used"


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed version per package; `None` when the package is absent."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_info() -> dict:
    """Read GPU name, driver and memory from `nvidia-smi`.

    Uses the CLI rather than torch on purpose: the recording path no longer
    imports torch (see CLAUDE.md, VAD on ONNX Runtime), and a fingerprint should
    not be the thing that drags a 2 GB import into the process.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False, "error": "nvidia-smi not on PATH"}
    try:
        proc = subprocess.run(
            [exe, f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    line = proc.stdout.strip().splitlines()
    if not line:
        return {"available": False, "error": "nvidia-smi returned no rows"}
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) != 4:
        return {"available": False, "error": f"unexpected nvidia-smi output: {line[0]!r}"}
    name, driver, total, used = parts
    return {
        "available": True,
        "name": name,
        "driver_version": driver,
        "memory_total_mib": _as_int(total),
        "memory_used_mib": _as_int(used),
    }


def _as_int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


def os_info() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def python_info() -> dict:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable_name": sys.executable.rsplit("\\", 1)[-1].rsplit("/", 1)[-1],
    }


def collect(model: dict | None = None) -> dict:
    """Assemble the full fingerprint.

    `model` carries whatever identifies the engine for this run (name, size,
    compute type, revision) — the caller knows that, this module does not.
    """
    gpu = gpu_info()
    return {
        "os": os_info(),
        "python": python_info(),
        "packages": package_versions(),
        "gpu": gpu,
        # Idle memory before the run: a benchmark started with another process
        # already holding VRAM is not comparable with one started clean.
        "gpu_idle_memory_used_mib": gpu.get("memory_used_mib") if gpu.get("available") else None,
        "model": model or {},
    }


def format_fingerprint(fingerprint: dict) -> str:
    """One-line human summary, for the top of a Markdown report."""
    gpu = fingerprint.get("gpu", {})
    gpu_text = (
        f"{gpu.get('name')} (driver {gpu.get('driver_version')})"
        if gpu.get("available")
        else f"GPU unavailable: {gpu.get('error')}"
    )
    os_data = fingerprint.get("os", {})
    py = fingerprint.get("python", {})
    return (
        f"{gpu_text} | {os_data.get('system')} {os_data.get('release')} "
        f"build {os_data.get('version')} | Python {py.get('version')}"
    )
