"""Where the project's data lives.

These paths used to be relative to the current working directory. Both Windows
launchers set cwd to the repo root (`start_tray.vbs` sets `CurrentDirectory`,
`start_tray.bat` does `cd /d "%~dp0.."`), so that worked in practice — but it
breaks for anything that does not, such as a macOS LaunchAgent or running
`python -m asr_input.tray` from another directory. Resolving from the package
location instead makes the location independent of how the process was started.

Resolution order:

1. `ASR_INPUT_HOME`, if set — the escape hatch for unusual installs.
2. The repo root inferred from this file, when it still looks like a checkout.
3. The current working directory — the historical behaviour, kept as the
   fallback for a non-editable install where there is no checkout to find.
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def project_root() -> Path:
    override = os.environ.get("ASR_INPUT_HOME")
    if override:
        return Path(override).expanduser().resolve()

    # src layout: <root>/src/asr_input/paths.py
    candidate = _PACKAGE_DIR.parent.parent
    if (candidate / "pyproject.toml").is_file():
        return candidate

    return Path.cwd()


def data_dir() -> Path:
    return project_root() / "data"


def logs_dir() -> Path:
    return data_dir() / "logs"
