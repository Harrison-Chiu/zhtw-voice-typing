"""Tests for the environment fingerprint.

A fingerprint is only useful if it never becomes the reason a benchmark run
fails, so every probe has to degrade into a recorded reason rather than an
exception.
"""

from __future__ import annotations

import subprocess

from asr_input.eval import environment


def test_missing_nvidia_smi_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(environment.shutil, "which", lambda _: None)
    info = environment.gpu_info()
    assert info["available"] is False
    assert "PATH" in info["error"]


def test_nvidia_smi_output_is_parsed(monkeypatch):
    monkeypatch.setattr(environment.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 0, stdout="NVIDIA GeForce RTX 4060, 566.36, 8188, 1024\n", stderr=""
        ),
    )
    assert environment.gpu_info() == {
        "available": True,
        "name": "NVIDIA GeForce RTX 4060",
        "driver_version": "566.36",
        "memory_total_mib": 8188,
        "memory_used_mib": 1024,
    }


def test_nvidia_smi_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(environment.shutil, "which", lambda _: "nvidia-smi")

    def explode(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 20)

    monkeypatch.setattr(environment.subprocess, "run", explode)
    info = environment.gpu_info()
    assert info["available"] is False
    assert "TimeoutExpired" in info["error"]


def test_unexpected_nvidia_smi_output_is_reported(monkeypatch):
    monkeypatch.setattr(environment.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="garbage\n", stderr=""),
    )
    assert environment.gpu_info()["available"] is False


def test_absent_package_records_none_rather_than_failing():
    versions = environment.package_versions(("numpy", "definitely-not-installed-xyz"))
    assert versions["definitely-not-installed-xyz"] is None
    assert versions["numpy"] is not None


def test_collect_has_every_section_even_without_a_gpu(monkeypatch):
    monkeypatch.setattr(environment, "gpu_info", lambda: {"available": False, "error": "none"})
    fingerprint = environment.collect(model={"name": "large-v3-turbo"})
    assert set(fingerprint) >= {"os", "python", "packages", "gpu", "model"}
    assert fingerprint["gpu_idle_memory_used_mib"] is None
    assert fingerprint["model"]["name"] == "large-v3-turbo"


def test_collect_records_idle_gpu_memory(monkeypatch):
    monkeypatch.setattr(
        environment,
        "gpu_info",
        lambda: {"available": True, "name": "x", "driver_version": "1", "memory_used_mib": 733},
    )
    assert environment.collect()["gpu_idle_memory_used_mib"] == 733


def test_fingerprint_summary_mentions_the_gpu_problem(monkeypatch):
    monkeypatch.setattr(environment, "gpu_info", lambda: {"available": False, "error": "no smi"})
    text = environment.format_fingerprint(environment.collect())
    assert "GPU unavailable: no smi" in text
    assert "Python" in text


def test_fingerprint_carries_no_user_identifying_paths():
    fingerprint = environment.collect()
    name = fingerprint["python"]["executable_name"]
    assert "\\" not in name
    assert "/" not in name
