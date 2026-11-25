"""Tests for autonerves.environment: the hardware probe and PYAUTO_SETUP parsing.

Every detect branch runs with the environment monkeypatched; no test imports JAX or
needs a GPU.
"""
import builtins
import subprocess
import sys
import types

import pytest

from autonerves import environment
from autonerves.environment import Environment, parse_setup_intent


# --- parse_setup_intent -------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "  ", "auto", "AUTO", " Auto "])
def test_parse_auto(value):
    assert parse_setup_intent(value) == "auto"


@pytest.mark.parametrize("value, intent", [("cpu", "cpu"), ("gpu", "gpu"), ("CPU", "cpu")])
def test_parse_device(value, intent):
    assert parse_setup_intent(value) == intent


@pytest.mark.parametrize(
    "value, intent",
    [
        ("numba_cpu_sparse_fp64", "cpu"),
        ("numba_cpu_sparse_fp64_wtilde", "cpu"),
        ("jax_cpu_na_fp64", "cpu"),
        ("numpy_cpu_dense_fp32", "cpu"),
        ("jax_gpu_dense_fp64", "gpu"),
        ("jax_gpu_sparse_fp64_wtilde", "gpu"),
        ("jax_gpu_dense_mp_nufft", "gpu"),
        ("JAX_GPU_DENSE_FP64", "gpu"),
    ],
)
def test_parse_setup_id(value, intent):
    assert parse_setup_intent(value) == intent


@pytest.mark.parametrize(
    "value",
    [
        "tpu",
        "jax",
        "jax_cpu",
        "jax_cpu_dense",
        "torch_cpu_dense_fp64",
        "jax_tpu_dense_fp64",
        "jax_cpu_dense_fp16",
        "jax_cpu__fp64",
        "jax_cpu_dense_fp64_wtilde_extra",
        "jax-cpu-dense-fp64",
    ],
)
def test_parse_invalid_raises(value):
    with pytest.raises(ValueError, match="PYAUTO_SETUP") as info:
        parse_setup_intent(value)
    assert "setup_id" in str(info.value)


# --- cores --------------------------------------------------------------------


def test_cores_from_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    monkeypatch.setattr(environment.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    assert environment.detect_cores() == 7


def test_cores_bad_slurm_falls_back_to_affinity(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "lots")
    monkeypatch.setattr(environment.os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    assert environment.detect_cores() == 3


def test_cores_from_affinity(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr(environment.os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr(environment.os, "cpu_count", lambda: 64)
    assert environment.detect_cores() == 3


def test_cores_from_cpu_count(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delattr(environment.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(environment.os, "cpu_count", lambda: 12)
    assert environment.detect_cores() == 12


def test_cores_at_least_one(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delattr(environment.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(environment.os, "cpu_count", lambda: None)
    assert environment.detect_cores() == 1


# --- RAM ----------------------------------------------------------------------


def test_ram_from_proc_meminfo(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16777216 kB\nMemFree: 1 kB\n")
    real_open = builtins.open
    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, *a, **k: real_open(meminfo if path == "/proc/meminfo" else path, *a, **k),
    )
    assert environment.detect_ram_gb() == pytest.approx(16.0)


def _no_proc(monkeypatch):
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/meminfo":
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_ram_sysconf_fallback(monkeypatch):
    _no_proc(monkeypatch)
    values = {"SC_PHYS_PAGES": 2 * 1024**2, "SC_PAGE_SIZE": 4096}
    monkeypatch.setattr(environment.os, "sysconf", lambda name: values[name], raising=False)
    assert environment.detect_ram_gb() == pytest.approx(8.0)


def test_ram_unknown(monkeypatch):
    _no_proc(monkeypatch)
    monkeypatch.delattr(environment.os, "sysconf", raising=False)
    assert environment.detect_ram_gb() is None


# --- optional packages --------------------------------------------------------


def test_optional_packages_absent(monkeypatch):
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda name: None)
    assert environment._package_available("numba") is False
    assert environment._package_available("nufftax") is False


def test_optional_packages_present(monkeypatch):
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda name: object())
    assert environment._package_available("numba") is True


# --- GPU ----------------------------------------------------------------------


@pytest.fixture
def no_jax(monkeypatch):
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)


def _completed(stdout, returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_gpu_nvidia_smi_absent(monkeypatch, no_jax):
    def run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(environment.subprocess, "run", run)
    assert environment.detect_gpu() == (False, None, None)


def test_gpu_nvidia_smi_timeout(monkeypatch, no_jax):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", kwargs.get("timeout"))

    monkeypatch.setattr(environment.subprocess, "run", run)
    assert environment.detect_gpu() == (False, None, None)


def test_gpu_nvidia_smi_failing(monkeypatch, no_jax):
    monkeypatch.setattr(environment.subprocess, "run", lambda *a, **k: _completed("", 9))
    assert environment.detect_gpu() == (False, None, None)


def test_gpu_nvidia_smi_present(monkeypatch, no_jax):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _completed("NVIDIA A100-SXM4-40GB, 40960\nNVIDIA A100-SXM4-40GB, 40960\n")

    monkeypatch.setattr(environment.subprocess, "run", run)
    assert environment.detect_gpu() == (True, "NVIDIA A100-SXM4-40GB", 40.0)
    assert len(calls) == 1
    assert calls[0][1]["timeout"] > 0


@pytest.mark.parametrize(
    "variable, value",
    [
        ("JAX_PLATFORMS", "cpu"),
        ("JAX_PLATFORMS", "CPU"),
        ("JAX_PLATFORMS", "cpu,tpu"),
        ("JAX_PLATFORM_NAME", "cpu"),
    ],
)
def test_gpu_jax_platforms_without_gpu_skips_nvidia_smi(monkeypatch, no_jax, variable, value):
    def run(*args, **kwargs):
        raise AssertionError("nvidia-smi must not run when JAX is restricted to CPU")

    monkeypatch.setattr(environment.subprocess, "run", run)
    monkeypatch.setenv(variable, value)
    assert environment.detect_gpu() == (False, None, None)


@pytest.mark.parametrize("value", ["cuda", "gpu", "cpu,cuda", "rocm", ""])
def test_gpu_jax_platforms_with_gpu_or_empty_queries_nvidia_smi(monkeypatch, no_jax, value):
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda *a, **k: _completed("NVIDIA A100-SXM4-40GB, 40960\n"),
    )
    monkeypatch.setenv("JAX_PLATFORMS", value)
    assert environment.detect_gpu() == (True, "NVIDIA A100-SXM4-40GB", 40.0)


def test_gpu_via_already_imported_jax(monkeypatch):
    class Device:
        platform = "gpu"
        device_kind = "NVIDIA A100"

        def memory_stats(self):
            return {"bytes_limit": 8 * 1024**3}

    fake_jax = types.SimpleNamespace(devices=lambda: [Device()])
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    def run(*args, **kwargs):
        raise AssertionError("nvidia-smi must not run when jax is imported")

    monkeypatch.setattr(environment.subprocess, "run", run)
    assert environment.detect_gpu() == (True, "NVIDIA A100", 8.0)


def test_gpu_via_already_imported_jax_cpu_only(monkeypatch):
    fake_jax = types.SimpleNamespace(
        devices=lambda: [types.SimpleNamespace(platform="cpu")]
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    assert environment.detect_gpu() == (False, None, None)


# --- site ---------------------------------------------------------------------


def test_site_from_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "RAL")
    assert environment.detect_site() == "ral"


def test_site_from_hostname_matching_shipped_site(monkeypatch):
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    monkeypatch.setattr(environment.socket, "gethostname", lambda: "ral-login01")
    assert environment.detect_site() == "ral"


def test_site_colab(monkeypatch):
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    monkeypatch.setattr(environment.socket, "gethostname", lambda: "abc123")
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-1")
    assert environment.detect_site() == "colab"


def test_site_local(monkeypatch):
    for key in ("SLURM_CLUSTER_NAME", "COLAB_RELEASE_TAG", "COLAB_GPU"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(environment.socket, "gethostname", lambda: "my-laptop")
    assert environment.detect_site() == "local"


# --- Environment.detect -------------------------------------------------------


def test_detect_assembles_fields(monkeypatch, no_jax):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "ral")
    monkeypatch.setattr(environment.subprocess, "run", lambda *a, **k: _completed("", 1))
    monkeypatch.setattr(
        environment.importlib.util, "find_spec", lambda name: object() if name == "numba" else None
    )
    env = Environment.detect()
    assert env.gpu is False and env.gpu_name is None and env.gpu_memory_gb is None
    assert env.cores == 4
    assert env.numba is True
    assert env.nufftax is False
    assert env.site == "ral"
    assert env.ram_gb is None or env.ram_gb > 0
    with pytest.raises(Exception):
        env.cores = 8  # frozen


def test_detect_does_not_import_jax(monkeypatch, no_jax):
    monkeypatch.setattr(environment.subprocess, "run", lambda *a, **k: _completed("", 1))
    Environment.detect()
    assert "jax" not in sys.modules
