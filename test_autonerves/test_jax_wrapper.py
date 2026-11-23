"""Tests for the env-var handling in autonerves.jax_wrapper.

The wrapper's env logic runs at import time, so each in-process test reloads the
module under a controlled os.environ and does not import jax. The PYAUTO_SETUP and
JAX_ENABLE_X64 tests at the end import autonerves (and jax) in a fresh subprocess,
because both only take effect before JAX initialises.
"""

import importlib
import json
import os
import subprocess
import sys

import pytest

import autonerves.jax_wrapper

CONSTANT_FOLDING = "--xla_disable_hlo_passes=constant_folding"


@pytest.fixture
def clean_env(monkeypatch):
    for key in (
        "XLA_FLAGS",
        "JAX_ENABLE_X64",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "XDG_CACHE_HOME",
        "PYAUTO_SETUP",
    ):
        monkeypatch.delenv(key, raising=False)
    yield monkeypatch
    importlib.reload(autonerves.jax_wrapper)


def reload_wrapper():
    return importlib.reload(autonerves.jax_wrapper)


def test_xla_flags_set_when_unset(clean_env):
    reload_wrapper()
    assert os.environ["XLA_FLAGS"].startswith(CONSTANT_FOLDING)


def test_xla_flags_appended_not_clobbered(clean_env):
    clean_env.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/foo --xla_gpu_autotune_level=0")
    reload_wrapper()
    flags = os.environ["XLA_FLAGS"]
    assert "--xla_dump_to=/tmp/foo" in flags
    assert "--xla_gpu_autotune_level=0" in flags
    assert CONSTANT_FOLDING in flags


def test_xla_flags_unchanged_when_already_present(clean_env):
    preset = f"--xla_dump_to=/tmp/foo {CONSTANT_FOLDING}"
    clean_env.setenv("XLA_FLAGS", preset)
    reload_wrapper()
    # constant_folding is not re-appended; only the autotune default is added.
    assert os.environ["XLA_FLAGS"].startswith(preset)
    assert os.environ["XLA_FLAGS"].count(CONSTANT_FOLDING) == 1


def test_cache_dir_defaulted_when_unset(clean_env):
    reload_wrapper()
    expected = os.path.join(os.path.expanduser("~"), ".cache", "pyauto_jax")
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == expected
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "1"


def test_cache_dir_respects_xdg_cache_home(clean_env):
    clean_env.setenv("XDG_CACHE_HOME", "/custom/cache")
    reload_wrapper()
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == os.path.join(
        "/custom/cache", "pyauto_jax"
    )


def test_cache_dir_respects_preset_value(clean_env):
    clean_env.setenv("JAX_COMPILATION_CACHE_DIR", "/my/cache")
    reload_wrapper()
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == "/my/cache"
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "1"


def test_cache_disabled_by_empty_string(clean_env):
    clean_env.setenv("JAX_COMPILATION_CACHE_DIR", "")
    reload_wrapper()
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == ""
    assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ


def test_min_compile_time_respects_preset_value(clean_env):
    clean_env.setenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "10")
    reload_wrapper()
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "10"


def test_x64_enabled_by_default(clean_env):
    reload_wrapper()
    assert os.environ["JAX_ENABLE_X64"] == "True"


AUTOTUNE_OFF = "--xla_gpu_autotune_level=0"
TRITON_OFF = "--xla_gpu_enable_triton_gemm=false"


def test_autotune_disabled_by_default(clean_env):
    reload_wrapper()
    flags = os.environ["XLA_FLAGS"]
    assert AUTOTUNE_OFF in flags
    # With autotuning off, XLA would otherwise emit an un-tuned Triton kernel
    # for dense float64 GEMMs (~5x slower than cuBLAS on an A100).
    assert TRITON_OFF in flags


def test_triton_gemm_preset_respected(clean_env):
    clean_env.setenv("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=true")
    reload_wrapper()
    flags = os.environ["XLA_FLAGS"]
    assert AUTOTUNE_OFF in flags
    assert "--xla_gpu_enable_triton_gemm=true" in flags
    assert TRITON_OFF not in flags
    assert flags.count("--xla_gpu_enable_triton_gemm") == 1


def test_autotune_preset_level_respected(clean_env):
    clean_env.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=3")
    reload_wrapper()
    flags = os.environ["XLA_FLAGS"]
    assert "--xla_gpu_autotune_level=3" in flags
    assert AUTOTUNE_OFF not in flags
    assert CONSTANT_FOLDING in flags
    # An explicit level means XLA's own Triton GEMM behaviour is left alone.
    assert TRITON_OFF not in flags


def test_autotune_composes_with_user_flags(clean_env):
    clean_env.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/foo")
    reload_wrapper()
    flags = os.environ["XLA_FLAGS"]
    assert "--xla_dump_to=/tmp/foo" in flags
    assert CONSTANT_FOLDING in flags
    assert AUTOTUNE_OFF in flags
    assert TRITON_OFF in flags


def test_x64_explicit_false_respected(clean_env):
    clean_env.setenv("JAX_ENABLE_X64", "False")
    reload_wrapper()
    assert os.environ["JAX_ENABLE_X64"] == "False"


def test_x64_explicit_one_left_alone(clean_env):
    clean_env.setenv("JAX_ENABLE_X64", "1")
    reload_wrapper()
    assert os.environ["JAX_ENABLE_X64"] == "1"


# --- subprocess import tests: PYAUTO_SETUP and JAX_ENABLE_X64 -----------------

SETUP_VARS = (
    "PYAUTO_SETUP",
    "JAX_PLATFORMS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "JAX_ENABLE_X64",
    "SLURM_CPUS_PER_TASK",
)
THREAD_VARS = SETUP_VARS[2:6]


def _run(code, **env_overrides):
    """Run ``code`` in a fresh interpreter with a controlled environment.

    ``-S`` skips the ``site`` module, so no ``.pth`` hook or ``sitecustomize`` in
    the venv can set JAX_PLATFORMS / *_NUM_THREADS / JAX_ENABLE_X64 behind the
    test's back; this process's ``sys.path`` is passed on instead so the same
    packages import.
    """
    env = {k: v for k, v in os.environ.items() if k not in SETUP_VARS}
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-S", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _env_after_import(**env_overrides):
    code = (
        "import json, os, autonerves; "
        f"print(json.dumps({{k: os.environ.get(k) for k in {list(SETUP_VARS)!r}}}))"
    )
    result = _run(code, **env_overrides)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("value", ["cpu", "numba_cpu_sparse_fp64", "jax_cpu_dense_fp64_nufft"])
def test_setup_cpu_sets_platform_and_threads(value):
    env = _env_after_import(PYAUTO_SETUP=value, SLURM_CPUS_PER_TASK="3")
    assert env["JAX_PLATFORMS"] == "cpu"
    for key in THREAD_VARS:
        assert env[key] == "3", key


def test_setup_cpu_never_overwrites_user_values():
    env = _env_after_import(
        PYAUTO_SETUP="cpu",
        SLURM_CPUS_PER_TASK="3",
        OMP_NUM_THREADS="1",
        JAX_PLATFORMS="cuda,cpu",
    )
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["JAX_PLATFORMS"] == "cuda,cpu"
    assert env["MKL_NUM_THREADS"] == "3"


@pytest.mark.parametrize("value", [None, "", "auto", "gpu", "jax_gpu_dense_fp64"])
def test_setup_auto_gpu_and_unset_set_nothing(value):
    overrides = {} if value is None else {"PYAUTO_SETUP": value}
    env = _env_after_import(**overrides)
    assert env["JAX_PLATFORMS"] is None
    for key in THREAD_VARS:
        assert env[key] is None, key


def test_setup_invalid_raises_at_import():
    result = _run("import autonerves", PYAUTO_SETUP="tpu_turbo")
    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert "PYAUTO_SETUP" in result.stderr


@pytest.mark.parametrize(
    "value, expected",
    [("False", False), ("0", False), ("false", False), (None, True), ("True", True), ("1", True)],
)
def test_x64_respected_by_jax(value, expected):
    pytest.importorskip("jax")
    overrides = {"JAX_PLATFORMS": "cpu"}
    if value is not None:
        overrides["JAX_ENABLE_X64"] = value
    result = _run(
        "import autonerves, jax; print(jax.config.read('jax_enable_x64'))", **overrides
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == str(expected)
