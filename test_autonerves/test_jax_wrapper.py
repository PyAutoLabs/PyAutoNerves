"""Tests for the env-var handling in autonerves.jax_wrapper.

The wrapper's env logic runs at import time, so each test reloads the module
under a controlled os.environ. No test imports jax — the wrapper only sets
environment variables.
"""

import importlib
import os

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
