import importlib.util
import logging

logger = logging.getLogger(__name__)

import os

if importlib.util.find_spec("jax") is None:
    logger.warning(
        """
        JAX is not installed, so all computations will run on the pure NumPy
        path. Performance is significantly reduced without JAX — model fits
        that take minutes with JAX can take hours without it.

        JAX is a default dependency of the PyAuto libraries; it is absent
        either because this platform has no JAX wheels (e.g. Intel macOS) or
        because it was uninstalled. On supported platforms, restore it with:

            pip install jax
        """
    )

xla_env = os.environ.get("XLA_FLAGS")

xla_env_set = True

if xla_env is None:
    xla_env_set = False
elif isinstance(xla_env, str):
    xla_env_set = "--xla_disable_hlo_passes=constant_folding" in xla_env

if not xla_env_set:
    logger.info(
        """
        For fast JAX compile times, the envirment variable XLA_FLAGS must include "--xla_disable_hlo_passes=constant_folding",
        which is currently not.

        In Python, to do this manually, use the code:

        import os
        os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=constant_folding"

        The environment variable has been set automatically for you now, however if JAX has already been imported,
        this change will not take effect and JAX function compiling times may be slow.

        Therefore, it is recommended to set this environment variable before running your script, e.g. in your terminal.
        """)

    # Append rather than overwrite: replacing the value silently discarded any
    # flags the user or a batch script had set (e.g. --xla_dump_to=...,
    # --xla_gpu_autotune_level=0, --xla_gpu_enable_triton_gemm=false), which is
    # indistinguishable from those flags having no effect.
    if xla_env:
        os.environ["XLA_FLAGS"] = f"{xla_env} --xla_disable_hlo_passes=constant_folding"
    else:
        os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=constant_folding"

if "--xla_gpu_autotune_level" not in os.environ.get("XLA_FLAGS", ""):

    os.environ["XLA_FLAGS"] = (
        f"{os.environ['XLA_FLAGS']} --xla_gpu_autotune_level=0"
    )

    if "--xla_gpu_enable_triton_gemm" not in os.environ["XLA_FLAGS"]:
        os.environ["XLA_FLAGS"] = (
            f"{os.environ['XLA_FLAGS']} --xla_gpu_enable_triton_gemm=false"
        )

    logger.info(
        """
        XLA GPU autotuning has been disabled by default (--xla_gpu_autotune_level=0):
        it dominates cold JAX compile times on GPU (measured up to ~7 minutes for a
        single fusion on an A100, autolens_profiling jax_compile finding 6). Because
        with autotuning off XLA otherwise lowers dense float64 matrix products to an
        un-tuned Triton kernel that is ~5x slower than cuBLAS (curvature matrix
        25.5 ms vs 4.8 ms on an A100, autolens_profiling job 342333), Triton GEMM
        emission is also disabled (--xla_gpu_enable_triton_gemm=false) so those
        products go to cuBLAS.

        To change either, include an explicit --xla_gpu_autotune_level=<n> and/or
        --xla_gpu_enable_triton_gemm=<true|false> in the XLA_FLAGS environment
        variable before running your script — pre-set flags are always respected,
        and with an explicit autotune level Triton GEMM is left to XLA (it is
        autotuned against cuBLAS there).
        """
    )

jax_enable_x64 = os.environ.get("JAX_ENABLE_X64")

if jax_enable_x64 is None:
    jax_enable_x64 = False
elif isinstance(jax_enable_x64, str):
    jax_enable_x64 = jax_enable_x64.lower() == "true"

if not jax_enable_x64:

    os.environ["JAX_ENABLE_X64"] = "True"

    logger.info(
        """"
        JAX 64-bit precision has been automatically enabled for you (JAX_ENABLE_X64=True),
        as double precision is required for most scientific computing applications.

        To enable 64 precision as default in JAX, set the environment variable
        JAX_ENABLE_X64=true before running your script.
        """
    )

if "JAX_COMPILATION_CACHE_DIR" not in os.environ:

    _cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    _cache_dir = os.path.join(_cache_root, "pyauto_jax")

    os.environ["JAX_COMPILATION_CACHE_DIR"] = _cache_dir

    logger.info(
        f"""
        The JAX persistent compilation cache has been enabled at {_cache_dir}
        (JAX_COMPILATION_CACHE_DIR). The first fit of a given model and data shape
        on this machine compiles its JAX functions, which can take minutes; the
        compiled code is cached on disk, so later runs (including after restarting
        Python) skip this cost.

        To use a different location, set JAX_COMPILATION_CACHE_DIR before running
        your script. To disable the cache entirely, set it to an empty string.
        """
    )

# An explicitly empty JAX_COMPILATION_CACHE_DIR means "cache disabled"; do not
# force a compile-time threshold in that case.
if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")


def register_pytree_node_class(cls):
    """Opt-in JAX pytree class registration that defers the JAX import.

    The previous eager registration in ``autofit.mapper.prior_model.prior_model``
    forced ``jax.tree_util`` to load whenever ``import autofit`` ran. To keep
    JAX an optional dependency, library code now exposes ``tree_flatten`` /
    ``tree_unflatten`` methods but does NOT register the class itself; callers
    that want JAX integration call this helper explicitly (typically via
    ``autofit.jax.enable_pytrees()``).

    No-ops if JAX is not installed.
    """
    try:
        from jax.tree_util import register_pytree_node_class as _r
    except ImportError:
        return cls
    return _r(cls)


def register_pytree_node(nodetype, flatten_func, unflatten_func):
    """Opt-in JAX pytree registration for an externally-defined class.

    Lazy counterpart to :func:`register_pytree_node_class` for the case where
    the class cannot be decorated directly. No-ops if JAX is not installed.
    """
    try:
        from jax.tree_util import register_pytree_node as _r
    except ImportError:
        return None
    return _r(nodetype, flatten_func, unflatten_func)
