"""
The environment layer: what hardware and optional packages this process can use,
and what compute setup the user asked for through ``PYAUTO_SETUP``.

This module only *detects* and *parses*; it never chooses a setup. Choosing (the
backend, device, operator and precision that ``setup="auto"`` resolves to) belongs to
``Setup.auto`` in PyAutoArray, which reads :class:`Environment` and
:func:`parse_setup_intent` from here.

It imports nothing heavy at module level (no JAX, numba or nufftax), because
``autonerves.jax_wrapper`` calls :func:`parse_setup_intent` and
:func:`detect_cores` before JAX may be imported.
"""
import importlib.util
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

SETUP_ENV_VAR = "PYAUTO_SETUP"

SETUP_BACKENDS = ("jax", "numba", "numpy")
SETUP_DEVICES = ("cpu", "gpu")
SETUP_PRECISIONS = ("fp64", "fp32", "mp")

_SETUP_ID = re.compile(
    r"^(?P<backend>jax|numba|numpy)"
    r"_(?P<device>cpu|gpu)"
    r"_(?P<operator>[a-z0-9]+)"
    r"_(?P<precision>fp64|fp32|mp)"
    r"(?:_(?P<transformer>[a-z0-9]+))?$"
)

_ALLOWED_FORMS = (
    "unset, '' or 'auto'; 'cpu'; 'gpu'; or a setup_id "
    "'<backend>_<device>_<operator>_<precision>[_<transformer>]' with backend in "
    f"{SETUP_BACKENDS}, device in {SETUP_DEVICES}, precision in {SETUP_PRECISIONS} "
    "and operator/transformer lowercase alphanumeric tokens "
    "(e.g. 'numba_cpu_sparse_fp64', 'jax_gpu_dense_fp64', "
    "'jax_gpu_sparse_fp64_wtilde')"
)


def parse_setup_intent(value: Optional[str]) -> str:
    """
    The device intent of a ``PYAUTO_SETUP`` value: ``"auto"``, ``"cpu"`` or ``"gpu"``.

    Accepted forms (case-insensitive, surrounding whitespace ignored):

    - ``None`` (unset), ``""`` or ``"auto"`` -> ``"auto"``: nothing is pinned; JAX
      picks the GPU when one is present.
    - ``"cpu"`` / ``"gpu"`` -> that device.
    - a setup_id ``<backend>_<device>_<operator>_<precision>[_<transformer>]``, the
      vocabulary of the profiling campaigns (e.g. ``numba_cpu_sparse_fp64``,
      ``jax_gpu_dense_fp64``, ``jax_gpu_sparse_fp64_wtilde``): ``backend`` is one of
      ``jax``, ``numba``, ``numpy``; ``device`` is ``cpu`` or ``gpu``; ``precision``
      is ``fp64``, ``fp32`` or ``mp``; ``operator`` (``dense``, ``sparse``, ``na``,
      ...) and the optional ``transformer`` (``wtilde``, ``nufft``, ...) are any
      lowercase alphanumeric token. The device token decides the intent.

    Raises
    ------
    ValueError
        If ``value`` is none of these; the message names the allowed forms.
    """
    if value is None:
        return "auto"
    text = value.strip().lower()
    if text in ("", "auto"):
        return "auto"
    if text in SETUP_DEVICES:
        return text
    match = _SETUP_ID.match(text)
    if match is None:
        raise ValueError(
            f"{SETUP_ENV_VAR}={value!r} is not a valid setup. Allowed: {_ALLOWED_FORMS}."
        )
    return match.group("device")


def detect_cores() -> int:
    """
    The number of CPU cores this process may use.

    In order: ``SLURM_CPUS_PER_TASK`` (the cores Slurm allocated to the task), then
    ``len(os.sched_getaffinity(0))`` (the cores this process is pinned to, which
    respects ``taskset`` and cgroup CPU sets), then ``os.cpu_count()``; at least 1.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            cores = int(slurm)
            if cores > 0:
                return cores
        except ValueError:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    return max(1, os.cpu_count() or 1)


def detect_ram_gb() -> Optional[float]:
    """
    Total physical memory in GB, from ``MemTotal`` in ``/proc/meminfo``, falling back
    to ``os.sysconf`` (pages x page size); ``None`` if neither is available.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (AttributeError, ValueError, OSError):
        return None


def _package_available(name: str) -> bool:
    """Whether ``name`` is importable, checked with ``find_spec`` (never imported)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


GPU_PLATFORMS = ("cuda", "rocm", "gpu")


def _jax_platforms_exclude_gpu() -> bool:
    """
    Whether ``JAX_PLATFORMS`` (or the older ``JAX_PLATFORM_NAME``) restricts JAX to
    platforms with no GPU, e.g. ``JAX_PLATFORMS=cpu``. Unset or empty means no
    restriction (``False``).
    """
    value = os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME") or ""
    platforms = {p.strip().lower() for p in value.split(",") if p.strip()}
    if not platforms:
        return False
    return not any(p in GPU_PLATFORMS for p in platforms)


def detect_gpu(timeout: float = 5.0):
    """
    ``(gpu, gpu_name, gpu_memory_gb)`` for the first GPU this process can see.

    This never imports JAX itself, so it is safe before JAX initialises:

    - If ``jax`` is already in ``sys.modules``, ``jax.devices()`` is asked (JAX has
      already chosen its platform, so this reflects what JAX will actually use,
      including ``JAX_PLATFORMS=cpu``). The name is the device kind; memory comes
      from ``memory_stats()["bytes_limit"]`` when the backend reports it.
    - Otherwise, if ``JAX_PLATFORMS`` (or the older ``JAX_PLATFORM_NAME``) is set and
      names no GPU platform (``cuda``, ``rocm`` or ``gpu``), JAX will not use a GPU
      even if one is installed, so the answer is ``gpu=False`` without querying it.
    - Otherwise one ``nvidia-smi --query-gpu=name,memory.total`` subprocess runs with
      ``timeout`` seconds. A missing ``nvidia-smi``, a non-zero exit, a timeout or
      empty output all mean ``gpu=False``.
    """
    if "jax" in sys.modules:
        try:
            jax = sys.modules["jax"]
            gpus = [d for d in jax.devices() if d.platform == "gpu"]
        except Exception:
            gpus = []
        if not gpus:
            return False, None, None
        device = gpus[0]
        memory_gb = None
        try:
            stats = device.memory_stats() or {}
            if "bytes_limit" in stats:
                memory_gb = stats["bytes_limit"] / 1024**3
        except Exception:
            pass
        return True, getattr(device, "device_kind", None), memory_gb

    if _jax_platforms_exclude_gpu():
        return False, None, None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return False, None, None
    if result.returncode != 0 or not result.stdout.strip():
        return False, None, None
    first = result.stdout.strip().splitlines()[0]
    name, _, memory = first.rpartition(",")
    if not name:
        name, memory = memory, ""
    try:
        memory_gb = float(memory.strip()) / 1024  # nvidia-smi reports MiB
    except ValueError:
        memory_gb = None
    return True, name.strip() or None, memory_gb


def detect_site() -> str:
    """
    A hint naming the computing site.

    ``SLURM_CLUSTER_NAME`` (lowercased) when Slurm sets it; otherwise a class of the
    hostname: the name of a site file shipped in ``autonerves/site/`` (the site layer
    of :mod:`autonerves.site_config`) when the hostname starts with it, ``"colab"``
    on Google Colab, else ``"local"``. It is a hint only; nothing is loaded.
    """
    cluster = os.environ.get("SLURM_CLUSTER_NAME")
    if cluster:
        return cluster.strip().lower()
    try:
        hostname = socket.gethostname().lower()
    except OSError:
        hostname = ""
    from autonerves.site_config import PACKAGE_SITE_DIRECTORY

    for path in sorted(PACKAGE_SITE_DIRECTORY.glob("*.yaml")):
        if hostname.startswith(path.stem.lower()):
            return path.stem.lower()
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    return "local"


@dataclass(frozen=True)
class Environment:
    """
    What this process can use, as detected by :meth:`Environment.detect`.

    Attributes
    ----------
    gpu
        Whether a GPU is visible.
    gpu_name
        The first GPU's name, or ``None``.
    gpu_memory_gb
        The first GPU's memory in GB, or ``None`` if unknown.
    cores
        CPU cores this process may use (see :func:`detect_cores`).
    ram_gb
        Total physical memory in GB, or ``None`` if unknown.
    numba
        Whether ``numba`` is installed (checked without importing it).
    nufftax
        Whether ``nufftax`` is installed (checked without importing it).
    site
        A site hint (see :func:`detect_site`).
    """

    gpu: bool
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    cores: int
    ram_gb: Optional[float]
    numba: bool
    nufftax: bool
    site: str

    @classmethod
    def detect(cls) -> "Environment":
        """
        Probe the current process. Reports only; chooses nothing.

        Never imports JAX: GPU facts come from ``jax.devices()`` only when JAX is
        already imported, otherwise from a single ``nvidia-smi`` query with a short
        timeout (see :func:`detect_gpu`).
        """
        gpu, gpu_name, gpu_memory_gb = detect_gpu()
        return cls(
            gpu=gpu,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            cores=detect_cores(),
            ram_gb=detect_ram_gb(),
            numba=_package_available("numba"),
            nufftax=_package_available("nufftax"),
            site=detect_site(),
        )
