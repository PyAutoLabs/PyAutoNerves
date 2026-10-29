"""
The site layer: facts about the computing site a campaign runs on, loaded from a
``site/<name>.yaml`` file by ``Config.push_site(name)``.

A site file names the scheduler partitions jobs may use and the constants an energy
and carbon estimate needs, in the vocabulary of GA4HPC's ``cluster_info.yaml``:

.. code-block:: yaml

    name: ral
    partitions:
      ral:
        gpu_only: false
        cores_per_node: {value: null, unverified: true}   # not yet known
        mem_per_node_mb: {value: null, unverified: true}
        max_time: {value: "2-00:00:00", unverified: true}
        max_array_size: {value: 1001, unverified: true}
        tdp_w_per_cpu: {low: 2.8, central: 3.5, high: 4.4, unverified: true}
        cpu_is_thread: {value: true, unverified: true}
      gpu:
        gpu_only: true            # CPU-only jobs must never be sent here
        gpu_model: A100
        gpu_tdp_w: {low: 250, high: 400, unverified: true}
    pue: {low: 1.14, central: 1.31, high: 1.45, unverified: true}
    grid_region: GB
    ci_basis: location_annual_average
    energy_accounting: false

Any value may be written plainly (``gpu_only: true``), as ``{value: ...}`` or as a
range ``{low: ..., high: ...}`` (with an optional ``central``), and any of these may
carry ``unverified: true``: a constant nobody has confirmed against the site is kept,
used as a range and reported as unverified, never silently trusted. ``pue`` is always
a range.

A partition with ``gpu_only: true`` takes GPU jobs only: a planner must never place a
CPU-only job there (``Site.cpu_partitions`` leaves it out, ``Site.check_cpu_job``
raises).

The site files shipped with autonerves are in ``autonerves/site/``; a
``site/<name>.yaml`` in any pushed config directory overrides the shipped one.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from autonerves.exc import ConfigException

PACKAGE_SITE_DIRECTORY = Path(__file__).parent / "site"

PARTITION_FIELDS = {
    "gpu_only": "bool",
    "cores_per_node": "int",
    "mem_per_node_mb": "int",
    "max_time": "time",
    "max_array_size": "int",
    "tdp_w_per_cpu": "number",
    "cpu_is_thread": "bool",
    "gpu_model": "str",
    "gpu_tdp_w": "number",
    "note": "str",
}

SITE_FIELDS = {
    "name": "str",
    "partitions": "partitions",
    "pue": "range",
    "grid_region": "str",
    "ci_basis": "str",
    "energy_accounting": "bool",
    "note": "str",
}

# Slurm's time formats: "minutes", "HH:MM:SS", "D-HH", "D-HH:MM", "D-HH:MM:SS",
# "MM:SS"; or "UNLIMITED".
_TIME = re.compile(r"^(\d+-)?\d+(:\d{2}){0,2}$|^UNLIMITED$", re.IGNORECASE)


class SiteConfigException(ConfigException):
    pass


class SiteValue:
    def __init__(
        self,
        value=None,
        low=None,
        high=None,
        central=None,
        unverified: bool = False,
    ):
        """
        One site constant: a plain ``value`` or a ``low``-``high`` range with an
        optional ``central`` value, and whether it is ``unverified``.
        """
        self.value = value
        self.low = low
        self.high = high
        self._central = central
        self.unverified = bool(unverified)

    @classmethod
    def from_yaml(cls, raw) -> "SiteValue":
        if isinstance(raw, dict):
            return cls(
                value=raw.get("value"),
                low=raw.get("low"),
                high=raw.get("high"),
                central=raw.get("central"),
                unverified=raw.get("unverified", False),
            )
        return cls(value=raw)

    @property
    def is_range(self) -> bool:
        return self.low is not None or self.high is not None

    @property
    def central(self):
        """
        The value to use when one number is needed: ``value``, else ``central``,
        else the middle of the range.
        """
        if self.value is not None:
            return self.value
        if self._central is not None:
            return self._central
        if self.low is not None and self.high is not None:
            return (self.low + self.high) / 2
        return self.low if self.low is not None else self.high

    @property
    def range(self):
        """
        ``(low, high)``; a plain value is the range of itself.
        """
        if self.is_range:
            return self.low, self.high
        return self.value, self.value

    def __repr__(self):
        flag = ", unverified" if self.unverified else ""
        if self.is_range:
            return f"SiteValue({self.low}-{self.high}{flag})"
        return f"SiteValue({self.value!r}{flag})"


class Partition:
    def __init__(self, name: str, raw: Dict):
        self.name = name
        self.values = {
            key: SiteValue.from_yaml(value)
            for key, value in raw.items()
            if key in PARTITION_FIELDS
        }

    def __getattr__(self, item):
        if item in PARTITION_FIELDS:
            value = self.__dict__["values"].get(item)
            return None if value is None else value.central
        raise AttributeError(item)

    @property
    def gpu_only(self) -> bool:
        value = self.values.get("gpu_only")
        return bool(value.central) if value is not None else False

    @property
    def accepts_cpu_jobs(self) -> bool:
        return not self.gpu_only

    def __repr__(self):
        return f"Partition({self.name!r}, gpu_only={self.gpu_only})"


class Site:
    def __init__(self, raw: Dict, source: Optional[Path] = None):
        """
        A validated site file (``load_site``).
        """
        self.raw = raw
        self.source = source
        self.name = raw.get("name")
        self.partitions = {
            name: Partition(name, partition or {})
            for name, partition in (raw.get("partitions") or {}).items()
        }
        self.values = {
            key: SiteValue.from_yaml(raw[key])
            for key in ("pue", "grid_region", "ci_basis", "energy_accounting")
            if key in raw
        }

    def partition(self, name: str) -> Partition:
        try:
            return self.partitions[name]
        except KeyError:
            raise SiteConfigException(
                f"Site {self.name!r} has no partition {name!r} "
                f"(it has {sorted(self.partitions)})."
            )

    @property
    def cpu_partitions(self) -> List[Partition]:
        """
        The partitions a CPU-only job may use (every partition not ``gpu_only``).
        """
        return [p for p in self.partitions.values() if p.accepts_cpu_jobs]

    def check_cpu_job(self, partitions: Union[str, List[str]]):
        """
        Raise if a CPU-only job would be sent to a ``gpu_only`` partition.
        ``partitions`` is a partition name, a Slurm list (``"ral,gpu"``) or a list.
        """
        if isinstance(partitions, str):
            partitions = partitions.split(",")
        for name in partitions:
            if self.partition(name.strip()).gpu_only:
                raise SiteConfigException(
                    f"Partition {name.strip()!r} of site {self.name!r} is gpu_only: "
                    f"CPU-only jobs must use {[p.name for p in self.cpu_partitions]}."
                )

    @property
    def pue(self) -> Optional[SiteValue]:
        return self.values.get("pue")

    @property
    def grid_region(self):
        value = self.values.get("grid_region")
        return None if value is None else value.central

    @property
    def ci_basis(self):
        value = self.values.get("ci_basis")
        return None if value is None else value.central

    @property
    def energy_accounting(self) -> bool:
        value = self.values.get("energy_accounting")
        return bool(value.central) if value is not None else False

    def unverified(self) -> List[str]:
        """
        The dotted names of every value marked ``unverified``.
        """
        names = [key for key, value in self.values.items() if value.unverified]
        for partition in self.partitions.values():
            names += [
                f"partitions.{partition.name}.{key}"
                for key, value in partition.values.items()
                if value.unverified
            ]
        return names

    def __repr__(self):
        return f"Site({self.name!r}, partitions={sorted(self.partitions)})"


def _check_type(kind: str, value) -> bool:
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "str":
        return isinstance(value, str)
    if kind == "time":
        return isinstance(value, (str, int)) and bool(_TIME.match(str(value)))
    return True


def _check_value(where: str, kind: str, raw, errors: List[str], range_only=False):
    if isinstance(raw, dict):
        unknown = set(raw) - {"value", "low", "high", "central", "unverified"}
        if unknown:
            errors.append(f"{where}: unknown keys {sorted(unknown)}")
        if "unverified" in raw and not isinstance(raw["unverified"], bool):
            errors.append(f"{where}.unverified must be true or false")
        has_value = raw.get("value") is not None
        has_range = raw.get("low") is not None or raw.get("high") is not None
        if has_value and has_range:
            errors.append(f"{where}: give value or low/high, not both")
        if range_only and not (raw.get("low") is not None and raw.get("high") is not None):
            errors.append(f"{where}: must be a range with low and high")
        if has_range and kind not in ("number", "int", "range"):
            errors.append(f"{where}: a {kind} cannot be a range")
        numeric = "number" if kind == "range" else kind
        for key in ("value", "low", "high", "central"):
            if raw.get(key) is not None and not _check_type(numeric, raw[key]):
                errors.append(f"{where}.{key} must be a {numeric}: {raw[key]!r}")
        low, high = raw.get("low"), raw.get("high")
        if (
            isinstance(low, (int, float))
            and isinstance(high, (int, float))
            and low > high
        ):
            errors.append(f"{where}: low {low} is above high {high}")
        return
    if range_only:
        errors.append(f"{where}: must be a range with low and high")
        return
    if raw is not None and not _check_type(kind, raw):
        errors.append(f"{where} must be a {kind}: {raw!r}")


def validate(raw) -> List[str]:
    """
    Every problem with a site file's contents, as messages (empty when valid).
    """
    errors = []
    if not isinstance(raw, dict):
        return ["a site file must be a mapping"]

    unknown = set(raw) - set(SITE_FIELDS)
    if unknown:
        errors.append(f"unknown site keys {sorted(unknown)}")

    partitions = raw.get("partitions")
    if not isinstance(partitions, dict) or not partitions:
        errors.append("partitions: at least one partition is required")
        partitions = {}

    for name, partition in partitions.items():
        where = f"partitions.{name}"
        if partition is None:
            partition = {}
        if not isinstance(partition, dict):
            errors.append(f"{where} must be a mapping")
            continue
        unknown = set(partition) - set(PARTITION_FIELDS)
        if unknown:
            errors.append(f"{where}: unknown keys {sorted(unknown)}")
        for key, kind in PARTITION_FIELDS.items():
            if key in partition:
                _check_value(f"{where}.{key}", kind, partition[key], errors)
        gpu_only = SiteValue.from_yaml(partition.get("gpu_only")).central
        if gpu_only and partition.get("gpu_model") is None:
            errors.append(f"{where}: a gpu_only partition must name its gpu_model")

    for key in ("grid_region", "ci_basis", "name", "note"):
        if key in raw:
            _check_value(key, "str", raw[key], errors)
    if "energy_accounting" in raw:
        _check_value("energy_accounting", "bool", raw["energy_accounting"], errors)
    if "pue" in raw:
        _check_value("pue", "range", raw["pue"], errors, range_only=True)
        pue = raw["pue"] if isinstance(raw["pue"], dict) else {}
        if isinstance(pue.get("low"), (int, float)) and pue["low"] < 1:
            errors.append("pue: a PUE is at least 1")

    return errors


def load_site(path: Union[str, Path]) -> Site:
    """
    Load and validate the site file at ``path``.

    Raises
    ------
    SiteConfigException
        If the file is missing, is not YAML or fails ``validate``.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exception:
        raise SiteConfigException(f"Site file {path} cannot be read: {exception}")
    except yaml.YAMLError as exception:
        raise SiteConfigException(f"Site file {path} is not valid YAML: {exception}")

    errors = validate(raw)
    if errors:
        raise SiteConfigException(
            f"Site file {path} is invalid:\n  " + "\n  ".join(errors)
        )
    raw.setdefault("name", path.stem)
    return Site(raw, source=path)


def site_path(name: str, config_paths=()) -> Path:
    """
    The file of site ``name``: ``site/<name>.yaml`` in the first config directory
    that has one, else the copy shipped in ``autonerves/site/``. ``name`` may also be a
    path to a site file.

    Raises
    ------
    SiteConfigException
        If no such site file exists.
    """
    direct = Path(name)
    if direct.suffix in (".yaml", ".yml") and direct.is_file():
        return direct

    candidates = [Path(path) / "site" / f"{name}.yaml" for path in config_paths]
    candidates.append(PACKAGE_SITE_DIRECTORY / f"{name}.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    shipped = sorted(p.stem for p in PACKAGE_SITE_DIRECTORY.glob("*.yaml"))
    raise SiteConfigException(
        f"No site file for {name!r} (looked in {[str(c) for c in candidates]}; "
        f"shipped sites: {shipped})."
    )
