"""
The site layer (``autonerves.site_config``, ``Config.push_site``): schema validation,
values with ``unverified`` flags and ranges, the shipped RAL site and its GPU-only rule.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from autonerves import conf
from autonerves.site_config import (
    PACKAGE_SITE_DIRECTORY,
    SiteConfigException,
    SiteValue,
    load_site,
    validate,
)

VALID = {
    "name": "toy",
    "partitions": {
        "cpu": {"gpu_only": False, "cores_per_node": 64, "max_time": "2-00:00:00"},
        "gpu": {"gpu_only": True, "gpu_model": "A100", "gpu_tdp_w": {"low": 250, "high": 400}},
    },
    "pue": {"low": 1.1, "high": 1.5},
    "grid_region": "GB",
    "ci_basis": "location_annual_average",
    "energy_accounting": False,
}


def test_push_site_loads_the_shipped_ral_site(config):
    site = config.push_site("ral")

    assert config.site is site
    assert site.name == "ral"
    assert site.source == PACKAGE_SITE_DIRECTORY / "ral.yaml"
    assert set(site.partitions) == {"ral", "gpu"}


def test_ral_gpu_partition_is_gpu_only(config):
    site = config.push_site("ral")

    assert site.partition("gpu").gpu_only
    assert site.partition("gpu").gpu_model == "A100"
    assert not site.partition("ral").gpu_only
    assert [p.name for p in site.cpu_partitions] == ["ral"]

    site.check_cpu_job("ral")
    for partitions in ("gpu", "ral,gpu", "gpu,ral", ["ral", "gpu"]):
        with pytest.raises(SiteConfigException, match="gpu_only"):
            site.check_cpu_job(partitions)


def test_ral_unconfirmed_constants_are_unverified_ranges(config):
    site = config.push_site("ral")

    unverified = site.unverified()
    assert "pue" in unverified
    assert "partitions.ral.tdp_w_per_cpu" in unverified
    assert "partitions.gpu.gpu_tdp_w" in unverified
    # The GPU-only rule is a human ruling, not an unconfirmed constant.
    assert "partitions.gpu.gpu_only" not in unverified

    assert site.pue.is_range and site.pue.range == (1.14, 1.45)
    assert site.pue.central == 1.31
    assert site.partition("ral").tdp_w_per_cpu == 3.5


def test_shipped_site_files_are_all_valid():
    files = list(PACKAGE_SITE_DIRECTORY.glob("*.yaml"))
    assert files
    for path in files:
        assert validate(yaml.safe_load(path.read_text())) == [], path


def test_config_directory_site_overrides_the_shipped_one(tmp_path, files_directory):
    (tmp_path / "site").mkdir()
    (tmp_path / "site" / "ral.yaml").write_text(yaml.safe_dump(VALID))
    config = conf.Config(tmp_path, files_directory / "default")

    site = config.push_site("ral")

    assert site.source == tmp_path / "site" / "ral.yaml"
    assert site.name == "toy"


def test_push_site_by_path_and_unknown_name(tmp_path, config):
    path = tmp_path / "mine.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in VALID.items() if k != "name"}))

    assert config.push_site(path).name == "mine"

    with pytest.raises(SiteConfigException, match="No site file"):
        config.push_site("no_such_site")


def test_site_value_forms():
    assert SiteValue.from_yaml(3).central == 3
    assert SiteValue.from_yaml({"value": 3, "unverified": True}).unverified
    ranged = SiteValue.from_yaml({"low": 2, "high": 4})
    assert ranged.is_range and ranged.central == 3 and ranged.range == (2, 4)
    assert not ranged.unverified


@pytest.mark.parametrize(
    "change, message",
    [
        ({"partitions": {}}, "at least one partition"),
        ({"surprise": 1}, "unknown site keys"),
        ({"pue": 1.3}, "must be a range"),
        ({"pue": {"low": 1.5, "high": 1.2}}, "is above high"),
        ({"pue": {"low": 0.9, "high": 1.2}}, "at least 1"),
        ({"energy_accounting": "no"}, "must be a bool"),
        ({"partitions": {"gpu": {"gpu_only": True}}}, "must name its gpu_model"),
        ({"partitions": {"cpu": {"cores_per_node": "many"}}}, "must be a int"),
        ({"partitions": {"cpu": {"max_time": "a week"}}}, "must be a time"),
        ({"partitions": {"cpu": {"cpu_is_thread": {"low": 1, "high": 2}}}}, "cannot be a range"),
        ({"partitions": {"cpu": {"cores_per_node": {"value": 1, "unverified": "yes"}}}}, "unverified must be"),
        ({"partitions": {"cpu": {"cores": 1}}}, "unknown keys"),
    ],
)
def test_validation_errors(change, message, tmp_path):
    raw = {**VALID, **change}
    assert any(message in error for error in validate(raw)), validate(raw)

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SiteConfigException, match=message):
        load_site(path)


def test_valid_toy_site():
    assert validate(VALID) == []


def test_nerves_imports_no_autofit():
    """
    Loading a site imports nothing from autofit (Nerves sits below the libraries).
    """
    code = (
        "import sys; from autonerves import conf; "
        "conf.Config().push_site('ral'); "
        "assert not [m for m in sys.modules if m.split('.')[0] == 'autofit'], 'autofit'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(conf.__file__).parents[1])
