"""
The config parse cache (``autonerves.parse_cache``) must be invisible: every
config object built through it compares equal to one built from fresh parses,
while each file is parsed at most once per process.
"""

import collections
import json
import os
from pathlib import Path

import pytest
import yaml

from autonerves import conf, parse_cache
from autonerves.json_prior.config import JSONPriorConfig

PACKAGE_ROOT = Path(__file__).parent.parent
CONFIG_SUFFIXES = (".yaml", ".yml", ".json", ".ini")


def _packaged_config_directories():
    """Every directory in the repo that ships config files (test fixtures included)."""
    candidates = sorted((PACKAGE_ROOT / "test_autonerves" / "files").iterdir())
    candidates += sorted(p for p in (PACKAGE_ROOT / "autonerves").iterdir())
    return [
        directory
        for directory in candidates
        if directory.is_dir()
        and any(
            f.endswith(CONFIG_SUFFIXES)
            for _, _, files in os.walk(directory)
            for f in files
        )
    ]


PACKAGED = _packaged_config_directories()


def _plain(value):
    if isinstance(value, conf.DictWrapper):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _snapshot(directory):
    """Everything a Config over ``directory`` parses, as plain data."""
    config = conf.Config(directory)
    priors = JSONPriorConfig.from_directory(Path(directory) / "priors")
    return {
        "dict": _plain(config.dict),
        "priors": priors.obj,
        "logging": config.logging_config,
    }


def _fresh_load(path, kind):
    with open(path) as f:
        return yaml.safe_load(f) if kind == "yaml" else json.load(f)


@pytest.fixture(autouse=True)
def _clear_cache():
    parse_cache.clear()
    yield
    parse_cache.clear()


@pytest.fixture(name="parse_counter")
def make_parse_counter(monkeypatch):
    counts = collections.Counter()
    original = yaml.safe_load

    def counting_safe_load(stream, *args, **kwargs):
        counts[getattr(stream, "name", None)] += 1
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(yaml, "safe_load", counting_safe_load)
    return counts


def test_packaged_directories_found():
    names = {d.name for d in PACKAGED}
    assert {"config", "default", "config_flip"} <= names


@pytest.mark.parametrize("directory", PACKAGED, ids=lambda d: d.name)
def test_cached_parse_deep_equals_fresh_parse(directory, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(parse_cache, "_load", _fresh_load)
        reference = _snapshot(directory)

    cold = _snapshot(directory)
    warm = _snapshot(directory)

    assert cold == reference
    assert warm == reference


def test_each_file_parsed_once_across_pushes(files_directory, parse_counter):
    config = conf.Config(files_directory / "default")
    for directory in ("config_flip", "config", "default", "config"):
        config.push(files_directory / directory)
        config.dict
        config.prior_config.prior_configs[0].path_value_map

    assert parse_counter
    assert max(parse_counter.values()) == 1


def test_returned_objects_are_independent(tmp_path):
    path = tmp_path / "general.yaml"
    path.write_text("section:\n  key: [1, 2]\n")

    first = parse_cache.load_yaml(path)
    first["section"]["key"].append(3)

    assert parse_cache.load_yaml(path) == {"section": {"key": [1, 2]}}


def test_changed_file_is_reparsed(tmp_path, parse_counter):
    path = tmp_path / "general.yaml"
    path.write_text("a: 1\n")
    assert parse_cache.load_yaml(path) == {"a": 1}

    path.write_text("a: 22\n")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

    assert parse_cache.load_yaml(path) == {"a": 22}
    assert sum(parse_counter.values()) == 2


def test_missing_file_raises_and_is_not_cached(tmp_path):
    path = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        parse_cache.load_yaml(path)
    path.write_text("a: 1\n")
    assert parse_cache.load_yaml(path) == {"a": 1}


def test_json_and_yaml_kinds_are_cached_separately(tmp_path):
    path = tmp_path / "both.json"
    path.write_text('{"a": 1}')
    assert parse_cache.load_json(path) == {"a": 1}
    assert parse_cache.load_yaml(path) == {"a": 1}
