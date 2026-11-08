"""
Process-level cache of parsed configuration files.

``Config.push`` / ``Config.register`` invalidate the lazily-built config
dictionary and prior configuration, so before this cache every library import
re-read every YAML / JSON config file from disk: hundreds of parses per script
run for a few dozen distinct files.

Each file is parsed at most once per process while it is unchanged on disk. The
cache key is the absolute path and the parser kind; an entry is reused only
while the file's ``st_mtime_ns`` and ``st_size`` match, so editing a file
re-parses it exactly as before.

Callers get a deep copy of the cached object, so mutating a loaded config can
never leak into a later load — the result is identical to a fresh parse.
"""

import copy
import json
import os

import yaml

_cache = {}


def clear():
    """Drop every cached parse."""
    _cache.clear()


def _parse_yaml(f):
    # Looked up at call time so a hook on ``yaml.safe_load`` sees every real parse.
    return yaml.safe_load(f)


def _parse_json(f):
    return json.load(f)


_PARSERS = {"yaml": _parse_yaml, "json": _parse_json}


def _load(path, kind):
    path = os.path.abspath(os.fspath(path))
    stat = os.stat(path)
    signature = (stat.st_mtime_ns, stat.st_size)
    key = (path, kind)

    entry = _cache.get(key)
    if entry is None or entry[0] != signature:
        with open(path) as f:
            parsed = _PARSERS[kind](f)
        entry = (signature, parsed)
        _cache[key] = entry

    return copy.deepcopy(entry[1])


def load_yaml(path):
    """The result of ``yaml.safe_load`` on the file at ``path``, parsed once."""
    return _load(path, "yaml")


def load_json(path):
    """The result of ``json.load`` on the file at ``path``, parsed once."""
    return _load(path, "json")
