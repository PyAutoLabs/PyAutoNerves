import datetime
import os
import warnings
from pathlib import Path

from autonerves import exc


class WorkspaceVersionMismatchError(exc.ConfigException):
    pass


_BYPASS_ENV_VAR = "PYAUTO_SKIP_WORKSPACE_VERSION_CHECK"

# The installed library may legitimately run ahead of a workspace clone —
# releases are frequent, `git pull`s are not. Only warn once the gap is large
# enough to suggest the clone is genuinely stale.
_STALENESS_WINDOW_DAYS = 30

# Every library init (autofit, autogalaxy, autolens, ...) calls check_version,
# so without dedup a byte-identical warning prints once per library in the
# import chain. Python's own warning registry does not dedupe here because
# third-party imports between the calls invalidate it.
_warned_messages = set()


def _warn_once(message):
    if message in _warned_messages:
        return
    _warned_messages.add(message)
    warnings.warn(message)


def _read_general_yaml(workspace_root):
    """
    Return the parsed ``config/general.yaml`` dict for ``workspace_root``.

    Returns an empty dict on any failure (missing file, missing yaml module,
    unreadable YAML) so the caller can fall through to legacy ``version.txt``
    handling without crashing the user's script on import.
    """
    try:
        import yaml

        config_path = workspace_root / "config" / "general.yaml"
        with config_path.open("r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _yaml_bypass_set(general_yaml):
    return general_yaml.get("version", {}).get("workspace_version_check") is False


def _yaml_version_value(general_yaml, key):
    value = general_yaml.get("version", {}).get(key)
    if value is None:
        return None
    return str(value).strip()


def _missing_version_warning(workspace_root, library_version):
    return (
        f"Cannot verify the workspace at {workspace_root} is compatible with "
        f"the installed library version ({library_version}): no "
        f"`version.minimum_library_version` or `version.workspace_version` "
        f"key in config/general.yaml and no version.txt at the workspace "
        f"root.\n\n"
        f"If you cloned the workspace from `main` rather than a release tag, "
        f"set `version.workspace_version_check: False` in "
        f"config/general.yaml to silence this warning. The `main` branch "
        f"updates more frequently than library releases, so version "
        f"mismatches are expected and not actionable for `main`-branch users.\n\n"
        f"You can also set the environment variable "
        f"{_BYPASS_ENV_VAR}=1 to disable temporarily."
    )


def _parse_version(version_string):
    try:
        return tuple(int(part) for part in version_string.split("."))
    except (AttributeError, ValueError):
        return None


def _version_date(parsed_version):
    """
    The ``(year, month, day)`` prefix of a parsed date-version as a
    ``datetime.date``, or None when the prefix is not a valid date.
    """
    try:
        return datetime.date(parsed_version[0], parsed_version[1], parsed_version[2])
    except (IndexError, TypeError, ValueError):
        return None


def _is_source_checkout(root):
    """
    True when ``root`` is a Python package source checkout (a ``setup.py`` or
    ``pyproject.toml`` at its top level) rather than a workspace clone.

    ``check_version`` is called unconditionally on library import with
    ``workspace_root`` defaulting to the current working directory, so any
    pytest run or script executed from inside a library's own repo would
    otherwise warn "Cannot verify the workspace ..." on every import — a
    false positive, since a source checkout is not a workspace and records
    no version floor to verify. Workspace clones ship neither file, so a
    genuine workspace missing its version keys still warns.

    The companion skip for directories that are not a workspace at all (a
    plain data directory a user runs a script from) is
    ``_looks_like_workspace``.
    """
    return (root / "setup.py").exists() or (root / "pyproject.toml").exists()


def _looks_like_workspace(root):
    """
    True when ``root`` has a ``config/`` directory, which every PyAuto
    workspace ships.

    ``check_version`` defaults ``workspace_root`` to the current working
    directory, so a plain script run from a data directory would otherwise
    warn "Cannot verify the workspace ..." on every library import even
    though no workspace is involved. Only the directory itself is inspected
    -- there is deliberately no walk-up to a parent, which could pick up an
    unrelated ``config/`` -- so a workspace script run from a subdirectory
    is also treated as outside a workspace (silent, floor unchecked).
    """
    return (root / "config").is_dir()


def _library_name_from_workspace(workspace_root):
    name = workspace_root.name
    suffix = "_workspace"
    if name.endswith(suffix) and len(name) > len(suffix):
        return name[: -len(suffix)]
    return None


def _bypass_block():
    return (
        f"To bypass this check, edit config/general.yaml:\n\n"
        f"    version:\n"
        f"      workspace_version_check: False\n\n"
        f"You can also set the environment variable "
        f"{_BYPASS_ENV_VAR}=1 to disable temporarily."
    )


def _below_floor_message(floor_version, library_version, workspace_root):
    package = _library_name_from_workspace(workspace_root) or "<library>"
    return (
        f"The installed library version ({library_version}) is older than the "
        f"minimum version this workspace requires ({floor_version}), so its "
        f"scripts may use API your install does not have. Update the "
        f"library:\n\n"
        f"    pip install --upgrade {package}\n\n"
        f"If the newest release on PyPI is still older than {floor_version}, "
        f"this workspace clone tracks code that has not been released yet — "
        f"either wait for the next release, or check out the workspace state "
        f"matching your installed version:\n\n"
        f"    cd {workspace_root} && git checkout {library_version}\n\n"
        f"IMPORTANT: If you cloned the workspace from `main` rather than a "
        f"release tag, `main` may require a newer library than the latest "
        f"release provides.\n\n"
        f"{_bypass_block()}"
    )


def _stale_workspace_message(floor_version, library_version, workspace_root):
    return (
        f"The workspace at {workspace_root} records library version "
        f"{floor_version}, but the installed library is {library_version} — "
        f"more than {_STALENESS_WINDOW_DAYS} days newer. The workspace "
        f"examples and configs may lag the installed API. Pull the latest "
        f"workspace:\n\n"
        f"    cd {workspace_root} && git pull origin main\n\n"
        f"{_bypass_block()}"
    )


def _unparseable_mismatch_message(floor_version, library_version, workspace_root):
    return (
        f"The workspace at {workspace_root} records library version "
        f"{floor_version}, which does not match the installed library "
        f"version ({library_version}); the two cannot be compared as date "
        f"versions, so compatibility is unverified. This is expected for "
        f"development installs.\n\n"
        f"{_bypass_block()}"
    )


def check_version(library_version, workspace_root=None):
    """
    Verify that the installed library is new enough for the workspace at
    ``workspace_root``.

    The workspace records the **minimum library version** its scripts
    require, resolved with the following precedence:

    1. ``config/general.yaml`` — ``version.minimum_library_version``, bumped
       deliberately when workspace scripts start depending on new API.
    2. ``config/general.yaml`` — ``version.workspace_version``, the legacy
       exact release stamp, reinterpreted as a floor so older clones keep
       working against newer libraries.
    3. ``version.txt`` at the workspace root — legacy fallback for clones
       that pre-date the YAML keys.

    An installed library **older** than the floor raises
    ``WorkspaceVersionMismatchError`` — the workspace's scripts may use API
    the install does not have. An installed library **newer** than the floor
    passes; if it is more than ``_STALENESS_WINDOW_DAYS`` days newer (by
    date-version comparison) a warning suggests pulling the workspace.
    Versions that cannot be parsed as date versions (e.g. development
    installs) warn on inequality rather than raising.

    If no floor source is found, the check is skipped. It is skipped
    silently when ``workspace_root`` is not a workspace at all: either a
    package source checkout (``setup.py``/``pyproject.toml`` at its top
    level — running from inside a library's own repo) or a directory with
    no ``config/`` (e.g. a plain data directory a user runs a script from).
    Otherwise — a ``config/`` directory with no version keys and no
    ``version.txt``, i.e. a misconfigured workspace — a warning is emitted.

    The check can be disabled in two ways:

    * Set ``version.workspace_version_check: False`` in
      ``config/general.yaml`` — the recommended path for users on
      ``main``-branch workspace clones.
    * Set ``PYAUTO_SKIP_WORKSPACE_VERSION_CHECK=1`` — intended for
      developers running source checkouts where workspace and library
      versions intentionally diverge.

    Defaults ``workspace_root`` to the current working directory, which is
    where users run workspace scripts from.
    """
    if os.environ.get(_BYPASS_ENV_VAR) == "1":
        return

    root = Path(workspace_root) if workspace_root else Path.cwd()

    general_yaml = _read_general_yaml(root)

    if _yaml_bypass_set(general_yaml):
        return

    floor_version = _yaml_version_value(general_yaml, "minimum_library_version")

    if floor_version is None:
        floor_version = _yaml_version_value(general_yaml, "workspace_version")

    if floor_version is None:
        version_file = root / "version.txt"
        if version_file.exists():
            floor_version = version_file.read_text().strip()

    if floor_version is None or floor_version == "":
        if _is_source_checkout(root) or not _looks_like_workspace(root):
            return
        _warn_once(_missing_version_warning(root, library_version))
        return

    if floor_version == library_version:
        return

    floor_parsed = _parse_version(floor_version)
    library_parsed = _parse_version(library_version)

    if floor_parsed is None or library_parsed is None:
        _warn_once(
            _unparseable_mismatch_message(floor_version, library_version, root)
        )
        return

    if library_parsed < floor_parsed:
        raise WorkspaceVersionMismatchError(
            _below_floor_message(floor_version, library_version, root)
        )

    floor_date = _version_date(floor_parsed)
    library_date = _version_date(library_parsed)

    if (
        floor_date is not None
        and library_date is not None
        and (library_date - floor_date).days > _STALENESS_WINDOW_DAYS
    ):
        _warn_once(_stale_workspace_message(floor_version, library_version, root))
