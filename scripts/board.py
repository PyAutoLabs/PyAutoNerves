#!/usr/bin/env python3
"""scripts/board.py — the PyAutoNerves Board: every config file and option.

A **read-only browser** of the configuration layer the Nerves own: every YAML
file under each library's ``<package>/config/`` and under each workspace's
``config/`` that overrides them, with its keys, its comments (the comments
*are* the option docs), its source and a link to it on GitHub. Nothing on the
board acts; it is a map, grown later if it needs to be.

**What it collects** (``collect()`` — the only I/O, degrading per source into
``errors[]``):

* every ``*.yaml`` / ``*.yml`` file of every source in ``SOURCES``: its line
  count, ``yaml.safe_load`` result (or the parse error), the top-level keys,
  every key path (dotted) with its line number and comment — the inline
  ``# …`` on the key's own line, else the comment block directly above it;
* **prior files** (under ``priors/``, or every top-level value a mapping of
  ``param → {type, …}``) as table rows — Class · param · type · mean/sigma
  or lower/upper · width modifier · limits;
* **the override map**: autonerves resolves a key workspace → last-imported
  library → … → PyAutoFit (``conf.Config.push(keep_first=True)``, keys
  lowercased), so every workspace file is looked up across that
  workspace's library stack in the same order — by relative path, where a
  library ``X.yaml``'s section ``f`` is the workspace's ``X/f.yaml`` and the
  other way round, as ``RecursiveConfig`` merges them (prior files: same
  path only). The *counterpart* is the first library holding the file;
  the diff is against the merged stack (the value autonerves would fall
  back to): keys whose value differs, keys only the workspace sets
  (**orphans** — no library defines them), keys only the stack sets. The
  version-handshake keys autonerves reads itself (``NERVES_OWNED``) are
  *owned*, never orphans. ``build/*.yaml`` is workspace tooling (CI/build
  lists, not library settings) and is grouped apart;
* **possibly unused library keys**: every library package's ``.py`` (and
  ``autonerves/``) is scanned for config lookups (``scan_lookups``) and each
  key of a library settings file is classed ``used`` / ``section-read`` /
  ``unused`` against the reads of the whole stack (``classify_files``) —
  static and untrusted, so it only ever adds ``info`` feed items;
* the ``PYAUTO_*`` environment variables read by ``autonerves/test_mode.py``,
  ``workspace.py`` and ``__init__.py`` — the non-YAML options the Nerves also
  own — each with the comment above it (or its function's docstring).

**Where it reads from.** Locally, each source resolves through the body map
(``PyAutoMind/repos.yaml`` ``path:``) under ``--root``; in the workflow,
``--sources DIR`` points at sparse clones laid out ``DIR/<Repo>/<config dir>``
(and ``DIR/<Repo>/<package>`` for a library's code).
GitHub slugs come from the body map's ``github:``; the Nerves' own owner comes
from ``git remote`` (the tenant firewall: no owner is written here).

**Shape** (mirrors ``PyAutoGut/scripts/board.py``): ``render(snapshot, fmt)``
is pure — ``md | md-brief | json | badge | state | html-index | html-repo`` —
and ``--site DIR`` writes the whole Pages site (``index.html``,
``repos/<Repo>.html``, ``badge.json``, ``board.json``, ``state.json``,
``dashboard.md``). This script is NOT part of the ``autonerves`` package.

Usage:
    python scripts/board.py [--brain P] [--mind P] [--root P] [--sources DIR]
                            (--collect OUT | --snapshot F)
                            (--md|--md-brief|--json|--badge|--state|
                             --html-index|--html-repo NAME|--site DIR)
"""

from __future__ import annotations

import ast
import datetime
import html as _html
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import yaml

NERVES_HOME = Path(__file__).resolve().parents[1]
CHECKIN_PROMPT = 'Review configuration work across the Nerves board in this chat. Read PyAutoNerves/AGENTS.md and inspect the relevant source repositories, YAML parse errors, option documentation, override relationships and scan coverage. Give a concise priority order and propose one bounded next step. Apply my direction while preserving the overall review. Treat possibly-unused keys as static scan hints, not permission to remove them. Route edits through the development workflow and preserve downstream API validation and human gates.'

BOARD_KEY = "nerves"  # this board's entry in the Brain's palette table
SCHEMA_VERSION = 1
STATE_ITEMS_MAX = 20

# The config sources, in one table. `stack` is the lookup order autonerves
# uses behind a workspace (the last-imported library first, PyAutoFit last),
# so a workspace file is compared against what the libraries would give.
# `package` is a library's source dir, scanned for the config keys its code
# reads (the "possibly unused keys" flag).
SOURCES = (
    {"repo": "PyAutoFit", "config": "autofit/config", "kind": "library",
     "package": "autofit", "stack": ()},
    {"repo": "PyAutoArray", "config": "autoarray/config", "kind": "library",
     "package": "autoarray", "stack": ()},
    {"repo": "PyAutoGalaxy", "config": "autogalaxy/config", "kind": "library",
     "package": "autogalaxy", "stack": ()},
    {"repo": "PyAutoLens", "config": "autolens/config", "kind": "library",
     "package": "autolens", "stack": ()},
    {"repo": "PyAutoCTI", "config": "autocti/config", "kind": "library",
     "package": "autocti", "stack": ()},
    {"repo": "autofit_workspace", "config": "config", "kind": "workspace",
     "stack": ("PyAutoFit",)},
    {"repo": "autogalaxy_workspace", "config": "config", "kind": "workspace",
     "stack": ("PyAutoGalaxy", "PyAutoArray", "PyAutoFit")},
    {"repo": "autolens_workspace", "config": "config", "kind": "workspace",
     "stack": ("PyAutoLens", "PyAutoGalaxy", "PyAutoArray", "PyAutoFit")},
    {"repo": "autocti_workspace", "config": "config", "kind": "workspace",
     "stack": ("PyAutoCTI", "PyAutoArray", "PyAutoFit")},
)

# Workspace files that are tooling (build/CI lists), not library settings.
TOOLING_DIRS = ("build",)
YAML_SUFFIXES = (".yaml", ".yml")
ENV_MODULES = ("autonerves/test_mode.py", "autonerves/workspace.py",
               "autonerves/__init__.py")
ENV_RE = re.compile(r"\bPYAUTO_[A-Z0-9_]*[A-Z0-9]\b")


# --- locating the Brain and the Mind ------------------------------------------
_THEME_BRAIN: list = [None]


def _brain_home(brain: str | None = None) -> Path | None:
    for cand in (brain, os.environ.get("PYAUTO_BRAIN"),
                 NERVES_HOME / "PyAutoBrain", NERVES_HOME.parent / "PyAutoBrain"):
        if cand and (Path(cand) / "board" / "_theme.py").is_file():
            return Path(cand)
    return None


def theme():
    """The shared theme module (PyAutoBrain/board/_theme.py). Only the html
    surfaces need it; every other surface renders with no Brain in reach."""
    home = _brain_home(_THEME_BRAIN[0])
    board = home / "board" if home else None
    if board and (board / "_theme.py").is_file():
        if str(board) not in sys.path:
            sys.path.insert(0, str(board))
        import _theme
        return _theme
    raise RuntimeError(
        "the shared board theme (PyAutoBrain/board/_theme.py) is not in reach "
        "— check PyAutoBrain out beside this repo or pass --brain")


def _repos_yaml(root: Path, mind: str | None = None) -> Path | None:
    for cand in (mind, os.environ.get("PYAUTO_MIND") if not mind else None,
                 root / "PyAutoMind", root / "organs" / "PyAutoMind"):
        if cand and (Path(cand) / "repos.yaml").is_file():
            return Path(cand) / "repos.yaml"
    return None


def load_body_map(root: Path, mind: str | None = None) -> dict:
    """``{repo: {path, github, …}}`` from the Mind's body map, or ``{}``."""
    path = _repos_yaml(root, mind)
    if not path:
        return {}
    try:
        return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) \
            .get("repos") or {}
    except (OSError, yaml.YAMLError):
        return {}


# --- identity (derived, never hardcoded — tenant firewall) -------------------
def parse_owner_repo(url: str) -> tuple[str, str]:
    m = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    return (m.group(1), m.group(2)) if m else ("", "")


def _self_owner_repo() -> tuple[str, str]:
    slug = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" in slug:
        o, r = slug.split("/", 1)
        return o, r
    try:
        out = subprocess.run(["git", "-C", str(NERVES_HOME), "remote",
                              "get-url", "origin"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    return parse_owner_repo(out)


# --- pure YAML reading ----------------------------------------------------------
_KEY_RE = re.compile(
    r"^(?P<ind>[ ]*)(?P<dash>-[ ]+)?"
    r"(?P<key>\"[^\"]*\"|'[^']*'|[^\s#:\-\[\]{}][^#:]*?|[\-][^\s#:][^#:]*?)"
    r"[ \t]*:(?=[ \t]|$)(?P<rest>.*)$")
_BLOCK_SCALAR = re.compile(r"^\s*[|>][+-]?\d*\s*(#.*)?$")


def _strip_quotes(text: str) -> str:
    return re.sub(r"\"[^\"]*\"|'[^']*'", lambda m: " " * len(m.group(0)), text)


def split_comment(line: str) -> tuple[str, str]:
    """``(code, comment)`` for one YAML line — the comment is the text after a
    ``#`` that starts the line or follows whitespace, outside quotes."""
    masked = _strip_quotes(line)
    m = re.search(r"(^|\s)#", masked)
    if not m:
        return line, ""
    at = m.start() + len(m.group(1))
    return line[:at], line[at:]


def _clip(text: object, limit: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _comment_text(raw: str) -> str:
    return raw.lstrip("#").strip()


def scan_keys(text: str) -> list[dict]:
    """Every mapping key in a YAML file, in order: ``{k, line, c}`` — ``k`` the
    dotted path, ``c`` the key's comment (its inline ``#``, else the comment
    lines directly above it). A line scan, so it keeps what ``safe_load``
    throws away: line numbers and comments."""
    out = []
    stack: list[tuple[int, str]] = []
    above: list[str] = []
    block_indent = None
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r")
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if block_indent is not None:
            if stripped and indent <= block_indent:
                block_indent = None
            else:
                continue
        if not stripped:
            above = []
            continue
        if stripped.startswith("#"):
            above.append(_comment_text(stripped))
            continue
        if stripped in ("---", "..."):
            above = []
            continue
        m = _KEY_RE.match(line)
        if not m:
            above = []
            continue
        key = m.group("key").strip().strip("\"'")
        ind = len(m.group("ind")) + len(m.group("dash") or "")
        while stack and stack[-1][0] >= ind:
            stack.pop()
        path = ".".join([k for _, k in stack] + [key])
        code, comment = split_comment(m.group("rest"))
        c = _comment_text(comment) if comment else " ".join(above[-3:])
        out.append({"k": path, "line": n, "c": _clip(c, 200)})
        stack.append((ind, key))
        above = []
        if _BLOCK_SCALAR.match(m.group("rest") or ""):
            block_indent = ind
    return out


def flatten(data, prefix: str = "", depth: int | None = None) -> dict:
    """Leaf key paths of parsed YAML, keys lowercased as autonerves reads
    them; an empty mapping is itself a leaf. ``depth`` stops the descent
    (a prior file compares whole ``Class.param`` specs, depth 2)."""
    out = {}
    stop = depth is not None and prefix.count(".") + 1 >= depth and prefix
    if isinstance(data, dict) and data and not stop:
        for k, v in data.items():
            path = f"{prefix}.{str(k).lower()}" if prefix else str(k).lower()
            out.update(flatten(v, path, depth))
    elif prefix:
        out[prefix] = data
    return out


def _is_prior_param(v) -> bool:
    return isinstance(v, dict) and "type" in v


def is_prior_file(rel: str, data) -> bool:
    """Under ``priors/``, or every top-level value a mapping of
    ``param → {type, …}``."""
    if rel.startswith("priors/") or "/priors/" in rel:
        return True
    if not isinstance(data, dict) or not data:
        return False
    return all(isinstance(v, dict) and v and all(_is_prior_param(p)
                                                  for p in v.values())
               for v in data.values())


def _num(v) -> str:
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(v)
    return str(v)


def prior_rows(data, keys: list[dict]) -> list[dict]:
    """One row per ``Class.param`` of a prior file."""
    lines = {k["k"]: k["line"] for k in keys}
    rows = []
    if not isinstance(data, dict):
        return rows
    for cls, params in data.items():
        if not isinstance(params, dict):
            continue
        for param, spec in params.items():
            if not isinstance(spec, dict):
                continue
            kind = str(spec.get("type", ""))
            if "mean" in spec or "sigma" in spec:
                a = f"mean {_num(spec.get('mean'))}"
                b = f"σ {_num(spec.get('sigma'))}"
            elif "lower_limit" in spec or "upper_limit" in spec:
                a = f"lower {_num(spec.get('lower_limit'))}"
                b = f"upper {_num(spec.get('upper_limit'))}"
            elif "value" in spec:
                a, b = f"value {_num(spec.get('value'))}", ""
            else:
                a, b = "", ""
            wm = spec.get("width_modifier")
            width = (f"{wm.get('type', '')} {_num(wm.get('value'))}".strip()
                     if isinstance(wm, dict) else ("" if wm is None else str(wm)))
            lim = spec.get("limits")
            limits = (f"[{_num(lim.get('lower'))}, {_num(lim.get('upper'))}]"
                      if isinstance(lim, dict) else "")
            rows.append({"cls": str(cls), "param": str(param), "type": kind,
                         "a": a, "b": b, "width": width, "limits": limits,
                         "line": lines.get(f"{cls}.{param}", 0)})
    return rows


def read_file(repo: str, rel: str, text: str, kind: str) -> dict:
    """One file's record — pure over its text."""
    rec = {"repo": repo, "path": rel, "lines": len(text.splitlines()),
           "text": text, "error": None, "top_keys": [], "keys": [],
           "prior": False, "priors": [],
           "tooling": kind == "workspace" and rel.split("/", 1)[0] in TOOLING_DIRS}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        rec["error"] = _clip(str(e).splitlines()[0] if str(e) else repr(e), 200)
        mark = getattr(e, "problem_mark", None)
        rec["error_line"] = (mark.line + 1) if mark is not None else 1
        data = None
    rec["keys"] = scan_keys(text)
    if isinstance(data, dict):
        rec["top_keys"] = [str(k) for k in data]
    elif rec["error"] is None:
        rec["top_keys"] = [k["k"] for k in rec["keys"] if "." not in k["k"]]
    if rec["error"] is None and not rec["tooling"] and is_prior_file(rel, data):
        rec["prior"] = True
        rec["priors"] = prior_rows(data, rec["keys"])
    rec["_data"] = data
    return rec


# Keys autonerves itself reads from a workspace's config — the version
# handshake (``autonerves/workspace.py``, ``autonerves/__init__.py``). The
# Nerves ship no YAML, so no library defines them: owned, never orphans.
NERVES_OWNED = {
    "general.yaml": ("version.workspace_version",
                     "version.minimum_library_version",
                     "version.workspace_version_check",
                     "version.python_version_check"),
}

_MISSING = object()


def _section(data, parts: list[str]):
    """``data`` descended through ``parts`` (keys lowercased, as autonerves
    reads them), else ``_MISSING``."""
    for p in parts:
        if not isinstance(data, dict):
            return _MISSING
        data = next((v for k, v in data.items() if str(k).lower() == p),
                    _MISSING)
        if data is _MISSING:
            return _MISSING
    return data


def resolve_in_library(lib_files: dict, rel: str) -> dict | None:
    """What one library gives for a workspace file ``rel`` (``lib_files`` is
    ``{path: record}``), resolved the way autonerves' ``RecursiveConfig``
    walks a config dir: at each level of the path a ``<name>.yml``/``.yaml``
    file shadows a ``<name>/`` directory, and a file's top-level sections are
    the files of the directory form. So a library ``visualize.yaml``'s
    ``general:`` section is the workspace's ``visualize/general.yaml``, and a
    library ``visualize/`` directory is a workspace ``visualize.yaml``.
    Returns a stack record (``repo``, ``path`` — the library file, with a
    ``§ section`` when one — and ``_data``), or ``None``."""
    # file stems reach autonerves lowercased (DictWrapper): match them so
    files = {p.lower(): r for p, r in lib_files.items()}
    parts = rel.rsplit(".", 1)[0].lower().split("/")
    for i in range(len(parts)):
        stem = "/".join(parts[:i + 1])
        rec = next((files[stem + s] for s in YAML_SUFFIXES[::-1]
                    if stem + s in files), None)
        if rec is not None:
            if rec["error"]:
                return None
            data = _section(rec.get("_data"), parts[i + 1:])
            if data is _MISSING:
                return None
            where = rec["path"] + (f" § {'.'.join(parts[i + 1:])}"
                                   if parts[i + 1:] else "")
            return {"repo": rec["repo"], "path": where, "_data": data}
        if not any(p.startswith(stem + "/") for p in files):
            return None
    # the whole path is a library directory: each entry is a section
    children = set()
    for p in files:
        if p.startswith(stem + "/"):
            head, _, tail = p[len(stem) + 1:].partition("/")
            children.add(head if tail else head.rsplit(".", 1)[0])
    data = {}
    for child in sorted(children - {"priors"}):  # RecursiveConfig skips priors
        sub = resolve_in_library(lib_files, f"{stem}/{child}.yaml")
        if sub is not None:
            data[child] = sub["_data"]
    if not data:
        return None
    repo = next(iter(files.values()))["repo"]
    return {"repo": repo, "path": stem + "/", "_data": data}


def diff_against_stack(ws: dict, stack_recs: list[dict]) -> dict:
    """The override record for one workspace file against its library stack
    (``stack_recs`` in autonerves lookup order, only the libraries that hold
    it — by relative path, or through ``resolve_in_library``)."""
    # A prior file is compared per parameter: a workspace that swaps a
    # Gaussian for a Uniform changes one spec, it does not orphan the new
    # prior's lower/upper fields.
    depth = 2 if ws.get("prior") else None
    wflat = flatten(ws.get("_data"), depth=depth)
    merged: dict = {}
    for rec in stack_recs:
        for k, v in flatten(rec.get("_data"), depth=depth).items():
            merged.setdefault(k, v)
    # An empty mapping (`fit_imaging: {}`) sets nothing, so it can neither
    # differ nor be orphaned.
    wflat = {k: v for k, v in wflat.items() if v != {}}
    lines = {k["k"].lower(): k["line"] for k in ws.get("keys") or []}
    differs = sorted(k for k in wflat if k in merged and merged[k] != wflat[k])
    ws_only = sorted(k for k in wflat if k not in merged)
    owned = set(NERVES_OWNED.get(ws["path"], ()))
    nerves_owned = [k for k in ws_only if k in owned]
    ws_only = [k for k in ws_only if k not in owned]
    lib_only = sorted(k for k in merged if k not in wflat)
    first = stack_recs[0] if stack_recs else None
    return {"repo": ws["repo"], "path": ws["path"],
            "counterpart": first["repo"] if first else None,
            "counterpart_path": first["path"] if first else None,
            "stack": [r["repo"] for r in stack_recs],
            "differs": differs, "workspace_only": ws_only,
            "nerves_owned": nerves_owned,
            "library_only": lib_only[:50], "library_only_count": len(lib_only),
            "first_orphan_line": min((lines.get(k, 0) for k in ws_only),
                                     default=0) or 1}


def env_vars_from(texts: dict) -> list[dict]:
    """``PYAUTO_*`` names from module texts (``{relpath: text}``), each with
    the ``#`` comment lines directly above a mention, else the first
    paragraph of the enclosing function's docstring — the first mention that
    has either."""
    seen: dict = {}
    for rel, text in texts.items():
        lines = text.splitlines()
        funcs = []
        try:
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node) or ""
                    funcs.append((node.lineno, getattr(node, "end_lineno",
                                                       node.lineno), doc))
        except SyntaxError:
            pass
        for n, line in enumerate(lines, start=1):
            for name in ENV_RE.findall(line):
                if name in seen:
                    if rel not in seen[name]["modules"]:
                        seen[name]["modules"].append(rel)
                    if seen[name]["comment"]:
                        continue
                comment = ""
                j = n - 2
                block = []
                while j >= 0 and lines[j].strip().startswith("#"):
                    block.insert(0, lines[j].strip().lstrip("#").strip())
                    j -= 1
                if block:
                    comment = " ".join(block)
                else:
                    inner = [f for f in funcs if f[0] <= n <= f[1] and f[2]]
                    if inner:
                        doc = min(inner, key=lambda f: f[1] - f[0])[2]
                        para = doc.strip().split("\n\n")[0]
                        comment = " ".join(para.split())
                if name in seen:
                    # the first mention had nothing to say (e.g. a bare
                    # constant); a later one inside a documented function does
                    seen[name]["comment"] = _clip(comment, 220)
                    continue
                seen[name] = {"name": name, "modules": [rel], "line": n,
                              "comment": _clip(comment, 220)}
    return sorted(seen.values(), key=lambda e: e["name"])


# --- config lookups in library code (the "not in use anymore" scan) ------------
# autonerves resolves ``conf.instance["general"]["output"]["remove_files"]`` as
# file ``general.yaml`` → section ``output`` → key ``remove_files`` (keys
# lowercased, every layer merged), so a read is a dotted path whose head is the
# config file's path without its suffix.
#
# Helpers that take a key name and read it under a fixed prefix — a literal
# first argument is a read of ``prefix.<arg>``.
HELPER_READS = {"should_output": ("output",)}
# Files autonerves loads whole outside the subscript API
# (``Config.logging_config`` opens ``logging.yaml`` and hands it to
# ``logging.config.dictConfig``) — every key in them is a section read.
WHOLESALE_FILES = ("logging",)
# Config lookups are only counted once every library's package was scanned:
# a partial scan would call every key of the missing library unused.
LOOKUP_CLASSES = ("used", "section-read", "unused")
# autonerves' own package — a config reader too (swappable for tests).
NERVES_PACKAGE = NERVES_HOME / "autonerves"


def _is_conf_root(node, names: set) -> bool:
    """``conf.instance`` / ``x.conf.instance`` / ``conf.instance.dict`` /
    a bare ``instance`` imported from ``autonerves.conf``."""
    if isinstance(node, ast.Attribute) and node.attr == "dict":
        node = node.value
    if isinstance(node, ast.Attribute) and node.attr == "instance":
        v = node.value
        return (isinstance(v, ast.Name) and v.id == "conf") or \
            (isinstance(v, ast.Attribute) and v.attr == "conf")
    return isinstance(node, ast.Name) and node.id in names


def _literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower()
    return None


def _chain(node, roots: set, local: dict):
    """Unwind a subscript chain ending at ``node``: ``(keys, open)`` where
    ``keys`` is the literal path from the root and ``open`` says a non-literal
    subscript ended it (everything under ``keys`` may be read); ``None`` when
    the chain does not start at the config."""
    steps = []
    cur = node
    while True:
        if isinstance(cur, ast.Subscript):
            steps.append(cur.slice)
            cur = cur.value
        elif (isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute)
              and cur.func.attr == "get" and cur.args):
            steps.append(cur.args[0])
            cur = cur.func.value
        else:
            break
    if isinstance(cur, ast.Name) and cur.id in local:
        base = list(local[cur.id])
    elif _is_conf_root(cur, roots):
        base = []
    else:
        return None
    keys = base
    for s in reversed(steps):
        lit = _literal(s)
        if lit is None:
            return keys, True
        keys.append(lit)
    return keys, False


def _parents(tree) -> dict:
    par = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            par[child] = node
    return par


def _is_chain_link(node, parent) -> bool:
    """Whether ``node`` continues into ``parent`` as part of a longer chain."""
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return True
    if isinstance(parent, ast.Attribute) and parent.value is node and \
            parent.attr in ("get", "dict"):
        return True
    return False


def lookups_from(text: str) -> tuple[set, set, set]:
    """``(reads, wildcards, bound)`` — the dotted config paths one module
    reads.

    * a literal subscript chain from ``conf.instance`` (``.get("k")`` counts as
      a subscript) is a read of that path;
    * a non-literal subscript ends the chain: its literal prefix becomes a
      wildcard (anything under it may be read);
    * ``x = conf.instance["a"]["b"]`` then ``x["c"]`` is followed within the
      function (or module, or a closure inside it) — the assignment reads
      ``a.b`` itself (``bound``: the path is used, its keys are not covered),
      each use of ``x`` reads ``a.b`` + its own literal keys, and a bare use
      of ``x`` (passed on, iterated) reads ``a.b`` whole;
    * ``HELPER_READS`` calls with a literal first argument.

    Raises ``SyntaxError`` on a module that does not parse."""
    tree = ast.parse(text)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                "conf") and (node.module or "").split(".")[0] in (
                "autonerves", "autoconf"):
            roots |= {a.asname or a.name for a in node.names
                      if a.name == "instance"}
    par = _parents(tree)
    reads, wild, bound = set(), set(), set()
    funcs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
    scopes = [n for n in ast.walk(tree) if isinstance(n, funcs)]
    scopes.append(tree)
    owned_by, own_locals, skip = {}, {}, set()
    for scope in scopes:
        # nodes owned by this scope (a nested def is its own scope)
        owned, stack = [], list(ast.iter_child_nodes(scope))
        while stack:
            n = stack.pop()
            if isinstance(n, funcs):
                continue
            owned.append(n)
            stack.extend(ast.iter_child_nodes(n))
        owned_by[scope] = owned
        local: dict = {}
        for n in owned:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and \
                    isinstance(n.targets[0], ast.Name):
                got = _chain(n.value, roots, {})
                if got and not got[1] and got[0]:
                    local[n.targets[0].id] = tuple(got[0])
                    bound.add(".".join(got[0]))
                    skip.add(n.value)
        own_locals[scope] = local
    seen: set = set()
    for scope in scopes:
        # a closure sees the names bound in the functions around it
        local, up = {}, par.get(scope)
        chain_up = []
        while up is not None:
            if isinstance(up, funcs):
                chain_up.append(up)
            up = par.get(up)
        for outer in reversed(chain_up):
            local.update(own_locals[outer])
        local.update(own_locals[scope])
        owned = owned_by[scope]
        for n in owned:
            if n in seen or n in skip:
                continue
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and \
                    n.func.id in HELPER_READS and n.args:
                lit = _literal(n.args[0])
                if lit is not None:
                    reads.add(".".join(HELPER_READS[n.func.id] + (lit,)))
                continue
            if not isinstance(n, (ast.Subscript, ast.Call, ast.Name,
                                  ast.Attribute)):
                continue
            if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
                continue
            if _is_chain_link(n, par.get(n)):
                continue
            got = _chain(n, roots, local)
            if not got:
                continue
            seen.add(n)
            keys, open_ = got
            if not keys:
                if open_:
                    wild.add("")
                continue
            (wild if open_ else reads).add(".".join(keys))
    return reads, wild, bound


def scan_lookups(pkg_dir) -> dict:
    """Every config read in a package's ``.py`` files:
    ``{files, reads, wildcards, bound, unparsed}`` (sorted lists)."""
    reads, wild, bound, unparsed, n = set(), set(), set(), [], 0
    for path in sorted(Path(pkg_dir).rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            unparsed.append(path.relative_to(pkg_dir).as_posix())
            continue
        if "instance" not in text and not any(h in text for h in HELPER_READS):
            n += 1
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # old escapes in library code
                r, w, b = lookups_from(text)
        except SyntaxError:
            unparsed.append(path.relative_to(pkg_dir).as_posix())
            continue
        n += 1
        reads |= r
        wild |= w
        bound |= b
    return {"files": n, "reads": sorted(reads), "wildcards": sorted(wild),
            "bound": sorted(bound), "unparsed": unparsed}


def config_path(rel: str, key: str) -> str:
    """A file key's full lookup path: ``visualize/general.yaml`` +
    ``general.backend`` → ``visualize.general.general.backend``."""
    stem = rel.rsplit(".", 1)[0].replace("/", ".")
    return f"{stem}.{key}".lower() if key else stem.lower()


def classify_key(path: str, reads: set, prefixes: set) -> str:
    """``used`` — the path is read (``reads``: exact reads, wildcard stems and
    bound locals), or a read runs through it (a section one of whose keys is
    read); ``section-read`` — a whole-section read or wildcard (``prefixes``)
    covers one of its ancestors; ``unused`` — nothing references it."""
    if path in reads:
        return "used"
    dotted = path + "."
    if any(r.startswith(dotted) for r in reads):
        return "used"
    parts = path.split(".")
    for i in range(len(parts) - 1, -1, -1):
        if ".".join(parts[:i]) in prefixes:
            return "section-read"
    return "unused"


def classify_files(files: list[dict], lookups: dict) -> None:
    """Stamp ``use`` on every key of every library settings file (in place),
    against the reads of the whole stack — autonerves merges every layer, so a
    PyAutoFit key read by PyAutoLens is used."""
    reads, prefixes = set(), set(WHOLESALE_FILES)
    for lk in lookups.values():
        # a chain that stops on a section reads it whole; a wildcard stem
        # ``a.b[x]`` reads ``a.b`` and may read anything under it
        prefixes |= set(lk["reads"]) | set(lk["wildcards"])
        reads |= set(lk["reads"]) | set(lk["wildcards"]) | \
            set(lk.get("bound") or ())
    for f in files:
        if not f.get("_library") or f.get("prior") or f.get("error") or \
                f.get("tooling"):
            continue
        counts = dict.fromkeys(LOOKUP_CLASSES, 0)
        for k in f["keys"]:
            k["use"] = classify_key(config_path(f["path"], k["k"]), reads,
                                    prefixes)
            counts[k["use"]] += 1
        f["use_counts"] = counts


# --- collection (the only I/O) --------------------------------------------------
def _source_dir(src: dict, root: Path, body: dict, sources: Path | None,
                field: str = "config") -> Path | None:
    sub = src.get(field)
    if not sub:
        return None
    if sources is not None:
        cand = sources / src["repo"] / sub
        return cand if cand.is_dir() else None
    rel = (body.get(src["repo"]) or {}).get("path")
    for base in ((root / rel) if rel else None, root / src["repo"]):
        if base and (base / sub).is_dir():
            return base / sub
    return None


def _now_z() -> str:
    return datetime.datetime.now(datetime.timezone.utc) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")


def build_snapshot(files: list[dict], sources: list[dict], env: list[dict],
                   errors: list[str], owner: str = "", repo: str = "",
                   generated: str | None = None,
                   lookups: dict | None = None) -> dict:
    """Assemble the snapshot from read file records (pure). ``lookups``
    (``{reader: scan_lookups(…)}``) classifies every library settings key —
    only when every library's package was scanned (``None`` skips it)."""
    if lookups is not None:
        classify_files(files, lookups)
    by_repo: dict = {}
    for f in files:
        by_repo.setdefault(f["repo"], {})[f["path"]] = f
    overrides = []
    for src in sources:
        if src["kind"] != "workspace":
            continue
        for rel, rec in sorted((by_repo.get(src["repo"]) or {}).items()):
            if rec["tooling"] or rec["error"]:
                continue
            if rec["prior"]:
                # priors resolve by class through JSONPriorConfig, not
                # RecursiveConfig: same relative path only
                stack = [by_repo[lib][rel] for lib in src.get("stack") or ()
                         if rel in (by_repo.get(lib) or {})
                         and not by_repo[lib][rel]["error"]]
            else:
                stack = [r for r in (resolve_in_library(by_repo[lib], rel)
                                     for lib in src.get("stack") or ()
                                     if by_repo.get(lib)) if r is not None]
            if not stack:
                continue
            overrides.append(diff_against_stack(rec, stack))
    for src in sources:
        recs = list((by_repo.get(src["repo"]) or {}).values())
        src["files"] = len(recs)
        src["lines"] = sum(r["lines"] for r in recs)
        src["keys"] = sum(len(r["keys"]) for r in recs)
        src["priors"] = sum(1 for r in recs if r["prior"])
        src["errors"] = sum(1 for r in recs if r["error"])
        if src["kind"] == "library" and lookups is not None:
            src["unused"] = sum((r.get("use_counts") or {}).get("unused", 0)
                                for r in recs)
    clean = [{k: v for k, v in f.items() if not k.startswith("_")}
             for f in files]
    return {"schema_version": SCHEMA_VERSION,
            "generated": generated or _now_z(),
            "owner": owner, "repo": repo or "PyAutoNerves",
            "sources": sources, "files": clean, "overrides": overrides,
            "env_vars": env, "errors": errors,
            "lookups": lookups or {}}


def collect(root=None, brain=None, mind=None, sources_dir=None,
            generated=None) -> dict:
    """Walk every ``SOURCES`` config dir and read the autonerves env vars."""
    root = Path(root or os.environ.get("PYAUTO_ROOT") or Path.cwd())
    sdir = Path(sources_dir) if sources_dir else None
    body = load_body_map(root, mind)
    owner, repo = _self_owner_repo()
    errors: list[str] = []
    if not body:
        errors.append("PyAutoMind/repos.yaml not found — GitHub links use this "
                      "repo's owner and local paths fall back to <root>/<Repo>")
    srcs, files = [], []
    scans: dict | None = {}
    for src in SOURCES:
        meta = body.get(src["repo"]) or {}
        github = meta.get("github") or (f"{owner}/{src['repo']}" if owner else "")
        entry = {"repo": src["repo"], "github": github, "config": src["config"],
                 "kind": src["kind"], "stack": list(src["stack"]),
                 "found": False}
        where = _source_dir(src, root, body, sdir)
        if where is None:
            errors.append(f"{src['repo']}: config dir {src['config']}/ not found")
            srcs.append(entry)
            if src["kind"] == "library":
                scans = None  # its code's reads are unknown: classify nothing
            continue
        entry["found"] = True
        for path in sorted(where.rglob("*")):
            if not path.is_file() or path.suffix not in YAML_SUFFIXES:
                continue
            rel = path.relative_to(where).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                errors.append(f"{src['repo']}/{rel}: unreadable ({e})")
                continue
            rec = read_file(src["repo"], rel, text, src["kind"])
            rec["_library"] = src["kind"] == "library"
            files.append(rec)
        srcs.append(entry)
        if src["kind"] == "library":
            pkg = _source_dir(src, root, body, sdir, "package")
            if pkg is None:
                errors.append(f"{src['repo']}: package {src.get('package')}/ "
                              "not found — config keys are not classified")
                scans = None
            elif scans is not None:
                scans[src["repo"]] = scan_lookups(pkg)
                for rel in scans[src["repo"]]["unparsed"]:
                    errors.append(f"{src['repo']}: {src['package']}/{rel} did "
                                  "not parse — its config reads are not counted")
    texts = {}
    for rel in ENV_MODULES:
        try:
            texts[rel] = (NERVES_HOME / rel).read_text(encoding="utf-8")
        except OSError as e:
            errors.append(f"env vars: {rel} unreadable ({e})")
    if scans is not None:
        # autonerves reads config itself (should_output, the backend) — its
        # own package is a reader too, whatever repo it runs from.
        scans["PyAutoNerves"] = scan_lookups(NERVES_PACKAGE)
    return build_snapshot(files, srcs, env_vars_from(texts), errors, owner,
                          repo, generated, scans)


# --- derived views ----------------------------------------------------------------
def repo_url(snap: dict) -> str:
    o, r = snap.get("owner"), snap.get("repo")
    return f"https://github.com/{o}/{r}" if o and r else ""


def pages_url(snap: dict) -> str:
    o, r = str(snap.get("owner") or "").lower(), snap.get("repo") or ""
    return f"https://{o}.github.io/{r}/" if o and r else ""


def _source(snap: dict, repo: str) -> dict:
    return next((s for s in snap.get("sources") or [] if s["repo"] == repo), {})


def file_url(snap: dict, repo: str, rel: str, line: int = 1) -> str:
    s = _source(snap, repo)
    if not s.get("github"):
        return ""
    return (f"https://github.com/{s['github']}/blob/main/{s['config']}/{rel}"
            f"#L{max(int(line or 1), 1)}")


def slug(rel: str) -> str:
    return "f-" + re.sub(r"[^A-Za-z0-9]+", "-", rel).strip("-").lower()


def unused_anchor(lib: str) -> str:
    return "unused-" + re.sub(r"[^A-Za-z0-9]+", "-", lib).strip("-").lower()


def orphans(snap: dict) -> list[dict]:
    return [o for o in snap.get("overrides") or [] if o["workspace_only"]]


def parse_errors(snap: dict) -> list[dict]:
    return [f for f in snap.get("files") or [] if f.get("error")]


def unused_keys(snap: dict) -> dict:
    """``{library: [{path, k, line}]}`` — the library settings keys no code
    in the stack reads, in source order (only classified snapshots)."""
    out: dict = {}
    for f in snap.get("files") or []:
        for k in f.get("keys") or []:
            if k.get("use") == "unused":
                out.setdefault(f["repo"], []).append(
                    {"path": f["path"], "k": k["k"], "line": k["line"]})
    return out


def status(snap: dict) -> str:
    if not snap.get("files"):
        return "grey"
    if parse_errors(snap) or orphans(snap):
        return "yellow"
    return "green"


def _summary(snap: dict) -> str:
    files = snap.get("files") or []
    found = sum(1 for s in snap.get("sources") or [] if s.get("found"))
    bits = [f"{len(files)} config files across {found} sources",
            f"{len(snap.get('overrides') or [])} workspace overrides"]
    if parse_errors(snap):
        bits.append(f"{len(parse_errors(snap))} unparseable")
    if orphans(snap):
        bits.append(f"{len(orphans(snap))} with orphan keys")
    unused = sum(len(v) for v in unused_keys(snap).values())
    if unused:
        bits.append(f"{unused} possibly unused library keys")
    return " · ".join(bits)


def _keys_text(keys: list[str], limit: int = 6) -> str:
    shown = ", ".join(keys[:limit])
    return shown + (f" (+{len(keys) - limit} more)" if len(keys) > limit else "")


UNUSED_CAVEAT = (
    "A static scan of every library's (and autonerves') Python for "
    "`conf.instance[...]` lookups: a key is *unused* when no literal lookup "
    "reads it, no read runs through it and no section read or non-literal "
    "subscript covers it. Keys read dynamically some other way are false "
    "positives — check before deleting.")


# --- markdown -----------------------------------------------------------------------
def _render_md(snap: dict) -> str:
    st = status(snap)
    out = [f"# PyAutoNerves Board — {st.upper()}", "",
           f"{_summary(snap)} · generated {snap.get('generated')}", "",
           "Read-only map of every config file and option across the library "
           "and workspace `config/` folders.", "",
           "| Source | Kind | Files | Lines | Keys | Prior files | Unparseable |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for s in snap.get("sources") or []:
        if not s.get("found"):
            out.append(f"| {s['repo']} | {s['kind']} | — | — | — | — | not found |")
            continue
        out.append(f"| {s['repo']} | {s['kind']} | {s['files']} | {s['lines']} "
                   f"| {s['keys']} | {s['priors']} | {s['errors']} |")
    ov = snap.get("overrides") or []
    if ov:
        out += ["", "## Workspace overrides", "",
                "| Workspace file | Counterpart | Differs | Workspace-only | "
                "Stack-only |", "|---|---|---:|---:|---:|"]
        for o in ov:
            out.append(f"| {o['repo']}/{o['path']} | {o['counterpart']} | "
                       f"{len(o['differs'])} | {len(o['workspace_only'])} | "
                       f"{o['library_only_count']} |")
    if orphans(snap):
        out += ["", "## Orphan keys (set by a workspace, defined by no library)", ""]
        for o in orphans(snap):
            out.append(f"- `{o['repo']}/{o['path']}`: "
                       f"{_keys_text(o['workspace_only'], 12)}")
    if parse_errors(snap):
        out += ["", "## Unparseable files", ""]
        for f in parse_errors(snap):
            out.append(f"- `{f['repo']}/{f['path']}`: {f['error']}")
    unused = unused_keys(snap)
    if unused:
        out += ["", "## Possibly unused config keys (no library code reads "
                "them)", "", UNUSED_CAVEAT, ""]
        for lib, keys in unused.items():
            out.append(f"- **{lib}** ({len(keys)}):")
            for u in keys:
                out.append(f"  - [`{u['path']}` `{u['k']}`]"
                           f"({file_url(snap, lib, u['path'], u['line'])})")
    if snap.get("env_vars"):
        out += ["", "## Environment variables", ""]
        for e in snap["env_vars"]:
            out.append(f"- `{e['name']}` ({', '.join(e['modules'])}) — "
                       f"{e['comment'] or '—'}")
    if snap.get("errors"):
        out += ["", "## Unavailable this render", ""]
        out += [f"- {e}" for e in snap["errors"]]
    return "\n".join(out) + "\n"


def _render_md_brief(snap: dict) -> str:
    st = status(snap)
    out = [f"**Nerves board: {st.upper()}** — {_summary(snap)}", ""]
    for s in snap.get("sources") or []:
        out.append(f"- {s['repo']}: " + (f"{s['files']} files"
                                         if s.get("found") else "not found"))
    items = to_state(snap)["items"]
    if items:
        out += ["", "Items:"]
        out += [f"- [{i['severity']}] {i['text']}" for i in items]
    return "\n".join(out) + "\n"


# --- badge / state ------------------------------------------------------------------
def badge_endpoint(snap: dict) -> dict:
    st = status(snap)
    if st == "grey":
        return {"schemaVersion": 1, "label": "nerves", "message": "unknown",
                "color": "lightgrey"}
    found = sum(1 for s in snap.get("sources") or [] if s.get("found"))
    return {"schemaVersion": 1, "label": "nerves",
            "message": f"{len(snap['files'])} files · {found} sources",
            "color": "yellow" if st == "yellow" else "green"}


def _iso_z(ts) -> str:
    try:
        t = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.timezone.utc)
    except (TypeError, ValueError):
        t = datetime.datetime.now(datetime.timezone.utc)
    return t.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_state(snap: dict) -> dict:
    """The organ-cockpit feed (contract v1, owned by PyAutoBrain
    ``board/_state.py``): one yellow item per unparseable file, then one per
    workspace file carrying orphan keys, then collection errors (info);
    one info item per library with possibly unused keys; capped at 20. Never
    red — the board is a map, not a gate."""
    st = status(snap)
    items = []
    for f in parse_errors(snap):
        items.append({"severity": "yellow",
                      "text": _clip(f"unparseable: {f['repo']}/{f['path']} — "
                                    f"{f['error']}"),
                      "url": file_url(snap, f["repo"], f["path"],
                                      f.get("error_line", 1)) or None,
                      "prompt": _clip(f"Fix the YAML parse error in "
                                      f"{f['repo']} {f['path']}: {f['error']}",
                                      400)})
    for o in orphans(snap):
        items.append({"severity": "yellow",
                      "text": _clip(f"orphan keys in {o['repo']}/{o['path']}: "
                                    f"{_keys_text(o['workspace_only'])} "
                                    f"(no library in the stack defines them)"),
                      "url": file_url(snap, o["repo"], o["path"],
                                      o.get("first_orphan_line", 1)) or None,
                      "prompt": _clip(
                          f"{o['repo']} config/{o['path']} sets keys no "
                          f"library defines ({', '.join(o['workspace_only'][:20])}"
                          f"); check {' → '.join(o['stack'])} and either add "
                          "them to the library config or drop them from the "
                          "workspace.", 600)})
    # Info only, never yellow: the scan is static and untrusted until a human
    # has reviewed its false positives.
    for lib, keys in unused_keys(snap).items():
        items.append({"severity": "info",
                      "text": _clip(f"{lib}: {len(keys)} possibly unused "
                                    f"config key{'s' if len(keys) != 1 else ''}"
                                    f" (no library code reads "
                                    f"{'them' if len(keys) != 1 else 'it'})"),
                      "url": (pages_url(snap) + "#" + unused_anchor(lib)
                              if pages_url(snap) else None),
                      "prompt": _clip(
                          f"Review the {lib} config keys the Nerves board "
                          f"flags as possibly unused ("
                          f"{', '.join(u['path'] + ':' + u['k'] for u in keys[:15])}"
                          "); confirm no code reads them, then remove them "
                          "from the library config and the workspaces.", 600)})
    items += [{"severity": "info", "text": _clip(f"unavailable this render: {e}"),
               "url": None, "prompt": None} for e in snap.get("errors") or []]
    if st == "grey":
        headline = "GREY — no config files could be collected"
    elif st == "green":
        headline = _summary(snap)
    else:
        headline = f"YELLOW — {_summary(snap)}"
    return {"schema_version": 1, "organ": BOARD_KEY,
            "repo": snap.get("repo") or "PyAutoNerves", "status": st,
            "headline": _clip(headline), "updated": _iso_z(snap.get("generated")),
            "pages_url": pages_url(snap) or "./",
            "items": items[:STATE_ITEMS_MAX]}


# --- html -----------------------------------------------------------------------------
def _esc(v) -> str:
    return _html.escape(str(v), quote=True)


_EXTRA_CSS = """
#q,#pfilter{width:100%;padding:.5rem .65rem;margin:.25rem 0 .6rem;
 background:var(--btn);color:var(--fg);border:1px solid var(--line);
 border-radius:8px;font:inherit;box-sizing:border-box}
#q:focus,#pfilter:focus{outline:none;border-color:var(--accent)}
ul.hits{list-style:none;padding:0;margin:0 0 1rem}
ul.hits li{padding:.35rem 0;border-bottom:1px solid var(--line)}
ul.hits .where{color:var(--muted);font-size:.85em}
table.grid{border-collapse:collapse;width:100%;font-size:.9em;display:block;
 overflow-x:auto}
table.grid th,table.grid td{text-align:left;padding:.3rem .45rem;
 border-bottom:1px solid var(--line);vertical-align:top}
table.grid td.n{text-align:right;font-variant-numeric:tabular-nums}
section.file{padding:.55rem 0;border-bottom:1px solid var(--line)}
section.file h3{margin:.1rem 0 .25rem;font-size:1rem}
.chips{display:flex;flex-wrap:wrap;gap:.3rem;margin:.3rem 0}
.chip{font:.8em ui-monospace,SFMono-Regular,Menlo,monospace;padding:.1rem .45rem;
 border:1px solid var(--line);border-radius:999px;background:var(--btn)}
.chip.y{border-color:var(--warn);color:var(--warn)}
.chip.r{border-color:var(--bad);color:var(--bad)}
.chip.u{border-style:dashed;color:var(--muted)}
pre.src{font:.8em/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
 background:var(--btn);border:1px solid var(--line);border-radius:8px;
 padding:.5rem 0;margin:.4rem 0}
pre.src span.l{display:block;padding:0 .6rem 0 0;white-space:pre}
pre.src span.l:target{background:rgba(127,127,127,.22)}
pre.src i{display:inline-block;min-width:3em;padding-right:.8em;
 text-align:right;color:var(--muted);font-style:normal;user-select:none}
pre.src em{color:var(--muted);font-style:normal}
tr:target{background:rgba(127,127,127,.22)}
.errline{color:var(--bad)}
"""

_INDEX_JS = """
var IDX=JSON.parse(document.getElementById('idx').textContent);
function flt(q){q=q.toLowerCase().trim();var out=document.getElementById('hits');
 var info=document.getElementById('hitinfo');
 if(q.length<2){out.innerHTML='';info.textContent='';return;}
 var terms=q.split(/\\s+/),hits=[],total=0;
 for(var i=0;i<IDX.k.length;i++){var e=IDX.k[i],f=IDX.f[e[0]],r=IDX.r[f[0]];
  var hay=(e[1]+' '+r+'/'+f[1]+' '+e[3]).toLowerCase(),ok=true;
  for(var t=0;t<terms.length;t++){if(hay.indexOf(terms[t])<0){ok=false;break;}}
  if(ok){total++;if(hits.length<80)hits.push(e);}}
 var esc=function(s){return String(s).replace(/[&<>"]/g,function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});};
 out.innerHTML=hits.map(function(e){var f=IDX.f[e[0]],r=IDX.r[f[0]];
  var href='repos/'+encodeURIComponent(r)+'.html#'+f[2]+(e[2]?'-L'+e[2]:'');
  return '<li><a href="'+href+'"><code>'+esc(e[1]||f[1])+'</code></a> '+
   (e[4]==='unused'?'<span class="chip u">unused</span> ':'')+
   '<span class="where">'+esc(r+'/'+f[1])+(e[2]?':'+e[2]:'')+'</span>'+
   (e[3]?'<br><span class="muted">'+esc(e[3])+'</span>':'')+'</li>';}).join('');
 info.textContent=total?(total+' match'+(total==1?'':'es')+
  (total>hits.length?' — first '+hits.length+' shown':'')):'no matches';}
"""

_REPO_JS = """
function reveal(){var id=decodeURIComponent(location.hash.slice(1));
 if(!id)return;var el=document.getElementById(id);if(!el)return;
 for(var p=el.parentElement;p;p=p.parentElement){if(p.tagName==='DETAILS')p.open=true;}
 el.scrollIntoView({block:'center'});}
window.addEventListener('hashchange',reveal);
document.addEventListener('DOMContentLoaded',reveal);
function flt(q){q=q.toLowerCase().trim();
 var secs=document.querySelectorAll('section.file');
 for(var i=0;i<secs.length;i++){var s=secs[i];
  var hit=!q||s.getAttribute('data-k').indexOf(q)>=0;
  s.style.display=hit?'':'none';}
 var groups=document.querySelectorAll('div.group');
 for(var j=0;j<groups.length;j++){var g=groups[j];
  var any=g.querySelector('section.file:not([style*="none"])');
  g.style.display=any?'':'none';}}
"""


def _page(snap: dict, title: str, body: str, js: str, depth: int = 0) -> str:
    t_ = theme()
    owner = str(snap.get("owner") or "").lower()
    footer = ""
    if owner and hasattr(t_, "board_links"):
        footer = t_.boards_footer(t_.board_links(f"https://{owner}.github.io",
                                                 BOARD_KEY), BOARD_KEY)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{t_.css(BOARD_KEY)}{_EXTRA_CSS}</style>
</head>
<body>
{body}
{footer}
<footer>Rendered by <code>scripts/board.py</code> from the config folders of
every library and workspace · read-only · generated
{_esc(snap.get('generated') or '?')}.</footer>
<script>{t_.JS}{js}</script>
</body></html>
"""


def key_index(snap: dict) -> dict:
    """The compact search index the index page embeds: repos ``r``, files
    ``f = [repoIdx, path, anchor]`` and entries ``k = [fileIdx, key, line,
    comment, use]`` (a file itself is an entry with an empty key; ``use`` is
    the lookup class of a library settings key, else ``""``). Prior files
    index their ``Class`` and ``Class.param`` paths, not every leaf field."""
    repos = [s["repo"] for s in snap.get("sources") or []]
    ri = {r: i for i, r in enumerate(repos)}
    files, entries = [], []
    for f in snap.get("files") or []:
        if f["repo"] not in ri:
            ri[f["repo"]] = len(repos)
            repos.append(f["repo"])
        fi = len(files)
        files.append([ri[f["repo"]], f["path"], slug(f["path"])])
        entries.append([fi, "", 0, "", ""])
        for k in f.get("keys") or []:
            if f.get("prior") and k["k"].count(".") > 1:
                continue
            entries.append([fi, k["k"], k["line"], _clip(k["c"], 120),
                            k.get("use", "")])
    return {"r": repos, "f": files, "k": entries}


def _render_html_index(snap: dict) -> str:
    t_ = theme()
    st = status(snap)
    tone = {"green": "ok", "yellow": "warn", "grey": "muted"}[st]
    files = snap.get("files") or []
    ov = snap.get("overrides") or []
    found = [s for s in snap.get("sources") or [] if s.get("found")]
    rows = []
    for s in snap.get("sources") or []:
        name = (f"<a href=\"repos/{_esc(s['repo'])}.html\">{_esc(s['repo'])}</a>"
                if s.get("found") else _esc(s["repo"]))
        if not s.get("found"):
            rows.append(f"<tr><td>{name}</td><td>{_esc(s['kind'])}</td>"
                        "<td colspan='5' class='muted'>not collected</td></tr>")
            continue
        err = (f"<b class='warn'>{s['errors']}</b>" if s["errors"] else "0")
        rows.append(f"<tr><td>{name}<br><span class='muted'><code>"
                    f"{_esc(s['config'])}/</code></span></td>"
                    f"<td>{_esc(s['kind'])}</td><td class='n'>{s['files']}</td>"
                    f"<td class='n'>{s['lines']}</td><td class='n'>{s['keys']}</td>"
                    f"<td class='n'>{s['priors']}</td><td class='n'>{err}</td></tr>")
    overview = ("<h2 id='sources'>Sources</h2><table class='grid'><tr><th>Repo</th><th>Kind"
                "</th><th>Files</th><th>Lines</th><th>Keys</th><th>Priors</th>"
                "<th>Unparseable</th></tr>" + "".join(rows) + "</table>")
    ov_rows = []
    for o in ov:
        link = (f"repos/{_esc(o['repo'])}.html#{slug(o['path'])}")
        chips = "".join(f"<span class='chip r'>{_esc(k)}</span>"
                        for k in o["workspace_only"][:8])
        chips += "".join(f"<span class='chip y'>{_esc(k)}</span>"
                         for k in o["differs"][:8])
        more = len(o["workspace_only"]) + len(o["differs"]) - \
            min(len(o["workspace_only"]), 8) - min(len(o["differs"]), 8)
        if more > 0:
            chips += f"<span class='muted'>+{more} more</span>"
        ov_rows.append(
            f"<tr><td><a href=\"{link}\">{_esc(o['repo'])}/<wbr>"
            f"{_esc(o['path'])}</a><div class='chips'>{chips}</div></td>"
            f"<td>{_esc(o['counterpart'])}</td>"
            f"<td class='n'>{len(o['differs'])}</td>"
            f"<td class='n'>{len(o['workspace_only'])}</td>"
            f"<td class='n'>{o['library_only_count']}</td></tr>")
    override = ""
    if ov_rows:
        override = (
            "<h2 id='overrides'>Override map</h2><p class='muted'>Each workspace file against "
            "the same path across its library stack, in autonerves lookup "
            "order (e.g. lens → galaxy → array → fit). <span class='chip y'>"
            "differs</span> the workspace changes the value the libraries give; "
            "<span class='chip r'>orphan</span> no library defines the key; "
            "stack-only keys fall through to the libraries.</p>"
            "<details><summary>" + f"{len(ov_rows)} overriding files"
            "</summary><table class='grid'><tr><th>Workspace file</th><th>"
            "Counterpart</th><th>Differs</th><th>Orphan</th><th>Stack-only"
            "</th></tr>" + "".join(ov_rows) + "</table></details>")
    tooling = [f for f in files if f.get("tooling")]
    tool_html = ""
    if tooling:
        tool_html = ("<h2 id='tooling'>Workspace tooling</h2><p class='muted'>"
                     "<code>build/</code> lists for CI and the release build — "
                     "workspace-local, not library settings.</p><ul>" + "".join(
                         f"<li><a href=\"repos/{_esc(f['repo'])}.html#"
                         f"{slug(f['path'])}\">{_esc(f['repo'])}/{_esc(f['path'])}"
                         f"</a> <span class='muted'>{f['lines']} lines</span></li>"
                         for f in tooling) + "</ul>")
    env_html = ""
    if snap.get("env_vars"):
        env_html = ("<h2 id='environment'>Environment variables</h2><p class='muted'>The "
                    "<code>PYAUTO_*</code> switches autonerves reads — options "
                    "that live outside the YAML.</p><table class='grid'>"
                    "<tr><th>Variable</th><th>What it does</th></tr>" + "".join(
                        f"<tr><td><code>{_esc(e['name'])}</code><br><span "
                        f"class='muted'>{_esc(', '.join(e['modules']))}</span></td>"
                        f"<td>{_esc(e['comment'] or '—')}</td></tr>"
                        for e in snap["env_vars"]) + "</table>")
    problems = ""
    if parse_errors(snap):
        problems = ("<h2 id='problems'>Unparseable files</h2><ul>" + "".join(
            f"<li><a href=\"repos/{_esc(f['repo'])}.html#{slug(f['path'])}\">"
            f"{_esc(f['repo'])}/{_esc(f['path'])}</a> — <span class='errline'>"
            f"{_esc(f['error'])}</span></li>" for f in parse_errors(snap))
            + "</ul>")
    unused_html = ""
    unused = unused_keys(snap)
    if unused:
        groups = []
        for lib, keys in unused.items():
            lis = "".join(
                f"<li><a href=\"repos/{_esc(lib)}.html#{slug(u['path'])}-L"
                f"{u['line']}\"><code>{_esc(u['k'])}</code></a> <span "
                f"class='where'>{_esc(u['path'])}:{u['line']}</span>"
                + (f" · <a href=\"{_esc(file_url(snap, lib, u['path'], u['line']))}"
                   f"\">GitHub</a>" if file_url(snap, lib, u['path']) else "")
                + "</li>" for u in keys)
            groups.append(f"<details id='{unused_anchor(lib)}'><summary><b>"
                          f"{_esc(lib)}</b> — {len(keys)} key"
                          f"{'s' if len(keys) != 1 else ''}</summary>"
                          f"<ul class='hits'>{lis}</ul></details>")
        unused_html = ("<h2 id='unused'>Possibly unused config keys</h2><p class='muted'>"
                       + _esc(UNUSED_CAVEAT).replace("`", "").replace("*", "") + "</p>"
                       + "".join(groups))
    elif snap.get("lookups"):
        unused_html = ("<h2 id='unused'>Possibly unused config keys</h2><p class='muted'>"
                       "none — every library settings key is read.</p>")
    errors = ""
    if snap.get("errors"):
        errors = ("<div class='errors'><p class='muted'>unavailable this "
                  "render:</p><ul class='muted'>" + "".join(
                      f"<li>{_esc(e)}</li>" for e in snap["errors"]) +
                  "</ul></div>")
    idx = json.dumps(key_index(snap), separators=(",", ":"),
                     ensure_ascii=False).replace("</", "<\\/")
    gh = repo_url(snap)
    gh_link = f' · <a href="{gh}">GitHub</a>' if gh else ""
    lede = ("Every config file and option across the libraries and the "
            "workspaces that override them — read-only.")
    navigation = [
        {"href": "#search", "label": "Search keys", "count": sum(len(f["keys"]) for f in files)},
        {"href": "#sources", "label": "Sources", "count": len(found)},
    ]
    for target, label, count, present in [
        ("overrides", "Overrides", len(ov), override),
        ("unused", "Possibly unused", sum(len(v) for v in unused.values()), unused_html),
        ("problems", "Unparseable files", len(parse_errors(snap)), problems),
        ("environment", "Environment variables", len(snap.get("env_vars") or []), env_html),
        ("tooling", "Workspace tooling", len(tooling), tool_html),
    ]:
        if present:
            navigation.append({"href": "#" + target, "label": label, "count": count})
    work_links = []
    if repo_url(snap):
        work_links.append({"label": "Nerves repository", "href": repo_url(snap)})
    for source in snap.get("sources") or []:
        work_slug = source.get("github") or ""
        if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", work_slug):
            work_links.append({"label": source["repo"] + " repository",
                               "href": "https://github.com/" + work_slug})
    panel = t_.orchestration_panel("nerves", "", "", CHECKIN_PROMPT,
                                   organ="nerves", work_links=work_links)
    body = f"""{t_.hero(BOARD_KEY, "Board", lede, navigation=navigation)}
{panel}
<p class="verdict {tone}"><b class="{tone}">{st.upper()}</b>
<span class="muted">{_esc(_summary(snap))}</span></p>
<p class="muted"><a href="dashboard.md">markdown version</a> ·
<a href="board.json">board.json</a>{gh_link}</p>
<h2 id='search'>Search</h2>
<input id="q" type="search" placeholder="search keys, files and comments…"
 oninput="flt(this.value)" autocomplete="off">
<p id="hitinfo" class="muted"></p>
<ul id="hits" class="hits"></ul>
{overview}
{override}
{unused_html}
{problems}
{env_html}
{tool_html}
{errors}
<script type="application/json" id="idx">{idx}</script>"""
    return _page(snap, "PyAutoNerves Board", body, _INDEX_JS)


def _source_html(f: dict, sid: str) -> str:
    err_line = f.get("error_line") if f.get("error") else None
    unused = {k["line"] for k in f.get("keys") or [] if k.get("use") == "unused"}
    out = []
    for n, line in enumerate(f.get("text", "").splitlines(), start=1):
        code, comment = split_comment(line)
        cls = " errline" if n == err_line else ""
        inner = _esc(code) + (f"<em>{_esc(comment)}</em>" if comment else "")
        if n in unused:
            inner += " <span class='chip u'>unused</span>"
        out.append(f'<span class="l{cls}" id="{sid}-L{n}"><i>{n}</i>{inner}</span>')
    return '<pre class="src">' + "".join(out) + "</pre>"


def _prior_html(f: dict, sid: str) -> str:
    rows = "".join(
        f'<tr id="{sid}-L{r["line"]}"><td>{_esc(r["cls"])}</td>'
        f'<td><code>{_esc(r["param"])}</code></td><td>{_esc(r["type"])}</td>'
        f'<td>{_esc(r["a"])}</td><td>{_esc(r["b"])}</td>'
        f'<td>{_esc(r["width"])}</td><td>{_esc(r["limits"])}</td></tr>'
        for r in f.get("priors") or [])
    return ("<table class='grid'><tr><th>Class</th><th>Param</th><th>Type</th>"
            "<th>Centre / lower</th><th>Width / upper</th><th>Width modifier"
            "</th><th>Limits</th></tr>" + rows + "</table>")


def _file_html(snap: dict, f: dict, ov: dict | None) -> str:
    sid = slug(f["path"])
    url = file_url(snap, f["repo"], f["path"], 1)
    gh = f' · <a href="{_esc(url)}">GitHub</a>' if url else ""
    kind = ("tooling" if f.get("tooling") else "prior file" if f.get("prior")
            else "settings")
    meta = [f"{f['lines']} lines", f"{len(f['keys'])} keys", kind]
    uc = f.get("use_counts")
    if uc:
        meta.append(f"{uc['used']} used · {uc['section-read']} section-read · "
                    f"{uc['unused']} unused")
    notes = []
    if f.get("error"):
        notes.append(f"<p class='errline'>does not parse: {_esc(f['error'])}</p>")
    if ov:
        owned = (f" · {len(ov['nerves_owned'])} owned by autonerves"
                 if ov.get("nerves_owned") else "")
        notes.append(f"<p class='muted'>overrides <b>{_esc(ov['counterpart'])}"
                     f"</b>/{_esc(ov.get('counterpart_path') or f['path'])} "
                     f"(stack: {_esc(' → '.join(ov['stack']))}) · "
                     f"{len(ov['differs'])} differ · "
                     f"{len(ov['workspace_only'])} orphan{owned} · "
                     f"{ov['library_only_count']} from the stack</p>")
    chips = "".join(f"<span class='chip'>{_esc(k)}</span>"
                    for k in f.get("top_keys", [])[:40])
    if len(f.get("top_keys", [])) > 40:
        chips += f"<span class='muted'>+{len(f['top_keys']) - 40} more</span>"
    diff_chips = ""
    if ov and (ov["workspace_only"] or ov["differs"]):
        diff_chips = "<div class='chips'>" + "".join(
            f"<span class='chip r' title='orphan: no library defines it'>"
            f"{_esc(k)}</span>" for k in ov["workspace_only"]) + "".join(
            f"<span class='chip y' title='differs from the library value'>"
            f"{_esc(k)}</span>" for k in ov["differs"]) + "</div>"
    unused = [k for k in f.get("keys") or [] if k.get("use") == "unused"]
    if unused:
        diff_chips += "<div class='chips'>" + "".join(
            f"<a class='chip u' href='#{sid}-L{k['line']}' title='possibly "
            f"unused: no library code reads this key'>unused: {_esc(k['k'])}"
            f"</a>" for k in unused) + "</div>"
    if f.get("prior") and f.get("priors"):
        body = (f"<details><summary>priors ({len(f['priors'])} params)"
                f"</summary>{_prior_html(f, sid)}</details>")
    else:
        body = (f"<details><summary>source ({f['lines']} lines)</summary>"
                f"{_source_html(f, sid)}</details>")
    hay = " ".join([f["path"]] + [k["k"] for k in f.get("keys") or []
                                  if not (f.get("prior") and k["k"].count(".") > 1)])
    return (f"<section class='file' id='{sid}' data-k=\"{_esc(hay.lower())}\">"
            f"<h3>{_esc(f['path'])}</h3><p class='muted'>{' · '.join(meta)}{gh}"
            f"</p>{''.join(notes)}<div class='chips'>{chips}</div>{diff_chips}"
            f"{body}</section>")


def _render_html_repo(snap: dict, name: str) -> str:
    t_ = theme()
    src = _source(snap, name)
    if not src:
        raise ValueError(f"unknown source: {name!r}")
    files = [f for f in snap.get("files") or [] if f["repo"] == name]
    ovs = {o["path"]: o for o in snap.get("overrides") or []
           if o["repo"] == name}
    groups: dict = {}
    for f in files:
        d = f["path"].rsplit("/", 1)[0] if "/" in f["path"] else "(top level)"
        groups.setdefault(d, []).append(f)
    parts = []
    for d in sorted(groups, key=lambda g: (g != "(top level)", g)):
        parts.append(f"<div class='group' id='group-{slug(d)}'><h2>{_esc(d)} <span class='muted'>"
                     f"({len(groups[d])})</span></h2>" + "".join(
                         _file_html(snap, f, ovs.get(f["path"]))
                         for f in groups[d]) + "</div>")
    stack = (f" · looks up {_esc(' → '.join(src['stack']))}"
             if src.get("stack") else "")
    gh = (f' · <a href="https://github.com/{_esc(src["github"])}/tree/main/'
          f'{_esc(src["config"])}">GitHub</a>' if src.get("github") else "")
    lede = f"{_esc(name)} · <code>{_esc(src['config'])}/</code>"
    navigation = [{"href": "../index.html", "label": "All sources"},
                  {"href": "#pfilter", "label": "Filter files", "count": len(files)}]
    navigation.extend({"href": "#group-" + slug(d), "label": d, "count": len(groups[d])}
                      for d in sorted(groups, key=lambda g: (g != "(top level)", g)))
    body = f"""{t_.hero(BOARD_KEY, "Board", lede, navigation=navigation)}
<p class="muted"><a href="../index.html">← all sources</a> · {src.get('kind')} ·
{len(files)} files · {src.get('lines', 0)} lines{stack}{gh}</p>
<input id="pfilter" type="search" placeholder="filter {len(files)} files by path or key…"
 oninput="flt(this.value)" autocomplete="off">
{''.join(parts) or "<p class='muted'>No config files collected.</p>"}"""
    return _page(snap, f"{name} · PyAutoNerves Board", body, _REPO_JS)


# --- dispatch ---------------------------------------------------------------------
def render(snap: dict, fmt: str = "md", name: str | None = None) -> str:
    if fmt == "md":
        return _render_md(snap)
    if fmt == "md-brief":
        return _render_md_brief(snap)
    if fmt == "json":
        # Compact: the snapshot carries every file's text (~2 MB), and this
        # is a machine surface, not a page anyone reads by eye.
        return json.dumps({**snap, "pages_url": pages_url(snap)},
                          separators=(",", ":"), sort_keys=True)
    if fmt == "badge":
        return json.dumps(badge_endpoint(snap))
    if fmt == "state":
        return json.dumps(to_state(snap), indent=2)
    if fmt == "html-index":
        return _render_html_index(snap)
    if fmt == "html-repo":
        return _render_html_repo(snap, name or "")
    raise ValueError(f"unknown board fmt: {fmt!r}")


def write_site(snap: dict, out: str | Path) -> list[Path]:
    """Write the whole Pages site; returns the paths written."""
    out = Path(out)
    (out / "repos").mkdir(parents=True, exist_ok=True)
    written = []

    def put(rel, text):
        p = out / rel
        p.write_text(text, encoding="utf-8")
        written.append(p)

    put("index.html", render(snap, "html-index"))
    for s in snap.get("sources") or []:
        if s.get("found"):
            put(f"repos/{s['repo']}.html", render(snap, "html-repo", s["repo"]))
    put("badge.json", render(snap, "badge") + "\n")
    put("board.json", render(snap, "json") + "\n")
    put("state.json", render(snap, "state") + "\n")
    put("dashboard.md", render(snap, "md"))
    return written


# --- CLI ----------------------------------------------------------------------------
FORMATS = ("md", "md-brief", "json", "badge", "state", "html-index")


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="board.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    for f in FORMATS:
        g.add_argument(f"--{f}", action="store_true")
    g.add_argument("--html-repo", metavar="NAME")
    g.add_argument("--site", metavar="DIR")
    ap.add_argument("--brain", help="PyAutoBrain checkout (theme)")
    ap.add_argument("--mind", help="PyAutoMind checkout (repos.yaml)")
    ap.add_argument("--root", help="workspace root holding the repos "
                                   "(default: $PYAUTO_ROOT or cwd)")
    ap.add_argument("--sources", metavar="DIR",
                    help="sparse clones laid out DIR/<Repo>/<config dir> "
                         "(overrides repos.yaml path resolution)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--collect", metavar="OUT")
    src.add_argument("--snapshot", metavar="F")
    ns = ap.parse_args(argv)
    _THEME_BRAIN[0] = ns.brain

    if ns.collect:
        snap = collect(ns.root, ns.brain, ns.mind, ns.sources)
        Path(ns.collect).write_text(json.dumps(snap, indent=2, sort_keys=True)
                                    + "\n", encoding="utf-8")
        print(f"collected → {ns.collect} ({_summary(snap)}; "
              f"{len(snap['errors'])} error(s))", file=sys.stderr)
        return 0
    snap = (json.loads(Path(ns.snapshot).read_text(encoding="utf-8"))
            if ns.snapshot else collect(ns.root, ns.brain, ns.mind, ns.sources))
    if ns.site:
        for p in write_site(snap, ns.site):
            print(f"{p} ({p.stat().st_size} bytes)", file=sys.stderr)
        return 0
    if ns.html_repo:
        print(render(snap, "html-repo", ns.html_repo))
        return 0
    fmt = next((f for f in FORMATS if getattr(ns, f.replace("-", "_"))), "md")
    print(render(snap, fmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
