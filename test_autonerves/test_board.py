"""Hermetic tests for ``scripts/board.py`` — the Nerves board.

A tmp workspace with two fake libraries and one workspace stands in for the
real config folders; ``SOURCES`` is swapped for it, a stub theme replaces the
Brain's, and every surface is rendered from the collected snapshot. No
network, no Brain checkout, no real repo names.
"""

import importlib.util
import json
import re
import types
from datetime import datetime
from pathlib import Path

import pytest

BOARD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "board.py"


def _load_board():
    spec = importlib.util.spec_from_file_location("nerves_board", BOARD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


board = _load_board()

FAKE_SOURCES = (
    {"repo": "LibLow", "config": "liblow/config", "kind": "library",
     "package": "liblow", "stack": ()},
    {"repo": "LibHigh", "config": "libhigh/config", "kind": "library",
     "package": "libhigh", "stack": ()},
    {"repo": "ws_demo", "config": "config", "kind": "workspace",
     "stack": ("LibHigh", "LibLow")},
)

REPOS_YAML = """\
repos:
  LibLow:
    path: libs/LibLow
    github: SomeOrg/LibLow
  LibHigh:
    path: libs/LibHigh
    github: SomeOrg/LibHigh
  ws_demo:
    path: workspaces/ws_demo
    github: SomeOrg/ws_demo
"""

LOW_GENERAL = """\
# The low library's general settings.
output:
  log_level: INFO          # Logging level for every search.
  remove_files: false
hpc:
  hpc_mode: false
"""

HIGH_GENERAL = """\
output:
  log_level: WARNING
"""

WS_GENERAL = """\
output:
  log_level: DEBUG         # Workspace default: verbose.
  # Kept for the tutorials.
  remove_files: false
legacy:
  stale_key: 3
"""

HIGH_PRIOR = """\
Blob:
  centre:
    type: Gaussian
    mean: 0.0
    sigma: 0.3
    width_modifier:
      type: Absolute
      value: 0.05
    limits:
      lower: -inf
      upper: inf
  size:
    type: Uniform
    lower_limit: 0.0
    upper_limit: 30.0
    width_modifier:
      type: Relative
      value: 1.0
"""

WS_PRIOR = """\
Blob:
  centre:
    type: Uniform
    lower_limit: -1.0
    upper_limit: 1.0
  size:
    type: Uniform
    lower_limit: 0.0
    upper_limit: 30.0
    width_modifier:
      type: Relative
      value: 1.0
"""

BROKEN = "key: [unclosed\nother: 1\n"

# The fake libraries' code — every config lookup style the scan follows.
# Style 1: a literal chain from ``conf.instance`` (split over lines), and a
# chain ended by a non-literal subscript (a wildcard over ``fit``).
LOW_CODE = """\
from autonerves import conf


def log_level():
    return conf.instance["general"]["output"][
        "log_level"
    ]


def fit_plot(name):
    return conf.instance["visualize"]["plots"]["fit"][name]
"""

# Style 2: a bare ``instance`` imported from autonerves.conf, read with
# ``.get`` — a cross-library read: ``hpc.hpc_mode`` lives only in LibLow.
# Style 3: a section bound to a local, then indexed (inside a closure).
HIGH_CODE = """\
from autonerves.conf import instance


def hpc_mode():
    return instance["general"]["hpc"].get("hpc_mode", False)


def dataset_image():
    plots = instance["visualize"]["plots"]

    def inner():
        return plots["dataset"]["image"]

    return inner()
"""

# autonerves' own reads (a helper with a literal name).
NERVES_CODE = """\
from autonerves.output import should_output


def go():
    return should_output("samples")
"""

LOW_PLOTS = """\
fit:
  subplot: true
  data: false
dataset:
  image: true
retired:
  old_flag: true   # nothing reads this any more
"""

FAKE_MODULE = '''\
import os


def is_quiet():
    """Return True if the demo output is silenced.

    More detail that is not the summary.
    """
    return os.environ.get("PYAUTO_DEMO_QUIET", "0") == "1"


# Upper bound on demo samples.
LIMIT = os.environ.get("PYAUTO_DEMO_LIMIT")
'''


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture(name="tree")
def make_tree(tmp_path):
    root = tmp_path / "root"
    _write(root / "PyAutoMind" / "repos.yaml", REPOS_YAML)
    low = root / "libs" / "LibLow" / "liblow" / "config"
    high = root / "libs" / "LibHigh" / "libhigh" / "config"
    ws = root / "workspaces" / "ws_demo" / "config"
    _write(low / "general.yaml", LOW_GENERAL)
    _write(high / "general.yaml", HIGH_GENERAL)
    _write(high / "priors" / "blob.yaml", HIGH_PRIOR)
    _write(high / "README.md", "not yaml, not collected\n")
    _write(ws / "general.yaml", WS_GENERAL)
    _write(ws / "priors" / "blob.yaml", WS_PRIOR)
    _write(ws / "broken.yaml", BROKEN)
    _write(ws / "build" / "no_run.yaml", "# skip list\n- slow_script\n")
    _write(root / "libs" / "LibLow" / "liblow" / "plot.py", LOW_CODE)
    _write(root / "libs" / "LibHigh" / "libhigh" / "util.py", HIGH_CODE)
    _write(root / "nerves_pkg" / "helper.py", NERVES_CODE)
    return root


FAKE_THEME = types.SimpleNamespace(
    css=lambda key: "body{}",
    hero=lambda key, kind, lede="": f"<header class='hero'>{kind} {lede}</header>",
    stats=lambda *pairs: "".join(f"<b>{n}</b>{label}" for n, label in pairs),
    board_links=lambda base, current=None: {"mind": f"{base}/Mind/"},
    boards_footer=lambda links, current: "<ul class='boards'></ul>",
    JS="/*js*/",
)


@pytest.fixture(name="snap")
def make_snap(tree, monkeypatch):
    monkeypatch.setattr(board, "SOURCES", FAKE_SOURCES)
    monkeypatch.setattr(board, "NERVES_PACKAGE", tree / "nerves_pkg")
    monkeypatch.setattr(board, "theme", lambda: FAKE_THEME)
    monkeypatch.setenv("GITHUB_REPOSITORY", "SomeOrg/PyAutoNerves")
    monkeypatch.delenv("PYAUTO_MIND", raising=False)
    return board.collect(root=tree, generated="2026-01-02T03:04:05Z")


def _file(snap, repo, path):
    return next(f for f in snap["files"] if f["repo"] == repo
                and f["path"] == path)


# The Brain's cockpit contract (board/_state.py), copied: required keys + enums.
def _state_ok(s):
    req = ("schema_version", "organ", "repo", "status", "headline", "updated",
           "pages_url", "items")
    assert all(k in s for k in req) and s["schema_version"] == 1
    assert s["status"] in ("green", "yellow", "red", "stale", "grey")
    assert "\n" not in s["headline"] and s["updated"].endswith("Z")
    datetime.fromisoformat(s["updated"].replace("Z", "+00:00"))
    assert all(i["severity"] in ("red", "yellow", "info") and i["text"]
               and "\n" not in i["text"] for i in s["items"])


def test_collect_reads_every_yaml_file_and_only_yaml(snap):
    paths = sorted((f["repo"], f["path"]) for f in snap["files"])
    assert paths == [("LibHigh", "general.yaml"), ("LibHigh", "priors/blob.yaml"),
                     ("LibLow", "general.yaml"), ("ws_demo", "broken.yaml"),
                     ("ws_demo", "build/no_run.yaml"),
                     ("ws_demo", "general.yaml"), ("ws_demo", "priors/blob.yaml")]
    counts = {s["repo"]: s["files"] for s in snap["sources"]}
    assert counts == {"LibLow": 1, "LibHigh": 2, "ws_demo": 4}
    assert all(s["found"] for s in snap["sources"])
    assert snap["errors"] == []


def test_keys_are_dotted_with_their_comments(snap):
    f = _file(snap, "LibLow", "general.yaml")
    assert f["top_keys"] == ["output", "hpc"]
    keys = {k["k"]: k for k in f["keys"]}
    assert set(keys) == {"output", "output.log_level", "output.remove_files",
                         "hpc", "hpc.hpc_mode"}
    # inline comment wins; a comment block directly above is the fallback
    assert keys["output.log_level"]["c"] == "Logging level for every search."
    assert keys["output"]["c"] == "The low library's general settings."
    assert keys["output.log_level"]["line"] == 3
    ws = {k["k"]: k["c"] for k in _file(snap, "ws_demo", "general.yaml")["keys"]}
    assert ws["output.remove_files"] == "Kept for the tutorials."


def test_a_broken_file_is_recorded_not_fatal(snap):
    f = _file(snap, "ws_demo", "broken.yaml")
    assert f["error"] and f["error_line"] >= 1
    assert f["text"] == BROKEN


def test_prior_files_render_as_rows(snap):
    f = _file(snap, "LibHigh", "priors/blob.yaml")
    assert f["prior"] is True
    rows = {r["param"]: r for r in f["priors"]}
    assert rows["centre"] == {"cls": "Blob", "param": "centre",
                              "type": "Gaussian", "a": "mean 0.0",
                              "b": "σ 0.3", "width": "Absolute 0.05",
                              "limits": "[-inf, inf]", "line": 2}
    assert rows["size"]["a"] == "lower 0.0" and rows["size"]["b"] == "upper 30.0"
    assert not _file(snap, "LibLow", "general.yaml")["prior"]


def test_prior_heuristic_without_a_priors_folder():
    data = {"Blob": {"size": {"type": "Uniform", "lower_limit": 0}}}
    assert board.is_prior_file("custom.yaml", data)
    assert not board.is_prior_file("general.yaml", {"output": {"log": 1}})


def test_override_diff_runs_across_the_whole_stack(snap):
    ov = {o["path"]: o for o in snap["overrides"]}
    g = ov["general.yaml"]
    # LibHigh is first in the stack, so it is the counterpart…
    assert g["counterpart"] == "LibHigh" and g["stack"] == ["LibHigh", "LibLow"]
    # …but remove_files lives only in LibLow and is NOT an orphan, and the
    # value compared for log_level is LibHigh's (it shadows LibLow's).
    assert g["differs"] == ["output.log_level"]
    assert g["workspace_only"] == ["legacy.stale_key"]
    assert g["library_only"] == ["hpc.hpc_mode"]
    # priors compare whole Class.param specs: a changed prior differs, it does
    # not orphan the new prior's fields
    p = ov["priors/blob.yaml"]
    assert p["differs"] == ["blob.centre"] and p["workspace_only"] == []
    # tooling and unparseable files are not overrides
    assert "build/no_run.yaml" not in ov and "broken.yaml" not in ov
    assert _file(snap, "ws_demo", "build/no_run.yaml")["tooling"] is True


def test_env_var_panel_from_module_text():
    env = {e["name"]: e for e in board.env_vars_from({"demo.py": FAKE_MODULE})}
    assert set(env) == {"PYAUTO_DEMO_QUIET", "PYAUTO_DEMO_LIMIT"}
    assert env["PYAUTO_DEMO_QUIET"]["comment"] == \
        "Return True if the demo output is silenced."
    assert env["PYAUTO_DEMO_LIMIT"]["comment"] == "Upper bound on demo samples."
    assert env["PYAUTO_DEMO_LIMIT"]["modules"] == ["demo.py"]


def test_state_is_yellow_per_broken_file_and_per_orphaned_file(snap):
    s = board.to_state(snap)
    _state_ok(s)
    assert s["status"] == "yellow" and s["organ"] == "nerves"
    texts = [i["text"] for i in s["items"]]
    assert any(t.startswith("unparseable: ws_demo/broken.yaml") for t in texts)
    orphan = next(i for i in s["items"] if "orphan keys" in i["text"])
    assert "legacy.stale_key" in orphan["text"]
    assert orphan["url"] == ("https://github.com/SomeOrg/ws_demo/blob/main/"
                             "config/general.yaml#L6")
    assert s["pages_url"] == "https://someorg.github.io/PyAutoNerves/"
    assert all(i["severity"] != "red" for i in s["items"])


def test_state_is_green_when_clean_and_grey_when_empty(snap):
    clean = dict(snap, files=[f for f in snap["files"] if not f["error"]],
                 overrides=[dict(o, workspace_only=[])
                            for o in snap["overrides"]])
    assert board.to_state(clean)["status"] == "green"
    empty = board.build_snapshot([], [], [], ["nothing"], "SomeOrg",
                                 "PyAutoNerves", "2026-01-02T03:04:05Z")
    s = board.to_state(empty)
    _state_ok(s)
    assert s["status"] == "grey"
    assert board.badge_endpoint(empty)["color"] == "lightgrey"


def test_state_items_are_capped(snap):
    many = dict(snap, errors=[f"e{i}" for i in range(40)])
    assert len(board.to_state(many)["items"]) == board.STATE_ITEMS_MAX


def test_badge(snap):
    b = board.badge_endpoint(snap)
    assert b == {"schemaVersion": 1, "label": "nerves",
                 "message": "7 files · 3 sources", "color": "yellow"}


def test_index_embeds_a_searchable_key_index(snap):
    page = board.render(snap, "html-index")
    raw = re.search(r'id="idx">(.*?)</script>', page, re.S).group(1)
    idx = json.loads(raw.replace("<\\/", "</"))
    entries = [(idx["r"][idx["f"][e[0]][0]], idx["f"][e[0]][1], e[1], e[2], e[3])
               for e in idx["k"]]
    assert ("LibLow", "general.yaml", "output.log_level", 3,
            "Logging level for every search.") in entries
    # a prior file indexes Class and Class.param, not every leaf field
    assert ("LibHigh", "priors/blob.yaml", "Blob.centre", 2, "") in entries
    assert not any(k == "Blob.centre.type" for _, _, k, _, _ in entries)
    assert "function flt(q)" in page and 'id="q"' in page
    assert "legacy.stale_key" in page  # the override map names the orphan
    assert "PYAUTO_" in page or snap["env_vars"] == []


def test_repo_page_shows_source_with_comments_and_links(snap):
    page = board.render(snap, "html-repo", "ws_demo")
    assert 'id="f-general-yaml-L2"' in page
    assert "<em># Workspace default: verbose.</em>" in page
    assert ("https://github.com/SomeOrg/ws_demo/blob/main/config/general.yaml"
            "#L1") in page
    assert "overrides <b>LibHigh</b>" in page
    assert "class='chip r'" in page  # orphan chip
    prior = board.render(snap, "html-repo", "LibHigh")
    assert "<td><code>centre</code></td><td>Gaussian</td>" in prior
    with pytest.raises(ValueError):
        board.render(snap, "html-repo", "nope")


def test_markdown_surfaces(snap):
    md = board.render(snap, "md")
    assert "| ws_demo | workspace | 4 |" in md
    assert "legacy.stale_key" in md
    brief = board.render(snap, "md-brief")
    assert brief.startswith("**Nerves board: YELLOW**")


def test_site_writes_every_surface(snap, tmp_path):
    out = tmp_path / "site"
    board.write_site(snap, out)
    for rel in ("index.html", "badge.json", "board.json", "state.json",
                "dashboard.md", "repos/LibLow.html", "repos/LibHigh.html",
                "repos/ws_demo.html"):
        assert (out / rel).is_file(), rel
    _state_ok(json.loads((out / "state.json").read_text()))
    assert json.loads((out / "board.json").read_text())["files"]


def test_sources_dir_overrides_body_map_resolution(tree, tmp_path, monkeypatch):
    monkeypatch.setattr(board, "SOURCES", FAKE_SOURCES)
    monkeypatch.setattr(board, "NERVES_PACKAGE", tree / "nerves_pkg")
    monkeypatch.setenv("GITHUB_REPOSITORY", "SomeOrg/PyAutoNerves")
    sources = tmp_path / "sources"
    _write(sources / "LibLow" / "liblow" / "config" / "general.yaml", LOW_GENERAL)
    snap = board.collect(root=tree, sources_dir=sources)
    assert [f["repo"] for f in snap["files"]] == ["LibLow"]
    assert any("LibHigh" in e for e in snap["errors"])
    # slugs still come from the body map
    assert snap["sources"][0]["github"] == "SomeOrg/LibLow"
    # a library whose code was not fetched: nothing is classified
    assert snap["lookups"] == {}
    assert not any("use" in k for f in snap["files"] for k in f["keys"])


# --- the "possibly unused keys" scan -------------------------------------------
def test_lookups_from_every_style():
    reads, wild, bound = board.lookups_from(LOW_CODE + HIGH_CODE)
    assert reads == {"general.output.log_level", "general.hpc.hpc_mode",
                     "visualize.plots.dataset.image"}
    assert wild == {"visualize.plots.fit"}
    # the local's own path is used, but its keys are not blanket-covered
    assert bound == {"visualize.plots"}
    r, w, b = board.lookups_from(
        "from autonerves import conf\n"
        "cfg = conf.instance['output']\n"
        "for k in cfg:\n    pass\n"
        "x = conf.instance[name]['a']\n"
        "conf.instance.output_path\n")
    # a bare use of a bound section reads it whole; a non-literal root is an
    # everything-wildcard; attribute access on the config is not a key read
    assert r == {"output"} and w == {""} and b == {"output"}
    r, _, _ = board.lookups_from(NERVES_CODE)
    assert r == {"output.samples"}


def test_classify_key():
    reads = {"general.output.log_level"}
    prefixes = {"visualize.plots.fit", "general.output.log_level"}
    assert board.classify_key("general.output.log_level", reads, prefixes) \
        == "used"
    assert board.classify_key("general.output", reads, prefixes) == "used"
    assert board.classify_key("visualize.plots.fit.subplot", reads,
                              prefixes) == "section-read"
    assert board.classify_key("general.hpc.hpc_mode", reads, prefixes) \
        == "unused"
    assert board.config_path("visualize/plots.yaml", "fit.data") == \
        "visualize.plots.fit.data"


@pytest.fixture(name="usnap")
def make_usnap(tree, monkeypatch):
    _write(tree / "libs" / "LibLow" / "liblow" / "config" / "visualize"
           / "plots.yaml", LOW_PLOTS)
    monkeypatch.setattr(board, "SOURCES", FAKE_SOURCES)
    monkeypatch.setattr(board, "NERVES_PACKAGE", tree / "nerves_pkg")
    monkeypatch.setattr(board, "theme", lambda: FAKE_THEME)
    monkeypatch.setenv("GITHUB_REPOSITORY", "SomeOrg/PyAutoNerves")
    monkeypatch.delenv("PYAUTO_MIND", raising=False)
    return board.collect(root=tree, generated="2026-01-02T03:04:05Z")


def _uses(snap, repo, path):
    return {k["k"]: k.get("use") for k in _file(snap, repo, path)["keys"]}


def test_library_keys_are_classified_across_the_stack(usnap):
    assert set(usnap["lookups"]) == {"LibLow", "LibHigh", "PyAutoNerves"}
    g = _uses(usnap, "LibLow", "general.yaml")
    assert g["output.log_level"] == "used"          # literal chain
    assert g["hpc.hpc_mode"] == "used"              # read by LibHigh's code
    assert g["output"] == "used" and g["hpc"] == "used"
    assert g["output.remove_files"] == "unused"
    p = _uses(usnap, "LibLow", "visualize/plots.yaml")
    assert p["fit"] == "used"                       # the wildcard's stem
    assert p["fit.subplot"] == p["fit.data"] == "section-read"
    assert p["dataset.image"] == "used"             # via the bound local
    assert p["dataset"] == "used"
    assert p["retired"] == p["retired.old_flag"] == "unused"
    assert _file(usnap, "LibLow", "visualize/plots.yaml")["use_counts"] == \
        {"used": 3, "section-read": 2, "unused": 2}
    # workspace files and priors are never classified
    for repo, path in (("ws_demo", "general.yaml"),
                       ("LibHigh", "priors/blob.yaml")):
        assert not any("use" in k for k in _file(usnap, repo, path)["keys"])
    assert board.unused_keys(usnap) == {"LibLow": [
        {"path": "general.yaml", "k": "output.remove_files", "line": 4},
        {"path": "visualize/plots.yaml", "k": "retired", "line": 6},
        {"path": "visualize/plots.yaml", "k": "retired.old_flag", "line": 7}]}


def test_unused_keys_render_on_every_surface(usnap):
    page = board.render(usnap, "html-index")
    assert "Possibly unused config keys" in page
    assert "id='unused-liblow'" in page
    assert ("https://github.com/SomeOrg/LibLow/blob/main/liblow/config/"
            "visualize/plots.yaml#L7") in page
    section = page.split("Possibly unused config keys", 1)[1] \
        .split('id="idx"', 1)[0]
    assert "retired.old_flag" in section
    assert "dataset.image" not in section and "hpc.hpc_mode" not in section
    raw = re.search(r'id="idx">(.*?)</script>', page, re.S).group(1)
    idx = json.loads(raw.replace("<\\/", "</"))
    use = {e[1]: e[4] for e in idx["k"] if idx["r"][idx["f"][e[0]][0]] == "LibLow"}
    assert use["retired.old_flag"] == "unused" and use["dataset.image"] == "used"
    repo = board.render(usnap, "html-repo", "LibLow")
    assert "unused: retired.old_flag" in repo
    assert "3 used · 2 section-read · 2 unused" in repo
    assert "<span class='chip u'>unused</span>" in repo
    md = board.render(usnap, "md")
    assert "## Possibly unused config keys" in md and "`retired.old_flag`" in md
    j = json.loads(board.render(usnap, "json"))
    assert any(k.get("use") == "unused" for f in j["files"] for k in f["keys"])


def test_unused_keys_are_one_info_item_per_library_never_yellow(usnap):
    s = board.to_state(usnap)
    _state_ok(s)
    items = [i for i in s["items"] if "possibly unused" in i["text"]]
    assert len(items) == 1 and items[0]["severity"] == "info"
    assert items[0]["text"].startswith("LibLow: 3 possibly unused config keys")
    assert items[0]["url"] == \
        "https://someorg.github.io/PyAutoNerves/#unused-liblow"
    # the unused keys alone never colour the board
    clean = dict(usnap, files=[f for f in usnap["files"] if not f["error"]],
                 overrides=[dict(o, workspace_only=[])
                            for o in usnap["overrides"]])
    assert board.to_state(clean)["status"] == "green"
