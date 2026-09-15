import logging
import re
import subprocess
import sys
import tomllib
import types
from pathlib import Path
from unittest import mock

import pytest

from autonerves import setup_colab


def _normalise(name):
    """
    PEP 503 name normalisation. ``_SHARED_EXTRAS`` spells one entry
    ``timeout_decorator`` while autofit declares ``timeout-decorator``; without
    this the two would compare as different packages and read as a drift.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _split_requirement(requirement):
    """
    Split a requirement string into ``(normalised name, specifier)``.

    The environment marker is dropped: autofit declares ``optax`` as
    ``optax>=0.2.5; sys_platform != "darwin" or ...`` and the Colab entry is
    deliberately unmarked, so comparing the raw strings would fail spuriously.
    Any extras bracket is dropped with it.
    """
    requirement = requirement.split(";", maxsplit=1)[0].strip()
    requirement = re.sub(r"\[[^\]]*\]", "", requirement)

    match = re.search(r"[<>=!~]", requirement)

    if match is None:
        return _normalise(requirement), ""

    return (
        _normalise(requirement[: match.start()]),
        requirement[match.start():].strip(),
    )


@pytest.fixture(name="no_ipython")
def make_no_ipython(monkeypatch):
    """A plain-interpreter run: no IPython module loaded at all."""
    monkeypatch.delitem(sys.modules, "IPython", raising=False)


@pytest.fixture(name="fake_ipython")
def make_fake_ipython(monkeypatch):
    """
    Stub the ``IPython`` module a live notebook / kernel would have loaded.
    ``get_ipython`` returns a shell object, marking the session interactive.
    """
    module = types.ModuleType("IPython")
    module.get_ipython = lambda: object()
    monkeypatch.setitem(sys.modules, "IPython", module)
    return module


class FakeDevice:
    def __init__(self, kind):
        self.kind = kind

    def __str__(self):
        return self.kind


@pytest.fixture(name="fake_jax")
def make_fake_jax(monkeypatch):
    """
    Install a stub ``jax`` module (library unit tests never import real JAX)
    whose device list the test controls.
    """
    module = types.ModuleType("jax")
    module.devices = lambda: []
    monkeypatch.setitem(sys.modules, "jax", module)
    return module


class TestCheckJaxUsingGpu:
    def test_gpu_detected(self, fake_jax):
        fake_jax.devices = lambda: [FakeDevice("cuda:0")]
        assert setup_colab.check_jax_using_gpu() is False

    def test_tpu_detected(self, fake_jax):
        fake_jax.devices = lambda: [FakeDevice("TPU_0")]
        assert setup_colab.check_jax_using_gpu() is False

    def test_cpu_only(self, fake_jax):
        fake_jax.devices = lambda: [FakeDevice("TFRT_CPU_0")]
        assert setup_colab.check_jax_using_gpu() is True

    def test_any_accelerator_counts_regardless_of_order(self, fake_jax):
        # Regression: the old implementation kept only the last device's
        # status, so [gpu, cpu] wrongly reported no accelerator.
        fake_jax.devices = lambda: [FakeDevice("cuda:0"), FakeDevice("TFRT_CPU_0")]
        assert setup_colab.check_jax_using_gpu() is False

    def test_empty_device_list(self, fake_jax):
        # Regression: the old implementation raised UnboundLocalError here.
        fake_jax.devices = lambda: []
        assert setup_colab.check_jax_using_gpu() is True


class TestRegistry:
    def test_all_projects_have_required_fields(self):
        required = {
            "project_name",
            "top_package",
            "packages",
            "workspace_repo",
            "workspace_dir",
            "gpu_note",
        }
        for project, spec in setup_colab._PROJECTS.items():
            assert required <= set(spec), project
            assert spec["packages"][0] == "autonerves", project
            assert spec["workspace_repo"].startswith(
                "https://github.com/PyAutoLabs/"
            ), project

    def test_every_project_installs_every_sampler(self):
        # Regression: `--no-deps` means a dependency absent from the install
        # list never lands, and the notebook cell that reaches it dies with
        # ModuleNotFoundError (HowToFit chapter 1 tutorials 4, 5 and 6 on
        # Colab). Widened past the samplers to every autofit dependency that is
        # imported lazily, inside a function, and so survives `import autofit`:
        # `corner` (the reported failure), `optax`, `xxhash` and `blackjax`.
        # Match on the package name only, so a future re-pin of any of them
        # does not break this test — drift is `TestSpecifiersTrackAutofit`'s
        # job, this one guards the "missing entirely" class.
        required = {
            "dynesty",
            "emcee",
            "nautilus-sampler",
            "corner",
            "optax",
            "xxhash",
            "blackjax",
        }
        for project, spec in setup_colab._PROJECTS.items():
            names = {
                re.split(r"[<>=!~\[]", package, maxsplit=1)[0].strip()
                for package in spec["packages"]
            }
            assert required <= names, (project, sorted(names))

    def test_every_project_has_a_wrapper(self):
        for project in setup_colab._PROJECTS:
            assert hasattr(setup_colab, f"for_{project}"), project

    def test_unknown_project_raises_with_choices(self):
        with pytest.raises(KeyError, match="autogalaxy"):
            setup_colab.setup("not_a_project")


# An exact pin, e.g. `==1.0.5`. Deliberately does not match `===1.0.5`
# (arbitrary equality) or a wildcard pin like `==1.0.*`, neither of which names
# a single version that can be tested against a range; both fall through to the
# string-equality arm below.
_EXACT_PIN = re.compile(r"==(?!=)\s*([^,\s*]+)$")


class TestSpecifiersTrackAutofit:
    def test_shared_extras_match_autofits_declared_specifiers(self):
        """
        Every `_SHARED_EXTRAS` entry autofit also declares must agree with
        autofit's specifier, under a deliberately ASYMMETRIC rule:

        - An EXACT pin (`==X`) need only be COMPATIBLE: `X` must satisfy
          autofit's declared specifier. This lets a deliberate narrowing stand.
          The worked example is `dill`: the Colab list pins `dill==0.4.0` while
          autofit declares the floor `dill>=0.3.1.1`. 0.4.0 satisfies that
          floor, so it is a narrowing and not a drift, and it passes.
        - Anything that is NOT an exact pin (a range: `optax>=0.2.5`,
          `xxhash<=3.4.1`) must match autofit's specifier as an exact STRING.
          Merely overlapping autofit's range is not enough — for these the list
          is meant to MIRROR the file, and a looser-but-overlapping range is
          exactly how it would quietly drift away from it.

        Do not "simplify" this back to plain string equality on both arms: that
        is what `dill` fails, and it fails for no good reason.

        The expectations are DERIVED from PyAutoFit's pyproject.toml at run
        time, never restated here. Two entries had already drifted from the
        file the list's own comment claims to track (`nautilus-sampler` a patch
        behind autofit's pin, `anesthetic` pinned BELOW autofit's floor)
        precisely because the list repeats literals nobody re-checks. Copying
        those literals into this test would reproduce that failure mode one
        layer up — the test would go stale alongside the list it guards.
        """
        try:
            from packaging.specifiers import SpecifierSet
        except ImportError:  # pragma: no cover - `packaging` ships with pip
            pytest.skip("`packaging` is not importable, so specifiers cannot be compared")

        pyproject = Path(__file__).parents[2] / "PyAutoFit" / "pyproject.toml"

        if not pyproject.is_file():
            # PyAutoNerves CI may run with no sibling PyAutoFit checkout; there
            # is nothing to compare against, and that is not a failure.
            pytest.skip(
                f"no sibling PyAutoFit checkout at {pyproject} to read "
                "declared specifiers from"
            )

        with open(pyproject, "rb") as f:
            project = tomllib.load(f)["project"]

        # `blackjax` and `nautilus-sampler` are declared in the `optional`
        # extra rather than the base dependencies.
        declared = dict(
            _split_requirement(requirement)
            for requirement in (
                project["dependencies"]
                + project["optional-dependencies"]["optional"]
            )
        )

        mismatched = {}

        for entry in setup_colab._SHARED_EXTRAS:
            name, specifier = _split_requirement(entry)

            if name not in declared:
                # Not an autofit dependency at all (`jaxnnls`), so there is no
                # declared specifier for it to track.
                continue

            autofit_specifier = declared[name]
            pin = _EXACT_PIN.fullmatch(specifier)

            if pin is not None:
                # `prereleases=True` so a prerelease pin is not reported as
                # non-satisfying merely for being a prerelease.
                ok = SpecifierSet(autofit_specifier).contains(
                    pin.group(1), prereleases=True
                )
            else:
                # Covers the unpinned case too: an entry autofit declares but
                # the Colab list leaves bare is a mismatch, not a free pass.
                ok = specifier == autofit_specifier

            if not ok:
                mismatched[name] = {
                    "setup_colab": specifier or "(unpinned)",
                    "autofit": autofit_specifier,
                }

        assert not mismatched, mismatched


class TestNoImportSideEffects:
    def test_import_does_not_set_xla_flags(self):
        # Regression: the module used to set XLA_FLAGS at import time,
        # clobbering user environments on every `import autonerves.setup_colab`.
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(setup_colab))
        for node in tree.body:
            assert not isinstance(
                node, (ast.Assign, ast.Expr)
            ) or "environ" not in ast.dump(node), "module-level os.environ mutation"


class TestOutsideColab:
    def test_cli_run_is_a_silent_noop(self, no_ipython, capsys):
        # The Witness: google.colab is not importable and no notebook shell is
        # active (a plain CLI run of a workspace start_here.py), so setup()
        # must return cleanly without installing or cloning anything AND
        # without printing the "not running in a Google Colab" block.
        with mock.patch.object(subprocess, "check_call") as check_call:
            setup_colab.setup("autolens")
        check_call.assert_not_called()
        assert capsys.readouterr().out == ""

    def test_notebook_run_still_surfaces_the_message(self, fake_ipython, capsys):
        # In a notebook the setup cell was a no-op and the user needs telling
        # they can carry on — the message must survive there.
        with mock.patch.object(subprocess, "check_call") as check_call:
            setup_colab.setup("autolens")
        check_call.assert_not_called()
        assert "not running in a Google Colab" in capsys.readouterr().out

    def test_imported_but_inactive_ipython_stays_silent(
        self, fake_ipython, capsys
    ):
        # IPython merely importable (or imported by a library) with no shell
        # driving the process is still the CLI path.
        fake_ipython.get_ipython = lambda: None
        setup_colab.setup("autolens")
        assert capsys.readouterr().out == ""

    def test_cli_message_retrievable_at_debug_level(self, no_ipython, caplog):
        # The silent path demotes rather than deletes: the text lands on the
        # module logger at DEBUG for anyone who asks for it.
        with caplog.at_level(logging.DEBUG, logger="autonerves.setup_colab"):
            setup_colab.setup("autolens")
        assert any(
            "not running in a Google Colab" in record.message
            for record in caplog.records
        )

    def test_wrappers_delegate(self):
        with mock.patch.object(setup_colab, "setup") as setup_mock:
            setup_colab.for_autolens(raise_error_if_not_gpu=False)
            setup_mock.assert_called_once_with(
                "autolens", raise_error_if_not_gpu=False
            )
            setup_mock.reset_mock()
            setup_colab.for_howtofit()
            setup_mock.assert_called_once_with(
                "howtofit", raise_error_if_not_gpu=False
            )


class TestCloneWorkspace:
    def test_existing_dir_skips_clone(self, tmp_path, capsys):
        with mock.patch.object(subprocess, "run") as run:
            setup_colab._clone_workspace(
                "https://github.com/PyAutoLabs/autolens_workspace",
                str(tmp_path),
                "autolens",
            )
        run.assert_not_called()
        assert "not cloning again" in capsys.readouterr().out

    def test_clones_release_tag_when_available(self, tmp_path):
        target = str(tmp_path / "ws")
        with mock.patch.object(setup_colab, "_installed_version", return_value="2026.7.22.1"):
            with mock.patch.object(
                subprocess, "run", return_value=mock.Mock(returncode=0)
            ) as run:
                setup_colab._clone_workspace("repo_url", target, "autolens")
        run.assert_called_once()
        args = run.call_args[0][0]
        assert args[:4] == ["git", "clone", "--depth", "1"]
        assert ["--branch", "2026.7.22.1"] == args[4:6]

    def test_falls_back_to_default_branch_when_tag_missing(self, tmp_path):
        target = str(tmp_path / "ws")
        with mock.patch.object(setup_colab, "_installed_version", return_value="1.2.3"):
            with mock.patch.object(
                subprocess, "run", return_value=mock.Mock(returncode=1)
            ) as run:
                setup_colab._clone_workspace("repo_url", target, "autolens")
        assert run.call_count == 2
        fallback = run.call_args_list[1]
        assert "--branch" not in fallback[0][0]
        assert fallback[1] == {"check": True}

    def test_falls_back_when_version_unknown(self, tmp_path):
        target = str(tmp_path / "ws")
        with mock.patch.object(setup_colab, "_installed_version", return_value=None):
            with mock.patch.object(
                subprocess, "run", return_value=mock.Mock(returncode=0)
            ) as run:
                setup_colab._clone_workspace("repo_url", target, "autolens")
        run.assert_called_once()
        assert "--branch" not in run.call_args[0][0]


class TestWorkspaceDirOverride:
    def test_override_threads_to_colab_setup(self):
        with mock.patch.object(setup_colab, "_colab_setup") as colab_setup:
            setup_colab.setup("autolens", workspace_dir="/tmp/sim_ws")
        assert colab_setup.call_args[1]["workspace_dir"] == "/tmp/sim_ws"

    def test_default_is_the_registry_colab_dir(self):
        with mock.patch.object(setup_colab, "_colab_setup") as colab_setup:
            setup_colab.setup("autolens")
        assert (
            colab_setup.call_args[1]["workspace_dir"]
            == "/content/autolens_workspace"
        )
