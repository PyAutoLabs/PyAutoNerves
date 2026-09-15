import logging
import re
import subprocess
import sys
import types
from unittest import mock

import pytest

from autonerves import setup_colab


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
        # Regression: `--no-deps` means a sampler absent from the install list
        # never lands, and the notebook cell constructing that search dies with
        # ModuleNotFoundError at fit time (HowToFit chapter 1 tutorials 4, 5
        # and 6 on Colab). Match on the package name only, so a future re-pin
        # of any of them does not break this test.
        #
        # `blackjax` joined the set on 2026-09-15: it was missing here, and
        # HowToFit chapter 1 tutorials 6 and 7 both died with
        # `ModuleNotFoundError: No module named 'blackjax'` at their
        # `af.BlackJAXNUTS` fit. This test is the guard that should have caught
        # it — the required set has to name every search the notebooks
        # construct, not just the ones that broke last time.
        required = {"dynesty", "emcee", "nautilus-sampler", "blackjax"}
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
