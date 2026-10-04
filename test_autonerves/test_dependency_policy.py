"""Installer contract: exclude the CPU deadlock without discarding older JAX."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
import pytest


@pytest.fixture(params=['jax', 'jaxlib'])
def requirement(request):
    project = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())
    return next(Requirement(value) for value in project['project']['dependencies']
                if Requirement(value).name == request.param)


@pytest.mark.parametrize('version,allowed', [
    ('0.7.0', True), ('0.9.2', True), ('0.10.0', False), ('0.10.1', False),
    ('0.10.2', False), ('0.10.99', False), ('0.11.0', False), ('0.11.1', True),
    ('0.11.2', True), ('0.12.0', False),
])
def test_cpu_deadlock_exclusion_keeps_older_versions(requirement, version, allowed):
    assert (version in requirement.specifier) is allowed


@pytest.mark.parametrize('system,machine,required', [
    ('darwin', 'x86_64', False), ('darwin', 'arm64', True),
    ('linux', 'x86_64', True), ('win32', 'AMD64', True),
])
def test_existing_platform_exemption_is_preserved(requirement, system, machine, required):
    assert requirement.marker.evaluate({'sys_platform': system, 'platform_machine': machine}) is required
