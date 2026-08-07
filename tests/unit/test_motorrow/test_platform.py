"""Regression tests for OpenMM platform selection in MotorRow.

MotorRow previously hardcoded ``Platform.getPlatformByName('OpenCL')`` at two
sites (minimisation and every propagation step). That raises outright on builds
where the OpenCL plugin ships but no OpenCL ICD is installed -- the ordinary
case on NVIDIA HPC nodes, including the Expanse GPU partition, where the
registered platforms are Reference, CPU and CUDA only.

These tests pin the replacement: prefer CUDA, fall back in a defined order, and
honour an explicit override. They monkeypatch the platform registry rather than
depending on what the test machine happens to have, so they are meaningful on a
CPU-only runner.
"""
import pytest

pytest.importorskip("openmm")

import openmm  # noqa: E402

from chimpss.motorrow.utils import platform_properties, select_platform  # noqa: E402


class _FakePlatform:
    def __init__(self, name):
        self._name = name

    def getName(self):
        return self._name


def _install(monkeypatch, names):
    """Pretend the OpenMM registry holds exactly ``names``."""
    fakes = [_FakePlatform(n) for n in names]
    monkeypatch.setattr(openmm.Platform, 'getNumPlatforms',
                        staticmethod(lambda: len(fakes)))
    monkeypatch.setattr(openmm.Platform, 'getPlatform',
                        staticmethod(lambda i: fakes[i]))

    def by_name(name):
        for f in fakes:
            if f.getName() == name:
                return f
        raise Exception(f'no platform {name}')

    monkeypatch.setattr(openmm.Platform, 'getPlatformByName',
                        staticmethod(by_name))
    return fakes


# ---------------------------------------------------------------------------
# Preference order
# ---------------------------------------------------------------------------

def test_prefers_cuda_over_opencl_and_cpu(monkeypatch):
    monkeypatch.delenv('CHIMPSS_PLATFORM', raising=False)
    _install(monkeypatch, ['Reference', 'CPU', 'OpenCL', 'CUDA'])
    assert select_platform().getName() == 'CUDA'


def test_this_environment_would_have_broken_the_old_code(monkeypatch):
    """Reference/CPU/CUDA only -- exactly what the campaign machine reports.

    The old hardcoded OpenCL lookup raises here; selection must return CUDA.
    """
    monkeypatch.delenv('CHIMPSS_PLATFORM', raising=False)
    _install(monkeypatch, ['Reference', 'CPU', 'CUDA'])
    assert select_platform().getName() == 'CUDA'

    with pytest.raises(Exception):
        openmm.Platform.getPlatformByName('OpenCL')


def test_falls_back_to_opencl_when_cuda_absent(monkeypatch):
    monkeypatch.delenv('CHIMPSS_PLATFORM', raising=False)
    _install(monkeypatch, ['Reference', 'CPU', 'OpenCL'])
    assert select_platform().getName() == 'OpenCL'


def test_falls_back_to_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.delenv('CHIMPSS_PLATFORM', raising=False)
    _install(monkeypatch, ['Reference', 'CPU'])
    assert select_platform().getName() == 'CPU'


def test_reference_is_the_last_resort(monkeypatch):
    monkeypatch.delenv('CHIMPSS_PLATFORM', raising=False)
    _install(monkeypatch, ['Reference'])
    assert select_platform().getName() == 'Reference'


def test_raises_when_nothing_is_registered(monkeypatch):
    monkeypatch.delenv('CHIMPSS_PLATFORM', raising=False)
    _install(monkeypatch, [])
    with pytest.raises(RuntimeError, match='No usable OpenMM platform'):
        select_platform()


# ---------------------------------------------------------------------------
# Explicit override
# ---------------------------------------------------------------------------

def test_argument_override_wins(monkeypatch):
    monkeypatch.delenv('CHIMPSS_PLATFORM', raising=False)
    _install(monkeypatch, ['Reference', 'CPU', 'OpenCL', 'CUDA'])
    assert select_platform('CPU').getName() == 'CPU'


def test_env_override_wins(monkeypatch):
    _install(monkeypatch, ['Reference', 'CPU', 'OpenCL', 'CUDA'])
    monkeypatch.setenv('CHIMPSS_PLATFORM', 'OpenCL')
    assert select_platform().getName() == 'OpenCL'


def test_argument_beats_env(monkeypatch):
    _install(monkeypatch, ['Reference', 'CPU', 'OpenCL', 'CUDA'])
    monkeypatch.setenv('CHIMPSS_PLATFORM', 'OpenCL')
    assert select_platform('CUDA').getName() == 'CUDA'


def test_unavailable_override_degrades_instead_of_raising(monkeypatch):
    """An override for a platform that is not registered must not be fatal."""
    monkeypatch.delenv('CHIMPSS_PLATFORM', raising=False)
    _install(monkeypatch, ['Reference', 'CPU', 'CUDA'])
    assert select_platform('OpenCL').getName() == 'CUDA'


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['CUDA', 'OpenCL'])
def test_mixed_precision_on_accelerators(name):
    assert platform_properties(_FakePlatform(name)) == {'Precision': 'mixed'}


@pytest.mark.parametrize('name', ['CPU', 'Reference'])
def test_no_properties_on_cpu_platforms(name):
    assert platform_properties(_FakePlatform(name)) == {}


def test_motorrow_calls_selection_not_a_hardcoded_platform():
    """Guard against the hardcoded OpenCL lookup being reintroduced."""
    import inspect

    from chimpss.motorrow import motorrow

    source = inspect.getsource(motorrow)
    assert "getPlatformByName('OpenCL')" not in source
    assert 'getPlatformByName("OpenCL")' not in source
    assert source.count('select_platform()') >= 2
