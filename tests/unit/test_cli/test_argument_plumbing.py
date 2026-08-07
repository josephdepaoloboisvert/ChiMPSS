"""Regression tests for CLI arguments that exist but were not reachable.

Two gaps of the same shape: the library accepted a parameter, the console
script gave no way to set it, so the capability was unusable from a batch job.

* ``MotorRow.main(dt=...)`` has always taken a timestep, but ``chimpss-motorrow``
  exposed no ``--dt``. Any hydrogen-mass-repartitioned system therefore had to
  be driven from Python, and the repartitioned system XML that Bridgeport writes
  unconditionally could not be used from the CLI at all.
* ``chimpss-generate-pca`` superposed every structure onto ``selected[0]`` --
  whichever survived filtering first, which depends on dict iteration order.
  Two runs over the same inputs could therefore produce different coordinate
  frames, and there was no way to request the conventional 2RH1 reference.

These tests assert the flags exist, carry sane defaults, and reach the call.
"""

import pytest

# ---------------------------------------------------------------------------
# chimpss-motorrow
# ---------------------------------------------------------------------------

def _motorrow_args(monkeypatch, argv):
    from chimpss.cli import motorrow
    monkeypatch.setattr('sys.argv', ['chimpss-motorrow'] + argv)
    return motorrow.parse_args()


BASE = ['sys.pdb', 'sys.xml', 'out/']


def test_motorrow_dt_defaults_to_two_femtoseconds(monkeypatch):
    args = _motorrow_args(monkeypatch, BASE)
    assert args.dt == 2.0


def test_motorrow_dt_is_settable(monkeypatch):
    args = _motorrow_args(monkeypatch, BASE + ['--dt', '3.25'])
    assert args.dt == 3.25


def test_motorrow_dt_is_a_float_not_a_string(monkeypatch):
    """A string would silently break LangevinMiddleIntegrator's unit maths."""
    args = _motorrow_args(monkeypatch, BASE + ['--dt', '3.5'])
    assert isinstance(args.dt, float)


def test_motorrow_step5_controls_exist(monkeypatch):
    args = _motorrow_args(monkeypatch, BASE)
    assert args.step_5_nsteps == 1250000      # 2.5 ns at 2 fs
    assert args.restrain_step_5 is False

    args = _motorrow_args(
        monkeypatch, BASE + ['--step_5_nsteps', '625000', '--restrain_step_5'])
    assert args.step_5_nsteps == 625000
    assert args.restrain_step_5 is True


def test_motorrow_passes_dt_through_to_main(monkeypatch):
    """The flag must actually reach MotorRow.main, not just be parsed."""
    from chimpss.cli import motorrow

    seen = {}

    class FakeMotorRow:
        def __init__(self, **kwargs):
            seen['init'] = kwargs

        def main(self, pdb_in, **kwargs):
            seen['main'] = dict(kwargs, pdb_in=pdb_in)

    import chimpss.motorrow as mr_pkg
    monkeypatch.setattr(mr_pkg, 'MotorRow', FakeMotorRow, raising=False)
    monkeypatch.setattr('os.makedirs', lambda *a, **k: None)
    monkeypatch.setattr(
        'sys.argv',
        ['chimpss-motorrow'] + BASE + ['--dt', '3.5', '--lig_resname', 'UNL'])

    motorrow.main()

    assert seen['main']['dt'] == 3.5
    assert seen['main']['step_5_nsteps'] == 1250000
    assert seen['main']['restrain_step_5'] is False
    assert seen['init']['lig_resname'] == 'UNL'


# ---------------------------------------------------------------------------
# chimpss-generate-pca
# ---------------------------------------------------------------------------

def test_generate_pca_reference_flag_exists(monkeypatch):
    from chimpss.cli import generate_pca

    monkeypatch.setattr('sys.argv', ['chimpss-generate-pca', 'list.txt'])
    args = generate_pca.parse_args()
    assert args.reference is None            # opt-in; old behaviour preserved

    monkeypatch.setattr('sys.argv',
                        ['chimpss-generate-pca', 'list.txt',
                         '--reference', '2RH1'])
    assert generate_pca.parse_args().reference == '2RH1'


def test_generate_pca_defaults_are_unchanged(monkeypatch):
    """Adding --reference must not have moved any existing default.

    The campaign overrides threshold and selection explicitly precisely because
    these defaults do not reproduce the published basis set; that override is
    only meaningful if the defaults stay put.
    """
    from chimpss.cli import generate_pca

    monkeypatch.setattr('sys.argv', ['chimpss-generate-pca', 'list.txt'])
    args = generate_pca.parse_args()
    assert args.threshold == 0.90
    assert args.selection == 'name CA'
    assert args.outlier_cutoff == 100.0
    assert args.species == 'Homo sapiens'


def test_reference_reordering_puts_the_requested_structure_first():
    """The mechanism --reference relies on: move the match to index 0.

    Reproduces the reordering step so the invariant is pinned even though the
    surrounding main() needs network access and MDAnalysis universes.
    """
    selected = [('1ABC', 'ag1'), ('2RH1', 'ag2'), ('3XYZ', 'ag3')]

    want = '2RH1'
    match = [i for i, (code, _) in enumerate(selected) if code.upper() == want]
    assert match, 'reference must be found before reordering'
    selected.insert(0, selected.pop(match[0]))

    assert selected[0][0] == '2RH1'
    # Every other structure survives, and none is duplicated.
    assert [c for c, _ in selected] == ['2RH1', '1ABC', '3XYZ']
    assert len(selected) == 3


def test_reference_reordering_is_a_noop_when_already_first():
    selected = [('2RH1', 'ag1'), ('1ABC', 'ag2')]
    match = [i for i, (code, _) in enumerate(selected) if code.upper() == '2RH1']
    selected.insert(0, selected.pop(match[0]))
    assert [c for c, _ in selected] == ['2RH1', '1ABC']


# ---------------------------------------------------------------------------
# Console script registration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('module,attr', [
    ('chimpss.cli.motorrow', 'main'),
    ('chimpss.cli.fultonmarket', 'main'),
    ('chimpss.cli.fultonmarket_mpi', 'main'),
    ('chimpss.cli.generate_pca', 'main'),
    ('chimpss.cli.project_pca', 'main'),
    ('chimpss.cli.retro_analysis', 'main'),
    ('chimpss.cli.fetch_pdbs', 'main'),
    ('chimpss.cli.recovery', 'main'),
    ('chimpss.cli.motifs', 'main'),
    ('chimpss.cli.run_stats', 'main'),
])
def test_entry_points_are_importable_and_callable(module, attr):
    mod = pytest.importorskip(module)
    assert callable(getattr(mod, attr))


def test_new_entry_points_are_declared_in_pyproject():
    """chimpss-motifs / chimpss-run-stats must be installable, not just importable."""
    import os
    import re

    # tests/unit/test_cli/<file> -> repo root
    root = os.path.abspath(__file__)
    for _ in range(4):
        root = os.path.dirname(root)
    with open(os.path.join(root, 'pyproject.toml')) as fh:
        text = fh.read()

    scripts = re.search(r'\[project\.scripts\](.*?)(\n\[|\Z)', text, re.S)
    assert scripts, '[project.scripts] section missing'
    block = scripts.group(1)

    assert 'chimpss.cli.motifs:main' in block
    assert 'chimpss.cli.run_stats:main' in block


def test_motorrow_parser_help_is_wellformed(monkeypatch):
    """--help must not raise; a broken format string here breaks every job."""
    from chimpss.cli import motorrow

    monkeypatch.setattr('sys.argv', ['chimpss-motorrow', '--help'])
    with pytest.raises(SystemExit) as exc:
        motorrow.parse_args()
    assert exc.value.code == 0


def test_argparse_rejects_a_nonnumeric_dt(monkeypatch):
    from chimpss.cli import motorrow

    monkeypatch.setattr('sys.argv',
                        ['chimpss-motorrow'] + BASE + ['--dt', 'fast'])
    with pytest.raises(SystemExit):
        motorrow.parse_args()
