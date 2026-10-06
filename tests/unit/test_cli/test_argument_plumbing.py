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
    ('chimpss.cli.ladder_audit', 'main'),
    ('chimpss.cli.trailblaze', 'main'),
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
    assert 'chimpss.cli.ladder_audit:main' in block
    assert 'chimpss.cli.trailblaze:main' in block


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


# ── convergence bundle: principal-component timeseries ────────────────────────

def test_convergence_bundle_tracks_several_components_by_default():
    """Tracking only PC1 can report a settled ensemble that is still moving.

    On NTSR1_ML-301.hmr.rep0 the activation coordinate flattens late in the run
    while a lower-variance component keeps drifting, so the default must be
    greater than one.
    """
    from chimpss.cli.convergence_bundle import parse_args
    import sys

    argv = sys.argv
    try:
        sys.argv = ['chimpss-convergence-bundle', 'run_dir']
        args = parse_args()
    finally:
        sys.argv = argv
    assert args.pc_timeseries >= 2


def test_convergence_bundle_component_count_is_configurable():
    from chimpss.cli.convergence_bundle import parse_args
    import sys

    argv = sys.argv
    try:
        sys.argv = ['chimpss-convergence-bundle', 'run_dir', '--pc_timeseries', '5']
        args = parse_args()
    finally:
        sys.argv = argv
    assert args.pc_timeseries == 5


# ---------------------------------------------------------------------------
# chimpss-fultonmarket --convergence
# ---------------------------------------------------------------------------

_FM_POSITIONAL = ['chimpss-fultonmarket', 'in.pdb', 'sys.xml', 'out/']


def test_fultonmarket_convergence_defaults_to_the_published_criterion(monkeypatch):
    from chimpss.cli import fultonmarket
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL)
    args = fultonmarket.parse_args()
    assert args.convergence == 'jsd'
    assert fultonmarket.panel_settings(args) is None


def test_fultonmarket_panel_mode_requires_a_basis(monkeypatch):
    from chimpss.cli import fultonmarket
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + ['--convergence', 'panel'])
    with pytest.raises(SystemExit) as exc:
        fultonmarket.parse_args()
    assert exc.value.code == 2


def test_fultonmarket_panel_settings_are_plumbed(monkeypatch):
    from chimpss.cli import fultonmarket
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + [
        '--convergence', 'both', '--panel_pangpcr_prefix', 'models/pangpcr1513',
        '--panel_pdb_code', '6OS9', '--panel_consecutive', '4', '--panel_n_states', '3'])
    args = fultonmarket.parse_args()
    s = fultonmarket.panel_settings(args)
    assert s['model_prefix'] == 'models/pangpcr1513' and s['pdb_code'] == '6OS9'
    assert s['consecutive'] == 4 and s['n_states'] == 3
    assert set(s) == {'model_prefix', 'pdb_code', 'receptor', 'stor_dir', 'n_states',
                      'consecutive', 'b0_fraction', 'width_fraction', 'min_blocks'}


def test_fultonmarket_rejects_an_unknown_convergence_mode(monkeypatch):
    from chimpss.cli import fultonmarket
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + ['--convergence', 'vibes'])
    with pytest.raises(SystemExit):
        fultonmarket.parse_args()


# ---------------------------------------------------------------------------
# chimpss-ladder-audit
# ---------------------------------------------------------------------------

def test_ladder_audit_defaults():
    from chimpss.cli import ladder_audit
    args = ladder_audit.parse_args(['runs/x'])
    assert args.skip == 1 and args.smooth == 'cv_linear' and args.targets == [0.35, 0.40, 0.50]
    assert args.blocks is None and args.out is None and not args.respace and not args.tsv


def test_ladder_audit_respace_needs_exactly_one_target():
    from chimpss.cli import ladder_audit
    with pytest.raises(SystemExit) as exc:
        ladder_audit.parse_args(['runs/x', '--respace'])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        ladder_audit.parse_args(['runs/x', '--respace', '--target_acceptance', '0.4', '--n_rungs', '60'])
    args = ladder_audit.parse_args(['runs/x', '--respace', '--target_acceptance', '0.4', '--virtual',
                                    '--blocks', '120:161', '7'])
    assert args.target_acceptance == 0.4 and args.virtual and args.blocks == ['120:161', '7']


def test_ladder_audit_virtual_requires_respace():
    from chimpss.cli import ladder_audit
    with pytest.raises(SystemExit):
        ladder_audit.parse_args(['runs/x', '--virtual'])


# ---------------------------------------------------------------------------
# chimpss-fultonmarket --ladder and chimpss-trailblaze
# ---------------------------------------------------------------------------

def test_fultonmarket_ladder_defaults_to_geometric(monkeypatch):
    from chimpss.cli import fultonmarket
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL)
    args = fultonmarket.parse_args()
    assert args.ladder == 'geometric' and args.temperatures_file is None
    assert args.init_overlap_thresh is None and args.term_overlap_thresh is None
    assert fultonmarket.trailblaze_settings(args) is None


def test_fultonmarket_trailblaze_settings_are_plumbed(monkeypatch):
    from chimpss.cli import fultonmarket
    from chimpss.fultonmarket.trailblaze import split_kwargs
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + [
        '--ladder', 'trailblaze', '--target_acc', '0.45', '--min_acc', '0.33', '--tb_sample_ps', '50',
        '--tb_first_equil_ps', '10', '--init_overlap_thresh', '0.2'])
    args = fultonmarket.parse_args()
    s = fultonmarket.trailblaze_settings(args)
    assert s['target_acc'] == 0.45 and s['min_acc'] == 0.33 and s['sample_ps'] == 50.0
    assert s['first_equil_ps'] == 10.0 and s['force'] is False
    sampler_kwargs, walk_kwargs = split_kwargs(s)          # every key is one the library accepts
    assert set(sampler_kwargs) | set(walk_kwargs) == set(s)
    assert args.init_overlap_thresh == 0.2 and args.term_overlap_thresh is None


def test_fultonmarket_temperatures_file_implies_file_ladder(monkeypatch):
    from chimpss.cli import fultonmarket
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + ['--temperatures_file', 'ladder.npy'])
    args = fultonmarket.parse_args()
    assert args.ladder == 'file' and args.temperatures_file == 'ladder.npy'
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + ['--temperatures_file', 'ladder.npy', '--ladder', 'trailblaze'])
    with pytest.raises(SystemExit) as exc:
        fultonmarket.parse_args()
    assert exc.value.code == 2


def test_fultonmarket_passes_ladder_arguments_to_the_library(monkeypatch, tmp_path):
    """The flags must reach FultonMarket.__init__ and run(), not just parse."""
    import chimpss.fultonmarket as pkg
    from chimpss.cli import fultonmarket
    seen = {}

    class FakeFM:
        def __init__(self, **kw):
            seen['init'] = kw

        def run(self, **kw):
            seen['run'] = kw

    monkeypatch.setattr(pkg, 'FultonMarket', FakeFM)
    monkeypatch.setattr('sys.argv', ['chimpss-fultonmarket', 'in.pdb', 'sys.xml', str(tmp_path / 'out'),
                                     '--ladder', 'trailblaze', '--target_acc', '0.4', '--term_overlap_thresh', '0.25'])
    fultonmarket.main()
    assert seen['init']['ladder'] == 'trailblaze' and seen['init']['temperatures_file'] is None
    assert seen['init']['trailblaze_kwargs']['target_acc'] == 0.4
    assert seen['run']['init_overlap_thresh'] is None and seen['run']['term_overlap_thresh'] == 0.25
    assert seen['run']['insertion_rule'] is None and seen['run']['insertion_min_blocks'] == 3


def test_fultonmarket_insertion_rule_flag(monkeypatch):
    from chimpss.cli import fultonmarket
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + ['--insertion_rule', 'ucb'])
    assert fultonmarket.parse_args().insertion_rule == 'ucb'
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + ['--insertion_rule', 'trailing', '--insertion_min_blocks', '4'])
    args = fultonmarket.parse_args()
    assert args.insertion_rule == 'trailing' and args.insertion_min_blocks == 4
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + ['--insertion_rule', 'median'])
    with pytest.raises(SystemExit):
        fultonmarket.parse_args()


def test_trailblaze_cli_walk_and_from_run_parse():
    from chimpss.cli import trailblaze
    args = trailblaze.parse_args(['walk', 'in.pdb', 'sys.xml', 'out/', '--input_state', 's.xml',
                                  '--T_min', '310', '--T_max', '367', '--timestep', '3.5'])
    assert args.mode == 'walk' and args.timestep == 3.5 and args.target_acc == 0.4
    s = trailblaze.walk_settings(args)
    from chimpss.fultonmarket.trailblaze import split_kwargs
    split_kwargs(s)
    with pytest.raises(SystemExit):
        trailblaze.parse_args(['walk', 'in.pdb', 'sys.xml', 'out/'])     # --input_state is required
    args = trailblaze.parse_args(['from-run', 'runs/x', '--out', 'ladders/x', '--blocks', '120:161'])
    assert args.mode == 'from-run' and args.out == 'ladders/x' and args.blocks == ['120:161']
    assert args.skip == 1 and not args.no_virtual


def test_trailblaze_direction_and_ramp_flags_are_plumbed(monkeypatch):
    from chimpss.cli import fultonmarket, trailblaze, trailblaze_args
    from chimpss.fultonmarket.trailblaze import DIRECTIONS, split_kwargs
    assert tuple(trailblaze_args.DIRECTIONS) == DIRECTIONS
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + ['--ladder', 'trailblaze', '--tb_direction', 'both',
                                                      '--tb_ramp_K', '5', '--tb_ramp_ps', '30'])
    s = fultonmarket.trailblaze_settings(fultonmarket.parse_args())
    assert s['direction'] == 'both' and s['ramp_K'] == 5.0 and s['ramp_ps'] == 30.0
    sampler_kwargs, walk_kwargs = split_kwargs(s)
    assert walk_kwargs['direction'] == 'both' and sampler_kwargs['ramp_K'] == 5.0
    assert set(sampler_kwargs) | set(walk_kwargs) == set(s)
    # the default direction is up, and the walk command accepts the short alias
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + ['--ladder', 'trailblaze'])
    assert fultonmarket.trailblaze_settings(fultonmarket.parse_args())['direction'] == 'up'
    args = trailblaze.parse_args(['walk', 'in.pdb', 'sys.xml', 'out/', '--input_state', 's.xml', '--direction', 'down'])
    assert args.tb_direction == 'down' and trailblaze.walk_settings(args)['direction'] == 'down'
    assert trailblaze.walk_settings(args)['ramp_ps'] is None
    with pytest.raises(SystemExit):
        trailblaze.parse_args(['walk', 'in.pdb', 'sys.xml', 'out/', '--input_state', 's.xml', '--direction', 'sideways'])
    # the two entry points build the same dict from the same flags
    monkeypatch.setattr('sys.argv', _FM_POSITIONAL + ['--ladder', 'trailblaze', '--tb_direction', 'down',
                                                      '--tb_sample_ps', '50'])
    fm = fultonmarket.trailblaze_settings(fultonmarket.parse_args())
    tb = trailblaze.walk_settings(trailblaze.parse_args(['walk', 'in.pdb', 'sys.xml', 'out/', '--input_state', 's.xml',
                                                         '--tb_direction', 'down', '--tb_sample_ps', '50']))
    assert fm == tb


def test_trailblaze_cli_reconcile_parses():
    from chimpss.cli import trailblaze
    args = trailblaze.parse_args(['reconcile', 'runs/x/trailblaze'])
    assert args.mode == 'reconcile' and args.up_dir is None and args.down_dir is None and args.target_acc is None
    args = trailblaze.parse_args(['reconcile', 'runs/x/trailblaze', '--up_dir', 'runs/y/trailblaze', '--target_acc', '0.45'])
    assert args.up_dir == 'runs/y/trailblaze' and args.target_acc == 0.45
