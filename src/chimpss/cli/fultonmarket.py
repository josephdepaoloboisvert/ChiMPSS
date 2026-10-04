"""
CLI entry point for FultonMarket parallel tempering REMD simulation.

Console script: chimpss-fultonmarket
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(
        description='Run a FultonMarket parallel tempering replica exchange simulation.')

    # System inputs
    p.add_argument('input_pdb',    type=str,
                   help='Path to PDB file with starting coordinates')
    p.add_argument('input_system', type=str,
                   help='Path to XML file with the serialized system')
    p.add_argument('output_dir',   type=str,
                   help='Directory to store output')
    p.add_argument('--input_state', default=None, type=str,
                   help='Path to XML file of a recently written state')

    # Temperature ladder
    p.add_argument('--T_min',        default=300,  type=float,
                   help='Lowest temperature on the ladder (K). Default 300.')
    p.add_argument('--T_max',        default=367,  type=float,
                   help='Highest temperature on the ladder (K). Default 367.')
    p.add_argument('--n_replicates', default=68,   type=int,
                   help='Number of replica exchange states. Default 68. Ignored when the '
                        'ladder is trailblazed or read from a file.')
    p.add_argument('--ladder', default='geometric', choices=['geometric', 'trailblaze', 'file'],
                   help="How the block-0 ladder is chosen. 'geometric' (default) is the "
                        "historical spacing. 'trailblaze' walks the ladder before block 0 so "
                        "every neighbour pair sits at the same thermodynamic length, caching it "
                        "under <output_dir>/trailblaze/ (chimpss-trailblaze can do this as a "
                        "separate job). 'file' reads --temperatures_file. On resume the saved "
                        "ladder always wins.")
    p.add_argument('--temperatures_file', default=None, type=str,
                   help='Ladder as .npy or one temperature per line; implies --ladder file.')
    p.add_argument('--init_overlap_thresh', default=None, type=float,
                   help='Neighbour acceptance below which block 0 inserts a rung and restarts. '
                        'Default: 0.5, or the walk\'s --min_acc for a trailblazed ladder.')
    p.add_argument('--term_overlap_thresh', default=None, type=float,
                   help='Same for later blocks. Default: 0.35, or --min_acc for a trailblazed ladder.')

    # Trailblazing
    t = p.add_argument_group('trailblazing (--ladder trailblaze)')
    t.add_argument('--target_acc', default=0.40, type=float,
                   help='Neighbour acceptance the walk aims for; sets the spacing '
                        'dL* = 2 erfcinv(A). Default 0.40 (dL* 1.19).')
    t.add_argument('--min_acc', default=0.30, type=float,
                   help='Measured acceptance below which a proposed rung is rejected and the '
                        'step shrunk. Default 0.30.')
    t.add_argument('--max_acc', default=0.75, type=float,
                   help='Acceptance above which a proposed rung is rejected and the step '
                        'stretched. Default 0.75.')
    t.add_argument('--tb_first_equil_ps', default=200.0, type=float,
                   help='Equilibration of the starting configuration at T_min. Default 200.')
    t.add_argument('--tb_equil_ps', default=20.0, type=float,
                   help='Equilibration at each new rung. Default 20.')
    t.add_argument('--tb_sample_ps', default=100.0, type=float,
                   help='Enthalpy sampling per rung, extended until --tb_neff_min effective '
                        'samples or --tb_max_sample_ps. Default 100.')
    t.add_argument('--tb_sample_interval_ps', default=1.0, type=float,
                   help='Spacing of enthalpy samples. Default 1.')
    t.add_argument('--tb_max_sample_ps', default=400.0, type=float, help='Default 400.')
    t.add_argument('--tb_neff_min', default=50, type=int, help='Default 50.')
    t.add_argument('--tb_max_rungs', default=400, type=int,
                   help='Abort if the walk has not reached T_max after this many rungs. Default 400.')
    t.add_argument('--tb_sigma_window', default=5, type=int,
                   help='Rungs over which sigma_H is pooled for the step (1 = each rung alone). Default 5.')
    t.add_argument('--tb_sigma_source', default='pooled', choices=['pooled', 'slope'],
                   help="What the step uses: 'pooled' (default) the pooled enthalpy fluctuation; "
                        "'slope' the fluctuation-dissipation value from the enthalpy slope d<H>/dT "
                        "over the trailing rungs, which tracks the production metric more closely "
                        "when single-trajectory windows miss slow fluctuations.")
    t.add_argument('--tb_force', action='store_true',
                   help='Resume a trailblaze record written with different settings.')

    # Simulation timing
    p.add_argument('--iter_length', default=0.001, type=float,
                   help='Time between replica swaps (ns). Default 0.001.')
    p.add_argument('--timestep',    default=2.0,   type=float,
                   help='Integration timestep (fs). Default 2.0.')
    p.add_argument('--sim_length',  default=25,    type=int,
                   help='Sub-simulation duration / checkpoint interval (ns). Default 25.')
    p.add_argument('--total_sim_time', default=1200, type=int,
                   help='Maximum aggregate simulation time (ns). Default 1200.')
    p.add_argument('--minimum_sim_fraction', default=0.35, type=float,
                   help='Minimum fraction of total_sim_time to run before auto-stop '
                        'is attempted. Default 0.35.')

    # Convergence thresholds
    p.add_argument('--n_resample', default=1000, type=int,
                   help='Frames to importance-resample per convergence check. Default 1000.')
    p.add_argument('--equil_method', default='max_neff', type=str,
                   help="Equilibration estimator. 'max_neff' (default) is "
                        "pymbar detect_equilibration, the published behaviour. "
                        "Alternatives come from chimpss.analysis.equilibration, "
                        "e.g. 'mser_sqrt' (the sqrt(n)-windowed marginal standard "
                        "error rule), 'mser_5', 'mser_w5'.")
    p.add_argument('--equil_monotone', action='store_true',
                   help='Forbid the equilibration index from moving backwards '
                        'between blocks. Without this a downward revision of t0 '
                        're-admits blocks that had already left the comparison '
                        'window, so the criterion has no ratchet.')
    p.add_argument('--window_blocks', default=None, type=int,
                   help='Compare only against the preceding N blocks, instead of every '
                        'block since the equilibration point. The published rule ties '
                        'the window to t0, so it widens exactly when the discard falls '
                        'and is not monotone in block number. Replayed on NTSR1 the '
                        'published rule reached 53 blocks wide and never passed, while '
                        'a trailing 3-block window converged at block 39.')
    p.add_argument('--equil_freeze_after', default=None, type=int,
                   help='Stop re-deriving t0 once this many blocks have been completed, '
                        'holding the value reached. Equilibration happens once; '
                        're-estimating it every block asks a noisy estimator to '
                        'rediscover a fixed point, and a downward revision re-admits '
                        'blocks that had already left the comparison window.')
    p.add_argument('--window_fraction', default=None, type=float,
                   help='Fix the convergence comparison window at this trailing '
                        'fraction of completed blocks instead of deriving it from '
                        't0. Decouples the window from the equilibration gate, '
                        'which otherwise oppose each other through a shared '
                        'variable. e.g. 0.25 for the trailing quarter.')
    p.add_argument('--min_pool_frames', default=None, type=int,
                   help='Cap t0 so the post-equilibration pool never falls below this '
                        'many frames. Pool size is what predicts MBAR weight collapse '
                        '(collapsed blocks had 1-29 frames, healthy ones ~106); this '
                        'prevents the failure rather than detecting it after the fact, '
                        'and unlike --min_ess it is knowable before the solve. 35 caught '
                        'every collapse observed on NTSR1 with no false negatives.')
    p.add_argument('--min_ess', default=None, type=float,
                   help='Reject a block whose MBAR weight column collapses below this '
                        'effective sample size, instead of resampling a handful of '
                        'distinct configurations. The block is still saved and '
                        'resumable; it simply contributes no convergence comparison. '
                        'Omit to preserve the previous behaviour. Measured on NTSR1: '
                        'collapsed blocks had ESS 1-10, healthy blocks 50-150.')
    p.add_argument('--max_equil_fraction', default=0.75, type=float,
                   help='Max fraction of simulation discardable as equilibration '
                        'before convergence is considered unreliable. Default 0.75.')
    p.add_argument('--frobenius_thresh', default=0.05, type=float,
                   help='Normalised Frobenius norm threshold for distance matrix '
                        'convergence. Default 0.05.')
    p.add_argument('--jsd_thresh', default=0.10, type=float,
                   help='Jensen-Shannon divergence threshold for distance matrix '
                        'convergence. Default 0.10.')

    # getContacts
    p.add_argument('--getcontacts_script', default=None, type=str,
                   help='Path to get_dynamic_contacts.py. Required for contact '
                        'distance matrix convergence.')
    p.add_argument('--conda_env', default=None, type=str,
                   help='Name of the conda environment containing getContacts. '
                        'Required when --getcontacts_python is not given.')
    p.add_argument('--getcontacts_python', default=None, type=str,
                   help='Explicit path to the Python interpreter for getContacts. '
                        'When provided, --conda_env is ignored.')
    p.add_argument('--getcontacts_cores', default=10, type=int,
                   help='CPU cores passed to getContacts via --cores. Default 10.')

    # Convergence decision
    p.add_argument('--convergence', default='jsd', choices=['jsd', 'panel', 'both', 'either'],
                   help="Which criterion stops the run. 'jsd' (default) is the published "
                        "distance-matrix criterion. 'panel' hands the decision to the "
                        "effect-size convergence panel (mean-structure drift, state "
                        "stability, ensemble width and basin populations against "
                        "experimental tolerances, on independent halves of the "
                        "low-temperature ensemble); the JSD machinery still runs every "
                        "block for its resampled artifacts and its report is still "
                        "printed. 'both' requires agreement, 'either' stops on the first. "
                        "The aggregate-time ceiling stops the run in every mode.")
    p.add_argument('--panel_pangpcr_prefix', default=None, type=str,
                   help='Pan-GPCR PCA basis prefix (e.g. models/pangpcr1513). Required '
                        'for every mode other than jsd.')
    p.add_argument('--panel_pdb_code', default=None, type=str,
                   help='Parent structure code; numbers the topology against GPCRdb and '
                        'selects the receptor for its tolerances. Required unless '
                        '<output_dir>/analysis/live_panel_config.json already exists.')
    p.add_argument('--panel_receptor', default=None, type=str,
                   help='GPCRdb protein slug (ntr1_human) for receptor-specific tolerances. '
                        'Default: looked up from the parent structure.')
    p.add_argument('--panel_stor_dir', default='./many_structures/', type=str,
                   help='Directory holding cached GPCRdb records. Compute nodes have no '
                        'network: run chimpss-convergence-panel --prepare from a login '
                        'node first so the numbering is cached here.')
    p.add_argument('--panel_n_states', default=5, type=int,
                   help='Lowest temperature rungs pooled for the panel. Default 5 '
                        '(310.0-312.3 K on the campaign ladder).')
    p.add_argument('--panel_consecutive', default=3, type=int,
                   help='Blocks the CONVERGED verdict must hold before the run stops. '
                        'Default 3.')
    p.add_argument('--panel_b0_fraction', default=0.25, type=float,
                   help='Fraction of completed blocks discarded as equilibration before '
                        'the halves are formed. Default 0.25.')
    p.add_argument('--panel_width_fraction', default=0.25, type=float,
                   help='Ensemble-width tolerance as a fraction of the same-state '
                        'experimental RMSD. Default 0.25.')
    p.add_argument('--panel_min_blocks', default=16, type=int,
                   help='Equilibrated blocks below which the panel declines to answer. '
                        'Default 16 (8 per half).')

    p.add_argument('--protein-name', default=None, type=str,
                   help='Protein name for output file naming (no underscores or periods). '
                        'When combined with --ligand-name, outputs are named '
                        'PROTEIN_LIGAND.trajectory.ncdf / PROTEIN_LIGAND.checkpoint.ncdf.')
    p.add_argument('--ligand-name', default=None, type=str,
                   help='Ligand name for output file naming (no underscores or periods).')

    args = p.parse_args()
    if args.convergence != 'jsd' and not args.panel_pangpcr_prefix:
        p.error(f'--convergence {args.convergence} requires --panel_pangpcr_prefix')
    if args.temperatures_file and args.ladder == 'trailblaze':
        p.error('--temperatures_file cannot be combined with --ladder trailblaze')
    if args.temperatures_file:
        args.ladder = 'file'
    return args


def trailblaze_settings(args):
    """The trailblaze_kwargs dict implied by the parsed flags, or None unless --ladder trailblaze."""
    if args.ladder != 'trailblaze':
        return None
    return dict(target_acc=args.target_acc, min_acc=args.min_acc, max_acc=args.max_acc,
                first_equil_ps=args.tb_first_equil_ps, equil_ps=args.tb_equil_ps,
                sample_ps=args.tb_sample_ps, sample_interval_ps=args.tb_sample_interval_ps,
                max_sample_ps=args.tb_max_sample_ps, neff_min=args.tb_neff_min,
                max_rungs=args.tb_max_rungs, sigma_window=args.tb_sigma_window, sigma_source=args.tb_sigma_source,
                force=args.tb_force)


def panel_settings(args):
    """The LivePanel keyword arguments implied by the parsed flags, or None."""
    if args.convergence == 'jsd':
        return None
    return dict(model_prefix=args.panel_pangpcr_prefix,
                pdb_code=args.panel_pdb_code,
                receptor=args.panel_receptor,
                stor_dir=args.panel_stor_dir,
                n_states=args.panel_n_states,
                consecutive=args.panel_consecutive,
                b0_fraction=args.panel_b0_fraction,
                width_fraction=args.panel_width_fraction,
                min_blocks=args.panel_min_blocks)


def main():
    args = parse_args()

    from chimpss.fultonmarket import FultonMarket as FM

    market = FM(input_pdb=args.input_pdb,
                input_system=args.input_system,
                input_state=args.input_state,
                T_min=args.T_min,
                T_max=args.T_max,
                n_replicates=args.n_replicates,
                protein_name=args.protein_name,
                ligand_name=args.ligand_name,
                ladder=args.ladder,
                temperatures_file=args.temperatures_file,
                trailblaze_kwargs=trailblaze_settings(args))

    os.makedirs(args.output_dir, exist_ok=True)

    getContacts_Info = None
    if args.getcontacts_script is not None:
        getContacts_Info = {
            'getcontacts_script': args.getcontacts_script,
            'conda_env':          args.conda_env,
            'getcontacts_python': args.getcontacts_python,
            'cores':              args.getcontacts_cores,
        }

    market.run(iter_length=args.iter_length,
               dt=args.timestep,
               sim_length=args.sim_length,
               total_sim_time=args.total_sim_time,
               output_dir=args.output_dir,
               minimum_fraction=args.minimum_sim_fraction,
               init_overlap_thresh=args.init_overlap_thresh,
               term_overlap_thresh=args.term_overlap_thresh,
               n_resample=args.n_resample,
               max_equil_fraction=args.max_equil_fraction,
               min_ess=args.min_ess,
               min_pool_frames=args.min_pool_frames,
               equil_method=args.equil_method,
               equil_monotone=args.equil_monotone,
               window_fraction=args.window_fraction,
               window_blocks=args.window_blocks,
               equil_freeze_after=args.equil_freeze_after,
               frobenius_thresh=args.frobenius_thresh,
               jsd_thresh=args.jsd_thresh,
               getContacts_Info=getContacts_Info,
               convergence=args.convergence,
               panel=panel_settings(args))


if __name__ == '__main__':
    main()
