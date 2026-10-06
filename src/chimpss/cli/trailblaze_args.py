"""The trailblazing flags shared by ``chimpss-fultonmarket`` and ``chimpss-trailblaze walk``.

One definition so the two entry points cannot drift apart: every flag here
maps onto a key of ``trailblaze.SAMPLER_KEYS + WALK_KEYS``, which
:func:`trailblaze_settings` assembles into the ``trailblaze_kwargs`` dict
``FultonMarket`` takes.
"""
#: Mirrors ``chimpss.fultonmarket.trailblaze.DIRECTIONS`` without importing the package (and openmm)
#: at argument-parsing time; the plumbing tests check the two agree.
DIRECTIONS = ('up', 'down', 'both')


def add_trailblaze_arguments(target, direction_flags=('--tb_direction',)):
    """Add the trailblazing flags to an argparse parser or argument group.

    ``direction_flags`` lists the option strings of the direction flag; the
    standalone walk command also accepts the shorter ``--direction``.
    """
    target.add_argument('--target_acc', default=0.40, type=float,
                        help='Neighbour acceptance the walk aims for; sets the spacing dL* = 2 erfcinv(A). '
                             'Default 0.40 (dL* 1.19).')
    target.add_argument('--min_acc', default=0.30, type=float,
                        help='Measured acceptance below which a proposed rung is rejected and the step shrunk. '
                             'Default 0.30.')
    target.add_argument('--max_acc', default=0.75, type=float,
                        help='Acceptance above which a proposed rung is rejected and the step stretched. '
                             'Default 0.75.')
    target.add_argument(*direction_flags, dest='tb_direction', default='up', choices=list(DIRECTIONS),
                        help="Which way to walk the ladder: 'up' from T_min (default, AlGDock's direction), "
                             "'down' from T_max (the starting configuration is heated there in stages), or "
                             "'both': the up walk, then the down walk seeded from its hottest rung, reconciled "
                             "into one ladder placed by the integral of the merged metric and seeded from the "
                             "down walk.")
    target.add_argument('--tb_first_equil_ps', default=200.0, type=float,
                        help='Equilibration of the starting configuration at the first rung. Default 200.')
    target.add_argument('--tb_equil_ps', default=20.0, type=float,
                        help='Equilibration at each new rung. Default 20.')
    target.add_argument('--tb_sample_ps', default=100.0, type=float,
                        help='Enthalpy sampling per rung, extended until --tb_neff_min effective samples or '
                             '--tb_max_sample_ps. Default 100.')
    target.add_argument('--tb_sample_interval_ps', default=1.0, type=float,
                        help='Spacing of enthalpy samples. Default 1.')
    target.add_argument('--tb_max_sample_ps', default=400.0, type=float, help='Default 400.')
    target.add_argument('--tb_neff_min', default=50, type=int, help='Default 50.')
    target.add_argument('--tb_max_rungs', default=400, type=int,
                        help='Abort if the walk has not reached the far end after this many rungs. Default 400.')
    target.add_argument('--tb_sigma_window', default=5, type=int,
                        help='Rungs over which sigma_H is pooled for the step (1 = each rung alone). Default 5.')
    target.add_argument('--tb_sigma_source', default='pooled', choices=['pooled', 'slope'],
                        help="What the step uses: 'pooled' (default) the pooled enthalpy fluctuation; "
                             "'slope' the fluctuation-dissipation value from the enthalpy slope d<H>/dT "
                             "over the trailing rungs, which tracks the production metric more closely "
                             "when single-trajectory windows miss slow fluctuations.")
    target.add_argument('--tb_ramp_K', default=10.0, type=float,
                        help='Stage size when a down-only walk heats the starting configuration to T_max. '
                             'Default 10.')
    target.add_argument('--tb_ramp_ps', default=None, type=float,
                        help='Dynamics per heating stage. Default: --tb_equil_ps.')
    target.add_argument('--tb_force', action='store_true',
                        help='Resume a trailblaze record written with different settings.')


def trailblaze_settings(args):
    """The ``trailblaze_kwargs`` dict implied by the parsed flags."""
    return dict(target_acc=args.target_acc, min_acc=args.min_acc, max_acc=args.max_acc,
                direction=args.tb_direction,
                first_equil_ps=args.tb_first_equil_ps, equil_ps=args.tb_equil_ps,
                sample_ps=args.tb_sample_ps, sample_interval_ps=args.tb_sample_interval_ps,
                max_sample_ps=args.tb_max_sample_ps, neff_min=args.tb_neff_min,
                max_rungs=args.tb_max_rungs, sigma_window=args.tb_sigma_window, sigma_source=args.tb_sigma_source,
                ramp_K=args.tb_ramp_K, ramp_ps=args.tb_ramp_ps, force=args.tb_force)
