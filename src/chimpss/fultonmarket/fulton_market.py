# Package Imports
import os

import mpiplus
import netCDF4 as nc
import numpy as np
import openmm.unit as unit
from openmm import *
from openmm.app import *
from openmmtools.states import SamplerState, ThermodynamicState

from chimpss.fultonmarket.analysis import FultonMarketAnalysis
from chimpss.fultonmarket.randolph import Randolph
from chimpss.fultonmarket.utils import *
from chimpss.fultonmarket.utils import (
    MBARWeightsDegenerate,
    _interpolate_new_positions,
    _interpolate_new_states,
)
from chimpss.shared.io import build_output_path, file_exists_skip, validate_name


# Arrays written by Randolph.save_simulation for every completed block, in write
# order. A block directory missing any of these is not resumable.
REQUIRED_BLOCK_ARRAYS = (
    'positions.npy',
    'velocities.npy',
    'box_vectors.npy',
    'states.npy',
    'energies.npy',
    'temperatures.npy',
)


class FultonMarket():
    """
    Unrestrained Parallel Tempering Replica Exchange.

    Parameters
    ----------
    input_pdb : str
        Path to input PDB file.
    input_system : str
        Path to OpenMM system XML file containing force field parameters.
    input_state : str, optional
        Path to OpenMM state XML file for initial context.
    sele_str : str, optional
        MDAnalysis-style selection string for atoms of interest.
    T_min : float
        Minimum temperature in Kelvin. Default 300.
    T_max : float
        Maximum temperature in Kelvin. Default 367.447.
    n_replicates : int
        Number of replicas across the temperature ladder. Default 12.
        Ignored when the ladder is trailblazed or read from a file.
    ladder : str
        How the block-0 ladder is chosen. ``'geometric'`` (default) is the
        historical ``geometric_distribution(T_min, T_max, n_replicates)``;
        ``'trailblaze'`` walks the ladder at the start of ``run()`` so every
        neighbour pair sits at the same thermodynamic length
        (:mod:`chimpss.fultonmarket.trailblaze`), caching the result under
        ``<output_dir>/trailblaze/``; ``'file'`` reads ``temperatures_file``.
        On resume the ladder saved with the last block always wins.
    temperatures_file : str, optional
        A ``.npy`` or text ladder; implies ``ladder='file'``.
    trailblaze_kwargs : dict, optional
        Settings for the walk and its sampler; see
        ``trailblaze.SAMPLER_KEYS`` and ``trailblaze.WALK_KEYS``.
    """

    def __init__(self,
                 input_pdb: str,
                 input_system: str,
                 input_state: str = None,
                 sele_str: str = None,
                 T_min: float = 300,
                 T_max: float = 367.447,
                 n_replicates: int = 12,
                 protein_name: str = None,
                 ligand_name: str = None,
                 ladder: str = 'geometric',
                 temperatures_file: str = None,
                 trailblaze_kwargs: dict = None):

        printf('Welcome to FultonMarket.')

        from chimpss.fultonmarket.trailblaze import LADDER_MODES, load_ladder_file, split_kwargs

        if temperatures_file is not None:
            ladder = 'file'
        if ladder not in LADDER_MODES:
            raise ValueError(f'unknown ladder mode {ladder!r}; choose from {LADDER_MODES}')
        self.ladder_mode = ladder
        self.ladder_meta = {}
        self.T_min, self.T_max = float(T_min), float(T_max)
        self.trailblaze_kwargs = dict(trailblaze_kwargs or {})
        split_kwargs(self.trailblaze_kwargs)          # fail on unknown keys before any work

        if ladder == 'file':
            temps = load_ladder_file(temperatures_file)
            if n_replicates != len(temps):
                printf(f'NOTE: n_replicates={n_replicates} ignored; {temperatures_file} holds {len(temps)} rungs')
            if abs(temps[0] - self.T_min) > 1e-6 or abs(temps[-1] - self.T_max) > 1e-6:
                printf(f'NOTE: T_min/T_max ({T_min}/{T_max}) ignored; {temperatures_file} spans '
                       f'{temps[0]:.2f}-{temps[-1]:.2f} K')
            self.temperatures = [float(t) * unit.kelvin for t in temps]
            self.n_replicates = len(temps)
            self.ladder_meta = {'ladder': 'file', 'source': temperatures_file}
        else:
            # For 'trailblaze' this is a placeholder until run() knows output_dir.
            self.temperatures = [temp * unit.kelvin for temp in geometric_distribution(T_min, T_max, n_replicates)]
            self.n_replicates = n_replicates
        self.sele_str = sele_str
        self.protein_name = validate_name(protein_name) if protein_name else None
        self.ligand_name = validate_name(ligand_name) if ligand_name else None

        self.input_pdb = input_pdb
        self.pdb = PDBFile(input_pdb)
        self._set_init_positions()
        printf(f'Found input_pdb: {input_pdb}')

        self.system = XmlSerializer.deserialize(open(input_system, 'r').read())
        self._set_init_box_vectors()
        printf(f'Found input_system: {input_system}')

        if input_state is not None:
            integrator = LangevinIntegrator(300, 0.01, 2)
            sim = Simulation(self.pdb.topology, self.system, integrator)
            sim.loadState(input_state)
            self.context = sim.context
            printf(f'Found input_state: {input_state}')


    def _set_init_positions(self):
        """
        Extract atomic positions from the input PDB and replicate them across
        all replicas.

        Sets
        ----
        self.init_positions : list of Quantity
            One copy of the PDB positions per replica, each of shape
            (n_atoms, 3) in nanometers.
        self.n_atoms : int
            Total number of atoms in the system.
        """
        self.init_positions = self.pdb.getPositions(asNumpy=True)
        self.n_atoms = self.init_positions.shape[0]
        self.init_positions = [self.init_positions for _ in range(self.n_replicates)]


    def _set_init_box_vectors(self):
        """
        Read the default periodic box vectors from the OpenMM system and
        replicate them across all replicas.

        Sets
        ----
        self.init_box_vectors : list of tuple
            One copy of the system's default box vectors per replica.
        """
        self.init_box_vectors = self.system.getDefaultPeriodicBoxVectors()
        self.init_box_vectors = [self.init_box_vectors for _ in range(self.n_replicates)]


    def run(self,
            iter_length: float,
            dt: float = 2.0,
            sim_length: int = 25,
            total_sim_time: int = None,
            minimum_fraction: float = 0.25,
            init_overlap_thresh: float = None,
            term_overlap_thresh: float = None,
            insertion_rule: str = None,
            insertion_min_blocks: int = 3,
            output_dir: str = os.path.join(os.getcwd(), 'FultonMarket_output/'),
            n_resample: int = 1000,
            max_equil_fraction: float = 0.75,
            min_ess: float = None,
            equil_method: str = 'max_neff',
            equil_monotone: bool = False,
            min_pool_frames: int = None,
            equil_freeze_after: int = None,
            window_fraction: float = None,
            window_blocks: int = None,
            frobenius_thresh: float = 0.05,
            jsd_thresh: float = 0.10,
            getContacts_Info: dict = None,
            skip_contacts: bool = False,
            convergence: str = 'jsd',
            panel: dict = None):
        """
        Run parallel tempering replica exchange.

        Parameters
        ----------
        iter_length : float
            Time between replica swap attempts, in nanoseconds.
        dt : float
            Integration timestep in femtoseconds. Default 2.0.
        sim_length : int
            Duration of each sub-simulation in nanoseconds. Controls how often
            trajectory files are truncated, data is saved, resampling occurs,
            and convergence is evaluated.
            Recomended Value: int(10*timestep)
        total_sim_time : int, optional
            Total aggregate simulation time across all replicates, in
            nanoseconds. Used as a terminal stopping criterion if reached.
        init_overlap_thresh : float, optional
            Minimum acceptance rate during the first sub-simulation before
            triggering a restart. None (default) resolves to 0.5, or to the
            walk's ``min_acc`` when the ladder was trailblazed.
        term_overlap_thresh : float, optional
            Minimum acceptance rate at any point after the first sub-simulation
            before triggering a restart. None (default) resolves to 0.35, or to
            the walk's ``min_acc`` when the ladder was trailblazed.
        insertion_rule : str, optional
            ``'mean'`` (the historical test: insert when the block's mean
            acceptance so far is below threshold) or ``'ucb'`` (insert only
            when the mean plus two standard errors across iterations is below
            threshold, after at least ten iterations) or ``'trailing'`` (no
            mid-block insertion; after each block, a pair whose acceptance
            pooled over the trailing blocks with the same ladder, at least
            ``insertion_min_blocks`` of them, is below threshold by more than
            two standard errors gets a rung before the next block, so nothing
            is discarded). None (default) resolves to ``'mean'`` for geometric
            and file ladders and ``'trailing'`` for a trailblazed one, whose
            rungs were already measured at target.
        insertion_min_blocks : int
            Blocks the ``'trailing'`` rule needs before it may insert. Default 3.
        output_dir : str
            Path to output directory. Default: 'FultonMarket_output/' in cwd.
        n_resample : int
            Number of frames to importance resample per convergence check.
            Default 1000.
        max_equil_fraction : float
            Maximum fraction of the simulation that can be discarded as
            equilibration before convergence is considered unreliable.
            Default 0.75.
        frobenius_thresh : float
            Maximum normalised Frobenius norm between the current and any
            previous checkpoint distance matrix for convergence. Default 0.05.
        jsd_thresh : float
            Maximum Jensen-Shannon Divergence between the current and any
            previous checkpoint pairwise distance distribution for convergence.
            Default 0.10.
        getContacts_Info : dict, optional
            Keyword arguments forwarded to getContactDistanceMatrix.
            Required keys: ``getcontacts_script`` (path to
            get_dynamic_contacts.py), ``conda_env`` (conda env name
            containing getContacts). Optional keys: ``getcontacts_python``
            (explicit interpreter path), ``cores`` (CPU cores, default 10).
            If None and skip_contacts is False, contact matrix computation
            will raise a RuntimeError when first attempted.
        skip_contacts : bool
            If True, skip the contact distance matrix in convergence checking.
            Contact convergence is then treated as passing. Must be explicitly
            set to True — contacts are required by default. Default False.
        """

        # Store run parameters
        self.total_sim_time = total_sim_time
        self.iter_length = iter_length
        self.dt = dt
        self.sim_length = sim_length
        self.init_overlap_thresh = init_overlap_thresh
        self.term_overlap_thresh = term_overlap_thresh
        self.insertion_rule = insertion_rule
        self.insertion_min_blocks = int(insertion_min_blocks)
        self._pending_insert_after = []
        self.minimum_fraction4convergence = minimum_fraction
        self.n_resample = n_resample
        self.max_equil_fraction = max_equil_fraction
        self.min_ess = min_ess
        self.equil_method = equil_method
        self.equil_monotone = equil_monotone
        self.min_pool_frames = min_pool_frames
        self.equil_freeze_after = equil_freeze_after
        self.window_fraction = window_fraction
        self.window_blocks = window_blocks
        self.frobenius_thresh = frobenius_thresh
        self.jsd_thresh = jsd_thresh
        self.getContacts_Info = getContacts_Info if getContacts_Info is not None else {}
        self.skip_contacts = skip_contacts
        self.convergence = convergence
        self.panel_kwargs = dict(panel or {})
        self.live_panel = None

        # Prepare output directories
        self.output_dir = output_dir
        self.name = output_dir.split('/')[-1]
        if self.protein_name and self.ligand_name:
            self.output_ncdf = build_output_path(output_dir, self.protein_name, self.ligand_name, 'trajectory', 'ncdf')
        else:
            self.output_ncdf = os.path.join(output_dir, 'output.ncdf')
        # Derive the checkpoint path exactly as MultiStateReporter does when it is
        # constructed with checkpoint_storage=None (multistatereporter.py:138-141):
        # splitext the storage name and splice in '_checkpoint'. Do not spell this
        # name independently. Randolph.save_simulation opens it by path at the end
        # of every block, so a divergence between the two spellings is invisible
        # until a full block of sampling has already been paid for.
        _stem, _ext = os.path.splitext(self.output_ncdf)
        self.checkpoint_ncdf = _stem + '_checkpoint' + _ext
        self.save_dir = os.path.join(output_dir, 'saved_variables')
        os.makedirs(self.save_dir, exist_ok=True)

        # The trajectory NetCDF is scratch, not a completion marker: Randolph
        # deletes it on every ladder rebuild (_remove_ncdf) and MultiStateReporter
        # reopens it with mode='w'. Durable state lives in saved_variables/.
        # Gating the whole run on this file made every wall-clock resubmit a
        # silent no-op that still exited 0 -- indistinguishable from success.
        # Only bail out when there is genuinely nothing to resume from.
        if not os.listdir(self.save_dir) and file_exists_skip(self.output_ncdf, 'FultonMarket trajectory'):
            print('Trajectory exists but saved_variables/ is empty, so there is no '
                  'block to resume from. Move or delete the trajectory to re-run.')
            return

        printf(f'total_sim_time      : {self.total_sim_time} ns')
        printf(f'iter_length         : {self.iter_length} ns')
        printf(f'dt                  : {self.dt} fs')
        printf(f'sim_length          : {self.sim_length} ns')
        printf(f'n_replicates        : {self.n_replicates}')
        printf(f'init_overlap_thresh : {self.init_overlap_thresh}')
        printf(f'term_overlap_thresh : {self.term_overlap_thresh}')
        printf(f'output_dir          : {self.output_dir}')
        printf(f'temperatures        : {[np.round(T._value, 1) for T in self.temperatures]} K')
        printf(f'n_resample          : {self.n_resample}')
        printf(f'max_equil_fraction  : {self.max_equil_fraction}')
        printf(f'min_ess             : {self.min_ess}')
        printf(f'equil_method        : {self.equil_method}')
        printf(f'equil_monotone      : {self.equil_monotone}')
        printf(f'min_pool_frames     : {self.min_pool_frames}')
        printf(f'equil_freeze_after  : {self.equil_freeze_after}')
        printf(f'window_fraction     : {self.window_fraction}')
        printf(f'window_blocks       : {self.window_blocks}')
        printf(f'frobenius_thresh    : {self.frobenius_thresh}')
        printf(f'jsd_thresh          : {self.jsd_thresh}')
        printf(f'getContacts_Info    : {self.getContacts_Info}')
        printf(f'skip_contacts       : {self.skip_contacts}')
        printf(f'convergence         : {self.convergence}')
        printf(f'panel               : {self.panel_kwargs or None}')

        self._configure_experiment_parameters()
        self._resolve_ladder()
        self._resolve_overlap_thresholds()
        printf(f'ladder              : {self.ladder_mode}, {self.n_replicates} rungs '
               f'{self.temperatures[0]._value:.2f}-{self.temperatures[-1]._value:.2f} K; '
               f'insertion thresholds {self.init_overlap_thresh}/{self.term_overlap_thresh} '
               f'(rule {self.insertion_rule})')
        self._setup_panel()

        while not self.finished:
            if self.sim_no > 0:
                self._load_initial_args()
                self._apply_pending_insertions()

            self._build_states()
            self._set_parameters()

            self.simulation = Randolph(**self.params)
            self.simulation.main(init_overlap_thresh=self.init_overlap_thresh,
                                 term_overlap_thresh=self.term_overlap_thresh)

            self._save_sub_simulation()
            self._audit_ladder()
            if self.insertion_rule == 'trailing':
                self._pending_insert_after = list(self._plan_insertions())
            at_max_time = (self.total_sim_time is not None
                           and self.sim_no >= self.total_n_sims)
            # The published criterion runs in every mode: it writes the
            # resampled trajectory and distance matrices the campaign reports,
            # and its verdict is printed even when it does not decide.
            jsd_stop = self._evaluate_stopping_criterion(
                n_resample=self.n_resample,
                max_equil_fraction=self.max_equil_fraction,
                min_ess=self.min_ess,
                frobenius_thresh=self.frobenius_thresh,
                jsd_thresh=self.jsd_thresh,
                getContacts_Info=self.getContacts_Info,
                skip_contacts=self.skip_contacts,
            )
            panel_stop = None
            if self.live_panel is not None and not at_max_time:
                panel_stop = self._evaluate_panel()
            from chimpss.analysis.convergence_panel import combine_stop
            self.finished = combine_stop(self.convergence, jsd_stop, panel_stop, at_max_time)
            if self.live_panel is not None:
                printf(f"Stop decision ({self.convergence}): jsd={jsd_stop} panel={panel_stop} "
                       f"max_time={at_max_time} -> {'STOP' if self.finished else 'continue'}")
            self.sim_no += 1


    # ------------------------------------------------------------------
    # Ladder resolution
    # ------------------------------------------------------------------
    @property
    def trailblaze_dir(self):
        return os.path.join(self.output_dir, 'trailblaze')

    def _resolve_ladder(self):
        """Settle the block-0 ladder once ``output_dir`` and ``sim_no`` are known.

        ``'trailblaze'`` walks the ladder on rank 0 (or reuses a complete cache
        under ``trailblaze/``) when no block has been saved yet; on resume it
        only reads the walk's metadata, because ``_load_initial_args`` takes
        the ladder from the last saved block.  The other modes need nothing.
        """
        if self.ladder_mode != 'trailblaze':
            return
        if self.sim_no == 0:
            temps, meta = self._trailblaze_rank0()
            self.temperatures = [float(t) * unit.kelvin for t in temps]
            self.n_replicates = len(temps)
            self._set_init_positions()
            self._set_init_box_vectors()
            self.ladder_meta = dict(meta)
        else:
            self.ladder_meta = self._read_trailblaze_meta()

    def _read_trailblaze_meta(self):
        from chimpss.fultonmarket.trailblaze import RECORD_NAME, TrailblazeRecord
        path = os.path.join(self.trailblaze_dir, RECORD_NAME)
        rec = TrailblazeRecord.load_if_complete(path) if os.path.exists(path) else None
        if rec is None:
            printf(f'WARNING: ladder=trailblaze but no complete {path}; insertion thresholds fall back '
                   f'to the geometric defaults')
            return {}
        return dict(rec.meta)

    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)
    def _trailblaze_rank0(self):
        """The walked ladder as plain floats plus its metadata; cached across launches."""
        from chimpss.fultonmarket.trailblaze import (
            RECORD_NAME,
            OpenMMRungSampler,
            TrailblazeRecord,
            run_trailblaze,
            split_kwargs,
        )
        os.makedirs(self.trailblaze_dir, exist_ok=True)
        cached = TrailblazeRecord.load_if_complete(os.path.join(self.trailblaze_dir, RECORD_NAME))
        if cached is not None:
            printf(f'trailblaze: using the cached {cached.direction} ladder of {len(cached.rungs)} rungs in '
                   f'{self.trailblaze_dir}')
            return cached.temperatures.tolist(), cached.meta
        if not hasattr(self, 'context'):
            raise RuntimeError('ladder=trailblaze needs input_state: the walk starts from its configuration')
        sampler_kwargs, walk_kwargs = split_kwargs(self.trailblaze_kwargs)
        sampler = OpenMMRungSampler(self.system, dt_fs=self.dt, **sampler_kwargs)
        seed = sampler.seed_from_sampler_state(SamplerState.from_context(self.context))
        # direction 'up' (default) or 'down' is one walk; 'both' walks up, then down from the hot
        # end, and reconciles the two into the ladder and seeds FultonMarket reads from trailblaze/
        rec = run_trailblaze(sampler, self.T_min, self.T_max, seed, out_dir=self.trailblaze_dir, printf=printf,
                             **walk_kwargs)
        return rec.temperatures.tolist(), rec.meta

    def trailblaze(self, output_dir: str, dt: float = 2.0):
        """Run (or reuse) the walk alone and return ``(temperatures, meta)``.

        What ``run()`` does before block 0 with ``ladder='trailblaze'``, as a
        separate step so a batch job can pay for the walk on one GPU and the
        production job then finds the cache.
        """
        if self.ladder_mode != 'trailblaze':
            raise ValueError("trailblaze() needs ladder='trailblaze'")
        self.output_dir = output_dir
        self.dt = dt
        os.makedirs(output_dir, exist_ok=True)
        temps, meta = self._trailblaze_rank0()
        self.temperatures = [float(t) * unit.kelvin for t in temps]
        self.n_replicates = len(temps)
        self._set_init_positions()
        self._set_init_box_vectors()
        self.ladder_meta = dict(meta)
        return temps, meta

    def _resolve_overlap_thresholds(self):
        """None means: the geometric defaults, or the walk's floor for a trailblazed ladder."""
        from chimpss.analysis.ladder import INSERTION_RULES
        floor = self.ladder_meta.get('min_acc')
        if self.init_overlap_thresh is None:
            self.init_overlap_thresh = float(floor) if floor is not None else 0.5
        if self.term_overlap_thresh is None:
            self.term_overlap_thresh = float(floor) if floor is not None else 0.35
        if self.insertion_rule is None:
            self.insertion_rule = 'trailing' if floor is not None else 'mean'
        if self.insertion_rule not in INSERTION_RULES:
            raise ValueError(f'unknown insertion_rule {self.insertion_rule!r}; choose from {INSERTION_RULES}')
        if self.ladder_mode == 'file' and floor is None:
            printf('NOTE: a file ladder carries no acceptance target; pass init/term_overlap_thresh '
                   'if the geometric defaults (0.5/0.35) would insert rungs it does not need')

    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)
    def _plan_insertions(self):
        """Pairs to split before the next block, from the trailing per-block audits (rank 0)."""
        try:
            from chimpss.analysis.ladder import insertion_from_audits
            res = insertion_from_audits(self.output_dir, self.sim_no, self.term_overlap_thresh,
                                        min_blocks=self.insertion_min_blocks)
        except Exception as exc:
            printf(f'WARNING: trailing insertion test could not run for block {self.sim_no}: {exc}')
            return []
        if res['n_blocks'] < self.insertion_min_blocks:
            printf(f"Trailing insertion test deferred: {res['n_blocks']} of {self.insertion_min_blocks} blocks "
                   f"on this ladder")
            return []
        T = res['temperatures']
        for k in res['insert_after']:
            printf(f"Trailing insertion: pair {T[k]:.2f}-{T[k + 1]:.2f} K reads {res['mean'][k]:.3f} "
                   f"(upper {res['upper'][k]:.3f}) over {res['n_blocks']} blocks; a rung will be added "
                   f"before the next block")
        if res['insert_after'].size == 0:
            printf(f"Trailing insertion test over {res['n_blocks']} block(s): no pair below "
                   f"{self.term_overlap_thresh} by more than two standard errors")
        return [int(k) for k in res['insert_after']]

    def _apply_pending_insertions(self):
        """Add the rungs the trailing audit asked for, seeded from the rung below, before the block starts."""
        if not self._pending_insert_after:
            return
        insert_inds = np.array([k + 1 for k in self._pending_insert_after], dtype=int)
        self._pending_insert_after = []
        prev_temps = np.array([t._value for t in self.temperatures])
        self.temperatures, self.n_replicates = _interpolate_new_states(prev_temps, insert_inds)
        velos = None if self.init_velocities is None else np.asarray(self.init_velocities._value)
        self.init_positions, self.init_box_vectors, self.init_velocities = _interpolate_new_positions(
            np.asarray(self.init_positions._value), np.asarray(self.init_box_vectors._value), velos,
            insert_inds, self.n_replicates)
        printf(f'Ladder now {self.n_replicates} rungs after inserting at {[round(float(t._value), 2) for i, t in enumerate(self.temperatures) if i in set(insert_inds + np.arange(len(insert_inds)))]} K')

    def _set_parameters(self):
        """
        Assemble the keyword-argument dictionary passed to the Randolph
        sampler constructor.

        Sets
        ----
        self.params : dict
            All simulation parameters required to instantiate a Randolph obj.
        """
        self.params = dict(sampler_states=self.sampler_states,
                           reference_thermo_state=self.thermodynamic_states[0],
                           sim_no=self.sim_no,
                           sim_time=self.sim_length,
                           temperatures=self.temperatures,
                           output_dir=self.output_dir,
                           output_ncdf=self.output_ncdf,
                           checkpoint_ncdf=self.checkpoint_ncdf,
                           iter_length=self.iter_length,
                           dt=self.dt,
                           insertion_rule='off' if self.insertion_rule == 'trailing' else self.insertion_rule)


    def _build_states(self):
        """
        Build both sampler and thermodynamic states required by Randolph.

        Delegates to `_build_sampler_states` and
        `_build_thermodynamic_states`.
        """
        self._build_sampler_states()
        self._build_thermodynamic_states()


    def _build_sampler_states(self):
        """
        Construct OpenMMTools SamplerState objects for each replica.

        On the first sub-simulation (sim_no == 0), positions and box vectors
        are read directly from the OpenMM Context (populated from the input
        state file). On subsequent sub-simulations they are loaded from saved
        numpy arrays and velocities are preserved.

        Sets
        ----
        self.sampler_states : list of SamplerState
            One SamplerState per replica, containing positions, box vectors,
            and (after sim_no > 0) velocities.
        """
        if self.sim_no == 0 and self.ladder_mode == 'trailblaze' and self._trailblaze_seeds_available():
            printf('Setting initial positions from the trailblaze seeds (one equilibrated configuration per rung).')
            self.sampler_states = self._sampler_states_from_seeds()
        elif self.sim_no == 0:
            printf('Setting initial positions via Context method.')
            self.sampler_states = [SamplerState(positions=self.init_positions,
                                                box_vectors=self.init_box_vectors).from_context(self.context)
                                   for _ in range(self.n_replicates)]
        else:
            printf('Setting initial positions via Velocity method.')
            self.sampler_states = build_sampler_states(self.n_replicates,
                                                       self.init_positions,
                                                       self.init_box_vectors,
                                                       self.init_velocities)


    def _trailblaze_seeds_available(self):
        from chimpss.fultonmarket.trailblaze import SeedStore
        return SeedStore(self.trailblaze_dir).has_all(self.n_replicates)

    def _sampler_states_from_seeds(self):
        """One SamplerState per rung from the walk's seeds, so block 0 starts each replica
        from a configuration already equilibrated at its own temperature."""
        from chimpss.fultonmarket.trailblaze import SeedStore
        store = SeedStore(self.trailblaze_dir)
        states = []
        for k in range(self.n_replicates):
            seed, _ = store.load(k)
            if seed.positions.shape[0] != self.n_atoms:
                raise RuntimeError(f'trailblaze seed {k} has {seed.positions.shape[0]} atoms, system has {self.n_atoms}')
            kwargs = dict(positions=seed.positions * unit.nanometer,
                          box_vectors=seed.box_vectors * unit.nanometer)
            if seed.velocities is not None:
                kwargs['velocities'] = seed.velocities * (unit.nanometer / unit.picosecond)
            states.append(SamplerState(**kwargs))
        return states

    def _build_thermodynamic_states(self):
        """
        Construct the OpenMMTools ThermodynamicState for the reference
        (lowest-temperature) replica, if not already initialised.

        The thermodynamic states for the remaining replicas are managed
        internally by Randolph using `self.temperatures`. This method only
        runs on the first call; subsequent calls are no-ops.

        Sets
        ----
        self.thermodynamic_states : list of ThermodynamicState
            Single-element list containing the NPT state at T_min and 1 bar.
        """
        if not hasattr(self, 'thermodynamic_states'):
            self.thermodynamic_states = [ThermodynamicState(system=self.system,
                                                            temperature=self.temperatures[0],
                                                            pressure=1.0 * unit.bar)]


    def _save_sub_simulation(self):
        """
        Persist the completed sub-simulation to disk via Randolph and update
        replica metadata.

        Calls `Randolph.save_simulation`, which writes positions, velocities,
        box vectors, state indices, and temperatures to the save directory,
        then updates `self.n_replicates` and `self.temperatures` to reflect
        any changes made during the run (e.g. replica insertion).

        Sets
        ----
        self.n_replicates : int
            Updated replica count after saving.
        self.temperatures : list of Quantity
            Updated temperature ladder after saving.
        """
        self.n_replicates, self.temperatures = self.simulation.save_simulation(self.save_dir)


    def _load_initial_args(self):
        """
        Load positions, velocities, box vectors, and state indices from the
        most recently completed sub-simulation, then reorder them from replica
        index to state (temperature) index ready for the next sub-simulation.

        Positions are loaded as a memory-mapped array if a direct `np.load`
        fails (e.g. large files). If numpy loading fails entirely, arguments
        are recovered directly from the NetCDF trajectory via
        `_recover_arguments`.

        Sets
        ----
        self.temperatures : list of Quantity
            Temperature ladder in Kelvin for the upcoming sub-simulation.
        self.n_replicates : int
            Number of replicas for the upcoming sub-simulation.
        self.init_positions : TrackedQuantity
            Positions ordered by state index, shape (n_replicates, n_atoms, 3)
            in nanometers.
        self.init_box_vectors : TrackedQuantity
            Box vectors ordered by state index, shape (n_replicates, 3, 3)
            in nanometers.
        self.init_velocities : TrackedQuantity or None
            Velocities ordered by state index in nm/ps, or None if not saved.
        """
        load_dir = os.path.join(self.save_dir, str(self.sim_no - 1))
        self.temperatures = np.load(os.path.join(load_dir, 'temperatures.npy'))
        self.temperatures = [t * unit.kelvin for t in self.temperatures]
        self.n_replicates = len(self.temperatures)
        printf(f'Loaded n_replicates: {self.n_replicates}')

        try:
            box_vectors = np.load(os.path.join(load_dir, 'box_vectors.npy'))
            n_frames = box_vectors.shape[0]
            init_box_vectors = box_vectors[-1]
            positions_path = os.path.join(load_dir, 'positions.npy')
            try:
                init_positions = np.load(positions_path)[-1]
            except Exception:
                init_positions = np.array(np.memmap(positions_path, mode='r', dtype='float32',
                                                    shape=(n_frames, self.n_replicates, self.n_atoms, 3))[-1])
            velocities_path = os.path.join(load_dir, 'velocities.npy')
            init_velocities = np.load(velocities_path) if os.path.exists(velocities_path) else None
            state_inds = np.load(os.path.join(load_dir, 'states.npy'))[-1]
        except Exception as exc:
            # This fallback re-reads the last frame out of the NetCDF instead of
            # the saved arrays. It used to fire silently, so a run could resume
            # from the fallback path for every block and look completely normal.
            printf(f'WARNING: could not load saved arrays from {load_dir} '
                   f'({type(exc).__name__}: {exc}); falling back to _recover_arguments() '
                   f'against {self.output_ncdf}.')
            init_velocities, init_positions, init_box_vectors, state_inds = self._recover_arguments()

        # Reorder arrays from replica index to state index
        reshaped_positions = np.empty_like(init_positions)
        reshaped_box_vectors = np.empty_like(init_box_vectors)
        for state in range(self.n_replicates):
            rep_ind = np.where(state_inds == state)[0]
            reshaped_positions[state] = init_positions[rep_ind]
            reshaped_box_vectors[state] = init_box_vectors[rep_ind]
        self.init_positions = convert_to_TrackedQuantity(reshaped_positions, unit.nanometer)
        self.init_box_vectors = convert_to_TrackedQuantity(reshaped_box_vectors, unit.nanometer)

        if init_velocities is not None:
            reshaped_velocities = np.empty_like(init_velocities)
            for state in range(self.n_replicates):
                rep_ind = np.where(state_inds == state)[0]
                reshaped_velocities[state] = init_velocities[rep_ind]
            self.init_velocities = convert_to_TrackedQuantity(reshaped_velocities, unit.nanometer / unit.picosecond)
        else:
            self.init_velocities = None


    def _configure_experiment_parameters(self):
        """
        Determine how many sub-simulations have already completed, validate
        the save directory, and set the initial stopping-criterion state.

        Raises
        ------
        RuntimeError
            If any sub-simulation save directory contains fewer than 5 files,
            indicating an incomplete or corrupted save.

        Sets
        ----
        self.sim_no : int
            Index of the next sub-simulation to run (0-based).
        self.total_n_sims : int
            Total number of sub-simulations required (only set when
            `total_sim_time` is not None).
        self.finished : bool
            Whether the stopping criterion is already satisfied before the
            main loop begins.
        self.converged : bool
            Initialised to False when resuming; updated by
            `_evaluate_stopping_criterion`.
        """
        # Validate saved blocks by NAME, not by file count. save_simulation writes
        # six arrays in a fixed order and is not atomic, so a crash partway through
        # can leave five files behind -- which the old `len(...) < 5` test accepted
        # as complete. The block would then resume with a missing or half-written
        # array and fail much later, or worse, quietly use garbage.
        block_dirs = sorted(d for d in os.listdir(self.save_dir)
                            if os.path.isdir(os.path.join(self.save_dir, d)))
        incomplete = {}
        for d in block_dirs:
            missing = [f for f in REQUIRED_BLOCK_ARRAYS
                       if not os.path.exists(os.path.join(self.save_dir, d, f))]
            if missing:
                incomplete[d] = missing
        if incomplete:
            raise RuntimeError(
                f'Incomplete save directories detected: {incomplete}. '
                f'Each block must contain all of {list(REQUIRED_BLOCK_ARRAYS)}. '
                f'Remove or fix them before continuing.')

        # Count directories only. A stray file in saved_variables/ used to inflate
        # sim_no and silently skip a block of production.
        self.sim_no = len(block_dirs)
        printf(f'Resuming from sim_no: {self.sim_no}')
        if self.total_sim_time is not None:
            self.total_n_sims = int(np.ceil(self.total_sim_time / self.sim_length))
            printf(f'Total sub-simulations required: {self.total_n_sims}')
        self.finished = False
        if self.sim_no > 0:
            self.converged = False

    def _recover_arguments(self):
        """
        Fall-back loader that reads the final frame of each trajectory
        variable directly from the NetCDF output file.

        Used when the numpy save files are unavailable or corrupt. Reads
        from `self.output_ncdf`.

        Returns
        -------
        velocities : np.ndarray
            Velocities of the last frame, shape (n_replicates, n_atoms, 3).
        positions : np.ndarray
            Positions of the last frame, shape (n_replicates, n_atoms, 3).
        box_vectors : np.ndarray
            Box vectors of the last frame, shape (n_replicates, 3, 3).
        state_inds : np.ndarray
            State index for each replica in the last frame, shape
            (n_replicates,).
        """
        with nc.Dataset(self.output_ncdf, 'r') as ncfile:
            velocities = ncfile.variables['velocities'][-1].data
            positions = ncfile.variables['positions'][-1].data
            box_vectors = ncfile.variables['box_vectors'][-1].data
            state_inds = ncfile.variables['states'][-1].data
        return velocities, positions, box_vectors, state_inds


    # Runs on rank 0 only, exactly as Randolph.save_simulation does. Without this
    # every rank executed the whole convergence analysis against the SAME output
    # paths -- four concurrent get_dynamic_contacts.py processes writing one
    # resampled_contacts.tsv, plus four writers each of resampled_trj.dcd,
    # resampled_top.pdb, resampled_wghts.npy, resampled_indcs.npy and the three
    # distance matrices. The race is nondeterministic: block 1 of run 53753497
    # happened to win it (1000 frames, 112 MB) and block 2 lost, leaving a
    # 1-frame TSV that collapsed the contact matrix to (1, 1) and killed the job
    # on a shape mismatch after eight hours of sampling. broadcast_result puts
    # the same verdict on every rank so they agree on whether to stop.
    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)
    def _frozen_t0(self):
        """The stored high-water t0, or None when nothing has been recorded yet.

        Reads the same state file the ratchet writes, so freezing and ratcheting
        compose: the ratchet keeps t0 from regressing while the run is young,
        and the freeze stops re-deriving it once the estimate has settled.
        """
        import json

        path = os.path.join(self.output_dir, 'equilibration_state.json')
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                return int(json.load(fh)['t0_high_water'])
        except Exception as exc:
            printf(f'WARNING: could not read the frozen t0 from {path} ({exc}).')
            return None


    def _ratchet_t0(self, t0):
        """Forbid the equilibration index from moving backwards between blocks.

        detect_equilibration maximises (T - t0)/g(t0), an objective that is not
        guaranteed unimodal; where the autocorrelation estimate is noisy it has
        competing maxima and the returned t0 alternates between them as the
        series grows.  Because t0 sets the first block of the comparison
        window, a downward revision re-admits blocks that had already aged out,
        so the criterion has no ratchet and progress is handed back.

        The high-water mark lives at the run root rather than inside
        saved_variables/, because FultonMarketAnalysis globs that directory and
        int()s each basename -- a stray file there would crash every subsequent
        analysis.
        """
        import json

        path = os.path.join(self.output_dir, 'equilibration_state.json')
        previous = None
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    previous = json.load(fh).get('t0_high_water')
            except Exception as exc:
                printf(f'WARNING: could not read {path} ({exc}); starting the '
                       f'ratchet from this block.')

        kept = int(t0) if previous is None else max(int(t0), int(previous))
        if previous is not None and kept != int(t0):
            printf(f'Equilibration ratchet: detector returned t0={int(t0)}, '
                   f'holding the high-water mark t0={kept}.')
        try:
            with open(path, 'w') as fh:
                json.dump({'t0_high_water': kept, 'sim_no': self.sim_no}, fh)
        except Exception as exc:
            printf(f'WARNING: could not persist the equilibration ratchet ({exc}).')
        return kept


    def _setup_panel(self):
        """Build the live convergence panel when a mode other than 'jsd' is requested.

        Preparation resolves the conserved atoms against the topology and the
        tolerances against the training set.  It runs on rank 0 and the
        resulting config is broadcast, so every rank holds the same atom set
        and the same thresholds; a construct that cannot be numbered fails
        here, before the first block is paid for.
        """
        from chimpss.analysis.convergence_panel import STOP_MODES, LivePanel

        if self.convergence not in STOP_MODES:
            raise ValueError(f'unknown convergence mode {self.convergence!r}; '
                             f'choose from {STOP_MODES}')
        if self.convergence == 'jsd':
            self.live_panel = None
            return
        required = ('model_prefix',)
        missing = [k for k in required if not self.panel_kwargs.get(k)]
        if missing:
            raise ValueError(f"convergence={self.convergence!r} needs panel settings "
                             f"{missing} (chimpss-fultonmarket --panel_pangpcr_prefix ...)")
        self.live_panel = LivePanel(run_dir=self.output_dir, topology=self.input_pdb,
                                    printf=printf, **self.panel_kwargs)
        cfg = self._panel_prepare()
        self.live_panel.apply_config(cfg)
        printf(f"convergence panel armed: stop when CONVERGED for "
               f"{self.live_panel.consecutive} consecutive blocks "
               f"(mode {self.convergence!r})")

    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)
    def _panel_prepare(self):
        cfg = self.live_panel.prepare()
        # catch up on blocks a previous submission saved but never ingested
        new = self.live_panel.ingest_all()
        if new:
            printf(f"convergence panel: ingested {len(new)} existing block(s)")
        return cfg

    @mpiplus.on_single_node(rank=0, sync_nodes=True)
    def _audit_ladder(self):
        """Record the block's per-pair exchange acceptance and thermodynamic length.

        Randolph prints the per-pair acceptance it reads from the reporter and
        then the NetCDF is truncated, so nothing on disk said how well the
        ladder mixed.  This reads the block just saved and writes
        ``analysis/ladder/<block>.json`` (chimpss.analysis.ladder.block_audit):
        the exact Metropolis acceptance of the configurations that sat at each
        neighbour pair, the pair's thermodynamic length from the enthalpy
        fluctuation, and a trailing pooled acceptance with less noise than one
        block.  A diagnostic, never a reason to lose the sampling.
        """
        try:
            import json

            from chimpss.analysis.ladder import block_audit

            out_dir = os.path.join(self.output_dir, 'analysis', 'ladder')
            os.makedirs(out_dir, exist_ok=True)
            record = block_audit(self.output_dir, self.sim_no, threshold=self.term_overlap_thresh)
            path = os.path.join(out_dir, f'{self.sim_no:04d}.json')
            tmp = f'{path}.tmp.{os.getpid()}'
            with open(tmp, 'w') as fh:
                json.dump(record, fh)
            os.replace(tmp, path)
            low = record.get('pairs_below_threshold', [])
            printf(f"ladder block {self.sim_no}: neighbour acceptance min/mean/max "
                   f"{record['A_min']:.3f}/{record['A_mean']:.3f}/{record['A_max']:.3f} "
                   f"over {record['n_iterations_used']} iterations; trailing "
                   f"{len(record['trailing_blocks'])} block(s): {len(low)} pair(s) below "
                   f"{self.term_overlap_thresh}")
        except Exception as exc:
            printf(f'WARNING: ladder audit failed for block {self.sim_no}: {exc}')

    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)
    def _evaluate_panel(self):
        """Ingest the block just saved, evaluate the panel, print, return stop."""
        try:
            self.live_panel.ingest_all()
            past_minimum = (self.total_sim_time is None
                            or self.sim_no >= self.total_n_sims * self.minimum_fraction4convergence)
            result = self.live_panel.evaluate(sim_no=self.sim_no, past_minimum=past_minimum)
            self.live_panel.report(result)
            return bool(result['stop'])
        except Exception as exc:
            # The panel is a stopping rule, not a reason to lose the sampling:
            # a block that cannot be evaluated is reported and the run goes on.
            printf(f"WARNING: convergence panel could not evaluate block {self.sim_no}: {exc}")
            return False

    def _evaluate_stopping_criterion(self, n_resample=1000, max_equil_fraction=0.75, frobenius_thresh=0.05, jsd_thresh=0.10, getContacts_Info=None, skip_contacts=False, min_ess=None):
        """
        Check whether the simulation should stop by evaluating a series of
        convergence criteria against the current and all previously saved
        resampled distance matrices.

        For each of the three distance matrix types (torsional, alpha-carbon,
        contact), two complementary metrics are computed against every previous
        checkpoint:

        - **Frobenius norm** — element-wise structural divergence between the
          full matrices. Sensitive to the magnitude of changes.
        - **Jensen-Shannon Divergence** — divergence between the distributions
          of pairwise distances (upper triangle). Sensitive to the shape of the
          conformational ensemble distribution, robust to outliers.

        Both metrics must be below their respective thresholds across all
        previous checkpoints for a matrix type to be declared converged.

        Parameters
        ----------
        n_resample : int
            Number of frames to importance resample. Default 1000.
        max_equil_fraction : float
            Maximum fraction of the simulation discardable as equilibration
            before convergence is considered unreliable. Default 0.75.
        frobenius_thresh : float
            Maximum normalised Frobenius norm between the current and any
            previous checkpoint matrix for convergence. Default 0.05.
        jsd_thresh : float
            Maximum Jensen-Shannon Divergence between the current and any
            previous checkpoint pairwise distance distribution for convergence.
            Bounded in [0, 1]. Default 0.10.
        getContacts_Info : dict, optional
            Keyword arguments forwarded directly to getContactDistanceMatrix.
            See run() docstring for accepted keys. Default None.

        Returns
        -------
        bool
            True if the simulation should stop.
        """
        at_max_time = self.total_sim_time is not None and self.sim_no >= self.total_n_sims
        if at_max_time:
            printf("Max simulation time reached — skipping convergence checks.")
            return True

        printf("Gathering convergence related data...")
        analyzer = FultonMarketAnalysis(input_dir=self.output_dir, pdb=self.input_pdb, sele_str=self.sele_str)

        # Equilibration
        ave_ener = analyzer.get_average_energy()
        # 'max_neff' is detect_equilibration, i.e. the published behaviour.
        # Anything else selects an alternative estimator from
        # chimpss.analysis.equilibration; see that module for the validation
        # of each against series with a known truncation point.
        if self.equil_method in (None, 'max_neff'):
            t0 = detect_energy_equil(ave_ener)
        else:
            from chimpss.analysis.equilibration import DETECTORS
            if self.equil_method not in DETECTORS:
                raise ValueError(f'unknown equil_method {self.equil_method!r}; '
                                 f'choose from {sorted(DETECTORS)}')
            t0 = DETECTORS[self.equil_method](ave_ener)
        if self.equil_freeze_after is not None and self.sim_no >= self.equil_freeze_after:
            frozen = self._frozen_t0()
            if frozen is not None:
                if frozen != t0:
                    printf(f'Equilibration frozen after block '
                           f'{self.equil_freeze_after}: detector returned t0={t0}, '
                           f'holding t0={frozen}.')
                t0 = frozen
        if self.equil_monotone:
            t0 = self._ratchet_t0(t0)
        if self.min_pool_frames:
            from chimpss.analysis.equilibration import bound_pool
            capped = bound_pool(t0, ave_ener.shape[0], self.min_pool_frames)
            if capped != t0:
                printf(f'Pool bound: t0={t0} would leave '
                       f'{ave_ener.shape[0] - t0} frames, below the required '
                       f'{self.min_pool_frames}; capping t0 to {capped}.')
                t0 = capped
        equil_fraction = t0 / ave_ener.shape[0]
        printf(f"Currently would discard {100*equil_fraction:.2f}% to energy equilibration")

        # Importance resampling and trajectory
        sim_dir = os.path.join(self.output_dir, 'saved_variables', str(self.sim_no))
        try:
            _ = analyzer.importance_resampling(n_samples=n_resample, min_ess=min_ess)
        except MBARWeightsDegenerate as exc:
            # A convergence check that cannot be computed is not a reason to
            # destroy the sampling. Three runs (53855971, 53855972, 53856114)
            # died here after 17 h to 1 d 22 h of production because the raw
            # numpy ValueError propagated out through MPI_Abort. The block's six
            # state arrays are already saved and resumable; report the block as
            # not-converged and carry on, leaving the operator to decide whether
            # a ladder this fine is worth reweighting at all.
            printf(f"WARNING: convergence check skipped for block {self.sim_no}: {exc}")
            printf("WARNING: sampling continues; this block contributes no "
                   "convergence comparison and has no resampled_* artifacts.")
            return False
        traj = analyzer.write_resampled_traj(
            pdb_out=os.path.join(sim_dir, 'resampled_top.pdb'),
            dcd_out=os.path.join(sim_dir, 'resampled_trj.dcd'),
            weights_out=os.path.join(sim_dir, 'resampled_wghts.npy'),
            inds_out=os.path.join(sim_dir, 'resampled_indcs.npy'),
            return_traj=True,
        )

        # Compute current distance matrices
        torsional    = getTorsionalDistanceMatrix(traj, selection_string='protein or resname UNK')
        alpha_carbon = getAlphaCarbonDistanceMatrix(traj, selection_string='protein or resname UNK')
        current_matrices = {
            'torsion':      torsional,
            'alpha_carbon': alpha_carbon,
        }

        if not skip_contacts:
            contact_distance, _ = getContactDistanceMatrix(
                top_fn=os.path.join(sim_dir, 'resampled_top.pdb'),
                traj_fn=os.path.join(sim_dir, 'resampled_trj.dcd'),
                output_fn=os.path.join(sim_dir, 'resampled_contacts.tsv'),
                **(getContacts_Info if getContacts_Info is not None else {}),
            )
            current_matrices['contact'] = contact_distance
        else:
            printf("skip_contacts=True — contact distance matrix omitted from convergence check.")

        # Save current matrices
        for name, matrix in current_matrices.items():
            np.save(os.path.join(sim_dir, f'resampled_{name}_matrix.npy'), matrix)

        # Effective post-equilibration fraction — whichever keeps more data
        min_post_equil_fraction = 1.0 - max_equil_fraction  # guaranteed minimum
        actual_post_equil_fraction = 1.0 - equil_fraction         # what MBAR detected
        effective_post_equil_fraction = max(actual_post_equil_fraction, min_post_equil_fraction)

        # Map the effective fraction back to the first valid sim_no.
        #
        # Deriving the window from t0 couples it to the equilibration gate
        # through a single variable, and the two then pull against each other:
        # a low discard satisfies the gate but widens the window to re-admit
        # early blocks, while a high discard narrows the window but fails the
        # gate. window_fraction breaks that coupling by fixing the window at a
        # trailing fraction of completed blocks, which also makes it monotone
        # in sim_no -- a block that leaves the window cannot re-enter it.
        from chimpss.fultonmarket.retro_convergence import resolve_window
        first_valid_sim_no = resolve_window(
            self.sim_no, equil_fraction, max_equil_fraction,
            window_blocks=self.window_blocks, window_fraction=self.window_fraction)

        # Compare against previous checkpoints in the post-equilibration window only
        frob_results = {name: {} for name in current_matrices}
        jsd_results  = {name: {} for name in current_matrices}

        for prev_sim_no in range(first_valid_sim_no, self.sim_no):
            prev_dir = os.path.join(self.output_dir, 'saved_variables', str(prev_sim_no))
            for name, current_matrix in current_matrices.items():
                prev_path = os.path.join(prev_dir, f'resampled_{name}_matrix.npy')
                if not os.path.exists(prev_path):
                    continue
                prev_matrix = np.load(prev_path)
                frob_results[name][prev_sim_no] = frobenius_norm(current_matrix, prev_matrix)
                jsd_results[name][prev_sim_no]  = jsd_distance_matrices(current_matrix, prev_matrix)

        # A matrix type is converged if ALL previous comparisons are below BOTH thresholds
        matrix_converged = {
            name: (
                len(frob_results[name]) > 0
                and all(v < frobenius_thresh for v in frob_results[name].values())
                and all(v < jsd_thresh      for v in jsd_results[name].values())
            )
            for name in current_matrices
        }

        # Build checks (at_max_time already handled above with early return)
        past_minimum    = self.sim_no >= (self.total_n_sims * self.minimum_fraction4convergence)
        equil_ok        = equil_fraction < max_equil_fraction
        torsion_ok      = matrix_converged['torsion']
        alpha_carbon_ok = matrix_converged['alpha_carbon']
        contact_ok      = matrix_converged.get('contact', True)

        checks = [
            ('Past minimum simulation fraction', past_minimum),
            ('Equilibration discard < 75%',      equil_ok),
            ('Torsion matrix converged',         torsion_ok),
            ('Alpha-carbon matrix converged',    alpha_carbon_ok),
            ('Contact matrix converged',         contact_ok),
        ]

        # Pretty print convergence report
        width = max(len(label) for label, _ in checks)
        printf("=" * (width + 12))
        printf(f"  {'Convergence Report':^{width}}")
        printf("=" * (width + 12))
        for label, result in checks:
            icon = "PASS" if result else "FAIL"
            printf(f"  [{icon}]  {label:<{width}}")
        printf("-" * (width + 12))

        # Per-matrix metric detail
        for name in current_matrices:
            frob_scores = frob_results[name]
            jsd_scores  = jsd_results[name]
            if frob_scores:
                frob_str = '  '.join(f'vs sim {k}: {v:.4f}' for k, v in sorted(frob_scores.items()))
                jsd_str  = '  '.join(f'vs sim {k}: {v:.4f}' for k, v in sorted(jsd_scores.items()))
                printf(f"  {name:>12} Frobenius -- {frob_str}  (thresh={frobenius_thresh})")
                printf(f"  {name:>12}       JSD -- {jsd_str}  (thresh={jsd_thresh})")
            else:
                printf(f"  {name:>12} -- no previous matrices to compare")
        printf("=" * (width + 12))

        # Stop if all convergence checks pass (at_max_time exits early above)
        if all(result for _, result in checks):
            return True
        return False
