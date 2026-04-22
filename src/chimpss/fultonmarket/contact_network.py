import os
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Literal

import mdtraj as md
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


printf = lambda x: print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}//{x}", flush=True)

# Interaction subtypes emitted by GetContacts TSV that belong to the hbond family
_HB_FAMILY           = {"hb", "hbbb", "hbsb", "hbss", "hblb", "hbls"}
_WATER_BRIDGE_FAMILY = {"wb", "wb2", "lwb", "lwb2"}
_WATER_RESNAMES      = {"HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE", "TIP4", "TIP4P"}


class ContactNetworkBuilder:
    """
    Runs GetContacts on a trajectory, parses the TSV output, and returns a
    binary contact fingerprint for every frame.

    The resolution of the fingerprint is controlled by ``token_mode``:

    - ``'residue'``           -- one bit per (interaction_type, residue_A, residue_B)
    - ``'residue_atomclass'`` -- one bit per (interaction_type, residue_A:atomclass, residue_B:atomclass)

    Two usage modes
    ---------------
    Run GetContacts and parse in one step::

        builder = ContactNetworkBuilder(
            pdb_fn='system.pdb',
            dcd_fn='system.dcd',
            getcontacts_script='/path/to/get_dynamic_contacts.py',
            selected_resseqs=[100, 101, 102, 200],
        )
        X, features = builder.get_contact_vectors()

    Parse an existing GetContacts TSV directly::

        builder = ContactNetworkBuilder.from_tsv(
            tsv_fn='system.getcontacts.tsv',
            pdb_fn='system.pdb',
            selected_resseqs=[100, 101, 102, 200],
        )
        X, features = builder.get_contact_vectors()

    In both cases ``X`` is an ``(n_frames, n_features)`` bool array and
    ``features`` is the corresponding list of contact token strings.

    Parameters
    ----------
    pdb_fn : str
        Path to the topology PDB file. Always required -- used to build the
        residue metadata map.
    dcd_fn : str, optional
        Path to the trajectory DCD file. Required when running GetContacts;
        not needed when parsing an existing TSV via ``from_tsv``.
    getcontacts_script : str, optional
        Path to ``get_dynamic_contacts.py``. Required when running
        GetContacts; not needed when parsing an existing TSV via ``from_tsv``.
    selected_resseqs : list of int, optional
        Residue sequence numbers to include. Controls ``scope``. If None,
        all protein residues are included and ``scope`` has no effect.
    scope : {'selected_any', 'selected_both'}
        ``'selected_any'``  -- keep interactions where at least one endpoint
        residue is in ``selected_resseqs``.
        ``'selected_both'`` -- keep interactions where both endpoints are in
        ``selected_resseqs``.
        Default ``'selected_any'``.
    token_mode : {'residue', 'residue_atomclass'}
        Resolution of each contact token. Default ``'residue'``.
    interaction_types : list of str
        GetContacts interaction type flags. Default ``['hb', 'sb', 'pc', 'ps', 'ts']``.
    include_water_bridges : bool
        Whether to include water-mediated hydrogen bonds. Default True.
    min_freq : float
        Minimum fraction of frames a contact must appear in to be retained.
        Default 0.01.
    max_freq : float
        Maximum fraction of frames a contact can appear in before being
        discarded as uninformative. Default 0.95.
    stride : int
        Frame stride passed to GetContacts. Default 1.
    cores : int
        Number of CPU cores passed to GetContacts. Default 1.
    getcontacts_python : str, optional
        Full path to the Python interpreter to use. If None, falls back to
        ``conda_env`` or the current ``python``.
    conda_env : str, optional
        Name of the conda environment containing GetContacts, used via
        ``conda run -n <env> python``.
    out_tsv : str, optional
        Path for the GetContacts output TSV. Defaults to ``<dcd_fn>.tsv``.
        When using ``from_tsv``, this is set to the provided TSV path.
    use_cache : bool
        If True and ``out_tsv`` already exists and is non-empty, skip running
        GetContacts and parse the existing file. Default True.
    """

    def __init__(
        self,
        pdb_fn: str,
        selected_resseqs: List[int] = None,
        dcd_fn: str = None,
        getcontacts_script: str = None,
        scope: Literal['selected_any', 'selected_both'] = 'selected_any',
        token_mode: Literal['residue', 'residue_atomclass'] = 'residue',
        interaction_types: List[str] = None,
        include_water_bridges: bool = True,
        min_freq: float = 0.01,
        max_freq: float = 0.95,
        stride: int = 1,
        cores: int = 1,
        getcontacts_python: str = None,
        conda_env: str = None,
        out_tsv: str = None,
        use_cache: bool = True,
    ):
        if scope not in ('selected_any', 'selected_both'):
            raise ValueError(f"scope must be 'selected_any' or 'selected_both', got {scope!r}")
        if token_mode not in ('residue', 'residue_atomclass'):
            raise ValueError(f"token_mode must be 'residue' or 'residue_atomclass', got {token_mode!r}")

        # Validate that enough info is provided to actually do something
        if out_tsv is None and dcd_fn is None:
            raise ValueError('Either out_tsv (pre-computed TSV) or dcd_fn (trajectory to run GetContacts on) must be provided.')
        if out_tsv is None and getcontacts_script is None:
            raise ValueError('getcontacts_script is required when not providing a pre-computed out_tsv.')
        if getcontacts_script is not None and not os.path.exists(getcontacts_script):
            raise FileNotFoundError(f'get_dynamic_contacts.py not found at {getcontacts_script}')

        self.pdb_fn              = pdb_fn
        self.dcd_fn              = dcd_fn
        self.getcontacts_script  = getcontacts_script
        self.selected_resseqs    = set(int(r) for r in selected_resseqs) if selected_resseqs is not None else None
        self.scope               = scope
        self.token_mode          = token_mode
        self.interaction_types   = list(interaction_types or ['hb', 'sb', 'pc', 'ps', 'ts'])
        self.include_water_bridges = include_water_bridges
        self.min_freq            = min_freq
        self.max_freq            = max_freq
        self.stride              = int(stride)
        self.cores               = int(cores)
        self.getcontacts_python  = getcontacts_python
        self.conda_env           = conda_env
        self.out_tsv             = out_tsv or os.path.splitext(dcd_fn)[0] + '.getcontacts.tsv'
        self.use_cache           = use_cache

        # Build residue metadata from topology
        self._resmap = self._build_resmap()


    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_tsv(
        cls,
        tsv_fn: str,
        pdb_fn: str,
        selected_resseqs: List[int] = None,
        scope: Literal['selected_any', 'selected_both'] = 'selected_any',
        token_mode: Literal['residue', 'residue_atomclass'] = 'residue',
        interaction_types: List[str] = None,
        include_water_bridges: bool = True,
        min_freq: float = 0.01,
        max_freq: float = 0.95,
    ) -> 'ContactNetworkBuilder':
        """
        Construct a ContactNetworkBuilder from an existing GetContacts TSV,
        skipping the GetContacts execution step entirely.

        Parameters
        ----------
        tsv_fn : str
            Path to a pre-computed GetContacts TSV output file.
        pdb_fn : str
            Path to the topology PDB file used to build the residue metadata
            map. Must match the topology used when GetContacts was run.
        selected_resseqs : list of int
            Residue sequence numbers to include. Controls ``scope``.
        scope : {'selected_any', 'selected_both'}
            Interaction endpoint scope. Default ``'selected_any'``.
        token_mode : {'residue', 'residue_atomclass'}
            Resolution of each contact token. Default ``'residue'``.
        interaction_types : list of str, optional
            Subset of interaction types to retain from the TSV. If None, all
            of ``['hb', 'sb', 'pc', 'ps', 'ts']`` are kept.
        include_water_bridges : bool
            Whether to retain water-bridge interactions. Default True.
        min_freq : float
            Minimum frequency threshold for retaining a contact. Default 0.01.
        max_freq : float
            Maximum frequency threshold for retaining a contact. Default 0.95.

        Returns
        -------
        ContactNetworkBuilder
            Instance configured to parse ``tsv_fn`` directly, with
            ``getcontacts_script`` and ``dcd_fn`` set to None.

        Raises
        ------
        FileNotFoundError
            If ``tsv_fn`` or ``pdb_fn`` does not exist.

        Examples
        --------
        >>> builder = ContactNetworkBuilder.from_tsv(
        ...     tsv_fn='run1.getcontacts.tsv',
        ...     pdb_fn='system.pdb',
        ...     selected_resseqs=[100, 101, 102],
        ...     token_mode='residue_atomclass',
        ... )
        >>> X, features = builder.get_contact_vectors()
        """
        if not os.path.exists(tsv_fn):
            raise FileNotFoundError(f'GetContacts TSV not found: {tsv_fn}')
        if not os.path.exists(pdb_fn):
            raise FileNotFoundError(f'PDB file not found: {pdb_fn}')

        return cls(
            pdb_fn=pdb_fn,
            selected_resseqs=selected_resseqs,
            dcd_fn=None,
            getcontacts_script=None,
            scope=scope,
            token_mode=token_mode,
            interaction_types=interaction_types,
            include_water_bridges=include_water_bridges,
            min_freq=min_freq,
            max_freq=max_freq,
            out_tsv=tsv_fn,
            use_cache=True,  # TSV already exists — always skip execution
        )


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_contact_vectors(self) -> tuple:
        """
        Run GetContacts if needed, then parse, filter, and return a binary
        frame-by-feature matrix.

        If the instance was created via ``from_tsv``, the GetContacts
        execution step is skipped and the provided TSV is parsed directly.

        Returns
        -------
        X : np.ndarray of shape (n_frames, n_features), dtype bool
            Binary contact fingerprint per frame. Each column corresponds to
            one contact token (see ``features``).
        features : list of str
            Human-readable label for each column of ``X``, e.g.
            ``'hb|A:ASP:155|A:ARG:200'``.
        """
        self._run_getcontacts()
        frame_tokens = self._parse_tsv()
        frame_tokens = self._filter_by_frequency(frame_tokens)

        n_frames = max(frame_tokens.keys()) + 1 if frame_tokens else 0
        ordered  = [frame_tokens.get(f, set()) for f in range(n_frames)]

        mlb      = MultiLabelBinarizer()
        X        = mlb.fit_transform(ordered).astype(bool)
        features = list(mlb.classes_)

        printf(f'Contact matrix: {X.shape[0]} frames x {X.shape[1]} features')
        return X, features


    # ------------------------------------------------------------------
    # GetContacts execution
    # ------------------------------------------------------------------

    def _run_getcontacts(self):
        """
        Run GetContacts, skipping if a valid cached TSV already exists or if
        ``getcontacts_script`` is None (i.e. TSV-only mode via ``from_tsv``).
        """
        if os.path.exists(self.out_tsv) and os.path.getsize(self.out_tsv) > 0:
            printf(f'Using existing contacts TSV: {self.out_tsv}')
            return

        if self.getcontacts_script is None:
            raise RuntimeError(
                f'TSV not found at {self.out_tsv} and no getcontacts_script was provided. '
                'Pass a valid tsv_fn to from_tsv, or provide getcontacts_script and dcd_fn.'
            )

        cmd = self._build_cmd()
        printf(f'Running GetContacts: {" ".join(cmd)}')
        subprocess.run(cmd, check=True, cwd=os.path.dirname(self.getcontacts_script))


    def _build_cmd(self) -> List[str]:
        """Assemble the GetContacts command list."""
        if self.getcontacts_python:
            prefix = [self.getcontacts_python]
        elif self.conda_env:
            prefix = ['conda', 'run', '-n', self.conda_env, 'python']
        else:
            prefix = ['python']

        return prefix + [
            self.getcontacts_script,
            '--topology',   self.pdb_fn,
            '--trajectory', self.dcd_fn,
            '--output',     self.out_tsv,
            '--cores',      str(self.cores),
            '--stride',     str(self.stride),
            '--itypes',
        ] + self.interaction_types


    # ------------------------------------------------------------------
    # TSV parsing
    # ------------------------------------------------------------------

    def _parse_tsv(self) -> dict:
        """
        Parse the GetContacts TSV into a dict of {local_frame: set(tokens)}.
        Applies interaction type filtering, water removal, scope filtering,
        and token resolution in one pass.
        """
        frame_tokens = defaultdict(set)

        with open(self.out_tsv) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                frame_no    = int(parts[0])
                itype       = parts[1]
                atom_labels = parts[2:]

                if not self._itype_is_requested(itype):
                    continue

                endpoint_tokens = []
                for atom in atom_labels:
                    tok = self._atom_to_token(atom)
                    if tok is not None:
                        endpoint_tokens.append(tok)
                endpoint_tokens = sorted(set(endpoint_tokens))

                if not self._passes_scope(endpoint_tokens):
                    continue

                token = f"{itype}|{'|'.join(endpoint_tokens)}"
                frame_tokens[frame_no].add(token)

        printf(f'Parsed {len(frame_tokens)} frames from {self.out_tsv}')
        return dict(frame_tokens)


    # ------------------------------------------------------------------
    # Frequency filtering
    # ------------------------------------------------------------------

    def _filter_by_frequency(self, frame_tokens: dict) -> dict:
        """
        Remove tokens that appear in fewer than ``min_freq`` or more than
        ``max_freq`` of frames.
        """
        n_frames  = len(frame_tokens)
        min_count = max(1, math.ceil(self.min_freq * n_frames))
        max_count = math.floor(self.max_freq * n_frames)

        counts = Counter(tok for tokens in frame_tokens.values() for tok in tokens)
        kept   = {tok for tok, c in counts.items() if min_count <= c <= max_count}
        printf(f'Frequency filter: kept {len(kept)} / {len(counts)} tokens '
               f'(min_freq={self.min_freq}, max_freq={self.max_freq})')

        return {f: tokens & kept for f, tokens in frame_tokens.items()}


    # ------------------------------------------------------------------
    # Token resolution
    # ------------------------------------------------------------------

    def _atom_to_token(self, atom_label: str):
        """
        Convert a raw GetContacts atom label (e.g. ``A:ASP:155:OD1``) to an
        endpoint token at the configured resolution, or None if the atom
        belongs to a water molecule.
        """
        residue_label = ':'.join(atom_label.split(':')[:3])
        info = self._resmap.get(residue_label)

        if info is None:
            resname = atom_label.split(':')[1] if len(atom_label.split(':')) > 1 else ''
            if resname in _WATER_RESNAMES:
                return None
        elif info['is_water']:
            return None

        if self.token_mode == 'residue':
            return residue_label

        # residue_atomclass: collapse atom name to element class
        parts      = atom_label.split(':')
        atomname   = parts[3].upper() if len(parts) > 3 else ''
        atom_class = next(
            (c for c in ('N', 'O', 'S', 'P', 'C') if atomname.startswith(c)), 'X'
        )
        return f"{':'.join(parts[:3])}:{atom_class}"


    # ------------------------------------------------------------------
    # Scope filtering
    # ------------------------------------------------------------------

    def _passes_scope(self, endpoint_tokens: List[str]) -> bool:
        """
        Return True if the endpoint tokens satisfy the configured scope and
        both endpoints are protein residues.

        When ``selected_resseqs`` is None, the resSeq filter is skipped and
        all protein-protein interactions are retained.
        """
        if len(endpoint_tokens) < 2:
            return False

        residue_labels = [':'.join(t.split(':')[:3]) for t in endpoint_tokens]

        if not all(self._resmap.get(r, {}).get('is_protein', False) for r in residue_labels):
            return False

        # No resSeq filter — keep all protein interactions
        if self.selected_resseqs is None:
            return True

        resseqs = [self._resmap[r]['resseq'] for r in residue_labels if r in self._resmap]

        if self.scope == 'selected_any':
            return any(r in self.selected_resseqs for r in resseqs)
        return all(r in self.selected_resseqs for r in resseqs)


    # ------------------------------------------------------------------
    # Interaction type matching
    # ------------------------------------------------------------------

    def _itype_is_requested(self, itype: str) -> bool:
        """Return True if the TSV interaction subtype should be kept."""
        if itype in self.interaction_types:
            return True
        if 'hb' in self.interaction_types:
            if itype in _HB_FAMILY:
                return True
            if self.include_water_bridges and itype in _WATER_BRIDGE_FAMILY:
                return True
        return False


    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    def _build_resmap(self) -> dict:
        """
        Build a dict mapping residue label strings (``'A:ASP:155'``) to
        metadata dicts with keys ``is_protein``, ``is_water``, ``resseq``.
        """
        traj   = md.load(self.pdb_fn)
        resmap = {}
        for res in traj.topology.residues:
            chain_lbl = self._chain_label(res.chain)
            label     = f'{chain_lbl}:{res.name}:{res.resSeq}'
            resmap[label] = {
                'is_protein': bool(res.is_protein),
                'is_water':   bool(res.is_water or res.name in _WATER_RESNAMES),
                'resseq':     int(res.resSeq),
            }
        return resmap


    @staticmethod
    def _chain_label(chain) -> str:
        """Return a single-character or index-based chain label."""
        for attr in ('chain_id', 'id', 'name'):
            val = getattr(chain, attr, None)
            if val not in (None, ''):
                return str(val)
        idx = getattr(chain, 'index', None)
        if idx is not None:
            return chr(ord('A') + int(idx)) if int(idx) < 26 else str(idx)
        return 'A'