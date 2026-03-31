import os
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime

import mdtraj as md
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MultiLabelBinarizer


printf = lambda x: print(f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}//{x}", flush=True)


class ContactNetworkBuilder:
    """
    Runs GetContacts on per-system contact-shell trajectories, parses the TSV output,
    filters interactions to the scientifically relevant subset, and builds a global
    frame-by-frame Jaccard distance matrix in Bronzeville frame order.
    """

    def __init__(
        self,
        contact_cache_dir: str,
        frame_labels: np.ndarray,
        selected_resseqs: np.array,
        getcontacts_repo_dir: str,
        getcontacts_conda_env: str = None,
        getcontacts_python: str = None,
        interaction_types=None,
        include_water_bridges: bool = True,
        interaction_scope: str = "selected_any",   # "selected_any" or "selected_both"
        interaction_min_freq: float = 0.01,
        interaction_max_freq: float = 0.95,
        use_cache: bool = True,
        getcontacts_stride: int = 1,
        getcontacts_cores: int = 1,
        interaction_token_mode: str = "residue",   
    ):
        """
        contact_cache_dir: storage dir for getcontacts
        frame_labels: frame_labels > from supertraj Bronzeville
        selected_resseqs: Bronzeville resSeqs
        selected_any, or selected_both:  interactions are calcuated or one end (any) or both ends (both) selected
        
        """
        self.contact_cache_dir = contact_cache_dir
        self.frame_labels = np.asarray(frame_labels)
        self.selected_resseqs = set(int(x) for x in selected_resseqs)

        self.getcontacts_repo_dir = getcontacts_repo_dir
        self.getcontacts_conda_env = getcontacts_conda_env
        self.getcontacts_python = getcontacts_python

        self.interaction_types = list(interaction_types or ["hb", "sb", "pc", "ps", "ts"])
        self.include_water_bridges = include_water_bridges
        self.interaction_scope = interaction_scope
        self.interaction_min_freq = interaction_min_freq
        self.interaction_max_freq = interaction_max_freq
        self.use_cache = use_cache

        self.getcontacts_stride = int(getcontacts_stride)
        self.getcontacts_cores = int(getcontacts_cores)

        self.getcontacts_script = os.path.join(self.getcontacts_repo_dir, "get_dynamic_contacts.py")
        if not os.path.exists(self.getcontacts_script):
            raise FileNotFoundError(f"Could not find get_dynamic_contacts.py at {self.getcontacts_script}")

        # Build a fast mapping from (system_name, local_frame) -> global_frame_index
        self.global_frame_index = {}
        for i, (name, frame) in enumerate(self.frame_labels):
            self.global_frame_index[(str(name), int(frame))] = i

        self.interaction_token_mode = str(interaction_token_mode).lower()
        valid_modes = {"residue", "residue_atomclass"}
        if self.interaction_token_mode not in valid_modes:
            raise ValueError(
                f"interaction_token_mode must be one of {sorted(valid_modes)}, "
                f"got {self.interaction_token_mode!r}"
            )
        
        printf(f"DEBUG helper interaction_token_mode={self.interaction_token_mode}")

    # ---------- CLI helpers ----------

    def _build_getcontacts_command(self, pdb_fn: str, dcd_fn: str, out_tsv: str):
        if self.getcontacts_python:
            prefix = [self.getcontacts_python]
        elif self.getcontacts_conda_env:
            prefix = ["conda", "run", "-n", self.getcontacts_conda_env, "python"]
        else:
            prefix = ["python"]

        cmd = prefix + [
            self.getcontacts_script,
            "--topology", pdb_fn,
            "--trajectory", dcd_fn,
            "--output", out_tsv,
            "--cores", str(self.getcontacts_cores),
            "--stride", str(self.getcontacts_stride),
            "--itypes",
        ] + self.interaction_types

        printf(f"DEBUG helper stride={self.getcontacts_stride}, cores={self.getcontacts_cores}")
        printf("DEBUG helper cmd: " + " ".join(cmd))
        
        return cmd

    def _run_getcontacts_for_system(self, system_name: str):
        pdb_fn = os.path.join(self.contact_cache_dir, f"{system_name}.pdb")
        dcd_fn = os.path.join(self.contact_cache_dir, f"{system_name}.dcd")
        out_tsv = os.path.join(self.contact_cache_dir, f"{system_name}.getcontacts.tsv")

        if self.use_cache and os.path.exists(out_tsv) and os.path.getsize(out_tsv) > 0:
            return out_tsv

        if not os.path.exists(pdb_fn):
            raise FileNotFoundError(f"Missing contact-shell pdb for {system_name}: {pdb_fn}")
        if not os.path.exists(dcd_fn):
            raise FileNotFoundError(f"Missing contact-shell dcd for {system_name}: {dcd_fn}")

        cmd = self._build_getcontacts_command(pdb_fn, dcd_fn, out_tsv)
        printf(f"Running GetContacts for {system_name}")
        subprocess.run(cmd, check=True, cwd=self.getcontacts_repo_dir)
        return out_tsv

    # ---------- residue label helpers ----------

    @staticmethod
    def _atom_to_residue_label(atom_label: str):
        """
        Convert atom label like A:ASP:155:OD1 -> A:ASP:155
        """
        parts = atom_label.split(":")
        if len(parts) < 3:
            return atom_label
        return ":".join(parts[:3])

    @staticmethod
    def _resseq_from_residue_label(res_label: str):
        """
        A:ASP:155 -> 155
        """
        parts = res_label.split(":")
        try:
            return int(parts[2])
        except Exception:
            return None

    @staticmethod
    def _resname_from_residue_label(res_label: str):
        """
        A:ASP:155 -> ASP
        """
        parts = res_label.split(":")
        if len(parts) >= 2:
            return parts[1]
        return None

    @staticmethod
    def _is_water_resname(resname: str):
        return resname in {"HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE", "TIP4", "TIP4P"}

    def _build_residue_category_map(self, pdb_fn: str):
        traj = md.load(pdb_fn)
        resmap = {}
        for res in traj.topology.residues:
            chain_lbl = self._chain_label(res.chain)
            label = f"{chain_lbl}:{res.name}:{res.resSeq}"
            resmap[label] = {
                "is_protein": bool(res.is_protein),
                "is_water": bool(res.is_water or self._is_water_resname(res.name)),
                "resname": res.name,
                "resseq": int(res.resSeq),
            }
        return resmap

    # ---------- parsing / filtering ----------

    def _keep_interaction_token(self, interaction_type: str, endpoint_labels, resmap):
        """
        endpoint_labels: non-water endpoint tokens after collapsing raw atom labels.
        These may be residue labels (A:ASP:155) or residue+atomclass labels
        (A:ASP:155:O), depending on interaction_token_mode.
        """
        if len(endpoint_labels) < 2:
            return False

        endpoint_residue_labels = []
        for lbl in endpoint_labels:
            parts = lbl.split(":")
            if len(parts) >= 3:
                endpoint_residue_labels.append(":".join(parts[:3]))
            else:
                endpoint_residue_labels.append(lbl)

        protein_flags = [resmap.get(r, {}).get("is_protein", False) for r in endpoint_residue_labels]
        if not all(protein_flags):
            return False

        endpoint_resseqs = [resmap[r]["resseq"] for r in endpoint_residue_labels]

        if self.interaction_scope == "selected_any":
            return any(r in self.selected_resseqs for r in endpoint_resseqs)

        if self.interaction_scope == "selected_both":
            return all(r in self.selected_resseqs for r in endpoint_resseqs)

        raise ValueError(f"Unknown interaction_scope: {self.interaction_scope}")

    def _atom_label_to_endpoint_token(self, atom_label: str, resmap):
        """
        Convert a raw GetContacts atom label into the endpoint token used in the
        final interaction feature, while dropping waters.
        """
        residue_label = self._atom_to_residue_label(atom_label)
        info = resmap.get(residue_label, None)

        if info is None:
            resname = self._resname_from_residue_label(residue_label)
            if self._is_water_resname(resname):
                return None
        else:
            if info["is_water"]:
                return None

        if self.interaction_token_mode == "residue":
            return residue_label

        if self.interaction_token_mode == "residue_atomclass":
            return self._atom_to_residue_atomclass_label(atom_label)

        raise ValueError(f"Unknown interaction_token_mode: {self.interaction_token_mode}")

    def _parse_system_contacts(self, system_name: str):
        """
        Returns dict: local_frame -> set(tokens)
        """
        pdb_fn = os.path.join(self.contact_cache_dir, f"{system_name}.pdb")
        tsv_fn = self._run_getcontacts_for_system(system_name)
        resmap = self._build_residue_category_map(pdb_fn)

        frame_tokens = defaultdict(set)

        with open(tsv_fn, "r") as f:
            for raw in f:
                line = raw.strip()
                if (not line) or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                local_frame = int(parts[0])
                interaction_type = parts[1]
                atom_labels = parts[2:]

                # Keep only requested types
                if not self._interaction_type_matches_requested(
                    interaction_type,
                    self.interaction_types,
                    self.include_water_bridges
                ):
                    continue

                endpoint_tokens = []
                for a in atom_labels:
                    tok = self._atom_label_to_endpoint_token(a, resmap)
                    if tok is not None:
                        endpoint_tokens.append(tok)

                # Deduplicate equivalent atoms/classes while preserving the token mode
                endpoint_tokens = sorted(set(endpoint_tokens))

                if interaction_type in {"wb", "wb2", "lwb", "lwb2"} and not self.include_water_bridges:
                    continue

                if not self._keep_interaction_token(interaction_type, endpoint_tokens, resmap):
                    continue

                token = f"{interaction_type}|{'|'.join(endpoint_tokens)}"
                frame_tokens[local_frame].add(token)

        return frame_tokens

    # ---------- global feature set assembly ----------

    def build_global_frame_feature_sets(self):
        """
        Returns a list of sets, length == number of global Bronzeville frames.
        """
        n_global = len(self.frame_labels)
        global_features = [set() for _ in range(n_global)]

        system_names = sorted(set(str(x[0]) for x in self.frame_labels))
        for system_name in system_names:
            printf(f"Parsing GetContacts features for {system_name}")
            local_features = self._parse_system_contacts(system_name)
            system_frame_labels = {int(f) for name, f in self.frame_labels if str(name) == system_name}
            tsv_frames = set(int(x) for x in local_features.keys())
            matched = system_frame_labels & tsv_frames

            printf(
                f"{system_name}: Bronzeville frames={len(system_frame_labels)}, "
                f"TSV frames={len(tsv_frames)}, matched={len(matched)}"
            )
            printf(f"{system_name}: first Bronzeville frames={sorted(system_frame_labels)[:10]}")
            printf(f"{system_name}: first TSV frames={sorted(tsv_frames)[:10]}")
            for local_frame, tokens in local_features.items():
                key = (system_name, int(local_frame))
                if key in self.global_frame_index:
                    global_idx = self.global_frame_index[key]
                    global_features[global_idx].update(tokens)

        return global_features

    def filter_features_by_frequency(self, frame_feature_sets):
        n_frames = len(frame_feature_sets)
        feature_counts = Counter()

        for s in frame_feature_sets:
            feature_counts.update(s)

        min_count = max(1, math.ceil(self.interaction_min_freq * n_frames))
        max_count = math.floor(self.interaction_max_freq * n_frames)

        kept_features = {
            feat for feat, count in feature_counts.items()
            if (count >= min_count and count <= max_count)
        }

        printf(f"Keeping {len(kept_features)} interaction features after frequency filtering")

        filtered = [set(x for x in s if x in kept_features) for s in frame_feature_sets]
        return filtered, kept_features

    # ---------- matrix construction ----------

    def build_distance_matrix(self, filename: str):
        frame_feature_sets = self.build_global_frame_feature_sets()
        raw_counts = np.array([len(s) for s in frame_feature_sets], dtype=int)
        print(f"RAW frames with zero contacts: {(raw_counts == 0).sum()} / {len(raw_counts)}", flush=True)
        print(f"RAW mean contacts per frame: {raw_counts.mean():.2f}", flush=True)
        print(f"RAW median contacts per frame: {np.median(raw_counts):.2f}", flush=True)
        frame_feature_sets, kept_features = self.filter_features_by_frequency(frame_feature_sets)

        # Save metadata
        meta = {
            "interaction_types": self.interaction_types,
            "include_water_bridges": self.include_water_bridges,
            "interaction_scope": self.interaction_scope,
            "interaction_min_freq": self.interaction_min_freq,
            "interaction_max_freq": self.interaction_max_freq,
            "selected_resseqs": sorted(self.selected_resseqs),
            "n_frames": len(frame_feature_sets),
            "n_features_kept": len(kept_features),
            "interaction_token_mode": self.interaction_token_mode,
        }
        with open(os.path.join(os.path.dirname(filename), "getcontacts_matrix_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Edge case: no features survive
        if len(kept_features) == 0:
            dist = np.memmap(filename, dtype="float32", mode="w+",
                             shape=(len(frame_feature_sets), len(frame_feature_sets)))
            dist[:] = 0.0
            dist.flush()
            return dist

        # Binary matrix -> Jaccard distance
        mlb = MultiLabelBinarizer()
        X = mlb.fit_transform(frame_feature_sets).astype(bool)

        row_counts = X.sum(axis=1)
        print(f"Contact features kept: {X.shape[1]}", flush=True)
        print(f"Frames with zero kept contacts: {(row_counts == 0).sum()} / {X.shape[0]}", flush=True)
        print(f"Mean kept contacts per frame: {row_counts.mean():.2f}", flush=True)
        print(f"Median kept contacts per frame: {np.median(row_counts):.2f}", flush=True)

        unique_rows = np.unique(X, axis=0).shape[0]
        print(f"Unique contact fingerprints: {unique_rows} / {X.shape[0]}", flush=True)

        D = squareform(pdist(X, metric="jaccard")).astype(np.float32)

        maxval = D.max()
        if maxval > 0:
            D /= maxval

        dist = np.memmap(filename, dtype="float32", mode="w+",
                         shape=(D.shape[0], D.shape[1]))
        dist[:] = D
        dist.flush()
        return dist
    
    @staticmethod
    def _chain_label(chain):
        for attr in ("chain_id", "id", "name"):
            val = getattr(chain, attr, None)
            if val not in (None, ""):
                return str(val)
        # fallback for single-chain systems
        if getattr(chain, "index", None) is not None:
            return chr(ord("A") + int(chain.index)) if int(chain.index) < 26 else str(chain.index)
        return "A"
    
    @staticmethod
    def _interaction_type_matches_requested(interaction_type: str, requested_types, include_water_bridges: bool):
        """
        Returns True if a TSV interaction subtype should be kept based on the
        user-requested top-level GetContacts interaction classes.
        """
        requested = set(requested_types)

        # Exact matches for non-hbond classes
        if interaction_type in requested:
            return True

        # Hydrogen-bond family emitted by GetContacts TSV when CLI requested "hb"
        hb_family = {"hb", "hbbb", "hbsb", "hbss", "hblb", "hbls"}
        water_bridge_family = {"wb", "wb2", "lwb", "lwb2"}

        if "hb" in requested:
            if interaction_type in hb_family:
                return True
            if include_water_bridges and interaction_type in water_bridge_family:
                return True

        return False
    @staticmethod
    def _split_atom_label(atom_label: str):
        """
        A:ASP:155:OD1 -> (chain, resname, resseq, atomname)
        """
        parts = atom_label.split(":")
        if len(parts) >= 4:
            return parts[0], parts[1], parts[2], parts[3]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2], None
        return None, None, None, None


    @staticmethod
    def _atom_to_residue_atomclass_label(atom_label: str):
        """
        Convert atom label like A:ASP:155:OD1 -> A:ASP:155:O
        Convert atom label like A:ARG:42:NH2 -> A:ARG:42:N

        Falls back conservatively if atom name is missing.
        """
        chain, resname, resseq, atomname = ContactNetworkBuilder._split_atom_label(atom_label)
        if chain is None or resname is None or resseq is None:
            return atom_label

        if atomname is None or atomname == "":
            atom_class = "X"
        else:
            atomname = atomname.upper()

            # simple coarse element/class collapse
            if atomname.startswith("N"):
                atom_class = "N"
            elif atomname.startswith("O"):
                atom_class = "O"
            elif atomname.startswith("S"):
                atom_class = "S"
            elif atomname.startswith("P"):
                atom_class = "P"
            elif atomname.startswith("C"):
                atom_class = "C"
            else:
                atom_class = "X"

        return f"{chain}:{resname}:{resseq}:{atom_class}"
    
