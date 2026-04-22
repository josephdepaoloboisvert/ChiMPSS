# Changelog

All notable changes to ChiMPSS are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `src/chimpss/shared/__init__.py`: new shared utilities package
- `src/chimpss/shared/io.py`: consolidated I/O helpers from `utils/utils.py` and `utility/` — `ensure_exists`, `write_FASTA` (canonical version, returns path), `cif2pdb`, `remove_dummy_atoms`, `isolate_chains`, `slice_select`, `change_resname`, `describe_system`, `describe_state`; Python 3.8 f-string fix in `isolate_chains` (nested quote removed)
- `src/chimpss/shared/logging.py`: consolidated logging helpers from `utility/Reporting.py` — `timestamp`, `printf`, `unique_residues`, `report_chain_information`; Python 3.8 f-string fix in `printf`
- `utils/ProteinPreparer.py`: shim re-export → `chimpss.bridgeport.protein_preparer.ProteinPreparer` (retained for one release cycle)

### Changed
- `src/chimpss/bridgeport/ligand.py`: three lazy imports in `_prepare_peptide` updated — `from utils.utils import write_FASTA` → `chimpss.shared.io`, `from utils.ProteinPreparer import ProteinPreparer` → `chimpss.bridgeport.protein_preparer`, `from RepairProtein.RepairProtein import RepairProtein` → `chimpss.bridgeport.repair_protein`; all inter-package `sys.path` hacks now eliminated from migrated bridgeport code

### Notes (Phase 5)
- `utils/utils.py` retained as-is: still imported by `Bridgeport/Bridgeport.py` (legacy) and several notebooks; will be shimmed in Phase 7 during notebook cleanup
- `utility/` directory retained as-is: `Build(BP).ipynb`, `ChiMPSS-Showcase.ipynb`, and `NewDistanceMatrix.ipynb` import from it; will be converted to shims in Phase 7

### Added (Phase 6 — Analysis module + CLI entry points)
- `src/chimpss/analysis/__init__.py`: new analysis package
- `src/chimpss/analysis/contacts.py`: thin re-export of `ContactNetworkBuilder` from `chimpss.fultonmarket.contact_network`
- `src/chimpss/analysis/pdb_fetch.py`: library functions for querying GPCRdb — `get`, `fetch_family_proteins`, `list_families`, `classify_method`
- `src/chimpss/analysis/distance_matrix.py`: consolidated `DistanceMatrix.py` + `DistanceMatrixUtils.py` — `DistanceMatrix` class plus all `_calc_*` and `_compute_*` helpers; Python 3.8 f-string fix in `printf`
- `src/chimpss/analysis/grid_potentials.py`: library-only portion of root `grid_potentials.py` — `IO_Grid`, `select_whole_residues`, `parameterize_from_pdb`, `simulate_from_filepair`; module-level simulation script code (original lines 343–442) excluded; imports updated to `chimpss.bridgeport.*`
- `src/chimpss/analysis/gpcr_pca.py`: consolidated `gpcr_pca_utils.py` + library helpers from `generate_pca_gpcrs.py` + library helpers from `project_pca_gpcrs.py` — all BW-mapping helpers, `Structure_Analyzer`, `fetch_all_parallel`, PCA helper functions, projection helpers, verbose diagnostics
- `src/chimpss/cli/__init__.py`: CLI package
- `src/chimpss/cli/motorrow.py`: argparse wrapper → `chimpss-motorrow`
- `src/chimpss/cli/fultonmarket.py`: argparse wrapper → `chimpss-fultonmarket`
- `src/chimpss/cli/fultonmarket_mpi.py`: MPI GPU-assignment wrapper → `chimpss-fultonmarket-mpi`; delegates to `chimpss.cli.fultonmarket.main()` (no `runpy.run_path` dependency)
- `src/chimpss/cli/retro_analysis.py`: argparse wrapper → `chimpss-retro-analysis`; fixes `None` bug in original `RUN_RETRO_ANALYSIS.py` (`output_cache_dir` defaulted to `<input_dir>/retro_cache/`); adds `--sele_str`, `--getcontacts_script`, `--conda_env`, `--getcontacts_python` args (replacing hardcoded Expanse paths)
- `src/chimpss/cli/recovery.py`: argparse wrapper → `chimpss-recovery`; converts `sys.argv[1]` to proper argparse
- `src/chimpss/cli/fetch_pdbs.py`: CLI portion of `fetch_pdb_list.py` → `chimpss-fetch-pdbs`
- `src/chimpss/cli/generate_pca.py`: CLI portion of `generate_pca_gpcrs.py` → `chimpss-generate-pca`
- `src/chimpss/cli/project_pca.py`: CLI portion of `project_pca_gpcrs.py` → `chimpss-project-pca`
- `scripts/slurm/run_one.job`: Expanse single-GPU job script (updated `python RUN_FULTONMARKET.py` → `chimpss-fultonmarket`)
- `scripts/slurm/run_mpi.job`: Expanse 4-GPU MPI job script (updated `mpiexec.hydra -np 4 python RUN_FULTONMARKET_MPI.py` → `mpiexec.hydra -np 4 chimpss-fultonmarket-mpi`)
- `scripts/slurm/recovery.job`: recovery job script (updated `python recovery.py` → `chimpss-recovery`)

### Changed (Phase 6)
- `pyproject.toml`: `[project.scripts]` entries uncommented and active for all 8 entry points (`chimpss-bridgeport` deferred — no `RUN_BRIDGEPORT.py` source; Bridgeport is used interactively)

### Backward-compat shims added (Phase 6)
- `gpcr_pca_utils.py` → `from chimpss.analysis.gpcr_pca import *`
- `GetContactsHelper.py` → re-exports `ContactNetworkBuilder` from `chimpss.fultonmarket.contact_network`
- `DistanceMatrix/DistanceMatrix.py` → re-exports `DistanceMatrix` from `chimpss.analysis.distance_matrix`
- `DistanceMatrix/DistanceMatrixUtils.py` → re-exports all `_calc_*` / `_compute_*` helpers from `chimpss.analysis.distance_matrix`
- `fetch_pdb_list.py` → delegates to `chimpss.cli.fetch_pdbs.main()`
- `generate_pca_gpcrs.py` → delegates to `chimpss.cli.generate_pca.main()`
- `project_pca_gpcrs.py` → delegates to `chimpss.cli.project_pca.main()`


- `pyproject.toml`: package skeleton — `pip install -e .` now works; `requires-python = ">=3.8"`; loose dep pins; `[dev]`, `[mpi]`, `[docs]` extras
- `src/chimpss/__init__.py`: package entry point, `__version__ = "0.1.0"`
- `.pre-commit-config.yaml`: ruff + black + nbstripout hooks
- `.github/workflows/ci.yml`: lint + no-deps install smoke test across Python 3.8 / 3.10 / 3.12
- `CLAUDE.md`: agent context file — active reorganization status, resumption phrase, key preferences
- `src/chimpss/bridgeport/`: Bridgeport and all dependencies migrated from `Bridgeport/`, `utils/`, `RepairProtein/`, `Ligand/`, `ForceFields/`, `Minimizer/`; all `sys.path.append` hacks removed; cross-module imports rewritten to `chimpss.bridgeport.*`; IPython side effects guarded with try/except
- `Bridgeport/__init__.py`: shim re-export → `chimpss.bridgeport.Bridgeport` (retained for one release cycle)
- `tests/unit/test_bridgeport/test_imports.py`: smoke tests (skip-if-no-mdtraj)
- `src/chimpss/motorrow/`: MotorRow and MotorRow_utils migrated from repo root; `sys.path.append` removed; imports rewritten to `chimpss.motorrow.*`; wildcard `from MotorRow_utils import *` replaced with explicit named imports
- `MotorRow.py`: shim re-export → `chimpss.motorrow.MotorRow` (retained for one release cycle)
- `tests/unit/test_motorrow/test_imports.py`: smoke tests (skip-if-no-openmm)
- `src/chimpss/fultonmarket/`: FultonMarket stage migrated from `FultonMarket/`; all six submodules migrated (`fulton_market.py`, `randolph.py`, `analysis.py`, `utils.py`, `contact_network.py`, `retro_convergence.py`); module-level side effects (`faulthandler.enable`, `warnings.filterwarnings`, `np.seterr`) consolidated into `__init__.py`; `@staticmethod` decorators removed from module-level functions; JAX import-time side effects guarded with try/except; `sys.path` depth for `project_gpcr_pca` updated (2 → 4 levels); wildcard imports replaced with explicit named imports; Python 3.8 f-string backslash fixed with `chr(10)`.
- `FultonMarket/__init__.py`: shim re-export → `chimpss.fultonmarket` classes (retained for one release cycle)
- `FultonMarket/FultonMarketwithAnalyzer.py`, `FultonMarket/RandolphwithAnalyzer.py`, `FultonMarket/FultonMarketAnalysis.py`, `FultonMarket/FultonMarketUtils.py`, `FultonMarket/retro_convergence_utils.py`: shim re-exports (retained for one release cycle)
- `tests/unit/test_fultonmarket/test_imports.py`: smoke tests (skip-if-no-openmm/mdtraj)
