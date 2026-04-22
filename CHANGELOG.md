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
