# ChiMPSS — Chicago Membrane Protein Simulation Suite

ChiMPSS is an end-to-end Python pipeline for simulating membrane proteins using OpenMM.
It covers every stage from raw crystal structure to converged free-energy estimates:

```
Crystal structure
      │
      ▼
 [Bridgeport]   ──  system construction (solvation, lipid bilayer, force fields)
      │
      ▼
 [MotorRow]     ──  equilibration (5-step restrained → unrestrained NPT protocol)
      │
      ▼
 [FultonMarket] ──  production parallel-tempering REMD + auto-convergence detection
      │
      ▼
 [FultonMarketAnalysis] ── MBAR reweighting, PCA, contact/distance-matrix analysis
```

---

## Repository layout

```
ChiMPSS/
├── chimpss/                    Main Python package
│   ├── common/                 Shared utilities (logging, I/O, OpenMM helpers)
│   └── motor_row/              MotorRow equilibration class
│
├── Bridgeport/                 System construction package
│   └── Bridgeport.py           class Bridgeport
│
├── FultonMarket/               Production REMD package
│   ├── fulton_market.py        class FultonMarket
│   ├── randolph.py             class Randolph (inner simulation loop)
│   ├── analysis.py             class FultonMarketAnalysis
│   ├── contact_network.py      class ContactNetworkBuilder
│   ├── convergence.py          distance-matrix convergence utilities
│   └── FultonMarketUtils.py    low-level numerical helpers
│
├── Minimizer/                  Standalone energy minimizer
│   └── Minimizer.py            class Minimizer
│
├── Docking/                    Autodock Vina wrapper
│   └── Docking.py              class Docking
│
├── RepairProtein/              MODELLER-based loop/residue repair
│   └── repair_protein.py       class RepairProtein
│
├── ForceFields/                OpenMM force-field XML files + handlers
├── Ligand/                     Small-molecule, analogue, and peptide ligand classes
├── utils/                      Protein preparation utilities
├── utility/                    File I/O and reporting helpers
│
├── scripts/                    CLI entry points (see below)
├── notebooks/                  Curated Jupyter notebooks
│   ├── 00_showcase.ipynb
│   ├── 01_build_system.ipynb
│   ├── 02_minimize.ipynb
│   ├── 03_equilibrate_and_run.ipynb
│   ├── 04_analysis.ipynb
│   ├── examples/               Worked examples (small molecules, docking, …)
│   └── archive/                Legacy / one-off notebooks
│
├── test_data/                  Reference structures (AlphaFold3, raw RCSB)
├── pyproject.toml              Package metadata and console-script entry points
└── environment.yml             Conda environment specification
```

---

## Installation

### 1. Create the conda environment

Most dependencies are best installed via conda before pip:

```bash
conda create -n chimpss -c conda-forge -c salilab \
    python=3.10 pdbfixer mdtraj openff-toolkit pdb2pqr \
    mdanalysis openbabel openmmtools vina scikit-learn \
    parmed modeller py3dmol mpi4py openmm

conda activate chimpss
```

> **Note:** The `modeller` package requires the `salilab` conda channel and a
> free academic licence key set in `~/.modeller/config.py`.

### 2. Install the gridforce plugin (optional, required for membrane simulations)

```bash
pip install git+https://github.com/jimtufts/openmmgridforce.git
```

### 3. Install ChiMPSS

```bash
pip install -e .          # editable install from the repo root
```

This registers all `chimpss-*` CLI commands in your environment.

---

## Pipeline overview

### Stage 1 — System construction (`Bridgeport`)

`Bridgeport` takes a crystal structure and a JSON configuration file, and
produces a membrane-solvated OpenMM system ready for simulation.

**Steps performed automatically:**

1. Align the input structure to a reference (e.g. OPM membrane orientation).
2. Separate protein and ligand chains.
3. Repair missing residues / loops with MODELLER (`RepairProtein`).
4. Add hydrogens, water, and a lipid bilayer (`ProteinPreparer` / pdb2pqr).
5. Parameterise the ligand with OpenFF.
6. Merge protein and ligand force fields (`ForceFieldHandler` + `Joiner`).
7. Write `system.pdb` and `system.xml` to `<working_dir>/systems/`.

**CLI:**

```bash
chimpss-bridgeport input.json
```

**Programmatic:**

```python
from Bridgeport.Bridgeport import Bridgeport

bp = Bridgeport(input_json="input.json")
bp.run()
```

See [notebooks/01_build_system.ipynb](notebooks/01_build_system.ipynb) for a
complete worked example and JSON schema.

---

### Stage 2 — Equilibration (`MotorRow`)

`MotorRow` runs a five-step restrained equilibration protocol on the system
produced by Bridgeport, gradually releasing positional restraints:

| Step | Ensemble | Duration | Restraints |
|------|----------|----------|------------|
| 0    | Minimization | — | Protein + ligand |
| 1    | NVT | 250 ps | Membrane Z + protein XYZ |
| 2    | NPT (membrane) | 250 ps | Membrane Z + protein XYZ |
| 3    | NVT | 250 ps | Protein XYZ only |
| 4    | NPT (membrane) | 2.5 ns | Protein XYZ only |
| 5    | NPT (isotropic) | 2.5 ns | None |

Settings: T = 300 K, dt = 2 fs, DCD trajectory every 500 steps.

**CLI:**

```bash
chimpss-motor-row system.pdb system.xml output/ --lig_resname UNK
```

**Programmatic:**

```python
from chimpss.motor_row import MotorRow

mr = MotorRow(
    pdb_file="system.pdb",
    system_xml="system.xml",
    working_directory="output/",
    lig_resname="UNK",
)
mr.main("system.pdb")
```

See [notebooks/03_equilibrate_and_run.ipynb](notebooks/03_equilibrate_and_run.ipynb).

---

### Stage 3 — Production REMD (`FultonMarket`)

`FultonMarket` runs parallel-tempering replica exchange MD across a geometric
temperature ladder, with built-in auto-convergence detection.

**Key features:**

- Geometric temperature distribution between T_min and T_max.
- Automatic checkpointing to `saved_variables/` after each sub-simulation.
- Convergence detection via three independent metrics run in parallel:
  - Energy equilibration (pymbar `detect_equilibration`)
  - PCA of backbone dihedrals
  - Distance matrix comparison (Frobenius norm + Jensen-Shannon divergence)
    using residue contacts from [getContacts](https://getcontacts.github.io/),
    Cα distances, or torsion distances.
- Resumes automatically from the last checkpoint after a crash or OOM.
- MPI-ready via `run_fulton_market_mpi.py`.

**CLI (single process):**

```bash
chimpss-fulton-market system.pdb system.xml output/ \
    --T_min 300 --T_max 367 --n_replicates 68 \
    --sim_length 25 --total_sim_time 1200
```

**CLI (MPI, 4 GPUs):**

```bash
mpirun -np 4 chimpss-fulton-market-mpi system.pdb system.xml output/ \
    --T_min 300 --T_max 367 --n_replicates 68
```

**Programmatic:**

```python
from FultonMarket.fulton_market import FultonMarket

fm = FultonMarket(
    input_pdb="system.pdb",
    input_system="system.xml",
    T_min=300, T_max=367, n_replicates=68,
)
fm.run(
    iter_length=0.001,
    sim_length=25,
    total_sim_time=1200,
    output_dir="output/",
)
```

See [notebooks/03_equilibrate_and_run.ipynb](notebooks/03_equilibrate_and_run.ipynb).

---

### Analysis (`FultonMarketAnalysis`)

`FultonMarketAnalysis` loads the checkpointed trajectory data and provides
MBAR-reweighted observables, convergence diagnostics, and visualizations.

```python
from FultonMarket.analysis import FultonMarketAnalysis

ana = FultonMarketAnalysis(output_dir="output/")
ana.resample(n_resample=1000)            # MBAR importance resampling
ana.get_traj_pca()                       # PCA of backbone dihedrals
ana.calculate_weighted_rc(rc_values=…)  # weighted reaction coordinate
```

See [notebooks/04_analysis.ipynb](notebooks/04_analysis.ipynb).

---

### Optional tools

**Standalone minimizer:**

```python
from Minimizer.Minimizer import Minimizer

m = Minimizer(system_xml="system.xml", pdb_file="system.pdb",
              working_directory="min_output/")
m.run(lig_resname="UNK")
```

**Docking (Autodock Vina wrapper):**

```bash
chimpss-docking -r receptor.pdbqt -l ligand.pdbqt \
    --box_center 10,20,30 --box_dim 20,20,20
```

**System adjustment (force groups + HMR):**

```bash
chimpss-adjust-system system.pdb system_sys.xml
```

Outputs `system_FG_sys.xml` (force groups adjusted) and
`system_FG_HMR_sys.xml` (force groups + hydrogen mass repartitioning,
enabling 4 fs timesteps).

**OOM recovery:**

```bash
chimpss-recovery /path/to/output/replica_exchange/SIMNAME_REP
```

Truncates a corrupted netCDF and saves numpy checkpoints so that
`chimpss-fulton-market` can resume cleanly.

---

## CLI reference

| Command | Script | Description |
|---------|--------|-------------|
| `chimpss-bridgeport` | `scripts/run_bridgeport.py` | Build an OpenMM system from a JSON spec |
| `chimpss-motor-row` | `scripts/run_motor_row.py` | Run the 5-step equilibration protocol |
| `chimpss-fulton-market` | `scripts/run_fulton_market.py` | Run production REMD |
| `chimpss-fulton-market-mpi` | `scripts/run_fulton_market_mpi.py` | MPI wrapper for REMD (one GPU per rank) |
| `chimpss-docking` | `scripts/run_docking.py` | Run Autodock Vina docking |
| `chimpss-repair-protein` | `scripts/run_repair_protein.py` | Repair missing residues with MODELLER |
| `chimpss-adjust-system` | `scripts/adjust_system.py` | Add force groups and apply HMR |
| `chimpss-recovery` | `scripts/recovery.py` | Recover after an out-of-memory crash |

Run any command with `--help` for the full argument list.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [00_showcase.ipynb](notebooks/00_showcase.ipynb) | End-to-end pipeline demonstration |
| [01_build_system.ipynb](notebooks/01_build_system.ipynb) | Bridgeport system construction |
| [02_minimize.ipynb](notebooks/02_minimize.ipynb) | Standalone energy minimization |
| [03_equilibrate_and_run.ipynb](notebooks/03_equilibrate_and_run.ipynb) | MotorRow equilibration and FultonMarket REMD |
| [04_analysis.ipynb](notebooks/04_analysis.ipynb) | MBAR analysis and convergence diagnostics |
| [examples/](notebooks/examples/) | Worked examples: small molecules, peptides, analogues, docking |

---

## Key dependencies

| Package | Role |
|---------|------|
| [OpenMM](https://openmm.org) | MD engine |
| [openmmtools](https://openmmtools.readthedocs.io) | REMD / multistate sampler |
| [OpenFF Toolkit](https://docs.openforcefield.org) | Small-molecule force fields |
| [MDTraj](https://mdtraj.org) | Trajectory analysis |
| [MDAnalysis](https://mdanalysis.org) | Structure manipulation and selection |
| [pymbar](https://pymbar.readthedocs.io) | MBAR free-energy reweighting |
| [JAX](https://jax.readthedocs.io) | Vectorised geometry operations |
| [PDBFixer](https://github.com/openmm/pdbfixer) | PDB preparation |
| [MODELLER](https://salilab.org/modeller/) | Loop and residue repair |
| [Vina](https://vina.scripps.edu) | Molecular docking |
| [mpi4py](https://mpi4py.readthedocs.io) | MPI parallelism |
| [getContacts](https://getcontacts.github.io) | Dynamic contact networks (optional) |
