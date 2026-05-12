# CLAUDE.md

Context for Claude Code (and other agents) working in this repo.

## What this is

**ChiMPSS** — Chicago Membrane Protein Simulation Suite. A three-stage molecular-dynamics pipeline for membrane proteins (GPCRs, KOR as the primary test case):

1. **Bridgeport** — system construction (PDB + FASTA + SMILES → solvated OpenMM system XML)
2. **MotorRow** — 5-step equilibration (NVT/NPT with gradient restraints)
3. **FultonMarket** — production parallel-tempering REMD via openmmtools

Stages are file-coupled only — each stage consumes the previous one's PDB/XML outputs. No upward Python imports exist between them.

## ⚠️ Active reorganization — read before editing

This repo is mid-migration from a flat research layout into a `pip install -e .`-able package named `chimpss` (src-layout, submodule API).

**Before making any edits:**

1. Read the active plan: `C:\Users\josep\.claude\plans\hello-claude-i-ve-got-lazy-pudding.md`
2. Run `git log --oneline main..` to see which migration phases have already landed.
3. Check the plan's `## Done` section for completed phases.
4. Do not introduce new files at the repo root — new code goes under `src/chimpss/...` as the migration progresses.
5. Do not add upper version pins to dependencies — the MD stack (OpenMM, openmmtools, openff, rdkit, jax, mpi4py) breaks on caps. Loose pins only.

## Resumption phrase

To continue the migration in a new session, the user should say:

> *"continue the chimpss reorg — which phase is next?"*

On that prompt, the agent should:
1. Re-read the plan file above.
2. Check `git log --oneline main..` and the plan's `## Done` section.
3. Rebuild the TodoWrite list from whichever phases remain `pending`.
4. Confirm with the user which phase to work on before making any changes.

## Key preferences (also in auto-memory)

- Python floor: `>=3.8`
- Dependencies: loose/bare names, no upper bounds
- License: **not yet chosen** — do not add a `LICENSE` file
- Package name: `chimpss` (lowercase)
- API: `from chimpss.bridgeport import Bridgeport` (submodule style)
- Migration cadence: one PR per phase
- Public API of the three stage classes stays stable during the reorg; follow-on phases (9–11) can refactor them against the new shared utilities once the structure lands.

## Runtime targets

- Development: local (Windows, `bash` shell available)
- Tests: dev machine runs unit tests only (`pytest -m "not slow and not gpu"`)
- Regression / simulations: run asynchronously on SDSC Expanse (HPC) or cloud A100 GPUs — not blocking for merges
- In-flight Expanse jobs must keep working — old root paths get shim re-exports for one release cycle before deletion
