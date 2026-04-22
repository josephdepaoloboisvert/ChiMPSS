# Changelog

All notable changes to ChiMPSS are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `pyproject.toml`: package skeleton — `pip install -e .` now works; `requires-python = ">=3.8"`; loose dep pins; `[dev]`, `[mpi]`, `[docs]` extras
- `src/chimpss/__init__.py`: package entry point, `__version__ = "0.1.0"`
- `.pre-commit-config.yaml`: ruff + black + nbstripout hooks
- `.github/workflows/ci.yml`: lint + no-deps install smoke test across Python 3.8 / 3.10 / 3.12
- `CLAUDE.md`: agent context file — active reorganization status, resumption phrase, key preferences
