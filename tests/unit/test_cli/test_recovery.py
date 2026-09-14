"""Tests for chimpss-recovery's file discovery and block selection.

Both were wrong: the tool hardcoded 'output.ncdf' while FultonMarket derives
the name from --protein-name/--ligand-name, and it picked the block index from
len(os.listdir()), which counts files as blocks.
"""
import os
import pytest

from chimpss.cli.recovery import find_trajectory_ncdf, latest_block_dir


def test_finds_the_derived_trajectory_name(tmp_path):
    p = tmp_path / 'NTSR1_ML-301.trajectory.ncdf'
    p.write_text('')
    assert find_trajectory_ncdf(str(tmp_path)) == str(p)


def test_ignores_the_checkpoint_companion(tmp_path):
    """Truncating the checkpoint instead of the trajectory yields a bad block."""
    (tmp_path / 'NTSR1_ML-301.trajectory.ncdf').write_text('')
    (tmp_path / 'NTSR1_ML-301.trajectory_checkpoint.ncdf').write_text('')
    assert find_trajectory_ncdf(str(tmp_path)).endswith('NTSR1_ML-301.trajectory.ncdf')


def test_raises_when_no_trajectory_present(tmp_path):
    with pytest.raises(FileNotFoundError, match='trajectory.ncdf'):
        find_trajectory_ncdf(str(tmp_path))


def test_raises_when_ambiguous(tmp_path):
    (tmp_path / 'A_x.trajectory.ncdf').write_text('')
    (tmp_path / 'B_y.trajectory.ncdf').write_text('')
    with pytest.raises(RuntimeError, match='several'):
        find_trajectory_ncdf(str(tmp_path))


def test_explicit_override_wins(tmp_path):
    (tmp_path / 'A_x.trajectory.ncdf').write_text('')
    (tmp_path / 'B_y.trajectory.ncdf').write_text('')
    chosen = tmp_path / 'B_y.trajectory.ncdf'
    assert find_trajectory_ncdf(str(tmp_path), str(chosen)) == str(chosen)


def test_explicit_override_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_trajectory_ncdf(str(tmp_path), str(tmp_path / 'nope.ncdf'))


def test_latest_block_is_highest_numbered_not_count(tmp_path):
    for n in (0, 1, 2, 10):            # 4 dirs, but the latest is 10, not 3
        os.makedirs(tmp_path / str(n))
    assert latest_block_dir(str(tmp_path)).endswith('10')


def test_latest_block_ignores_stray_files(tmp_path):
    for n in (0, 1, 2):
        os.makedirs(tmp_path / str(n))
    (tmp_path / 'notes.txt').write_text('')
    assert latest_block_dir(str(tmp_path)).endswith('2')


def test_latest_block_raises_when_empty(tmp_path):
    with pytest.raises(RuntimeError, match='no numbered block'):
        latest_block_dir(str(tmp_path))
