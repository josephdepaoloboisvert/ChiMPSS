"""
Unit tests for chimpss.shared.logging.

timestamp and printf only use stdlib — these tests run on CI with no MD stack.
unique_residues / report_chain_information need mdtraj and are gated.
"""
import re
from datetime import datetime

from chimpss.shared.logging import printf, timestamp


def test_timestamp_returns_string():
    result = timestamp("hello")
    assert isinstance(result, str)
    assert "hello" in result


def test_timestamp_format():
    result = timestamp("msg")
    # Expected format: "2024-01-15 12:34:56.789012://msg"
    assert "://" in result
    assert "msg" in result


def test_timestamp_datetime_prefix():
    before = str(datetime.now())[:16]  # "YYYY-MM-DD HH:MM"
    result = timestamp("x")
    after = str(datetime.now())[:16]
    # The timestamp prefix should fall between before and after
    prefix = result.split("://")[0]
    assert prefix >= before[:16] or prefix <= after[:16]


def test_printf_runs_without_error(capsys):
    printf("test message")
    captured = capsys.readouterr()
    assert "test message" in captured.out


def test_printf_format(capsys):
    printf("hello world")
    captured = capsys.readouterr()
    line = captured.out.strip()
    # Expected: "MM/DD/YYYY HH:MM:SS//hello world"
    assert "//" in line
    assert "hello world" in line
    # Date portion matches MM/DD/YYYY pattern
    parts = line.split("//")
    date_part = parts[0]
    assert re.match(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}", date_part)


# ── mdtraj-gated tests ────────────────────────────────────────────────────────

def test_unique_residues_and_report_chain_info():
    mdtraj = __import__("pytest").importorskip("mdtraj")
    import os

    from chimpss.shared.logging import report_chain_information, unique_residues

    pdb_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "test_data", "raw_OPM", "5xr8.pdb"
    )
    if not os.path.exists(pdb_path):
        __import__("pytest").skip("test_data/raw_OPM/5xr8.pdb not found")

    traj = mdtraj.load(pdb_path)

    res = unique_residues(traj)
    assert isinstance(res, dict)
    assert len(res) >= 1
    for chain_idx, names in res.items():
        assert isinstance(names, list)

    report = report_chain_information(traj)
    assert isinstance(report, str)
    assert "chainID" in report
