"""
Unit tests for pure-logic functions in chimpss.fultonmarket.retro_convergence.

retro_convergence.py only imports os / tempfile / datetime / typing / numpy
at module level, so these tests run on CI with just numpy — no openmm needed.

We load the submodule directly via importlib to avoid triggering
chimpss.fultonmarket.__init__ (which imports openmm at module level).
"""
import importlib.util
import os

# ── load submodule without triggering the package __init__ ────────────────────
_RC_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "src", "chimpss", "fultonmarket", "retro_convergence.py",
)
_RC_PATH = os.path.normpath(_RC_PATH)


def _load_rc():
    spec = importlib.util.spec_from_file_location("retro_convergence", _RC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load_rc()


# ── resolve_write_dir ─────────────────────────────────────────────────────────

def test_resolve_write_dir_read_only():
    result = rc.resolve_write_dir("/some/dir", None, 0, read_only=True)
    assert result is None


def test_resolve_write_dir_with_cache(tmp_path):
    cache = str(tmp_path / "cache")
    result = rc.resolve_write_dir("/src", cache, 3, read_only=False)
    expected = os.path.join(cache, "saved_variables", "3")
    assert result == expected
    assert os.path.isdir(result)


def test_resolve_write_dir_no_cache(tmp_path):
    src = str(tmp_path)
    result = rc.resolve_write_dir(src, None, 0, read_only=False)
    assert result == src


# ── resolve_cache_dir ─────────────────────────────────────────────────────────

def test_resolve_cache_dir_none():
    assert rc.resolve_cache_dir(None, 5) is None


def test_resolve_cache_dir_returns_path():
    result = rc.resolve_cache_dir("/my/cache", 7)
    assert result == os.path.join("/my/cache", "saved_variables", "7")


# ── build_checks ─────────────────────────────────────────────────────────────

def _all_pass_frob():
    return {name: {0: 0.01} for name in ("torsion", "alpha_carbon", "contact")}


def _all_pass_jsd():
    return {name: {0: 0.01} for name in ("torsion", "alpha_carbon", "contact")}


def _all_fail_frob():
    return {name: {0: 0.5} for name in ("torsion", "alpha_carbon", "contact")}


def _all_fail_jsd():
    return {name: {0: 0.5} for name in ("torsion", "alpha_carbon", "contact")}


def test_build_checks_all_pass():
    checks = rc.build_checks(
        sim_no=50, total_n_sims=100, minimum_fraction=0.3,
        equil_fraction=0.3, max_equil_fraction=0.8,
        frob_results=_all_pass_frob(), jsd_results=_all_pass_jsd(),
        frobenius_thresh=0.1, jsd_thresh=0.1,
    )
    assert checks["STOP"] is True
    assert all(v for k, v in checks.items())


def test_build_checks_stop_false_when_early():
    checks = rc.build_checks(
        sim_no=5, total_n_sims=100, minimum_fraction=0.3,
        equil_fraction=0.3, max_equil_fraction=0.8,
        frob_results=_all_pass_frob(), jsd_results=_all_pass_jsd(),
        frobenius_thresh=0.1, jsd_thresh=0.1,
    )
    assert checks["STOP"] is False
    assert checks["Past minimum simulation fraction"] is False


def test_build_checks_stop_false_high_equil():
    checks = rc.build_checks(
        sim_no=50, total_n_sims=100, minimum_fraction=0.3,
        equil_fraction=0.9, max_equil_fraction=0.8,
        frob_results=_all_pass_frob(), jsd_results=_all_pass_jsd(),
        frobenius_thresh=0.1, jsd_thresh=0.1,
    )
    assert checks["STOP"] is False


def test_build_checks_stop_false_high_frob():
    checks = rc.build_checks(
        sim_no=50, total_n_sims=100, minimum_fraction=0.3,
        equil_fraction=0.3, max_equil_fraction=0.8,
        frob_results=_all_fail_frob(), jsd_results=_all_pass_jsd(),
        frobenius_thresh=0.1, jsd_thresh=0.1,
    )
    assert checks["STOP"] is False
    assert checks["Torsion Frobenius converged"] is False


def test_build_checks_empty_scores_fail():
    empty = {name: {} for name in ("torsion", "alpha_carbon", "contact")}
    checks = rc.build_checks(
        sim_no=50, total_n_sims=100, minimum_fraction=0.3,
        equil_fraction=0.3, max_equil_fraction=0.8,
        frob_results=empty, jsd_results=empty,
        frobenius_thresh=0.1, jsd_thresh=0.1,
    )
    # No previous checkpoints → all convergence checks fail
    assert checks["STOP"] is False
    assert checks["Torsion Frobenius converged"] is False


def test_build_checks_keys():
    checks = rc.build_checks(
        sim_no=50, total_n_sims=100, minimum_fraction=0.3,
        equil_fraction=0.3, max_equil_fraction=0.8,
        frob_results=_all_pass_frob(), jsd_results=_all_pass_jsd(),
        frobenius_thresh=0.1, jsd_thresh=0.1,
    )
    assert "STOP" in checks
    assert "Past minimum simulation fraction" in checks
    assert "Torsion Frobenius converged" in checks
    assert "Torsion JSD converged" in checks
    assert "Alpha-carbon Frobenius converged" in checks
    assert "Contact JSD converged" in checks


# ── print_summary_table ───────────────────────────────────────────────────────

def test_print_summary_table_empty(capsys):
    rc.print_summary_table({}, total_n_sims=100)
    assert capsys.readouterr().out == ""


def test_print_summary_table_output(capsys):
    checks_0 = rc.build_checks(
        sim_no=30, total_n_sims=100, minimum_fraction=0.3,
        equil_fraction=0.3, max_equil_fraction=0.8,
        frob_results=_all_pass_frob(), jsd_results=_all_pass_jsd(),
        frobenius_thresh=0.1, jsd_thresh=0.1,
    )
    checks_1 = rc.build_checks(
        sim_no=60, total_n_sims=100, minimum_fraction=0.3,
        equil_fraction=0.3, max_equil_fraction=0.8,
        frob_results=_all_pass_frob(), jsd_results=_all_pass_jsd(),
        frobenius_thresh=0.1, jsd_thresh=0.1,
    )
    report = {30: checks_0, 60: checks_1}
    log_lines = []
    rc.print_summary_table(report, total_n_sims=100, _printf=log_lines.append)
    combined = "\n".join(log_lines)
    assert "PASS" in combined or "FAIL" in combined
    assert "STOP" in combined


# ── log_mode ──────────────────────────────────────────────────────────────────

def test_log_mode_read_only(capsys):
    messages = []
    rc.log_mode(read_only=True, output_cache_dir=None, _printf=messages.append)
    assert any("read-only" in m for m in messages)


def test_log_mode_cache(capsys):
    messages = []
    rc.log_mode(read_only=False, output_cache_dir="/some/cache", _printf=messages.append)
    assert any("/some/cache" in m for m in messages)


def test_log_mode_normal(capsys):
    messages = []
    rc.log_mode(read_only=False, output_cache_dir=None, _printf=messages.append)
    assert any("normal" in m for m in messages)
