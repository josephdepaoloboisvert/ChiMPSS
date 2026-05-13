import matplotlib
import pytest

matplotlib.use('Agg')

pytest.importorskip("openmm", reason="openmm not installed — skipping fultonmarket plotting tests")


def test_imports():
    from chimpss.fultonmarket import add_equil_metric_to_plot, plot_convergence_metric
    assert callable(plot_convergence_metric)
    assert callable(add_equil_metric_to_plot)


def test_color_map():
    from chimpss.fultonmarket.retro_convergence import _RETRO_COLOR_MAP
    assert {'torsion', 'alpha_carbon', 'contact'} == set(_RETRO_COLOR_MAP)


def _minimal_metrics():
    return {
        0: {
            'jsd': {'torsion': [0.1, 0.08], 'alpha_carbon': [0.2], 'contact': [0.05]},
            'frobenius': {'torsion': [0.3], 'alpha_carbon': [0.4], 'contact': [0.1]},
            'equil_fraction': 0.3,
        },
        1: {
            'jsd': {'torsion': [0.07], 'alpha_carbon': [0.15], 'contact': [0.04]},
            'frobenius': {'torsion': [0.25], 'alpha_carbon': [0.35], 'contact': [0.09]},
            'equil_fraction': 0.25,
        },
    }


def test_plot_minimal_no_history():
    from chimpss.fultonmarket import plot_convergence_metric
    plot_convergence_metric(_minimal_metrics(), func='jsd', distype='torsion', history=False)


def test_plot_minimal_with_history():
    from chimpss.fultonmarket import plot_convergence_metric
    plot_convergence_metric(_minimal_metrics(), func='jsd', distype='torsion', history=True)


def test_plot_show_all():
    from chimpss.fultonmarket import plot_convergence_metric
    plot_convergence_metric(_minimal_metrics(), func='jsd', distype='alpha_carbon', show_all=True)


def test_plot_empty_dict():
    from chimpss.fultonmarket import plot_convergence_metric
    plot_convergence_metric({})


def test_add_equil_minimal():
    from chimpss.fultonmarket import add_equil_metric_to_plot
    add_equil_metric_to_plot(_minimal_metrics())


def test_add_equil_empty():
    from chimpss.fultonmarket import add_equil_metric_to_plot
    add_equil_metric_to_plot({})


def test_custom_color_map():
    from chimpss.fultonmarket import plot_convergence_metric
    custom = {'torsion': 'red', 'alpha_carbon': 'blue', 'contact': 'green'}
    plot_convergence_metric(_minimal_metrics(), distype='torsion', color_map=custom)
