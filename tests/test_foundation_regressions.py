"""Independent signal-model and finite-sample target checks from the audit."""

import numpy as np
import pytest
from scipy import stats

from src.math_models.energy_detection import analytical_pd, compute_theoretical_threshold
from src.math_models.hypothesis_testing import HypothesisConfig, HypothesisTestingSimulator, generate_pu_signal
from src.math_models.noise_models import compute_sample_complexity


def test_channel_gain_statistics_match_reference_snr_convention():
    """Fixed transmitter power implies received SNR is scaled by |h| squared."""
    simulator = HypothesisTestingSimulator(HypothesisConfig(noise_variance=1., snr_db=0.), seed=42)
    result = simulator.theoretical_statistics(channel_gain=.1j)
    assert result['var_h1'] == pytest.approx(1.01)
    assert result['snr_linear'] == pytest.approx(.01)
    assert result['snr_db'] == pytest.approx(-20.)


def test_real_qpsk_requested_power_is_not_halved():
    """Real-signal mode documents its own target power, rather than a half-power projection."""
    signal = generate_pu_signal(1000, signal_power=2., modulation='qpsk', complex_signal=False,
                                rng=np.random.default_rng(42))
    assert np.mean(signal**2) == pytest.approx(2.)


def test_fixed_energy_clt_uses_conditional_variance():
    """Gaussian approximation must use the same fixed-energy model as exact Pd."""
    n, snr, pfa = 32, -4., .05
    linear = 10**(snr/10)
    threshold = compute_theoretical_threshold(pfa, n, method='clt')
    expected = stats.norm.sf((threshold-1-linear)/np.sqrt((1+2*linear)/n))
    assert analytical_pd(pfa, snr, n, method='clt') == pytest.approx(expected)


@pytest.mark.parametrize('snr', [-3., -2., 0., 5.])
def test_returned_sample_requirement_attains_exact_worst_case_target(snr):
    """Check returned N with direct independent chi-square expressions, not the solver itself."""
    rho = 10**.1
    n = int(compute_sample_complexity(snr, 1., pfa_target=.05, pd_target=.9))
    threshold = rho * stats.chi2.isf(.05, 2*n)/(2*n)
    minimum_noise = 1/rho
    attained_pd = stats.ncx2.sf(2*n*threshold/minimum_noise, 2*n, 2*n*10**(snr/10)/minimum_noise)
    assert attained_pd >= .9 - 1e-12


@pytest.mark.parametrize('pfa,pd', [(0., .9), (.05, 1.), (.8, .5), (np.nan, .9)])
def test_invalid_sample_targets_rejected(pfa, pd):
    """Robust target search requires finite, ordered interior probabilities."""
    with pytest.raises(ValueError):
        compute_sample_complexity(0., .5, pfa_target=pfa, pd_target=pd)


@pytest.mark.parametrize('variance', [0., -1., np.nan, np.inf])
def test_threshold_invalid_variance_rejected(variance):
    """A nominal threshold with invalid power cannot be used for calibration."""
    with pytest.raises(ValueError):
        compute_theoretical_threshold(.05, 1024, noise_variance=variance)


def test_exact_pd_supports_gaussian_signal_distribution():
    """Independent Gaussian H1 samples have scaled central energy, not fixed-energy noncentral energy."""
    n, snr, pfa = 32, -4., .05
    threshold = stats.chi2.isf(pfa, 2*n)
    expected = stats.chi2.sf(threshold/(1+10**(snr/10)), 2*n)
    assert analytical_pd(pfa, snr, n, signal_model='gaussian') == pytest.approx(expected)
