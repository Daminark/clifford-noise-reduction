"""Tests for the noise models in ``clinr.noise_models``."""

import pytest

from clinr import noise_models

RATES = ("p1", "p2", "p_m", "p_prep")


@pytest.mark.parametrize("name", RATES)
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_noise_model_rejects_a_rate_outside_the_unit_interval(name, value):
    rates = dict.fromkeys(RATES, 0.001) | {name: value}

    with pytest.raises(ValueError, match=f"{name} must lie in"):
        noise_models.NoiseModel(**rates)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_noise_model_accepts_the_endpoints(value):
    noise_models.NoiseModel(p1=value, p2=value, p_m=value, p_prep=value)


def test_asymmetric_noise_is_the_papers_section_v_regime_by_default():
    # Two-qubit gates ten times noisier than every other operation.
    assert noise_models.asymmetric_noise(1e-3) == noise_models.NoiseModel(
        p1=1e-4, p2=1e-3, p_m=1e-4, p_prep=1e-4
    )


@pytest.mark.parametrize("ratio", [0.0, -2.0])
def test_asymmetric_noise_rejects_a_ratio_that_is_not_positive(ratio):
    with pytest.raises(ValueError, match="ratio must be positive"):
        noise_models.asymmetric_noise(1e-3, ratio)
