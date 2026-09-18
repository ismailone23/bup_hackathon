import math

import pytest

from main import HourPlan


@pytest.mark.parametrize("field", ["grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh"])
def test_fahim_rejects_non_finite_plan_values(field):
    values = {
        "hour": 0,
        "grid_kwh": 0.0,
        "solar_used_kwh": 0.0,
        "battery_action": "idle",
        "battery_kwh": 0.0,
        "battery_energy_after_kwh": 10.0,
    }
    values[field] = math.nan
    with pytest.raises(ValueError):
        HourPlan.model_validate(values)
