import math

import numpy as np
from scipy.optimize import linprog

from .errors import ServiceError
from .models import DirectiveInterpretation, DirectiveType, HourPlan, OptimizeRequest, OptimizeResponse

SUMMARY_PHRASES = {
    DirectiveType.SOLAR_REDUCTION: "the reduced solar availability",
    DirectiveType.MINIMUM_BATTERY_RESERVE: "the battery reserve floor",
    DirectiveType.NO_CHARGE_WINDOW: "the no-charge windows",
    DirectiveType.NO_DISCHARGE_WINDOW: "the no-discharge windows",
    DirectiveType.MAX_GRID_WINDOW: "the grid import caps",
}


def validate_directives(items: list[DirectiveInterpretation], note_count: int, capacity: float) -> None:
    if len(items) != note_count or [item.note_index for item in items] != list(range(note_count)):
        raise ValueError("interpretations must map one-to-one to notes")
    field_for = {
        DirectiveType.SOLAR_REDUCTION: "factor",
        DirectiveType.MINIMUM_BATTERY_RESERVE: "minimum_energy_kwh",
        DirectiveType.MAX_GRID_WINDOW: "max_grid_kwh",
    }
    for item in items:
        if not item.explanation.strip():
            raise ValueError("explanation must not be blank")
        if item.directive_type == DirectiveType.NO_OP:
            if item.applies or item.structured_adjustment is not None:
                raise ValueError("invalid no_op")
            continue
        adjustment = item.structured_adjustment
        if not item.applies or adjustment is None:
            raise ValueError("active directive requires an adjustment")
        if adjustment.hours != sorted(set(adjustment.hours)) or any(hour not in range(24) for hour in adjustment.hours):
            raise ValueError("invalid directive hours")
        populated = {name for name in ("factor", "minimum_energy_kwh", "max_grid_kwh") if getattr(adjustment, name) is not None}
        expected = field_for.get(item.directive_type)
        if populated != ({expected} if expected else set()):
            raise ValueError("adjustment does not match directive type")
        if expected == "factor" and not 0 <= adjustment.factor <= 1:
            raise ValueError("solar factor must be between zero and one")
        if expected == "minimum_energy_kwh" and adjustment.minimum_energy_kwh > capacity:
            raise ValueError("reserve exceeds battery capacity")


def apply_directives(payload: OptimizeRequest, directives: list[DirectiveInterpretation]):
    solar = np.array([hour.solar_kwh for hour in payload.hours], dtype=float)
    demand = np.array([hour.demand_kwh for hour in payload.hours], dtype=float)
    reserve = np.full(24, payload.battery.minimum_energy_kwh, dtype=float)
    charge_limit = np.full(24, payload.battery.max_charge_kwh_per_hour, dtype=float)
    discharge_limit = np.full(24, payload.battery.max_discharge_kwh_per_hour, dtype=float)
    # Grid import cannot exceed demand plus the maximum possible charging load.
    grid_cap = demand + charge_limit
    for directive in directives:
        if not directive.applies:
            continue
        adjustment = directive.structured_adjustment
        hours = adjustment.hours
        if directive.directive_type == DirectiveType.SOLAR_REDUCTION:
            solar[hours] *= adjustment.factor
        elif directive.directive_type == DirectiveType.MINIMUM_BATTERY_RESERVE:
            reserve[hours] = np.maximum(reserve[hours], adjustment.minimum_energy_kwh)
        elif directive.directive_type == DirectiveType.NO_CHARGE_WINDOW:
            charge_limit[hours] = 0
        elif directive.directive_type == DirectiveType.NO_DISCHARGE_WINDOW:
            discharge_limit[hours] = 0
        elif directive.directive_type == DirectiveType.MAX_GRID_WINDOW:
            grid_cap[hours] = np.minimum(grid_cap[hours], adjustment.max_grid_kwh)
    return solar, reserve, charge_limit, discharge_limit, grid_cap


def solve(payload: OptimizeRequest, directives: list[DirectiveInterpretation]) -> OptimizeResponse:
    try:
        validate_directives(directives, len(payload.operator_notes), payload.battery.capacity_kwh)
    except (TypeError, ValueError) as exc:
        raise ServiceError(422, "INVALID_INTERPRETATION", str(exc)) from exc
    solar, reserve, charge_limit, discharge_limit, grid_cap = apply_directives(payload, directives)
    demand = np.array([hour.demand_kwh for hour in payload.hours])
    tariff = np.array([hour.tariff_bdt_per_kwh for hour in payload.hours])
    # Variable blocks: grid, solar, charge, discharge, end-of-hour energy.
    # The objective is exactly the stated grid cost; charge/discharge carry no weight
    # so the LP optimum is the true minimum cost. Only the net battery flow is emitted.
    n = 120
    objective = np.zeros(n)
    objective[:24] = tariff
    equalities, targets = [], []
    for hour in range(24):
        balance = np.zeros(n)
        balance[hour] = 1
        balance[24 + hour] = 1
        balance[48 + hour] = -1
        balance[72 + hour] = 1
        equalities.append(balance)
        targets.append(demand[hour])

        transition = np.zeros(n)
        transition[48 + hour] = -1
        transition[72 + hour] = 1
        transition[96 + hour] = 1
        if hour:
            transition[96 + hour - 1] = -1
            target = 0
        else:
            target = payload.battery.initial_energy_kwh
        equalities.append(transition)
        targets.append(target)
    terminal = np.zeros(n)
    terminal[119] = 1
    equalities.append(terminal)
    targets.append(payload.battery.initial_energy_kwh)

    bounds = []
    for hour in range(24):
        bounds.append((0, None if math.isinf(grid_cap[hour]) else grid_cap[hour]))
    bounds += [(0, solar[h]) for h in range(24)]
    bounds += [(0, charge_limit[h]) for h in range(24)]
    bounds += [(0, discharge_limit[h]) for h in range(24)]
    bounds += [(reserve[h], payload.battery.capacity_kwh) for h in range(24)]

    lp = dict(A_eq=np.array(equalities), b_eq=np.array(targets), bounds=bounds)
    result = linprog(objective, method="highs", **lp)
    if not result.success or result.x is None:
        # HiGHS dual simplex can report an unrecognized status on degenerate LPs that the
        # interior-point solver handles; keep whichever feasible vector minimizes the
        # objective. The replay below gates validity.
        fallback = linprog(objective, method="highs-ipm", **lp)
        if fallback.x is not None and (result.x is None or fallback.fun <= result.fun):
            result = fallback
    if result.x is None:
        raise ServiceError(422, "INFEASIBLE_CONSTRAINTS", "Operator constraints cannot be satisfied.")

    grid, solar_used, charge, discharge, energy = np.split(result.x, 5)
    plan = []
    for hour in range(24):
        net = clean(charge[hour] - discharge[hour])
        if net > 0:
            action, amount = "charge", net
        elif net < 0:
            action, amount = "discharge", -net
        else:
            action, amount = "idle", 0.0
        plan.append(HourPlan(
            hour=hour,
            grid_kwh=clean(grid[hour]),
            solar_used_kwh=clean(solar_used[hour]),
            battery_action=action,
            battery_kwh=amount,
            battery_energy_after_kwh=clean(energy[hour]),
        ))
    replay(payload, directives, plan)
    total_grid = clean(sum(item.grid_kwh for item in plan))
    total_cost = clean(sum(item.grid_kwh * payload.hours[item.hour].tariff_bdt_per_kwh for item in plan))
    active = sorted({SUMMARY_PHRASES[item.directive_type] for item in directives if item.applies})
    if active:
        if len(active) == 1:
            satisfied = active[0]
        else:
            satisfied = ", ".join(active[:-1]) + f" and {active[-1]}"
        summary = f"Minimum-cost schedule satisfies {satisfied} and restores the initial battery energy."
    else:
        summary = "Minimum-cost schedule satisfies the base operating constraints and restores the initial battery energy."
    response = OptimizeResponse(
        scenario_id=payload.scenario_id,
        directive_interpretation=directives,
        hourly_plan=plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=clean(max(item.grid_kwh for item in plan)),
        plan_summary=summary,
    )
    if (
        response.scenario_id != payload.scenario_id
        or abs(response.total_grid_kwh - sum(item.grid_kwh for item in plan)) > 1e-6
        or abs(response.total_cost_bdt - sum(item.grid_kwh * payload.hours[item.hour].tariff_bdt_per_kwh for item in plan)) > 1e-6
        or abs(response.peak_grid_kwh - max(item.grid_kwh for item in plan)) > 1e-6
    ):
        raise ServiceError(500, "VALIDATION_FAILED", "Generated response failed consistency validation.")
    return response


def clean(value: float) -> float:
    value = round(float(value), 6)
    return 0.0 if abs(value) < 1e-7 else value


def replay(payload: OptimizeRequest, directives: list[DirectiveInterpretation], plan: list[HourPlan]) -> None:
    tol = 1e-5
    if len(plan) != 24 or [item.hour for item in plan] != list(range(24)):
        raise ServiceError(500, "VALIDATION_FAILED", "Generated schedule must contain hours 0 through 23 in order.")
    solar, reserve, charge_limit, discharge_limit, grid_cap = apply_directives(payload, directives)
    previous = payload.battery.initial_energy_kwh
    for item in plan:
        source = payload.hours[item.hour]
        charge = item.battery_kwh if item.battery_action == "charge" else 0
        discharge = item.battery_kwh if item.battery_action == "discharge" else 0
        valid = (
            math.isfinite(item.grid_kwh)
            and math.isfinite(item.solar_used_kwh)
            and math.isfinite(item.battery_kwh)
            and math.isfinite(item.battery_energy_after_kwh)
            and item.grid_kwh >= 0
            and item.solar_used_kwh >= 0
            and item.battery_kwh >= 0
            and item.battery_energy_after_kwh >= 0
            and not (item.battery_action == "idle" and item.battery_kwh != 0)
            and not (item.battery_action != "idle" and item.battery_kwh <= 0)
            and abs(item.grid_kwh + item.solar_used_kwh + discharge - source.demand_kwh - charge) <= tol
            and abs(item.battery_energy_after_kwh - previous - charge + discharge) <= tol
            and 0 <= item.solar_used_kwh <= solar[item.hour] + 1e-6
            and reserve[item.hour] - 1e-6 <= item.battery_energy_after_kwh <= payload.battery.capacity_kwh + 1e-6
            and charge <= charge_limit[item.hour] + 1e-6
            and discharge <= discharge_limit[item.hour] + 1e-6
            and item.grid_kwh <= grid_cap[item.hour] + 1e-6
        )
        if not valid:
            raise ServiceError(500, "VALIDATION_FAILED", "Generated schedule failed validation.")
        previous = item.battery_energy_after_kwh
    if abs(previous - payload.battery.initial_energy_kwh) > tol:
        raise ServiceError(500, "VALIDATION_FAILED", "Generated schedule failed terminal validation.")
