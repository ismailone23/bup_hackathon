"""Validate the service end-to-end against the official public sample cases.

Usage:
    python validate_public.py [cases.json] [base_url]

Compares interpreted directive semantics (ignoring explanation wording) and the
recalculated optimal cost against each reference within the 0.01 tolerance.
"""

import json
import sys
import urllib.error
import urllib.request

import main

DEFAULT_CASES = "BUP_CSE_FEST_2026_Participant_Docs/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
VALUE_FIELD = {
    "solar_reduction": "factor",
    "minimum_battery_reserve": "minimum_energy_kwh",
    "max_grid_window": "max_grid_kwh",
}


def post(base_url, payload):
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/optimize-energy",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def semantics(interpretation):
    out = []
    for item in interpretation:
        adjustment = item.get("structured_adjustment") or {}
        field = VALUE_FIELD.get(item["directive_type"])
        out.append((
            item["directive_type"],
            item["applies"],
            tuple(sorted(adjustment.get("hours", []))) if item["applies"] else None,
            None if field is None or not item["applies"] else round(float(adjustment[field]), 6),
        ))
    return out


def replay_plan(case_input, result):
    """Independently replay the returned schedule and recompute its totals."""
    payload = main.OptimizeRequest.model_validate(case_input)
    directives = [main.DirectiveInterpretation.model_validate(item) for item in result["directive_interpretation"]]
    plan = [main.HourPlan.model_validate(row) for row in result["hourly_plan"]]
    main.replay(payload, directives, plan)
    grid = sum(row.grid_kwh for row in plan)
    cost = sum(row.grid_kwh * case_input["hours"][row.hour]["tariff_bdt_per_kwh"] for row in plan)
    return grid, cost


def main_entry():
    cases_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASES
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"

    with open(cases_path, encoding="utf-8") as file:
        cases = json.load(file)["cases"]

    failures = 0
    for case in cases:
        status, result = post(base_url, case["input"])
        expected = case["expected_output"]
        if status != 200:
            failures += 1
            print(f"{case['id']:<10} FAIL status={status} {result}")
            continue
        semantics_ok = semantics(result["directive_interpretation"]) == semantics(expected["directive_interpretation"])
        try:
            grid, cost = replay_plan(case["input"], result)
            plan_ok = (
                abs(grid - float(result["total_grid_kwh"])) <= 1e-6
                and abs(cost - float(result["total_cost_bdt"])) <= 1e-6
            )
        except Exception:
            cost = None
            plan_ok = False
        reference = float(expected["total_cost_bdt"])
        cost_ok = cost is not None and abs(cost - reference) <= 0.01
        ok = semantics_ok and plan_ok and cost_ok
        failures += 0 if ok else 1
        mark = "PASS" if ok else "FAIL"
        detail = [w for w, s in (("semantics", semantics_ok), ("plan", plan_ok), ("cost", cost_ok)) if not s]
        shown = "  n/a" if cost is None else f"{cost:>10.2f}"
        print(f"{case['id']:<10} {mark} cost={shown} ref={reference:>10.2f} {' '.join(detail)}")

    print(f"\ncases={len(cases)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main_entry())
