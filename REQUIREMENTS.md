# GridWise LLM — Requirements

## 1. Goal and authoritative sources

Build one publicly reachable HTTP service that interprets campus operator notes with a language-capable generative model, validates the resulting directives, and returns a valid minimum-cost 24-hour energy schedule.

Source priority:

1. `BUP_CSE_FEST_2026_Preliminary_Problem_Statement_GridWise_LLM.pdf`: API contract, interpretation, energy rules, and validity.
2. `BUP_CSE_FEST_2026_Participant_Guide_&_Evaluation_Rubric_GridWise_LLM.pdf`: submission, deployment, performance, scoring, and tie-breaks.
3. `BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json`: ten worked public references; equivalent optimal schedules are accepted.

This document translates those sources into implementation requirements. Decisions and assumptions are explicitly identified. `IMPLEMENTATION_PLAN.md` describes how to code and verify them.

**Status:** planning specification; implementation and deployment are not yet verified.

## 2. Scope

Required product:

- `GET /health`.
- `POST /optimize-energy`.
- Real LLM interpretation of every operator note.
- Deterministic input/output validation and directive application.
- A mathematical optimizer and independent schedule replay validator.
- Public API deployment, source repository, reproducible README, pullable Docker fallback, and a maximum three-minute solution video.

A frontend, user accounts, database, live energy feeds, model training, and billing integration are outside the challenge requirements. The recommended implementation is a stateless API with optional bounded in-memory interpretation caching.

## 3. HTTP contract

| ID | Requirement | Acceptance criterion |
| --- | --- | --- |
| API-01 | Health readiness | `GET /health` returns HTTP 200 and `{"status":"ok"}` when ready. |
| API-02 | Main endpoint | `POST /optimize-energy` accepts one scenario and returns interpretation plus schedule as JSON. |
| API-03 | Successful response | HTTP 200 only after response-schema validation and schedule replay succeed. |
| API-04 | Structural errors | Malformed JSON, missing required fields, wrong types, or invalid structural shapes return controlled JSON HTTP 400. |
| API-05 | Semantic errors | Well-formed but semantically invalid requests may return HTTP 422; use this consistently and document it. |
| API-06 | Internal failures | Provider/model/solver failures return controlled JSON HTTP 500 without credentials or stack traces. |
| API-07 | Access | Both endpoints are callable from the submitted base URL without login, VPN, dashboard access, or manual intervention. |

Chosen error envelope, not an organizer-mandated schema:

```json
{"error":{"code":"INVALID_REQUEST","message":"Request validation failed."}}
```

Use stable codes such as `INVALID_REQUEST`, `INVALID_SCENARIO`, `INTERPRETATION_FAILED`, `OPTIMIZATION_FAILED`, and `VALIDATION_FAILED`. Never represent a failed optimization as a successful empty plan.

## 4. Request requirements

### 4.1 Top-level schema

| Field | Required type | Rules |
| --- | --- | --- |
| `scenario_id` | string | Echo unchanged in the response; never use it to select a stored answer. |
| `operator_notes` | array of strings | Exactly 1–3 non-empty natural-language notes. |
| `hours` | array of objects | Exactly 24 entries covering each integer hour 0–23 once. |
| `battery` | object | All five fields below are required. |

Each hour has:

- `hour`: integer 0–23.
- `demand_kwh`: number.
- `solar_kwh`: number, before note adjustments.
- `tariff_bdt_per_kwh`: number.

Battery fields:

- `capacity_kwh`.
- `initial_energy_kwh`.
- `minimum_energy_kwh`.
- `max_charge_kwh_per_hour`.
- `max_discharge_kwh_per_hour`.

### 4.2 Validation decisions

- Reject non-finite numeric values, including NaN and infinities.
- Accept ordinary JSON integer and decimal numbers for energy values. Reject booleans and numeric strings as numbers; reject booleans and fractions as hours/indexes.
- Require non-negative demand, solar, battery quantities, and hourly rate limits.
- Require `0 <= minimum_energy_kwh <= initial_energy_kwh <= capacity_kwh`.
- Support zero capacity and zero rate limits when the scenario remains valid; do not require strictly positive values unnecessarily.
- Sort valid input hour entries internally; uniqueness and completeness matter more than request ordering.
- Reject whitespace-only notes. Preserve original note text for interpretation and cache identity.
- Do not invent a narrow maximum value or a fixed campus profile.
- Proposed compatibility choice: accept and ignore extra request fields; strictly validate the fields used. Model directives must use exact supported shapes.
- Non-negative tariffs are the intended implementation assumption; confirm whether negative tariffs can appear before enforcing rejection. The optimizer formulation in the plan also supports finite negative tariffs with a derived physical grid bound.

The tariff-domain choice is an assumption, not an explicitly published numeric bound in the request table.

## 5. LLM interpretation requirements

| ID | Requirement |
| --- | --- |
| LLM-01 | A language-capable generative model must directly produce the structured interpretation used by the optimizer. |
| LLM-02 | Interpret all notes, including distractors; return exactly one interpretation entry per note. |
| LLM-03 | Preserve `note_index` mapping and return entries in order 0 through N−1. |
| LLM-04 | Handle paraphrased times, percentages, fractions, and irrelevant notes. |
| LLM-05 | Treat model output as untrusted until deterministic validation succeeds. |
| LLM-06 | Do not change demand, tariff, battery parameters, or introduce unsupported directive types. |
| LLM-07 | Model errors must not silently become `no_op` or an unconstrained schedule. |
| LLM-08 | Document the model/provider or local model identifier, prompt approach, and guardrails. |

Using an LLM only to write `plan_summary`, hard-coded phrase matching as the sole interpreter, or returning public sample answers by ID does not satisfy the mandatory requirement.

### 5.1 Supported directives

| `directive_type` | Exact `structured_adjustment` | Effect |
| --- | --- | --- |
| `solar_reduction` | `{"hours":[...],"factor":number}` | Available solar is original solar multiplied by the remaining fraction for each affected hour. |
| `minimum_battery_reserve` | `{"hours":[...],"minimum_energy_kwh":number}` | Raise the end-of-hour reserve floor. |
| `no_charge_window` | `{"hours":[...]}` | Charging must be zero in affected hours. |
| `no_discharge_window` | `{"hours":[...]}` | Discharging must be zero in affected hours. |
| `max_grid_window` | `{"hours":[...],"max_grid_kwh":number}` | Apply a per-hour grid-import ceiling. |
| `no_op` | `null` | No energy-model change. |

Every interpretation entry requires:

```json
{
  "note_index": 0,
  "applies": true,
  "directive_type": "solar_reduction",
  "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
  "explanation": "An 80% reduction leaves 20% available during hours 13 and 14."
}
```

### 5.2 Guardrails and semantics

- `no_op` requires `applies=false` and `structured_adjustment=null`.
- Every other type requires `applies=true` and the matching adjustment object.
- Hours are unique integers 0–23, returned in ascending order.
- Time windows are start-inclusive and end-exclusive: 1 PM–3 PM means `[13,14]`.
- Noon is 12; midnight at the start of the day is 0. End-of-day midnight may be a boundary of 24, but 24 is never an emitted hour.
- A reduction **by** 80% means factor 0.2; a reduction **to** 80% means factor 0.8.
- Factor is finite and in `[0,1]`, including both endpoints.
- Reserve is finite, non-negative, and no greater than battery capacity.
- A reserve expressed as a fraction of capacity must be converted to absolute kWh using the request battery capacity.
- Grid caps are finite and non-negative.
- Free-text explanations need not match reference wording.
- Official valid scoring scenarios are feasible, with one supported directive or `no_op` per note.

Cross-midnight windows, all-day wording, and ambiguous times need a documented interpretation policy. Do not claim an unpublished policy is judge ground truth.

## 6. Deterministic directive application

For each hour construct effective solar, active reserve, charge limit, discharge limit, and grid cap.

- Start from original request values.
- Apply every validated relevant directive before optimization.
- Combine reserve floors using the maximum of the base reserve and all applicable reserve directives.
- Combine grid ceilings using the minimum of all applicable grid caps.
- Combine no-charge/no-discharge hours by union. If both apply, the battery is idle that hour.
- Preserve original inputs for replay and debugging; do not mutate them in place.
- Overlapping solar reductions with different factors are not explicitly resolved by the published formula. Obtain clarification about multiplication versus another composition rule. Isolate this policy and document any provisional choice.

## 7. Energy and optimization requirements

Let `C[h]` and `D[h]` be non-negative charge and discharge magnitudes; the response permits only one battery action per hour.

```text
E_before[0] = initial_energy_kwh
E_before[h] = E_after[h-1]                  for h > 0
E_after[h] = E_before[h] + C[h] - D[h]

grid[h] + solar_used[h] + D[h] = demand[h] + C[h]

0 <= solar_used[h] <= effective_solar[h]
0 <= grid[h] <= active_grid_cap[h]          when a cap exists
active_reserve[h] <= E_after[h] <= capacity_kwh
0 <= C[h] <= active_charge_limit[h]
0 <= D[h] <= active_discharge_limit[h]

E_after[23] = initial_energy_kwh
```

The published equations model no efficiency losses. Do not add efficiency factors, grid export, battery degradation costs, unmet demand, or soft penalties for hard constraints.

Objective:

```text
minimize sum(grid[h] * tariff_bdt_per_kwh[h]) for h = 0..23
```

| ID | Acceptance criterion |
| --- | --- |
| OPT-01 | All 24 hourly energy balances hold. |
| OPT-02 | Battery transitions, reserves, capacity, actions, and rate limits hold. |
| OPT-03 | Solar availability and grid ceilings hold under the validated directives. |
| OPT-04 | Final battery energy equals initial energy. |
| OPT-05 | A solver-reported optimal solution passes independent replay. |
| OPT-06 | Serialized totals are recalculated from serialized hourly values. |
| OPT-07 | Public samples achieve the reference costs within the official absolute tolerance while allowing different schedules. |

Normal judge tolerance is absolute 0.01 kWh or 0.01 BDT, unless an official judge package specifies a stricter value. Aim for substantially smaller internal residuals; do not use tolerance to deliberately violate constraints.

## 8. Response requirements

Required top-level fields:

| Field | Type and meaning |
| --- | --- |
| `scenario_id` | string matching the request |
| `directive_interpretation` | one validated entry per note, in index order |
| `hourly_plan` | exactly 24 entries, emitted in hour order 0–23 |
| `total_grid_kwh` | sum of returned hourly grid purchases |
| `total_cost_bdt` | sum of returned grid purchases multiplied by request tariffs |
| `peak_grid_kwh` | maximum returned hourly grid purchase |
| `plan_summary` | short explanation of the actual schedule |

Required hourly fields:

```json
{
  "hour": 0,
  "grid_kwh": 90,
  "solar_used_kwh": 0,
  "battery_action": "idle",
  "battery_kwh": 0,
  "battery_energy_after_kwh": 110
}
```

- `battery_action` is exactly `charge`, `discharge`, or `idle`.
- `battery_kwh` is a non-negative magnitude; it must be zero for `idle`.
- Required response numeric values must be finite and non-negative under the published output checks.
- Generate `plan_summary` deterministically from the validated result to avoid an unnecessary extra model call.

## 9. Performance and reliability

| Requirement | Published standard / implementation target |
| --- | --- |
| Startup readiness | `/health` ready within 60 seconds of service start |
| Main-request timeout | Complete within 30 seconds; slower responses count as failures |
| p95 latency | ≤5 s earns 3/3 latency points; >5–15 s earns 2/3; >15–30 s earns 1/3; >30 s earns 0/3 |
| Stability | Valid requests should not produce 5xx, invalid JSON, or no response |
| Model failures | Bounded retries/fallback and controlled error handling |
| Repeated requests | No stale directives, cross-request state leakage, or resource exhaustion |
| Secrets | No API keys, `.env` values, raw sensitive prompts, or stack traces in repository, image, logs, or responses |

Use a total internal deadline below 30 seconds. A backup interpreter must also be a language-capable generative model. Never relax directives merely to avoid a failure.

## 10. Submission and documentation

- Submit a reachable public base URL serving both endpoints.
- Create the competition repository after question reveal; keep it private during the event and make it public after the submission deadline according to the official rulebook.
- Include all source, dependency/configuration files, tests, and environment-variable names.
- Provide a tested Docker registry image reference with exact tag or digest; keep it pullable during evaluation.
- Container binds to `0.0.0.0` and exposes the documented service port.
- Document a verified image pull/run command and health check.
- README includes clean local setup, exact run command, model/provider, LLM role, guardrails, solver, dependencies/credits, public-sample test command and expected results, request/response examples, limitations, and secret handling.
- Submit an accessible video of at most three minutes covering the problem, architecture, implementation choices, and run/test flow.

## 11. Scoring priorities

| Category | Points |
| --- | ---: |
| LLM directive interpretation | 25 |
| Directive application and constraint correctness | 25 |
| Optimization quality | 10 |
| API contract and schema | 10 |
| Performance and reliability | 10 |
| Deployment and Docker fallback | 10 |
| Documentation and local reproducibility | 10 |
| **Total** | **100** |

For valid optimization cases, the published ratio is `min(1, organizer_optimal_cost / recalculated_team_cost)`, averaged and multiplied by 10. Both costs approximately zero yield ratio 1. The guide's final sentence about a zero optimal cost and positive team cost is truncated; the displayed formula yields 0 for an exactly zero optimum. Confirm near-zero handling if reproducing the official scorer.

Invalid cases receive zero optimization credit. Missing the required LLM interpretation path makes a submission ineligible for the final preliminary shortlist.

The video carries no base points and is the first tie-breaker, followed by correctness, interpretation, optimization, API/schema validity, reliability/deployment, documentation, and exceptional engineering/verification.

## 12. Acceptance test suite

1. **Contract:** endpoint names, JSON types, status codes, field presence, ordering, scenario echo.
2. **Interpretation:** all six directive types, paraphrases, start/end boundaries, fractional reserves, distractors, charge/discharge clarifications.
3. **Guardrails:** unknown types, missing/duplicate note indexes, invalid hours, invalid applies/null combinations, non-finite values, out-of-range factors/reserves/caps.
4. **Optimizer:** terminal neutrality, tight hourly rate limits, future reserves, grid caps, solar curtailment, zero rates/capacity, zero-cost schedules, fractional quantities.
5. **Replay:** intentionally corrupt each constraint and ensure rejection, even if solver status is successful.
6. **End-to-end:** run all ten official public cases through the real model and service; compare interpretation semantics and valid costs.
7. **Operations:** cold startup, uncached p95, repeated requests, provider timeout/rate limiting, malformed model output, container startup, and external endpoint access.

Passing practice cases does not guarantee hidden-test performance; the exact hidden judge set is unpublished.

### Important correction to the earlier practice document

`HARD_ELIMINATION_TESTS.md` is a draft practice resource, not a trusted full-response oracle:

- **ELIM-03 is infeasible as written.** Hour 19 demands 215 kWh with no solar; grid cap 155 requires at least 60 kWh discharge, exceeding the 50 kWh limit. Across hours 18–20, at least 160 kWh must discharge while retaining 130 kWh, requiring at least 290 kWh at the start of hour 18, above capacity 220.
- **ELIM-06 is infeasible as written.** Hour 19 demand is 215, not 225. Cap 100 requires 115 kWh discharge, above the 55 kWh hourly limit. Earlier charging cannot overcome an hourly discharge-rate limit.
- **ELIM-04** uses SAMPLE-01 tariffs: hours 18 and 19 are 28 and 30 BDT/kWh, not 31 and 33.
- Some answer fragments omit required `explanation` fields; the file does not provide verified full optimal schedules/costs.

Keep the two infeasible originals only as deliberate failure-handling tests. The implementation plan describes proposed feasible replacements that must be solved and replayed before becoming reference fixtures.

## 13. Definition of done

- [ ] Every mandatory functional requirement above is implemented.
- [ ] Real LLM outputs drive the constraints.
- [ ] All ten public examples pass semantic interpretation and independent replay; optimal costs match within tolerance.
- [ ] Additional feasible stress fixtures have verified reference solutions.
- [ ] Invalid input and dependency failures are controlled and do not leak secrets.
- [ ] Performance meets the published timeout; uncached p95 is measured.
- [ ] Public URL and pullable Docker image work using documented commands.
- [ ] A fresh environment can follow the README without undocumented steps.
- [ ] Repository access timing and video submission follow the guide.
