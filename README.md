# GridWise LLM

GridWise LLM is a FastAPI service for the BUP CSE Fest 2026 preliminary
challenge. It interprets 1-3 synthetic campus operator notes with OpenAI
`gpt-4.1-mini`, validates the structured directives, solves the resulting
24-hour energy schedule with SciPy linear programming, and independently
replays the serialized schedule before returning it.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness check |
| `POST` | `/optimize-energy` | Interpret notes and optimize the schedule |

When configured and ready, `/health` returns:

```json
{"status":"ok"}
```

If the OpenAI key is not configured, the local service reports readiness as
HTTP `503` rather than claiming that model-backed requests are ready.

## Quickstart

Requirements: Python 3.12+, an OpenAI API key, and a network connection for
model-backed interpretation.

```bash
git clone https://github.com/ismailone23/bup_hackathon.git
cd bup_hackathon
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env            # Windows PowerShell: Copy-Item .env.example .env
```

Set the values in `.env` without committing the file:

```dotenv
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_SECONDS=4.5
```

Start the service exactly as the container does:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Check readiness from another terminal:

```bash
curl http://127.0.0.1:8000/health
```

## Request Contract

`POST /optimize-energy` accepts one JSON object containing:

- `scenario_id`: string
- `operator_notes`: 1-3 non-empty strings
- `hours`: exactly 24 entries covering integer hours 0 through 23 once
- `battery`: capacity, initial energy, minimum energy, and hourly charge/discharge limits

Each hour contains `hour`, `demand_kwh`, `solar_kwh`, and
`tariff_bdt_per_kwh`. Numeric values must be finite. Demand, solar, battery
quantities, and rate limits are non-negative. Input hour entries may arrive in
any order, but are sorted internally after uniqueness and completeness checks.

Example request shape:

```json
{
  "scenario_id": "GRID-101",
  "operator_notes": [
    "Solar output will drop to about 20% from 1 PM to 3 PM.",
    "Do not charge the battery between 2 PM and 4 PM.",
    "The cafeteria menu changes tomorrow."
  ],
  "hours": [
    {"hour": 0, "demand_kwh": 180, "solar_kwh": 0, "tariff_bdt_per_kwh": 7},
    {"hour": 1, "demand_kwh": 180, "solar_kwh": 0, "tariff_bdt_per_kwh": 7}
  ],
  "battery": {
    "capacity_kwh": 500,
    "initial_energy_kwh": 200,
    "minimum_energy_kwh": 50,
    "max_charge_kwh_per_hour": 100,
    "max_discharge_kwh_per_hour": 100
  }
}
```

The example is abbreviated for readability. A real request must include all
24 hour objects.

The response contains `scenario_id`, one `directive_interpretation` entry per
note, a 24-entry `hourly_plan`, `total_grid_kwh`, `total_cost_bdt`,
`peak_grid_kwh`, and a deterministic `plan_summary`.

## Supported Directives

The model may emit only these directive types:

| Type | Adjustment | Effect |
| --- | --- | --- |
| `solar_reduction` | `hours`, `factor` | Multiplies available solar by the remaining fraction |
| `minimum_battery_reserve` | `hours`, `minimum_energy_kwh` | Raises the end-of-hour reserve floor |
| `no_charge_window` | `hours` | Sets charging to zero |
| `no_discharge_window` | `hours` | Sets discharging to zero |
| `max_grid_window` | `hours`, `max_grid_kwh` | Sets a per-hour grid-import ceiling |
| `no_op` | `null` | Makes no energy-model change |

Time windows are start-inclusive and end-exclusive. For example, 1 PM to 3
PM means `[13, 14]`. Cross-midnight windows wrap through hour 23 and continue
at hour 0; 10 PM to 6 AM means `[0,1,2,3,4,5,22,23]`. Returned directive
hours are unique, valid, and ascending.

`no_op` must use `applies: false` and `structured_adjustment: null`. Every
other directive must use `applies: true` and its exact adjustment shape.

## Processing and Validation

The service follows the required pipeline:

```text
request -> OpenAI structured interpretation -> Pydantic guardrails
        -> directive application -> SciPy linprog -> independent replay
        -> response totals consistency check -> JSON response
```

The model output is untrusted until deterministic validation succeeds. The
guards verify note mapping, supported directive types, strict booleans and
integers, hours, finite numeric ranges, directive semantics, and adjustment
shapes. Provider failures, empty choices, empty content, invalid JSON, and
truncated responses are controlled failures and never become an unconstrained
`no_op` schedule.

The optimizer minimizes:

```text
sum(grid_kwh[h] * tariff_bdt_per_kwh[h]) for h = 0..23
```

It enforces hourly energy balance, solar availability, battery capacity and
reserve, charge/discharge limits, directive-specific limits, non-negative
serialized values, and end-of-day battery neutrality. A small regularization
term prevents unnecessary simultaneous charge/discharge cycles. Returned totals
are recalculated from the returned hourly values.

## Testing

Run all local tests:

```bash
python -m pytest -q
```

The suite includes optimizer/replay tests, strict guardrail tests, provider
failure tests, API validation tests, and teammate test files. A clean run is
expected to report all tests passed. Tests do not require a live OpenAI key
unless explicitly marked as a live-model test.

Ten additional hard interpretation cases run live against the real model. They
are skipped by default and enabled with an environment flag:

```bash
GRIDWISE_LIVE_TESTS=1 python -m pytest -q test_live_llm.py -v
```

### Public sample cases

The public pack contains 10 reference cases. They are not the hidden judge
set, and reference schedules do not need to match byte-for-byte. Compare
directive semantics, replay validity, and recalculated cost within the
challenge tolerance of `0.01` kWh/BDT.

With the service running and `OPENAI_API_KEY` configured, place the public
sample JSON beside the repository README as `public_sample_cases.json`, then
run all public inputs:

```bash
python - <<'PY'
import json
import urllib.request

path = "public_sample_cases.json"
with open(path, encoding="utf-8") as file:
    pack = json.load(file)

for case in pack["cases"]:
    request = urllib.request.Request(
        "http://127.0.0.1:8000/optimize-energy",
        data=json.dumps(case["input"]).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    assert result["scenario_id"] == case["input"]["scenario_id"]
    assert len(result["hourly_plan"]) == 24
    print(result["scenario_id"], result["total_cost_bdt"])
PY
```

Expected result: 10 successful JSON responses, each with a matching scenario
ID and a valid 24-hour plan. Exact costs depend on the valid interpretation
returned by the model and must be checked against the public case semantics.

## Docker Fallback

The image binds to `0.0.0.0:8000` and does not contain `.env` or credentials.
Build and run locally:

```bash
docker build -t gridwise:local .
docker run --rm --name gridwise \
  --env-file .env \
  -p 8000:8000 \
  gridwise:local
```

Verify the container:

```bash
curl http://127.0.0.1:8000/health
```

For submission, provide the organizers a pullable registry image with an
exact immutable tag or digest. The repository does not claim a registry image
that has not been published.

## Railway Deployment

Railway can deploy this repository using the included `Dockerfile`. Configure
these Railway service variables:

```text
OPENAI_API_KEY=<Railway secret>
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_SECONDS=4.5
```

Do not commit or print the key. Railway should use the container command from
the `Dockerfile`; the application listens on the Railway-provided port only if
the platform overrides it, otherwise it uses port 8000. Submit the resulting
public Railway base URL only after verifying both `/health` and

## Dependencies and Credits

- FastAPI and Uvicorn: HTTP API and ASGI serving
- Pydantic: strict request, model-output, and response validation
- OpenAI Python SDK: structured `gpt-4.1-mini` interpretation
- NumPy: linear-program arrays and numeric operations
- SciPy `linprog`: deterministic minimum-cost optimization
- python-dotenv: local `.env` loading
- Pytest: automated verification

These are public libraries used under their respective licenses. The solution
uses no live campus, utility, billing, or personal data. AI coding assistants
may have been used during development; the submitted architecture remains the
team's implementation.

## Errors and Limitations

- Malformed or structurally invalid requests return controlled HTTP `400` JSON.
- Semantically invalid model interpretations return controlled HTTP `422` JSON.
- Provider and solver failures return controlled HTTP `500` JSON without keys,
  prompts, provider errors, or stack traces.
- The service depends on OpenAI availability, quota, and a valid API key during
  live interpretation. `gpt-4.1-mini` remains the configured model.
- The service is stateless and does not store scenarios or user data.
- Cross-midnight policy is deterministic and uses whole-hour intervals with an
  exclusive ending boundary.
- Public sample cases are references, not hidden-case guarantees.

## Secret Handling

Never commit `.env`, API keys, tokens, passwords, or provider responses that
contain secrets. Use `.env.example` for variable names only. The Docker build
uses `.dockerignore` to exclude local secrets and repository caches. Configure
secrets in the runtime environment, such as Railway variables, rather than in
source code or image layers.
