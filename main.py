import asyncio
import json
import math
import os
from contextlib import asynccontextmanager
from enum import Enum
from typing import Annotated, Literal

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_serializer, model_validator
from scipy.optimize import linprog

load_dotenv()

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class DirectiveType(str, Enum):
    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


class HourInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    hour: Annotated[StrictInt, Field(ge=0, le=23)]
    demand_kwh: NonNegative
    solar_kwh: NonNegative
    tariff_bdt_per_kwh: Number


class BatteryInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    capacity_kwh: NonNegative
    initial_energy_kwh: NonNegative
    minimum_energy_kwh: NonNegative
    max_charge_kwh_per_hour: NonNegative
    max_discharge_kwh_per_hour: NonNegative

    @model_validator(mode="after")
    def valid_energy_range(self):
        if not self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh:
            raise ValueError("battery energy must satisfy minimum <= initial <= capacity")
        return self


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scenario_id: StrictStr
    operator_notes: Annotated[list[Annotated[StrictStr, Field(max_length=1000)]], Field(min_length=1, max_length=3)]
    hours: Annotated[list[HourInput], Field(min_length=24, max_length=24)]
    battery: BatteryInput

    @model_validator(mode="after")
    def valid_scenario(self):
        if any(not note.strip() for note in self.operator_notes):
            raise ValueError("operator notes must not be blank")
        if {item.hour for item in self.hours} != set(range(24)):
            raise ValueError("hours must contain every integer from 0 through 23 exactly once")
        self.hours.sort(key=lambda item: item.hour)
        return self


class StructuredAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: list[StrictInt] = Field(min_length=1)
    factor: Number | None = None
    minimum_energy_kwh: NonNegative | None = None
    max_grid_kwh: NonNegative | None = None

    @model_serializer
    def serialize(self):
        return {name: value for name, value in self.__dict__.items() if value is not None}


class DirectiveInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note_index: StrictInt
    applies: StrictBool
    directive_type: DirectiveType
    structured_adjustment: StructuredAdjustment | None
    explanation: StrictStr


class HourPlan(BaseModel):
    hour: StrictInt
    grid_kwh: Number
    solar_used_kwh: Number
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: Number
    battery_energy_after_kwh: Number

    @model_validator(mode="after")
    def valid_values(self):
        if self.grid_kwh < 0 or self.solar_used_kwh < 0 or self.battery_kwh < 0 or self.battery_energy_after_kwh < 0:
            raise ValueError("plan energy values must be non-negative")
        if self.battery_action == "idle" and self.battery_kwh != 0:
            raise ValueError("idle battery action must have zero battery_kwh")
        if self.battery_action != "idle" and self.battery_kwh <= 0:
            raise ValueError("non-idle battery action must have positive battery_kwh")
        return self


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message


class DirectiveValidationError(ValueError):
    pass


INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_index": {"type": "integer"},
                    "applies": {"type": "boolean"},
                    "directive_type": {"type": "string", "enum": [item.value for item in DirectiveType]},
                    "structured_adjustment": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {
                                    "hours": {"type": "array", "items": {"type": "integer"}},
                                    "factor": {"type": ["number", "null"]},
                                    "minimum_energy_kwh": {"type": ["number", "null"]},
                                    "max_grid_kwh": {"type": ["number", "null"]},
                                },
                                "required": ["hours", "factor", "minimum_energy_kwh", "max_grid_kwh"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["note_index", "applies", "directive_type", "structured_adjustment", "explanation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["interpretations"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You interpret campus operator notes for a 24-hour energy optimizer.
Return exactly one entry per note in note_index order. Supported directives are solar_reduction,
minimum_battery_reserve, no_charge_window, no_discharge_window, max_grid_window, and no_op.
Windows are start-inclusive and end-exclusive: '2 AM until 5 AM' means [2,3,4],
'6 PM until 9 PM' means [18,19,20], and '11 AM until 1 PM' means [11,12].
For a window that crosses midnight, continue through hour 23, wrap to hour 0,
and stop before the ending hour; '10 PM until 6 AM' means [0,1,2,3,4,5,22,23].
Paraphrased times may be words rather than numbers; map them to the same 24-hour clock.
The ending clock time is a boundary, never an included hour. Emit sorted unique hours 0 through 23.
A reduction BY 80% leaves factor 0.2; reduction TO 80% means factor 0.8.
Convert percentage/fraction reserves to kWh using the supplied battery capacity.
For no_op use applies=false and a null adjustment; otherwise use applies=true.
For an adjustment, always include hours and all three nullable value fields. Populate only the field
for that directive: factor, minimum_energy_kwh, or max_grid_kwh. Window directives use all null values.
Do not infer unsupported changes to demand, tariff, or battery parameters.
Before returning, independently check each note's directive type, boundary hours, percentage
normalization, note index, and whether the note is an unrelated distractor."""


OPENAI_CLIENT: AsyncOpenAI | None = None
REQUEST_DEADLINE_SECONDS = float(os.getenv("REQUEST_DEADLINE_SECONDS", "18"))


def _client() -> AsyncOpenAI:
    global OPENAI_CLIENT
    if OPENAI_CLIENT is not None:
        return OPENAI_CLIENT
    if not os.getenv("OPENAI_API_KEY"):
        raise ServiceError(500, "INTERPRETATION_FAILED", "Language model is not configured.")
    OPENAI_CLIENT = AsyncOpenAI(timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "4.5")), max_retries=0)
    return OPENAI_CLIENT


async def interpret_notes(payload: OptimizeRequest) -> list[DirectiveInterpretation]:
    prompt = {
        "battery_capacity_kwh": payload.battery.capacity_kwh,
        "operator_notes": payload.operator_notes,
    }
    for attempt in range(2):
        try:
            return await request_interpretation(prompt, 7 + attempt, SYSTEM_PROMPT)
        except DirectiveValidationError as exc:
            raise ServiceError(422, "INVALID_INTERPRETATION", str(exc)) from exc
        except (APIError, IndexError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            if attempt:
                raise ServiceError(500, "INTERPRETATION_FAILED", "Operator notes could not be interpreted.")
    raise ServiceError(500, "INTERPRETATION_FAILED", "Operator notes could not be interpreted.")


async def request_interpretation(prompt: dict, seed: int, system_prompt: str) -> list[DirectiveInterpretation]:
    completion = await _client().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "directive_interpretations", "strict": True, "schema": INTERPRETATION_SCHEMA},
            },
            temperature=0,
            seed=seed,
            max_tokens=600,
    )
    if not completion.choices:
        raise ValueError("model returned no choices")
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("model response was truncated")
    content = choice.message.content
    if not content:
        raise ValueError("empty model response")
    raw = json.loads(content)["interpretations"]
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ValueError("model response must be a list of interpretation objects")
    try:
        directives = [DirectiveInterpretation.model_validate(item) for item in raw]
        for directive in directives:
            adjustment = directive.structured_adjustment
            if adjustment and len(adjustment.hours) > 1:
                hours = adjustment.hours
                # Set-preserving reorder only; hours are never derived from note text.
                if all((hours[index] + 1) % 24 == hours[index + 1] for index in range(len(hours) - 1)):
                    directive.structured_adjustment = adjustment.model_copy(update={"hours": sorted(hours)})
        validate_directives(directives, len(prompt["operator_notes"]), prompt["battery_capacity_kwh"])
    except (TypeError, ValueError, KeyError) as exc:
        raise DirectiveValidationError(str(exc)) from exc
    return directives


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
    n = 120
    objective = np.zeros(n)
    objective[:24] = tariff
    objective[48:96] = 1e-5
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

    result = linprog(objective, A_eq=np.array(equalities), b_eq=np.array(targets), bounds=bounds, method="highs")
    if not result.success:
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
    active = [item.directive_type.value for item in directives if item.applies]
    summary = "Minimum-cost schedule satisfies " + (", ".join(active) if active else "the base operating constraints") + " and restores the initial battery energy."
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ponytail: keeps the first request off the cold-start path
    if os.getenv("OPENAI_API_KEY"):
        _client()
    yield


app = FastAPI(title="GridWise LLM", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_request: Request, _exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"error": {"code": "INVALID_REQUEST", "message": "Request validation failed."}})


@app.exception_handler(ServiceError)
async def service_error_handler(_request: Request, exc: ServiceError):
    return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})


@app.exception_handler(Exception)
async def internal_error_handler(_request: Request, _exc: Exception):
    return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_ERROR", "message": "Internal service failure."}})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(payload: OptimizeRequest):
    try:
        directives = await asyncio.wait_for(interpret_notes(payload), timeout=REQUEST_DEADLINE_SECONDS)
    except asyncio.TimeoutError:
        raise ServiceError(500, "INTERPRETATION_FAILED", "Operator notes could not be interpreted.")
    return solve(payload, directives)
