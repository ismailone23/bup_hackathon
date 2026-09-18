import math
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_serializer,
    model_validator,
)

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
    tariff_bdt_per_kwh: NonNegative

    @field_validator("tariff_bdt_per_kwh")
    @classmethod
    def reject_negative_zero(cls, value: float) -> float:
        if math.copysign(1.0, value) < 0:
            raise ValueError("tariff_bdt_per_kwh must be non-negative")
        return value


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
    scenario_id: Annotated[StrictStr, Field(min_length=1)]
    operator_notes: Annotated[list[Annotated[StrictStr, Field(max_length=1000)]], Field(min_length=1, max_length=3)]
    hours: Annotated[list[HourInput], Field(min_length=24, max_length=24)]
    battery: BatteryInput

    @model_validator(mode="after")
    def valid_scenario(self):
        if self.scenario_id != self.scenario_id.strip() or any(ch in self.scenario_id for ch in "\r\n\t"):
            raise ValueError("scenario_id must not contain surrounding whitespace or control characters")
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
