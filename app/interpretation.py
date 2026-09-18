import json
import os

from openai import APIError

from .config import _client
from .errors import DirectiveValidationError, ServiceError
from .models import DirectiveInterpretation, OptimizeRequest
from .optimizer import validate_directives
from .prompt import INTERPRETATION_SCHEMA, SYSTEM_PROMPT


async def interpret_notes(payload: OptimizeRequest) -> list[DirectiveInterpretation]:
    prompt = {
        "battery_capacity_kwh": payload.battery.capacity_kwh,
        "operator_notes": payload.operator_notes,
    }
    for attempt in range(2):
        try:
            return await request_interpretation(prompt, 7 + attempt, SYSTEM_PROMPT)
        except DirectiveValidationError as exc:
            if attempt:
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
