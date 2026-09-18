import math
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .errors import ServiceError

load_dotenv()

DEFAULT_REQUEST_DEADLINE_SECONDS = 18.0
DEFAULT_OPENAI_TIMEOUT_SECONDS = 4.5


def positive_seconds(name: str, default: float) -> float:
    """Read a positive seconds value, falling back to the documented default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value) or value <= 0:
        return default
    return value


REQUEST_DEADLINE_SECONDS = positive_seconds("REQUEST_DEADLINE_SECONDS", DEFAULT_REQUEST_DEADLINE_SECONDS)

OPENAI_CLIENT: AsyncOpenAI | None = None


def _client() -> AsyncOpenAI:
    global OPENAI_CLIENT
    if OPENAI_CLIENT is not None:
        return OPENAI_CLIENT
    if not os.getenv("OPENAI_API_KEY"):
        raise ServiceError(500, "INTERPRETATION_FAILED", "Language model is not configured.")
    OPENAI_CLIENT = AsyncOpenAI(
        timeout=positive_seconds("OPENAI_TIMEOUT_SECONDS", DEFAULT_OPENAI_TIMEOUT_SECONDS),
        max_retries=0,
    )
    return OPENAI_CLIENT
