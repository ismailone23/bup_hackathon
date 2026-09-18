import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .errors import ServiceError

load_dotenv()

REQUEST_DEADLINE_SECONDS = float(os.getenv("REQUEST_DEADLINE_SECONDS", "18"))

OPENAI_CLIENT: AsyncOpenAI | None = None


def _client() -> AsyncOpenAI:
    global OPENAI_CLIENT
    if OPENAI_CLIENT is not None:
        return OPENAI_CLIENT
    if not os.getenv("OPENAI_API_KEY"):
        raise ServiceError(500, "INTERPRETATION_FAILED", "Language model is not configured.")
    OPENAI_CLIENT = AsyncOpenAI(timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "4.5")), max_retries=0)
    return OPENAI_CLIENT
