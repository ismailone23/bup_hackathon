import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import REQUEST_DEADLINE_SECONDS, _client
from .errors import ServiceError
from .interpretation import interpret_notes
from .models import OptimizeRequest, OptimizeResponse
from .optimizer import solve


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
