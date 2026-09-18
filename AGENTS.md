# Agent Instructions

## Project shape

- This is a single-module FastAPI service; application code and the API entrypoint are in `main.py`.
- `POST /optimize-energy` interprets notes with an OpenAI model, applies validated directives, solves with SciPy `linprog`, and independently replays the returned schedule. Preserve that validation boundary when changing behavior.
- `REQUIREMENTS.md` is the detailed challenge specification. The participant PDFs and public sample JSON under `BUP_CSE_FEST_2026_Participant_Docs/` are the authoritative external references for contract and scoring questions.

## Setup and commands

- Create the environment with `python -m venv .venv` and install pinned dependencies with `pip install -r requirements.txt`.
- Run the complete local verification with `pytest -q`. Tests are split across `test_*.py`; `test_main.py` contains the main optimizer and replay fixtures.
- Start the service with `uvicorn main:app --host 0.0.0.0 --port 8000`.
- The container uses Python 3.12, installs `requirements.txt`, and runs the same Uvicorn command on port 8000; keep the bind address at `0.0.0.0`.
- No repository-configured lint, formatter, typecheck, CI, or task-runner command exists; do not invent one as a required verification step.

## Environment and boundaries

- Local configuration is loaded from `.env`; use `.env.example` for variable names. Never commit or copy `.env` into an image. Required variables are `OPENAI_API_KEY`, with optional `OPENAI_MODEL` and `OPENAI_TIMEOUT_SECONDS`.
- Tests that exercise `/health` set or remove `OPENAI_API_KEY` themselves. The optimizer tests use deterministic directives and do not require a live model; real end-to-end interpretation requires a valid accessible OpenAI model.
- Keep request/model failures controlled through the existing JSON error handlers; do not expose API keys, prompts, stack traces, or provider errors in responses or logs.
