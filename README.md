# GridWise LLM

FastAPI service using OpenAI Luna for structured operator-note interpretation and SciPy for deterministic minimum-cost scheduling.

## Run

```bash
python -m venv .venv
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` and, if needed, override `OPENAI_MODEL` (default: `gpt-4.1-mini`) and `OPENAI_TIMEOUT_SECONDS` (default: `12`). Then run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

- `GET /health`
- `POST /optimize-energy`

The LLM only translates notes into the six supported directive shapes. Pydantic validates its output, SciPy solves the linear program, and an independent replay check verifies every returned schedule.

Run the local deterministic check with `pytest -q`. End-to-end tests require a valid OpenAI API key and an account-accessible Luna model ID.
