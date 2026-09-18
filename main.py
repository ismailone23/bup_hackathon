"""GridWise LLM entry point.

Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""

from app.api import app

__all__ = ["app"]
