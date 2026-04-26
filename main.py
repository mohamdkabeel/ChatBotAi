# Entry point for Uvicorn: `uvicorn main:app`
# The real application is defined in app/main.py; this module re-exports it
# so that Railway's default start command (`uvicorn main:app`) resolves correctly.

from app.main import app  # noqa: F401  – re-export for Uvicorn

__all__ = ["app"]
