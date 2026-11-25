"""Compatibility entry point; the FastAPI application lives in ``app.main``."""

from app.main import app

__all__ = ["app"]
