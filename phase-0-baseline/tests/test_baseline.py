from fastapi import FastAPI

from phase_0_baseline import app


def test_app_instance() -> None:
    assert isinstance(app, FastAPI)
    assert app.title == "Phase 0 Baseline API"
