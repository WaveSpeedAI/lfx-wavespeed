"""Shared fakes. No test in this suite may touch the network."""

from __future__ import annotations

from typing import Any

import pytest

from lfx_wavespeed import _base


class FakeClient:
    """Stand-in for ``wavespeed.Client`` recording the call it received."""

    def __init__(self, outputs: Any = None, error: Exception | None = None) -> None:
        self._outputs = [{"url": "https://cdn.wavespeed.ai/out.png"}] if outputs is None else outputs
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def run(self, model: str, model_input: dict, **kwargs: Any) -> dict:
        self.calls.append({"model": model, "input": model_input, **kwargs})
        if self._error is not None:
            raise self._error
        return {"outputs": self._outputs}


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch):
    """Install a FakeClient factory and hand back a configure/inspect handle."""
    holder: dict[str, Any] = {"client": FakeClient(), "kwargs": None}

    def factory(api_key: str, **kwargs: Any) -> FakeClient:
        holder["api_key"] = api_key
        holder["kwargs"] = kwargs
        return holder["client"]

    monkeypatch.setattr(_base, "build_client", factory)
    return holder
