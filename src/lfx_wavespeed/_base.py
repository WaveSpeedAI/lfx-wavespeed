"""Shared plumbing for the WaveSpeed AI bundle.

Every component here talks to the platform through the official ``wavespeed``
Python SDK rather than raw HTTP: the SDK already carries the channel
attribution headers, the correct retry policy (a submission POST is never
retried, so a failed poll can never double-bill a task) and terminal-status
handling for ``failed`` / ``cancelled`` / ``timeout``.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import requests
from lfx.custom.custom_component.component import Component
from lfx.schema.data import Data

WAVESPEED_BASE_URL = "https://api.wavespeed.ai"
WAVESPEED_MODELS_PATH = "/api/v3/models"

#: Channel-attribution name sent as the ``X-Client-Name`` header so WaveSpeed
#: can attribute traffic to the Langflow integration.
CLIENT_NAME = "langflow"

#: Wait deadline for a single generation.  ``None`` would wait forever and
#: strand the flow; the task keeps running server-side either way.
DEFAULT_TIMEOUT = 600.0
DEFAULT_POLL_INTERVAL = 2.0

#: How long to wait on the (fast, cacheable) model-catalog request.
CATALOG_TIMEOUT = 10.0


def build_client(api_key: str, *, max_retries: int = 0) -> Any:
    """Construct a ``wavespeed.Client``.

    Split out as a module-level function so tests can substitute a fake client
    without touching the network.

    Args:
        api_key: WaveSpeed API key.
        max_retries: Task-level replacement attempts.  Defaults to ``0`` so a
            single component run can never turn into a second billed
            submission.

    Returns:
        A configured ``wavespeed.Client``.
    """
    from wavespeed import Client

    return Client(api_key=api_key, client_name=CLIENT_NAME, max_retries=max_retries)


def fetch_catalog(api_key: str) -> list[dict[str, Any]]:
    """Fetch the public WaveSpeed model catalog.

    Args:
        api_key: WaveSpeed API key (the catalog endpoint requires auth).

    Returns:
        The raw list of catalog entries, or an empty list on any failure.
    """
    response = requests.get(
        f"{WAVESPEED_BASE_URL}{WAVESPEED_MODELS_PATH}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Client-Name": CLIENT_NAME,
        },
        timeout=CATALOG_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    return data if isinstance(data, list) else []


class WaveSpeedBaseComponent(Component):
    """Base class holding the credential handling and run/format helpers."""

    documentation = "https://wavespeed.ai/docs"

    #: Catalog ``type`` values whose models belong in this component's dropdown.
    model_types: ClassVar[tuple[str, ...]] = ()
    #: Offline fallback shown when the catalog cannot be reached.
    fallback_models: ClassVar[tuple[str, ...]] = ()

    # -- credentials ----------------------------------------------------

    def _require_api_key(self) -> str:
        """Return the configured API key or raise."""
        api_key = getattr(self, "api_key", None)
        if isinstance(api_key, str):
            api_key = api_key.strip()
        if not api_key:
            msg = (
                "A WaveSpeed API key is required. Create one at "
                "https://wavespeed.ai/dashboard and set it on this component "
                "or in the WAVESPEED_API_KEY variable."
            )
            raise ValueError(msg)
        return api_key

    # -- model catalog --------------------------------------------------

    def get_models(self) -> list[str]:
        """Return the live model ids for this component, or the fallback list."""
        api_key = getattr(self, "api_key", None)
        if not api_key:
            return list(self.fallback_models)
        try:
            entries = fetch_catalog(api_key)
        except (requests.RequestException, ValueError) as e:
            self.status = f"Error fetching models: {e}"
            return list(self.fallback_models)

        wanted = set(self.model_types)
        model_ids = [
            entry["model_id"]
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("model_id"), str)
            and (not wanted or entry.get("type") in wanted)
        ]
        # Keep the curated defaults reachable even if the catalog omits them.
        for fallback in self.fallback_models:
            if fallback not in model_ids:
                model_ids.insert(0, fallback)
        return model_ids or list(self.fallback_models)

    def update_build_config(
        self,
        build_config: dict,
        field_value: str,  # noqa: ARG002
        field_name: str | None = None,
    ) -> dict:
        """Refresh the model dropdown when the key changes or Refresh is hit."""
        if field_name in {"api_key", "model_name"} and "model_name" in build_config:
            build_config["model_name"]["options"] = self.get_models()
        return build_config

    # -- execution ------------------------------------------------------

    def _run_model(self, model: str, model_input: dict[str, Any]) -> dict[str, Any]:
        """Submit a task, wait for it, and return the raw SDK result.

        ``None`` values are stripped: WaveSpeed's input whitelist rejects
        unknown or null fields rather than ignoring them.

        Raises:
            ValueError: On a missing key, a failed/cancelled/timed-out task, or
                a task that completed with no outputs.
        """
        model_id = (model or "").strip()
        if not model_id:
            msg = "A WaveSpeed model id is required."
            raise ValueError(msg)

        client = build_client(self._require_api_key())
        payload = {k: v for k, v in model_input.items() if v is not None}

        try:
            result = client.run(
                model_id,
                payload,
                # A blank field means "use the default deadline", not "wait
                # forever" -- an unbounded wait would strand the whole flow.
                timeout=getattr(self, "timeout", None) or DEFAULT_TIMEOUT,
                poll_interval=getattr(self, "poll_interval", None) or DEFAULT_POLL_INTERVAL,
            )
        except Exception as e:
            # SDK messages already carry "(task_id: ...)" plus the platform
            # error text; keep them verbatim so a billed failure stays
            # traceable from the Langflow error panel.
            self.status = f"Error: {e}"
            msg = f"WaveSpeed model {model_id!r} failed: {e}"
            raise ValueError(msg) from e

        outputs = result.get("outputs") or []
        if not outputs:
            msg = f"WaveSpeed model {model_id!r} returned no outputs."
            raise ValueError(msg)
        return result

    # -- formatting -----------------------------------------------------

    @staticmethod
    def _output_url(output: Any) -> str:
        """Render one platform output as a single usable line."""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            url = output.get("url")
            if isinstance(url, str):
                return url
        return json.dumps(output, default=str)

    def _to_data(self, model: str, model_input: dict[str, Any]) -> Data:
        """Run ``model`` and wrap the result as a ``Data`` payload."""
        result = self._run_model(model, model_input)
        urls = [self._output_url(o) for o in result["outputs"]]
        self.status = urls[0]
        return Data(
            data={
                "model": model,
                "url": urls[0],
                "urls": urls,
                "outputs": result["outputs"],
            }
        )
