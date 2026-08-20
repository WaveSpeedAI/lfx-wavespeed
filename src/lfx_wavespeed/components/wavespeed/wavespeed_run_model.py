"""Run any model in the WaveSpeed AI catalog with a raw JSON payload."""

from __future__ import annotations

import json
from typing import Any

from lfx.io import FloatInput, MessageTextInput, MultilineInput, Output, SecretStrInput
from lfx.schema.data import Data

from lfx_wavespeed._base import DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT, WaveSpeedBaseComponent

EXAMPLE_MODEL = "wavespeed-ai/z-image/turbo"


class WaveSpeedRunModelComponent(WaveSpeedBaseComponent):
    display_name = "WaveSpeed AI Run Model"
    description = (
        "Run any model from the WaveSpeed AI catalog by id with a JSON input "
        "payload. Browse the catalog at https://wavespeed.ai/models."
    )
    documentation = "https://wavespeed.ai/docs"
    icon = "zap"
    name = "WaveSpeedRunModel"

    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="WaveSpeed API Key",
            info="Your WaveSpeed API key, from https://wavespeed.ai/dashboard.",
            required=True,
            value="WAVESPEED_API_KEY",
        ),
        MessageTextInput(
            name="model_name",
            display_name="Model ID",
            info=f'WaveSpeed model id, e.g. "{EXAMPLE_MODEL}".',
            value=EXAMPLE_MODEL,
            required=True,
            tool_mode=True,
        ),
        MultilineInput(
            name="model_input",
            display_name="Input (JSON)",
            info=(
                "JSON object of model parameters, e.g. "
                '{"prompt": "A lighthouse at dusk"}. Fields the model does not '
                "declare are dropped by the platform."
            ),
            value='{"prompt": "A lighthouse at dusk"}',
            required=True,
            tool_mode=True,
        ),
        FloatInput(
            name="timeout",
            display_name="Timeout (s)",
            info="Maximum seconds to wait for the result. The task keeps running server-side.",
            value=DEFAULT_TIMEOUT,
            required=False,
            advanced=True,
        ),
        FloatInput(
            name="poll_interval",
            display_name="Poll Interval (s)",
            info="Seconds between result polls.",
            value=DEFAULT_POLL_INTERVAL,
            required=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Result", name="result", method="run_model"),
    ]

    @staticmethod
    def _parse_input(value: Any) -> dict[str, Any]:
        """Accept a dict, or the JSON *string* the UI and agents actually send."""
        if isinstance(value, dict):
            return value
        if value is None or (isinstance(value, str) and not value.strip()):
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as e:
                msg = f"Input must be a JSON object or a valid JSON string: {e}"
                raise ValueError(msg) from e
            if not isinstance(parsed, dict):
                msg = f"Input must decode to a JSON object, got {type(parsed).__name__}."
                raise ValueError(msg)
            return parsed
        msg = f"Input must be a JSON object or JSON string, got {type(value).__name__}."
        raise ValueError(msg)

    def run_model(self) -> Data:
        """Run the given model id with the given payload."""
        return self._to_data(self.model_name, self._parse_input(self.model_input))
