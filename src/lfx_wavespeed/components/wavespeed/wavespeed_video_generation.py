"""Text-to-video generation on the WaveSpeed AI platform."""

from __future__ import annotations

from typing import ClassVar

from lfx.io import DropdownInput, FloatInput, IntInput, MessageTextInput, Output, SecretStrInput
from lfx.schema.data import Data

from lfx_wavespeed._base import DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT, WaveSpeedBaseComponent

DEFAULT_VIDEO_MODEL = "bytedance/seedance-2.5/text-to-video"

#: Shown when the live catalog is unreachable (no key yet, offline, API down).
FALLBACK_VIDEO_MODELS = (DEFAULT_VIDEO_MODEL,)

#: Catalog ``type`` values that produce a video.
VIDEO_MODEL_TYPES = ("text-to-video", "image-to-video")


class WaveSpeedVideoGenerationComponent(WaveSpeedBaseComponent):
    display_name = "WaveSpeed AI Video Generation"
    description = (
        "Generate a video from a text prompt with WaveSpeed AI models such as Seedance, Kling, Wan and Hailuo."
    )
    documentation = "https://wavespeed.ai/docs"
    icon = "video"
    name = "WaveSpeedVideoGeneration"

    model_types: ClassVar[tuple[str, ...]] = VIDEO_MODEL_TYPES
    fallback_models: ClassVar[tuple[str, ...]] = FALLBACK_VIDEO_MODELS

    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="WaveSpeed API Key",
            info="Your WaveSpeed API key, from https://wavespeed.ai/dashboard.",
            required=True,
            value="WAVESPEED_API_KEY",
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="prompt",
            display_name="Prompt",
            info="The text prompt to generate the video from.",
            required=True,
            tool_mode=True,
        ),
        DropdownInput(
            name="model_name",
            display_name="Model",
            info="The WaveSpeed video model to run. Hit refresh to load the live catalog.",
            options=list(FALLBACK_VIDEO_MODELS),
            value=DEFAULT_VIDEO_MODEL,
            refresh_button=True,
            required=True,
        ),
        IntInput(
            name="duration",
            display_name="Duration (s)",
            info="Video length in seconds. Valid values are model-dependent, commonly 5 or 10.",
            value=5,
            required=False,
        ),
        FloatInput(
            name="timeout",
            display_name="Timeout (s)",
            info="Maximum seconds to wait for the video. The task keeps running server-side.",
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
        Output(display_name="Video", name="video", method="generate_video"),
    ]

    def generate_video(self) -> Data:
        """Run the selected video model and return its output URL(s)."""
        prompt = (self.prompt or "").strip()
        if not prompt:
            msg = "A non-empty prompt is required to generate a video."
            raise ValueError(msg)

        return self._to_data(
            self.model_name,
            {"prompt": prompt, "duration": self.duration or None},
        )
