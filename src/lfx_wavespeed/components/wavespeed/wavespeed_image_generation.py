"""Text-to-image generation on the WaveSpeed AI platform."""

from __future__ import annotations

from typing import ClassVar

from lfx.io import DropdownInput, FloatInput, MessageTextInput, Output, SecretStrInput
from lfx.schema.data import Data

from lfx_wavespeed._base import DEFAULT_POLL_INTERVAL, DEFAULT_TIMEOUT, WaveSpeedBaseComponent

DEFAULT_IMAGE_MODEL = "bytedance/seedream-v5.0-pro"

#: Shown when the live catalog is unreachable (no key yet, offline, API down).
FALLBACK_IMAGE_MODELS = (
    DEFAULT_IMAGE_MODEL,
    "wavespeed-ai/z-image/turbo",
)

#: Catalog ``type`` values that produce a still image.
IMAGE_MODEL_TYPES = ("text-to-image", "image-to-image")

RESOLUTIONS = ["1k", "1.5k", "2k", "4k"]
ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]


class WaveSpeedImageGenerationComponent(WaveSpeedBaseComponent):
    display_name = "WaveSpeed AI Image Generation"
    description = (
        "Generate an image from a text prompt with WaveSpeed AI models such as Seedream, Z-Image, FLUX and Qwen-Image."
    )
    documentation = "https://wavespeed.ai/docs"
    icon = "image"
    name = "WaveSpeedImageGeneration"

    model_types: ClassVar[tuple[str, ...]] = IMAGE_MODEL_TYPES
    fallback_models: ClassVar[tuple[str, ...]] = FALLBACK_IMAGE_MODELS

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
            info="The text prompt to generate the image from.",
            required=True,
            tool_mode=True,
        ),
        DropdownInput(
            name="model_name",
            display_name="Model",
            info="The WaveSpeed image model to run. Hit refresh to load the live catalog.",
            options=list(FALLBACK_IMAGE_MODELS),
            value=DEFAULT_IMAGE_MODEL,
            refresh_button=True,
            required=True,
        ),
        DropdownInput(
            name="aspect_ratio",
            display_name="Aspect Ratio",
            info="Aspect ratio of the generated image, when the model supports it.",
            options=ASPECT_RATIOS,
            value="1:1",
            required=False,
        ),
        DropdownInput(
            name="resolution",
            display_name="Resolution",
            info="Output resolution tier, when the model supports it. Higher tiers cost more.",
            options=RESOLUTIONS,
            value="1k",
            required=False,
            advanced=True,
        ),
        FloatInput(
            name="timeout",
            display_name="Timeout (s)",
            info="Maximum seconds to wait for the image. The task keeps running server-side.",
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
        Output(display_name="Image", name="image", method="generate_image"),
    ]

    def generate_image(self) -> Data:
        """Run the selected image model and return its output URL(s)."""
        prompt = (self.prompt or "").strip()
        if not prompt:
            msg = "A non-empty prompt is required to generate an image."
            raise ValueError(msg)

        # `size` and `seed` are deliberately not exposed: the platform's input
        # whitelist silently drops them for the default image models, so a
        # field here would look effective while doing nothing.
        return self._to_data(
            self.model_name,
            {
                "prompt": prompt,
                "aspect_ratio": self.aspect_ratio or None,
                "resolution": self.resolution or None,
            },
        )
