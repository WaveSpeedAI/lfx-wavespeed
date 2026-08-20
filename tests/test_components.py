"""Unit tests for the WaveSpeed AI Langflow bundle.

The SDK client is always faked; nothing here reaches api.wavespeed.ai.
"""

from __future__ import annotations

import json

import pytest
import requests

from lfx_wavespeed import (
    WaveSpeedImageGenerationComponent,
    WaveSpeedRunModelComponent,
    WaveSpeedVideoGenerationComponent,
    _base,
)
from lfx_wavespeed.components.wavespeed.wavespeed_image_generation import (
    DEFAULT_IMAGE_MODEL,
    FALLBACK_IMAGE_MODELS,
)
from lfx_wavespeed.components.wavespeed.wavespeed_video_generation import DEFAULT_VIDEO_MODEL

from .conftest import FakeClient


def make_image(**overrides):
    component = WaveSpeedImageGenerationComponent()
    defaults = {
        "api_key": "ws-test-key",
        "prompt": "a red panda drinking boba",
        "model_name": DEFAULT_IMAGE_MODEL,
        "aspect_ratio": "16:9",
        "resolution": "1k",
        "timeout": 30.0,
        "poll_interval": 1.0,
    }
    for key, value in {**defaults, **overrides}.items():
        setattr(component, key, value)
    return component


def make_video(**overrides):
    component = WaveSpeedVideoGenerationComponent()
    defaults = {
        "api_key": "ws-test-key",
        "prompt": "a drone shot over a glacier",
        "model_name": DEFAULT_VIDEO_MODEL,
        "duration": 5,
        "timeout": 30.0,
        "poll_interval": 1.0,
    }
    for key, value in {**defaults, **overrides}.items():
        setattr(component, key, value)
    return component


def make_runner(**overrides):
    component = WaveSpeedRunModelComponent()
    defaults = {
        "api_key": "ws-test-key",
        "model_name": "wavespeed-ai/z-image/turbo",
        "model_input": '{"prompt": "a lighthouse at dusk"}',
        "timeout": 30.0,
        "poll_interval": 1.0,
    }
    for key, value in {**defaults, **overrides}.items():
        setattr(component, key, value)
    return component


# --------------------------------------------------------------------------
# Argument mapping
# --------------------------------------------------------------------------


def test_image_arg_mapping(fake_client):
    result = make_image().generate_image()

    call = fake_client["client"].calls[0]
    assert call["model"] == DEFAULT_IMAGE_MODEL
    assert call["input"] == {
        "prompt": "a red panda drinking boba",
        "aspect_ratio": "16:9",
        "resolution": "1k",
    }
    assert call["timeout"] == 30.0
    assert call["poll_interval"] == 1.0
    assert result.data["url"] == "https://cdn.wavespeed.ai/out.png"


def test_image_never_sends_size_or_seed(fake_client):
    """The platform whitelist drops these; the component must not pretend."""
    make_image().generate_image()

    sent = fake_client["client"].calls[0]["input"]
    assert "size" not in sent
    assert "seed" not in sent


def test_blank_optional_args_are_dropped(fake_client):
    make_image(aspect_ratio="", resolution=None).generate_image()

    assert fake_client["client"].calls[0]["input"] == {"prompt": "a red panda drinking boba"}


def test_video_arg_mapping(fake_client):
    fake_client["client"] = FakeClient(outputs=["https://cdn.wavespeed.ai/out.mp4"])
    result = make_video().generate_video()

    call = fake_client["client"].calls[0]
    assert call["model"] == DEFAULT_VIDEO_MODEL
    assert call["input"] == {"prompt": "a drone shot over a glacier", "duration": 5}
    assert result.data["url"] == "https://cdn.wavespeed.ai/out.mp4"


def test_video_duration_omitted_when_unset(fake_client):
    make_video(duration=None).generate_video()

    assert fake_client["client"].calls[0]["input"] == {"prompt": "a drone shot over a glacier"}


def test_client_receives_the_configured_key(fake_client):
    make_image().generate_image()

    assert fake_client["api_key"] == "ws-test-key"


def test_real_client_factory_passes_client_name(monkeypatch):
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    module = type("m", (), {"Client": Client})
    monkeypatch.setitem(__import__("sys").modules, "wavespeed", module)

    _base.build_client("k")
    assert captured == {"api_key": "k", "client_name": "langflow", "max_retries": 0}


# --------------------------------------------------------------------------
# Run Model
# --------------------------------------------------------------------------


def test_run_model_parses_json_string(fake_client):
    make_runner().run_model()

    assert fake_client["client"].calls[0]["input"] == {"prompt": "a lighthouse at dusk"}


def test_run_model_accepts_dict(fake_client):
    make_runner(model_input={"prompt": "hi", "guidance_scale": 3.5}).run_model()

    assert fake_client["client"].calls[0]["input"] == {"prompt": "hi", "guidance_scale": 3.5}


def test_run_model_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON string"):
        make_runner(model_input="{not json").run_model()


def test_run_model_rejects_non_object_json():
    with pytest.raises(ValueError, match="JSON object"):
        make_runner(model_input="[1, 2]").run_model()


def test_run_model_requires_model_id():
    with pytest.raises(ValueError, match="model id is required"):
        make_runner(model_name="  ").run_model()


# --------------------------------------------------------------------------
# Terminal statuses and error surfacing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["failed", "cancelled", "timeout"])
def test_terminal_status_is_surfaced_with_task_id(fake_client, status):
    """The SDK raises on failed/cancelled/timeout; keep its message verbatim."""
    sdk_error = RuntimeError(f"Prediction {status} (task_id: abc123): out of capacity")
    fake_client["client"] = FakeClient(error=sdk_error)

    with pytest.raises(ValueError) as excinfo:
        make_image().generate_image()

    message = str(excinfo.value)
    assert status in message
    assert "task_id: abc123" in message
    assert DEFAULT_IMAGE_MODEL in message


def test_timeout_error_is_surfaced(fake_client):
    fake_client["client"] = FakeClient(error=TimeoutError("Prediction timed out (task_id: t-9)"))

    with pytest.raises(ValueError, match="task_id: t-9"):
        make_video().generate_video()


def test_empty_outputs_raise(fake_client):
    fake_client["client"] = FakeClient(outputs=[])

    with pytest.raises(ValueError, match="returned no outputs"):
        make_image().generate_image()


def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="API key is required"):
        make_image(api_key="").generate_image()


def test_blank_prompt_raises():
    with pytest.raises(ValueError, match="non-empty prompt"):
        make_image(prompt="   ").generate_image()


def test_blank_video_prompt_raises():
    with pytest.raises(ValueError, match="non-empty prompt"):
        make_video(prompt="").generate_video()


# --------------------------------------------------------------------------
# Output formatting
# --------------------------------------------------------------------------


def test_multiple_outputs_are_all_kept(fake_client):
    fake_client["client"] = FakeClient(
        outputs=[{"url": "https://cdn.wavespeed.ai/a.png"}, "https://cdn.wavespeed.ai/b.png"]
    )

    result = make_image().generate_image()

    assert result.data["urls"] == [
        "https://cdn.wavespeed.ai/a.png",
        "https://cdn.wavespeed.ai/b.png",
    ]
    assert result.data["url"] == "https://cdn.wavespeed.ai/a.png"


def test_opaque_output_is_json_encoded(fake_client):
    fake_client["client"] = FakeClient(outputs=[{"text": "no url here"}])

    result = make_image().generate_image()

    assert json.loads(result.data["url"]) == {"text": "no url here"}


# --------------------------------------------------------------------------
# Model catalog dropdown
# --------------------------------------------------------------------------


CATALOG = [
    {"model_id": "bytedance/seedream-v5.0-pro", "type": "text-to-image"},
    {"model_id": "wavespeed-ai/flux-dev", "type": "text-to-image"},
    {"model_id": "bytedance/seedance-2.5/text-to-video", "type": "text-to-video"},
    {"model_id": "some/llm", "type": "text-to-text"},
    {"not_a_model": True},
]


def test_get_models_filters_by_type(monkeypatch):
    monkeypatch.setattr(_base, "fetch_catalog", lambda key: CATALOG)

    assert make_image().get_models() == [
        "wavespeed-ai/z-image/turbo",
        "bytedance/seedream-v5.0-pro",
        "wavespeed-ai/flux-dev",
    ]
    assert make_video().get_models() == ["bytedance/seedance-2.5/text-to-video"]


def test_get_models_falls_back_on_network_error(monkeypatch):
    def boom(key):
        raise requests.RequestException("catalog down")

    monkeypatch.setattr(_base, "fetch_catalog", boom)
    component = make_image()

    assert component.get_models() == list(FALLBACK_IMAGE_MODELS)
    assert "catalog down" in component.status


def test_get_models_without_key_does_not_call_out(monkeypatch):
    def boom(key):
        raise AssertionError("must not hit the catalog without a key")

    monkeypatch.setattr(_base, "fetch_catalog", boom)

    assert make_image(api_key="").get_models() == list(FALLBACK_IMAGE_MODELS)


def test_update_build_config_refreshes_options(monkeypatch):
    monkeypatch.setattr(_base, "fetch_catalog", lambda key: CATALOG)
    build_config = {"model_name": {"options": []}}

    updated = make_image().update_build_config(build_config, "", "model_name")

    assert "wavespeed-ai/flux-dev" in updated["model_name"]["options"]


def test_update_build_config_ignores_other_fields(monkeypatch):
    monkeypatch.setattr(_base, "fetch_catalog", lambda key: CATALOG)
    build_config = {"model_name": {"options": ["kept"]}}

    updated = make_image().update_build_config(build_config, "", "prompt")

    assert updated["model_name"]["options"] == ["kept"]
