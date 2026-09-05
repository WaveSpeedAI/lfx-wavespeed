# lfx-wavespeed

[WaveSpeed AI](https://wavespeed.ai) components for [Langflow](https://langflow.org),
packaged as a Langflow Extension Bundle.

WaveSpeed AI is a high-performance inference platform for image and video
generation models. This bundle exposes it in the Langflow palette — no fork, no
PR to Langflow, just `pip install`.

## Install

```bash
pip install lfx-wavespeed
```

Restart Langflow. A **WaveSpeed AI** bundle appears in the component palette.

## Components

| Component | What it does |
| --- | --- |
| **WaveSpeed AI Image Generation** | Text-to-image. Defaults to `bytedance/seedream-v5.0-pro`; the model dropdown loads the live catalog. Exposes prompt, aspect ratio and resolution. |
| **WaveSpeed AI Video Generation** | Text-to-video. Defaults to `wavespeed-ai/minimax-h3/text-to-video` (cheap open-weights starting point; pick `bytedance/seedance-2.5/text-to-video` for the highest quality). Exposes prompt and duration. |
| **WaveSpeed AI Run Model** | Runs any model id from the catalog with a raw JSON input payload. |

All three take a `WaveSpeed API Key` (get one at
<https://wavespeed.ai/dashboard>) and can be driven by an Agent via tool mode.
Each returns a `Data` payload with `url`, `urls` and the raw platform `outputs`.

The model dropdowns refresh from `GET /api/v3/models` when you change the key or
click the refresh button; without a reachable catalog they fall back to a small
curated list.

## Why no `size` / `seed` inputs

WaveSpeed validates model inputs against each model's declared schema and drops
undeclared fields. For the default image models `size` and `seed` are not
declared, so a UI field for them would look effective while doing nothing. Use
**Run Model** with an explicit JSON payload if a specific model does declare
them.

## Development

```bash
pip install -e ".[test]" langflow
lfx extension validate src/lfx_wavespeed   # static manifest + bundle check
pytest                                     # mocked SDK, no network
ruff check .
lfx extension dev src/lfx_wavespeed        # Langflow with this bundle loaded
```

## License

MIT

---

**[WaveSpeed AI](https://wavespeed.ai/)** — AI image & video generation platform.
Try it in the browser: **[Image generator](https://wavespeed.ai/image-generator)** · **[Video generator](https://wavespeed.ai/video-generator)**
