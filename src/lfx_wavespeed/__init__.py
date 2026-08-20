"""lfx-wavespeed: WaveSpeed AI bundle for Langflow.

Distribution unit ``lfx-wavespeed``.  At runtime Langflow's loader discovers the
``extension.json`` shipped alongside this ``__init__.py`` and registers the
bundle's components under the namespaced IDs
``ext:wavespeed:WaveSpeedImageGeneration@official``,
``ext:wavespeed:WaveSpeedVideoGeneration@official`` and
``ext:wavespeed:WaveSpeedRunModel@official``.
"""

from lfx_wavespeed.components.wavespeed.wavespeed_image_generation import (
    WaveSpeedImageGenerationComponent,
)
from lfx_wavespeed.components.wavespeed.wavespeed_run_model import (
    WaveSpeedRunModelComponent,
)
from lfx_wavespeed.components.wavespeed.wavespeed_video_generation import (
    WaveSpeedVideoGenerationComponent,
)

__all__ = [
    "WaveSpeedImageGenerationComponent",
    "WaveSpeedRunModelComponent",
    "WaveSpeedVideoGenerationComponent",
]
