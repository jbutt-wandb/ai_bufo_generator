"""MiniMax-H3 video generation: load once per workflow, generate, mux to mp4.

Ported from the proven video_gen_demo/scripts/minimax_h3.py. The model is ~124GB
in bf16 (61.7GB DiT transformer + 62.1GB Qwen3-VL conditioner) and is split
across two H100s: prompt/media encoding on cuda:1, denoise/decode on cuda:0.
"""

import os

# Must be set before `import torch`: torch 2.13's _native ops otherwise register
# triton-JIT kernels that fail to compile on nodes without Python dev headers.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from typing import Annotated, Optional

import torch
import weave
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
from diffusers.utils.export_utils import encode_video
from PIL import Image
from weave import Content

from . import config

# workflow -> (conditioner, pipe). Each workflow has its own ~62GB transformer
# partition, so we cache per workflow and load lazily on first use.
_PIPELINES: dict[str, tuple] = {}

# The prompt guide's five task types run on three pipeline partitions: i2va and
# l2va are the fl2va pipeline conditioned on only the first / only the last
# keyframe. Mapping before _load() keeps them sharing one cached partition.
_PIPELINE_WORKFLOW = {"i2va": "fl2va", "l2va": "fl2va"}


def _load(workflow: str):
    if workflow in _PIPELINES:
        return _PIPELINES[workflow]

    if torch.cuda.device_count() < 2:
        raise RuntimeError("MiniMax-H3 needs the 2x H100 split (run inside the Slurm allocation).")

    blocks = ModularPipeline.from_pretrained(config.VIDEO_MODEL_ID).blocks.get_workflow(workflow)

    # The Qwen3-VL conditioner encodes the model's *presentation* of the request —
    # for fl2va/ref2va that includes the prepared keyframes or references — so the
    # media-preparation block rides along with the text encoder on cuda:1.
    conditioner_blocks = SequentialPipelineBlocks.from_blocks_dict(
        {
            name: blocks.sub_blocks.pop(name)
            for name in ("before_encode", "text_encoder")
            if name in blocks.sub_blocks
        }
    )
    text_manager = ComponentsManager()
    text_manager.enable_auto_cpu_offload(device="cuda:1")
    conditioner = conditioner_blocks.init_pipeline(config.VIDEO_MODEL_ID, components_manager=text_manager)
    conditioner.load_components(dtype=torch.bfloat16)

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda:0")
    pipe = blocks.init_pipeline(config.VIDEO_MODEL_ID, components_manager=manager)
    pipe.load_components(dtype=torch.bfloat16)

    try:
        # Flash Attention 3 on Hopper; silently falls back to SDPA on this cluster
        # (the `kernels` package can't compile on the compute nodes).
        transformer = getattr(pipe, "transformer", None) or getattr(pipe, "transformer_ref", None)
        transformer.set_attention_backend("_flash_3_hub")
    except Exception as e:
        print(f"Flash Attention 3 unavailable, using default backend ({e})")

    _PIPELINES[workflow] = (conditioner, pipe)
    return conditioner, pipe


@weave.op
def generate_video(
    workflow: str,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int = 50,
    seed: int = 42,
    # PIL images (not paths) so Weave renders each one viewable in the trace.
    first_frame: Optional[Image.Image] = None,
    last_frame: Optional[Image.Image] = None,
    reference_1: Optional[Image.Image] = None,
    reference_2: Optional[Image.Image] = None,
) -> Annotated[str, Content]:
    """Generate a video+audio mp4 and return its path (rendered playable in Weave).

    fl2va conditions on `first_frame`/`last_frame` keyframes; ref2va on
    `reference_1`/`reference_2` identity references (<Picture N> order).
    """
    # Validate against the measured hardware ceilings before touching the GPUs.
    num_frames = config.snap_frames(num_frames)
    if height * width * num_frames > config.MAX_PIXEL_FRAMES:
        raise ValueError(f"{width}x{height}x{num_frames} exceeds the ~53M pixel-frame memory ceiling.")

    media_kwargs = {}
    if workflow == "i2va":
        if first_frame is None:
            raise ValueError("i2va needs a start image.")
        media_kwargs["image"] = first_frame
    elif workflow == "l2va":
        if last_frame is None:
            raise ValueError("l2va needs an end image.")
        media_kwargs["last_image"] = last_frame
    elif workflow == "fl2va":
        if first_frame is None or last_frame is None:
            raise ValueError("fl2va needs both a start and an end image.")
        media_kwargs["image"] = first_frame
        media_kwargs["last_image"] = last_frame
    elif workflow == "ref2va":
        if reference_1 is None:
            raise ValueError("ref2va needs at least one reference image.")
        media_kwargs["references"] = [
            MiniMaxH3ImageReference(image=ref)
            for ref in (reference_1, reference_2)
            if ref is not None
        ]

    pipeline_workflow = _PIPELINE_WORKFLOW.get(workflow, workflow)
    conditioner, pipe = _load(pipeline_workflow)

    # MiniMax-H3 is guidance-distilled: no negative_prompt / guidance_scale.
    gen_kwargs = dict(
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(seed),
        output=["videos", "audio", "sampling_rate"],
    )
    # The conditioner resolves the canvas geometry for media workflows (and, for
    # ref2va, also the frame count), so those kwargs go to whichever stage owns them.
    cond_kwargs = dict(prompt=prompt, **media_kwargs)
    if workflow == "t2va":
        gen_kwargs.update(height=height, width=width)
    else:
        cond_kwargs.update(height=height, width=width)
        if workflow == "ref2va":
            cond_kwargs["num_frames"] = gen_kwargs.pop("num_frames")

    state = conditioner(**cond_kwargs)
    results = pipe(state=state, **gen_kwargs)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(
        config.OUTPUT_DIR, f"bufo_{workflow}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    )
    # Video frames and audio come back separately; encode_video muxes them.
    encode_video(
        results["videos"][0],
        fps=config.FPS,
        output_path=out_path,
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    return out_path
