"""Central configuration: environment, model ids, and hardware limits we must respect."""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

# Weave's local trace cache is SQLite-backed, which fails on the cluster's NFS
# home directories — disable it. Must be set before weave.init() is called.
os.environ.setdefault("WEAVE_USE_SERVER_CACHE", "false")

# --- Forge (Weights & Biases) ---
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "wandb-smle")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "jb_bufo_gen")
WEAVE_PROJECT = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

# Forge Inference is OpenAI-compatible: it authenticates with the W&B API key and
# attributes usage to a W&B project via the "OpenAI-Project" header.
INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
REWRITER_MODEL = "moonshotai/Kimi-K2.6"

# --- MiniMax-H3 ---
VIDEO_MODEL_ID = "MiniMaxAI/MiniMax-H3"
FPS = 24  # MiniMax-H3 always generates at 24 fps

# Ceilings measured on the 2x H100 split (see docs/minimax_prompt_guide.md and the
# video_gen_demo project this was ported from):
MAX_PIXEL_FRAMES = 53_000_000  # height * width * frames above this OOMs the denoise GPU
MAX_IMAGE_REFS = 2             # a 3rd image ref OOMs the Qwen3-VL conditioner GPU
MIN_FRAMES = 124               # the model generates 5.0-15.0 s only; 124 is the smallest 17n+5 >= 120
MAX_FRAMES = 345               # aligned duration must stay <= 15.0 s (345 / 24 = 14.375 s)

# Proven canvas/duration combos. The UI only offers these, so neither the user nor
# the agent can request a config that blows the memory ceiling.
PRESETS = {
    "default — 832x480, ~5s": dict(height=480, width=832, num_frames=124),
    "fast — 384x384, ~5s": dict(height=384, width=384, num_frames=124),
    "long — 768x448, ~5.9s": dict(height=448, width=768, num_frames=141),
}
DEFAULT_PRESET = "default — 832x480, ~5s"

OUTPUT_DIR = "generated_videos"
UPLOAD_DIR = "uploads"


def snap_frames(n: int) -> int:
    """Clamp to MiniMax-H3's valid frame counts: 17k+5, between 5.0 and 15.0 s."""
    k = -(-(n - 5) // 17)  # ceil((n-5)/17)
    return min(max(17 * k + 5, MIN_FRAMES), MAX_FRAMES)


# Every offered preset must survive the generator's own ceiling check — a preset
# that violates it fails every generation instantly (this bit us once).
for _name, _p in PRESETS.items():
    assert _p["height"] * _p["width"] * _p["num_frames"] <= MAX_PIXEL_FRAMES, (
        f"preset {_name!r} exceeds MAX_PIXEL_FRAMES"
    )
    assert snap_frames(_p["num_frames"]) == _p["num_frames"], (
        f"preset {_name!r} has a misaligned frame count"
    )
