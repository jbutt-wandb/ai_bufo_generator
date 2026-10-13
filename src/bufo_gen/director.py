"""The director agent: rewrites the prompt, routes the request, generates the video.

Flow per request (three weave-traced steps, orchestrated by the agent):
1. rewrite_prompt    — Kimi turns the user's idea into a structured MiniMax-H3 prompt
2. select_workflow   — Laya scores the five task types (t2va/i2va/l2va/fl2va/ref2va)
                       on the user's request and upload count
3. generate_video    — MiniMax-H3 renders the clip

Agent-level spans (the run, tool calls, model turns) are exported to Weave by the
WeaveTracingProcessor installed in tracing.init_tracing().
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from agents import Agent, OpenAIChatCompletionsModel, RunContextWrapper, function_tool
from openai import AsyncOpenAI
from PIL import Image

import weave

from . import classifier, config
from .prompt_guide import PROMPT_GUIDE


@dataclass
class BufoRun:
    """Shared state for one generation: UI settings in, step results out.

    Canvas/duration come from UI presets (not the agent), so a creative model
    can never pick a config that OOMs the GPUs.
    """

    user_prompt: str
    image_paths: list[str] = field(default_factory=list)
    height: int = 480
    width: int = 832
    num_frames: int = 124
    num_inference_steps: int = 50
    seed: int = 42
    # Filled in by the tools as the agent works:
    rewritten_prompt: Optional[str] = None
    workflow: Optional[str] = None
    workflow_probs: Optional[dict] = None
    video_path: Optional[str] = None


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=config.INFERENCE_BASE_URL,
        api_key=os.environ["WANDB_API_KEY"],
        default_headers={"OpenAI-Project": config.WEAVE_PROJECT},
    )


_REWRITER_SYSTEM = f"""
You rewrite casual requests into ONE structured MiniMax-H3 video prompt starring
Bufo, the beloved plump cheerful green cartoon toad (unless the user clearly asks
otherwise). Be cinematic and concrete; invent tasteful detail (shot framing,
lighting, sounds, score) that serves the user's idea. Use the stated duration for
any timestamp/keyframe alignment.

Choose the prompt format from the uploads:
- No uploaded images: the 3-section t2va format.
- Uploaded images used as literal keyframes (animating the picture itself): the
  3-section format with keyframe alignment — anchor the start frame (one image
  opening the video, i2va), the end frame (one image the video converges onto,
  l2va), or both (two images, fl2va), whichever the request implies.
- Uploaded images used as identity references (the subject appears in a NEW
  scene): the 6-section ref2va format.

Uploaded images are numbered in upload order: as keyframes the first is the start
frame and the second (if present) the end frame; as references they are
<Picture 1> and <Picture 2>.

Reply with the rewritten prompt ONLY — no commentary, no code fences.

{PROMPT_GUIDE}
"""


@weave.op
async def rewrite_prompt(user_prompt: str, model: str, n_images: int, duration_s: float) -> str:
    """Rewrite the user's idea into MiniMax-H3's structured prompt format."""
    response = await _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _REWRITER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Video request: {user_prompt}\n"
                    f"Uploaded images: {n_images}\n"
                    f"Video duration: {duration_s:.2f} seconds"
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def _tool_error(ctx: RunContextWrapper[BufoRun], error: Exception) -> str:
    # The SDK's default hides the exception ("An error occurred..."), which once let
    # the agent claim success on a failed generation. Give the model the real error.
    return f"{type(error).__name__}: {error}"


@function_tool(failure_error_function=_tool_error)
async def rewrite_bufo_prompt(ctx: RunContextWrapper[BufoRun]) -> str:
    """Rewrite the user's request into MiniMax-H3's structured prompt format."""
    run = ctx.context
    run.rewritten_prompt = await rewrite_prompt(
        user_prompt=run.user_prompt,
        model=config.REWRITER_MODEL,
        n_images=len(run.image_paths),
        duration_s=run.num_frames / config.FPS,
    )
    return run.rewritten_prompt


@function_tool(failure_error_function=_tool_error)
def pick_workflow(ctx: RunContextWrapper[BufoRun]) -> str:
    """Classify which MiniMax-H3 task type fits the user's request (t2va, i2va, l2va, fl2va or ref2va)."""
    run = ctx.context
    decision = classifier.select_workflow(run.user_prompt, len(run.image_paths))
    run.workflow, run.workflow_probs = decision["workflow"], decision["probs"]
    return f"workflow: {decision['workflow']} (probabilities: {decision['probs']})"


@function_tool(failure_error_function=_tool_error)
def generate_bufo_video(ctx: RunContextWrapper[BufoRun]) -> str:
    """Generate the video with MiniMax-H3 and return the output file path.

    Uses the rewritten prompt and workflow from the previous steps. Takes ~5-15
    minutes on the GPUs.
    """
    # Imported lazily so the agent pipeline can be smoke-tested without torch/GPUs.
    from . import generator

    run = ctx.context
    if not (run.rewritten_prompt and run.workflow):
        raise ValueError("Call rewrite_bufo_prompt and pick_workflow first.")

    # Image routing convention (the UI enforces <= MAX_IMAGE_REFS uploads):
    # i2va/l2va: the single upload is the start/end keyframe; fl2va: first upload
    # = start frame, second = end frame; ref2va: upload order = <Picture N>
    # numbering. Decoded to PIL here so each image renders viewable in the
    # generate_video Weave trace (paths would show as strings).
    uploads = [Image.open(p).convert("RGB") for p in run.image_paths]
    media: dict = {}
    if run.workflow == "i2va":
        media["first_frame"] = uploads[0]
    elif run.workflow == "l2va":
        media["last_frame"] = uploads[0]
    elif run.workflow == "fl2va":
        media["first_frame"] = uploads[0]
        media["last_frame"] = uploads[1]
    elif run.workflow == "ref2va":
        media["reference_1"] = uploads[0]
        if len(uploads) > 1:
            media["reference_2"] = uploads[1]

    run.video_path = generator.generate_video(
        workflow=run.workflow,
        prompt=run.rewritten_prompt,
        height=run.height,
        width=run.width,
        num_frames=run.num_frames,
        num_inference_steps=run.num_inference_steps,
        seed=run.seed,
        **media,
    )
    return run.video_path


INSTRUCTIONS = """
You are the Bufo Director: you turn casual requests into finished videos of Bufo,
the beloved cartoon toad from the Bufo emoji pack.

For each request, call these tools, in order, exactly once each:
1. rewrite_bufo_prompt — turns the request into a structured MiniMax-H3 prompt.
2. pick_workflow — routes the request to a MiniMax-H3 task type.
3. generate_bufo_video — renders the clip (takes several minutes).
Then reply with a one-sentence summary of the clip you directed. Do not include
the full prompt or file path in your reply.

If a tool returns an error, do NOT claim the video was made: retry once only if
the error names something you can change, otherwise reply stating plainly that
generation failed and quote the error.
"""


def build_director() -> Agent[BufoRun]:
    """Create the director agent on Kimi via Forge Inference (OpenAI-compatible)."""
    return Agent[BufoRun](
        name="Bufo Director",
        instructions=INSTRUCTIONS,
        model=OpenAIChatCompletionsModel(model=config.REWRITER_MODEL, openai_client=_client()),
        tools=[rewrite_bufo_prompt, pick_workflow, generate_bufo_video],
    )


def build_run_input(run: BufoRun) -> str:
    """Format the user's request (plus the facts the agent needs) as the run input."""
    duration = run.num_frames / config.FPS
    return (
        f"Video request: {run.user_prompt}\n"
        f"Uploaded images: {len(run.image_paths)}\n"
        f"Video duration: {duration:.2f} seconds ({run.num_frames} frames at {config.FPS} fps)"
    )
