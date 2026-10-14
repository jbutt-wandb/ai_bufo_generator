"""The three-agent production crew: Director, Screenwriter, Cameraman.

Flow per request (route first, then write, then render):
1. Director.pick_workflow    — Laya scores the task types   → select_workflow trace
2. Director.write_screenplay — delegates to the Screenwriter agent, which
   RETRIEVES its references (MiniMax prompt guide sliced to the routed task,
   plus the Bufo character description) and writes the structured prompt
                                                            → rewrite_prompt trace
3. Director.shoot_video      — delegates to the Cameraman agent, which renders
   the screenplay verbatim with MiniMax-H3                  → generate_video trace

All three agents run under one RunConfig group_id per generation, so the Weave
Agents view shows them as a single session of three named agents.
"""

import os
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal, Optional

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunContextWrapper,
    Runner,
    function_tool,
)
from openai import AsyncOpenAI
from PIL import Image

import weave

from . import classifier, config
from .prompt_guide import load_bufo_description, load_guide


@dataclass
class BufoRun:
    """Shared state for one generation: UI settings in, step results out.

    Canvas/duration come from UI presets (not the agents), so a creative model
    can never pick a config that OOMs the GPUs.
    """

    user_prompt: str
    image_paths: list[str] = field(default_factory=list)
    height: int = 480
    width: int = 832
    num_frames: int = 124
    num_inference_steps: int = 50
    seed: int = 42
    # Groups the three agents' runs into one Weave Agents-view session.
    group_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Filled in by the tools as the agents work:
    workflow: Optional[str] = None
    workflow_probs: Optional[dict] = None
    rewritten_prompt: Optional[str] = None
    video_path: Optional[str] = None


# The active run, reachable inside weave ops without appearing in their traced inputs.
_ACTIVE_RUN: ContextVar[Optional[BufoRun]] = ContextVar("_ACTIVE_RUN", default=None)


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=config.INFERENCE_BASE_URL,
        api_key=os.environ["WANDB_API_KEY"],
        default_headers={"OpenAI-Project": config.WEAVE_PROJECT},
    )


def _tool_error(ctx: RunContextWrapper[BufoRun], error: Exception) -> str:
    # The SDK's default hides the exception ("An error occurred..."), which once let
    # an agent claim success on a failed generation. Give the model the real error.
    return f"{type(error).__name__}: {error}"


# --- Screenwriter: writes the MiniMax prompt from RETRIEVED references ----------


@function_tool(failure_error_function=_tool_error)
def retrieve_reference(
    ctx: RunContextWrapper[BufoRun], name: Literal["prompt_guide", "bufo_description"]
) -> str:
    """Retrieve a reference document: the MiniMax-H3 prompt-writing guide
    (sliced to this production's task type) or the Bufo character description."""
    if name == "prompt_guide":
        return load_guide(ctx.context.workflow)
    return load_bufo_description()


SCREENWRITER_INSTRUCTIONS = """
You are the Bufo Screenwriter: you turn a video request into ONE structured
MiniMax-H3 prompt. You do not know MiniMax-H3's prompt format or what Bufo
looks like from memory — ALWAYS start by retrieving both references:
1. retrieve_reference("prompt_guide") — the exact prompt format for this
   production's task type. Follow it precisely.
2. retrieve_reference("bufo_description") — what Bufo looks like. Ground every
   visual mention of Bufo in this description (unless the request clearly asks
   for something else).

Then write the prompt: cinematic and concrete, inventing tasteful detail (shot
framing, lighting, diegetic sounds, score) that serves the request. Use the
stated duration for any timestamp/keyframe alignment. Uploaded images are
numbered in upload order: as keyframes the first is the start frame and the
second (if present) the end frame; as references they are <Picture 1> and
<Picture 2>.

Reply with the finished prompt ONLY — no commentary, no code fences.
"""


def build_screenwriter() -> Agent[BufoRun]:
    return Agent[BufoRun](
        name="Bufo Screenwriter",
        instructions=SCREENWRITER_INSTRUCTIONS,
        model=OpenAIChatCompletionsModel(model=config.REWRITER_MODEL, openai_client=_client()),
        tools=[retrieve_reference],
    )


@weave.op
async def rewrite_prompt(user_prompt: str, model: str, n_images: int, duration_s: float, workflow: str) -> str:
    """Rewrite the user's idea into MiniMax-H3's structured prompt format
    (delegated to the Screenwriter agent, which retrieves its references)."""
    run = _ACTIVE_RUN.get()
    result = await Runner.run(
        build_screenwriter(),
        (
            f"Video request: {user_prompt}\n"
            f"Task type: {workflow}\n"
            f"Uploaded images: {n_images}\n"
            f"Video duration: {duration_s:.2f} seconds"
        ),
        context=run,
        max_turns=8,
        run_config=RunConfig(group_id=run.group_id),
    )
    return str(result.final_output).strip()


# --- Cameraman: renders the screenplay verbatim ---------------------------------


@function_tool(failure_error_function=_tool_error)
def render_video(ctx: RunContextWrapper[BufoRun]) -> str:
    """Render the production's screenplay with MiniMax-H3 and return the output
    file path. Takes several minutes on the GPUs."""
    # Imported lazily so the agent pipeline can be smoke-tested without torch/GPUs.
    from . import generator

    run = ctx.context
    # Image routing convention (the UI enforces <= MAX_IMAGE_REFS uploads):
    # keyframe tasks use uploads as start/end frames; ref2va as <Picture N>
    # identity references. Decoded to PIL here so each image renders viewable
    # in the generate_video Weave trace (paths would show as strings).
    uploads = [Image.open(p).convert("RGB") for p in run.image_paths]
    media: dict = {}
    if run.workflow == "i2va":
        media["first_frame"] = uploads[0]
    elif run.workflow == "l2va":
        media["last_frame"] = uploads[0]
    elif run.workflow == "fl2va":
        media["first_frame"], media["last_frame"] = uploads[0], uploads[1]
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


CAMERAMAN_INSTRUCTIONS = """
You are the Bufo Cameraman: you shoot exactly what the screenplay says.
Call render_video exactly once — no creative changes, no re-renders — then
reply with one short line confirming the shot is in the can.
If render_video returns an error, do NOT claim the video was made: report the
error text plainly.
"""


def build_cameraman() -> Agent[BufoRun]:
    return Agent[BufoRun](
        name="Bufo Cameraman",
        instructions=CAMERAMAN_INSTRUCTIONS,
        model=OpenAIChatCompletionsModel(model=config.REWRITER_MODEL, openai_client=_client()),
        tools=[render_video],
    )


# --- Director: orchestrates the crew --------------------------------------------


@function_tool(failure_error_function=_tool_error)
def pick_workflow(ctx: RunContextWrapper[BufoRun]) -> str:
    """Classify which MiniMax-H3 task type fits this request (t2va, i2va, l2va,
    fl2va or ref2va)."""
    run = ctx.context
    decision = classifier.select_workflow(run.user_prompt, len(run.image_paths))
    run.workflow, run.workflow_probs = decision["workflow"], decision["probs"]
    return f"workflow: {decision['workflow']} (probabilities: {decision['probs']})"


@function_tool(failure_error_function=_tool_error)
async def write_screenplay(ctx: RunContextWrapper[BufoRun]) -> str:
    """Have the Screenwriter agent turn the request into a MiniMax-H3 prompt."""
    run = ctx.context
    if not run.workflow:
        raise ValueError("Call pick_workflow first.")
    token = _ACTIVE_RUN.set(run)
    try:
        run.rewritten_prompt = await rewrite_prompt(
            user_prompt=run.user_prompt,
            model=config.REWRITER_MODEL,
            n_images=len(run.image_paths),
            duration_s=run.num_frames / config.FPS,
            workflow=run.workflow,
        )
    finally:
        _ACTIVE_RUN.reset(token)
    return run.rewritten_prompt


@function_tool(failure_error_function=_tool_error)
async def shoot_video(ctx: RunContextWrapper[BufoRun]) -> str:
    """Have the Cameraman agent render the screenplay (takes several minutes)."""
    run = ctx.context
    if not (run.workflow and run.rewritten_prompt):
        raise ValueError("Call pick_workflow and write_screenplay first.")
    result = await Runner.run(
        build_cameraman(),
        "The screenplay is ready — shoot the video.",
        context=run,
        max_turns=4,
        run_config=RunConfig(group_id=run.group_id),
    )
    if not run.video_path:
        raise RuntimeError(f"The Cameraman produced no video: {result.final_output}")
    return run.video_path


DIRECTOR_INSTRUCTIONS = """
You are the Bufo Director: you run the crew that turns casual requests into
finished videos of Bufo, the beloved cartoon toad.

For each request, call these tools, in order, exactly once each:
1. pick_workflow — routes the request to a MiniMax-H3 task type.
2. write_screenplay — the Screenwriter turns the request into a structured prompt.
3. shoot_video — the Cameraman renders the clip (takes several minutes).
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
        instructions=DIRECTOR_INSTRUCTIONS,
        model=OpenAIChatCompletionsModel(model=config.REWRITER_MODEL, openai_client=_client()),
        tools=[pick_workflow, write_screenplay, shoot_video],
    )


def build_run_input(run: BufoRun) -> str:
    """Format the user's request (plus the facts the agents need) as the run input."""
    duration = run.num_frames / config.FPS
    return (
        f"Video request: {run.user_prompt}\n"
        f"Uploaded images: {len(run.image_paths)}\n"
        f"Video duration: {duration:.2f} seconds ({run.num_frames} frames at {config.FPS} fps)"
    )
