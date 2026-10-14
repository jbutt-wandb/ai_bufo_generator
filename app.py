"""Bufo Gen — describe a bufo video, optionally add reference images, get a clip.

Runs on the cluster GPU node (`marimo run app.py --host 0.0.0.0 --port 2718`);
users reach it through an SSH tunnel. Every generation is traced end-to-end in
Forge Weave (wandb-smle/jb_bufo_gen) with the playable video on the trace.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    # One-time setup: env, Weave tracing (incl. the Agents SDK processor), director.
    from bufo_gen import config
    from bufo_gen.director import BufoRun, build_director, build_run_input
    from bufo_gen.tracing import init_tracing

    init_tracing()
    director = build_director()

    mo.md(
        f"# 🐸 Bufo Gen\n"
        f"Describe the bufo video you want. Traces land in "
        f"[Forge Weave](https://wandb.ai/{config.WEAVE_PROJECT}/weave/traces)."
    )
    return BufoRun, build_run_input, config, director


@app.cell
def _(config, mo):
    prompt_box = mo.ui.text_area(
        label="What should Bufo do?",
        placeholder="e.g. Bufo rides a tiny skateboard off a ramp into a pond, slow-mo splash...",
        rows=4,
        full_width=True,
    )
    images = mo.ui.file(
        label=f"Reference images (optional, max {config.MAX_IMAGE_REFS})",
        filetypes=[".png", ".jpg", ".jpeg", ".webp"],
        multiple=True,
        kind="area",
    )
    preset = mo.ui.dropdown(
        options=list(config.PRESETS), value=config.DEFAULT_PRESET, label="Quality preset"
    )
    seed = mo.ui.number(value=42, start=0, step=1, label="Seed")
    steps = mo.ui.number(
        value=50, start=20, stop=100, step=1, label="Inference steps (model default 50)"
    )
    go = mo.ui.run_button(label="🎬 Generate bufo video")

    mo.vstack(
        [
            prompt_box,
            images,
            mo.hstack([preset, seed], justify="start"),
            mo.accordion({"Advanced parameters": steps}),
            go,
        ]
    )
    return go, images, preset, prompt_box, seed, steps


@app.cell
def _(images, mo):
    # Show the SERVER's view of the attachments: the browser droparea can look
    # populated from a previous session while the live kernel has nothing (uploads
    # reset on every app restart / page reload — this silently caused t2va runs).
    _files = list(images.value or [])
    mo.md(
        f"**Attached:** {', '.join(f.name for f in _files)}"
        if _files
        else "*No reference images attached — this will be a text-only (t2va) run.*"
    )
    return


@app.cell
async def _(BufoRun, build_run_input, config, director, go, images, mo, preset, prompt_box, seed, steps):
    import asyncio
    import os
    import re
    from pathlib import Path

    from agents import RunConfig, Runner

    # Only run on an explicit click, never on incidental UI changes.
    mo.stop(not go.value, mo.md("*Fill in a prompt and hit Generate.*"))
    mo.stop(not prompt_box.value.strip(), mo.md("**Please describe the video first.**"))
    mo.stop(
        len(images.value or []) > config.MAX_IMAGE_REFS,
        mo.md(
            f"**Please upload at most {config.MAX_IMAGE_REFS} reference images** — "
            f"MiniMax-H3 on this hardware supports a maximum of {config.MAX_IMAGE_REFS}."
        ),
    )
    # Uploads silently reset on every app restart/page reload; catch the resulting
    # "prompt talks about an image but none is attached" runs before wasting a render.
    mo.stop(
        not (images.value or [])
        and re.search(r"\b(image|picture|photo|reference|ref)s?\b", prompt_box.value, re.I),
        mo.callout(
            mo.md(
                "**Your prompt mentions an image, but no files are attached.** Uploads are "
                "cleared whenever the app restarts or the page reloads — re-attach the "
                "image(s) and click Generate again."
            ),
            kind="warn",
        ),
    )

    # The generation pipeline wants file paths, so persist the uploaded bytes.
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    image_paths = []
    for f in list(images.value or []):
        path = os.path.join(config.UPLOAD_DIR, f.name)
        Path(path).write_bytes(f.contents)
        image_paths.append(path)

    run_state = BufoRun(
        user_prompt=prompt_box.value,
        image_paths=image_paths,
        seed=int(seed.value),
        num_inference_steps=int(steps.value),
        **config.PRESETS[preset.value],
    )

    # The run is a single await; poll the shared run state to narrate progress.
    # (Route-first order: workflow is set before the screenplay exists.)
    with mo.status.spinner(
        title="Waiting...",
        subtitle="Step 1/3 — routing the task type with Laya...",
    ) as _spinner:
        _task = asyncio.create_task(
            Runner.run(
                director,
                build_run_input(run_state),
                context=run_state,
                max_turns=10,
                run_config=RunConfig(group_id=run_state.group_id),
            )
        )
        while not _task.done():
            if run_state.rewritten_prompt:
                _spinner.update(
                    subtitle=(
                        f"Step 3/3 — the Cameraman renders with MiniMax-H3 ({run_state.workflow}): "
                        "several minutes, plus ~124GB of weights loading on the first run per workflow..."
                    )
                )
            elif run_state.workflow:
                _spinner.update(
                    subtitle=f"Step 2/3 — the Screenwriter writes the {run_state.workflow} screenplay (Kimi)..."
                )
            await asyncio.sleep(1)
        result = _task.result()

    mo.stop(
        not run_state.video_path,
        mo.callout(
            mo.md(
                f"**Generation failed — no video was produced.**\n\n"
                f"The director said: {result.final_output}\n\n"
                f"Check the [Weave trace](https://wandb.ai/{config.WEAVE_PROJECT}/weave/traces) "
                f"for the underlying tool error."
            ),
            kind="danger",
        ),
    )

    _prob = (run_state.workflow_probs or {}).get(run_state.workflow, 0.0)

    mo.vstack(
        [
            mo.md(f"Suggested workflow: `{run_state.workflow}` with probability {_prob:.0%}"),
            # Clips are small (~1-3MB), so embed the bytes directly: marimo inlines
            # them as a data: URL, which needs no server route or service worker
            # (marimo 0.13's /public/ endpoint proved unreliable through the tunnel).
            mo.video(src=Path(run_state.video_path).read_bytes(), controls=True),
            mo.accordion(
                {
                    "Workflow probabilities": mo.md(
                        f"```python\n{run_state.workflow_probs}\n```"
                    ),
                    "Rewritten prompt": mo.md(f"```text\n{run_state.rewritten_prompt}\n```"),
                }
            ),
            mo.md(
                f"[View the full trace in Forge Weave]"
                f"(https://wandb.ai/{config.WEAVE_PROJECT}/weave/traces)"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
