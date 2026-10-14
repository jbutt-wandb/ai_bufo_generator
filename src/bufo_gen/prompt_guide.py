"""Retrieval loaders for the Screenwriter's reference documents.

The MiniMax prompt guide and the Bufo character description are deliberately
NOT baked into any system prompt: the Screenwriter agent retrieves them at run
time via its retrieve_reference tool, so the knowledge is external, versioned
in docs/, and visible in the trace exactly as the agent received it.
"""

from pathlib import Path

# The app runs with CWD ~/bufo_gen but the smoke test must not depend on CWD.
_DOCS = Path(__file__).parents[2] / "docs"

# Guide sections (## headings in docs/minimax_prompt_guide.md) by relevance:
_SHARED_SECTIONS = ("Shots and camera motion", "Dialogue and voiceover", "Audio")
_FORMAT_SECTIONS = {
    "t2va": ("Prompt structure",),
    "i2va": ("Prompt structure", "Keyframe alignment", "Example skeleton"),
    "l2va": ("Prompt structure", "Keyframe alignment", "Example skeleton"),
    "fl2va": ("Prompt structure", "Keyframe alignment", "Example skeleton"),
    "ref2va": ("Reference-guided prompts",),
}


def _sections() -> dict[str, str]:
    """Split the guide doc into {heading: body} by its ## headings."""
    text = (_DOCS / "minimax_prompt_guide.md").read_text()
    parts: dict[str, str] = {}
    heading = "_intro"
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            parts[heading] = "\n".join(body).strip()
            heading, body = line[3:].strip(), []
        else:
            body.append(line)
    parts[heading] = "\n".join(body).strip()
    return parts


def load_guide(workflow: str) -> str:
    """The MiniMax-H3 prompt-writing guide, sliced to one workflow's format.

    Returns the format section(s) for the given task type plus the shared
    craft sections (shots/camera, dialogue, audio) that apply to every prompt.
    """
    if workflow not in _FORMAT_SECTIONS:
        raise ValueError(f"Unknown workflow {workflow!r}; expected one of {sorted(_FORMAT_SECTIONS)}.")
    parts = _sections()

    def find(prefix: str) -> str:
        for heading, body in parts.items():
            if heading.startswith(prefix):
                return f"## {heading}\n{body}"
        raise ValueError(f"Guide section starting with {prefix!r} not found in minimax_prompt_guide.md.")

    wanted = _FORMAT_SECTIONS[workflow] + _SHARED_SECTIONS
    header = f"# MiniMax-H3 prompt-writing guide — sections for the `{workflow}` task\n"
    return header + "\n\n".join(find(p) for p in wanted)


def load_bufo_description() -> str:
    """The researched visual description of Bufo (docs/bufo_description.md)."""
    return (_DOCS / "bufo_description.md").read_text().strip()
