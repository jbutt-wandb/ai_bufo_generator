"""Laya "System 1" router: picks the MiniMax-H3 task type without a generative LLM.

Laya (convaiinnovations/laya) is a non-autoregressive classifier: it scores the
options we define and can never answer outside that set, so routing is safe by
construction.

Design: every candidate class gets its OWN independent, positively-phrased binary
(noul) question, and the class whose question scores highest wins. Earlier designs
asked one contrastive question ("END ... rather than STARTING?") and composed the
answers — probing showed that to be phrasing-brittle (the same meaning scored 0.92
vs 0.30 across paraphrases), misrouting clear start-frame requests to l2va.
Single-proposition questions with no contrast/negation are much more stable.
"""

import weave

LAYA_MODEL_ID = "convaiinnovations/laya"

# Every task type the prompt guide distinguishes. i2va/l2va run on the fl2va
# pipeline partition (first/last keyframe only); the split matters for prompt
# format and keyframe alignment, so we route on the full five.
WORKFLOWS = {
    "t2va": "text only — no pictures; the scene is invented from scratch",
    "i2va": "the uploaded picture is the FIRST frame — the video animates outward from it",
    "l2va": "the uploaded picture is the LAST frame — the video converges onto it",
    "fl2va": "two uploaded pictures anchor the FIRST and LAST frames of the video",
    "ref2va": "the uploaded picture(s) are identity references — the subject is taken out of them and appears in a NEW scene",
}

_router = None


def _get_router():
    """Load Laya once (first call downloads the ~843MB checkpoint from the Hub)."""
    global _router
    if _router is None:
        from laya import Router

        _router = Router()
    return _router


# One question per candidate class, asked only when the user uploaded images (the
# upload count fixes the candidate set in code, so no impossible option is ever
# offered). Each is a single positive proposition — no "rather than", no negation.
# A class may have SEVERAL paraphrased probes; its score is the max over them
# (different user phrasings light up different probes — e.g. "put the toad from
# my photo into a disco" vs "use the image to shape what bufo looks like" are
# both ref2va but trigger different wordings). Keyframe classes are listed
# before ref2va so exact ties fall to the keyframe reading, which still uses
# the pictures sensibly.
_PROBES_1_IMAGE = {
    "i2va": [
        "The user uploaded one picture with this video-generation request. Is the "
        "picture meant to be the video's FIRST frame — the scene starts exactly at "
        "the picture and animates onward from it?",
    ],
    "l2va": [
        "The user uploaded one picture with this video-generation request. Is the "
        "picture meant to be the video's LAST frame — the scene builds toward the "
        "picture and finishes exactly on it?",
        "The user uploaded one picture with this video-generation request. Does the "
        "video END at the picture — the final frame of the video is the picture?",
    ],
    "ref2va": [
        "The user uploaded one picture with this video-generation request. Does the "
        "request take the character or subject OUT of the picture and show it in a "
        "new scene, setting or activity?",
        "The user uploaded one picture with this video-generation request. Is the "
        "picture meant as a reference for the character's identity or appearance — "
        "the video shows what is IN the picture doing something new?",
    ],
}

_PROBES_2_IMAGES = {
    "fl2va": [
        "The user uploaded two pictures with this video-generation request. Are "
        "they meant to be the video's first and last frames — the video starts on "
        "picture 1 and finishes on picture 2?",
    ],
    "ref2va": [
        "The user uploaded two pictures with this video-generation request. Does "
        "the request take the characters or subjects OUT of the pictures and show "
        "them in a new scene, setting or activity?",
        "The user uploaded two pictures with this video-generation request. Are the "
        "pictures meant as references for the characters' identity or appearance — "
        "the video shows what is IN the pictures doing something new?",
    ],
}


@weave.op
def select_workflow(user_prompt: str, n_images: int, model: str = LAYA_MODEL_ID) -> dict:
    """Score the MiniMax-H3 task types for this request and pick one.

    Returns {"model", "scores", "probs", "workflow", "descriptions"}: `scores` is
    Laya's raw independent P(true) per candidate class; `probs` is those scores
    normalized over the candidates (impossible options for the upload count: 0.0).
    """
    scores = {}
    probs = {wf: 0.0 for wf in WORKFLOWS}
    if n_images == 0:
        # Without images there is nothing to condition on — t2va is the only
        # option, so we enforce that in code rather than asking the model.
        scores["t2va"] = 1.0
        probs["t2va"] = 1.0
        workflow = "t2va"
    else:
        probes = _PROBES_1_IMAGE if n_images == 1 else _PROBES_2_IMAGES
        questions = {
            f"{wf}_{i}": {"type": "noul", "instructions": text}
            for wf, texts in probes.items()
            for i, text in enumerate(texts)
        }
        # All probes go in ONE predict call (Laya batches them in a single
        # forward pass); a class's score is the max over its probes.
        answers = _get_router().predict(user_prompt, questions)["answers"]
        scores = {
            wf: max(float(answers[f"{wf}_{i}"]["noul"]) for i in range(len(texts)))
            for wf, texts in probes.items()
        }
        total = sum(scores.values()) or 1.0
        probs.update({wf: s / total for wf, s in scores.items()})
        # Argmax over the raw scores; dict order lists keyframe classes first, so
        # exact ties prefer the keyframe reading.
        workflow = max(scores, key=scores.get)
    return {
        "model": model,
        "scores": scores,
        "probs": probs,
        "workflow": workflow,
        "descriptions": WORKFLOWS,
    }
