"""The MiniMax-H3 prompt-writing guide, embedded as text for the director agent.

Condensed from the official guide on the MiniMaxAI/MiniMax-H3 model card; the
human-readable version lives in docs/minimax_prompt_guide.md.
"""

PROMPT_GUIDE = """
# How to write MiniMax-H3 prompts

The final prompt is ONE string of labeled sections. Which sections depend on the
workflow:

## t2va and the keyframe tasks (i2va, l2va, fl2va) — three sections, in this exact order
integrated_multimodal_description: <visuals, actions, dialogue and diegetic audio,
  chronological, organized into [Shot N] blocks>
overall_soundscape: <ambient + physical action sounds, 1-4 sentences>
non_diegetic_music: <background score description, 1-3 sentences>

## ref2va — six sections, in this exact order
subject_definitions: <define reusable labels (see Labels below)>
summary: [<task types>] <one-paragraph gist of the target video>
retention_analysis: <per label: how faithfully it is kept>
detailed_description: <the [Shot N] narrative, same rules as
  integrated_multimodal_description>
overall_soundscape: <as above>
non_diegetic_music: <as above>

## Keyframe alignment (i2va, l2va, fl2va)
State explicitly where each picture lands on the timeline, at the top of the
description.
- fl2va (both keyframes): e.g. "Picture 1 aligns with the 0.00-second mark;
  Picture 2 aligns with the 5.17-second mark." The action must BEGIN in the
  position established by Picture 1 and SETTLE INTO the pose/composition of
  Picture 2.
- i2va (start image only): open with "At 0.00 seconds into the target video,
  Picture 1 is fully referenced." and begin the scene exactly as that picture
  establishes it.
- l2va (end image only): anchor the single picture at the video's final
  timestamp (the stated duration) and describe how the scene CONVERGES onto
  its pose/composition.

## Shots and camera motion
- First shot has no timestamp; later shots open "At 00:XX.XXX, the camera cuts
  to...". Prefer hard cuts unless asked otherwise.
- Camera motion always has three dimensions woven into natural prose:
  type (push in, pull out, pan, tilt, truck, zoom...), amplitude ("with
  small/large amplitude") and speed ("at slow/fast speed").
  e.g. "the camera tilts up with large amplitude at slow speed".

## Dialogue and voiceover
- Stable speaker IDs (S1), (S2); dialogue in <d>[Language] spoken text</d> blocks.
- Voiceover must say "off-screen voiceover" and state that lips stay closed.

## Audio
The model generates the soundtrack from the prompt: name concrete diegetic sounds
inline with the action, keep ambience in overall_soundscape and score in
non_diegetic_music.

## ref2va labels
- <Picture N> = the N-th reference image in upload order.
- <Subject N> = a reusable element abstracted from the references ("appearance
  from <Picture 1>"); the normal way to ground a character's identity.
- summary task types (combine with " + "): keyframe completion, reference
  generation, video editing, video continuation, audio reuse, audio reference.
- retention_analysis markers (visual): fully_preserved, partially_preserved,
  attribute_transfer, weak_reference. Format:
  "<Subject 1> (appears in [Shot 1]): fully_preserved - <what is kept>."
- Reference labels appear at first mention in detailed_description and wherever
  their role applies. A speaking referenced subject: "<Subject 2> (S1) says,
  <d>[English] ...</d>".
"""
