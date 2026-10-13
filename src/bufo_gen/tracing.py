"""Weave tracing setup, shared by the app and the smoke test.

weave.init() autopatches the OpenAI Agents SDK: it registers an OTel processor
that feeds the Weave Agents view. Two things must NOT happen for that to work:
- `set_tracing_disabled(True)` anywhere (it no-ops ALL span creation, so no
  processor — Weave's included — ever sees the agent run), and
- `set_trace_processors(...)` after weave.init() (it would wipe Weave's processor).

We clear the processor list BEFORE weave.init() to drop the SDK's default OpenAI
backend exporter (we have no OPENAI_API_KEY, so it would only log warnings).
"""

import weave
from agents import set_trace_processors

from . import config

_initialized = False


def init_tracing() -> None:
    global _initialized
    if _initialized:
        return
    set_trace_processors([])  # drop the OpenAI backend exporter
    weave.init(config.WEAVE_PROJECT)  # adds Weave's Agents-view processor on top
    _initialized = True
