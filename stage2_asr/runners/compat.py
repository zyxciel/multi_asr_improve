from __future__ import annotations

import inspect


def accepted_kwargs(fn, kwargs: dict) -> dict:
    """Keep only kwargs the callable declares (unless it takes **kwargs)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def transcribe_unit_compat(runner, unit, turns, audio_path: str, **kwargs):
    """Call transcribe_unit without retrying real TypeErrors as signature misses.

    If the runner does not accept ``selected_models``, filter returned hyps.
    """
    fn = runner.transcribe_unit
    accepted = accepted_kwargs(fn, kwargs)
    hyps = fn(unit, turns, audio_path, **accepted)
    selected = kwargs.get("selected_models")
    if selected is not None and "selected_models" not in accepted:
        hyps = [h for h in hyps if h.model in selected]
    return hyps
