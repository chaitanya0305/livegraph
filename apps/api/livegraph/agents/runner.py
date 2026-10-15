"""``run_agent``: drive a Pydantic AI agent with ``agent.iter()`` and stream its final output.

``run_stream`` stops at the first output and skips later tool calls, so tool-using agents run through
``iter()`` instead. For every model request the response stream is validated as a partial output, rendered
with the node's deterministic ``render`` and the not-yet-sent suffix goes out as a ``node.delta`` (renderers
are append-only by construction, so the live buffer matches the final ``output_text``). Tool calls produce
no deltas: their ``tool.call``/``tool.result`` events come from the guarded tools themselves.

If a response streamed output text but the run continues anyway (it also called tools, or an output
validator raised ``ModelRetry``), the next request starts with ``on_reset("")`` (``node.delta {reset:true}``)
so the live buffer never shows superseded output. Whenever the final rendering is not an extension of what
was streamed, the buffer is reset to the final text.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from livegraph.agents.base import AgentResult, DeltaFn


def _noop(_: str) -> None:
    return None


async def run_agent(
    agent: Agent[Any, Any],
    prompt: str,
    *,
    model: Model,
    render: Callable[[Any], str],
    on_delta: DeltaFn = _noop,
    on_reset: DeltaFn = _noop,
    deps: Any = None,
    limits: UsageLimits | None = None,
    timeout: float | None = None,
    usage: RunUsage | None = None,
    debounce: float | None = 0.02,  # partial-validation cadence; the bus still batches deltas at ~50ms
) -> AgentResult:
    sent = ""

    async def _run() -> tuple[Any, RunUsage]:
        nonlocal sent
        async with agent.iter(prompt, model=model, deps=deps, usage_limits=limits, usage=usage) as run:
            async for node in run:
                if not Agent.is_model_request_node(node):
                    continue
                if sent:  # the previous response's streamed output did not end the run
                    on_reset("")
                    sent = ""
                async with node.stream(run.ctx) as stream:
                    async for response in stream.stream_response(debounce_by=debounce):
                        try:
                            partial = await stream.validate_response_output(response, allow_partial=True)
                            text = render(partial)
                        except Exception:  # no output yet / incomplete partial / validator retry
                            continue
                        if len(text) > len(sent):
                            on_delta(text[len(sent) :])  # append-only renderers: the new tail
                            sent = text
            assert run.result is not None
            return run.result.output, run.usage

    if timeout:
        async with asyncio.timeout(timeout):
            output, run_usage = await _run()
    else:
        output, run_usage = await _run()

    final_text = render(output)
    if final_text.startswith(sent):
        if len(final_text) > len(sent):
            on_delta(final_text[len(sent) :])
    else:
        on_reset(final_text)
    return AgentResult(
        output=output,
        output_text=final_text,
        model_name=getattr(model, "model_name", None),
        tokens_in=getattr(run_usage, "input_tokens", None),
        tokens_out=getattr(run_usage, "output_tokens", None),
    )
