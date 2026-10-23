"""An agent that keeps calling tools finalizes under its hard request cap instead of raising UsageLimitExceeded."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.usage import UsageLimits

from livegraph.agents.base import FINAL_REQUESTS, wind_down
from livegraph.agents.runner import run_agent


class Out(BaseModel):
    answer: str


@dataclass
class Deps:
    request_cap: int | None = None
    calls: int = 0


def greedy(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Calls the search tool whenever it is offered, otherwise answers."""
    if info.function_tools:
        return ModelResponse(parts=[ToolCallPart("search", {"q": "more"})])
    return ModelResponse(parts=[TextPart('{"answer": "done"}')])


async def greedy_stream(messages: list[ModelMessage], info: AgentInfo):
    if info.function_tools:
        calls: DeltaToolCalls = {0: DeltaToolCall(name="search", json_args='{"q": "more"}')}
        yield calls
    else:
        yield '{"answer": "done"}'


def model() -> FunctionModel:
    return FunctionModel(greedy, stream_function=greedy_stream)


def make_agent() -> Agent[Deps, Out]:
    from pydantic_ai import PromptedOutput

    from pydantic_ai.capabilities import PrepareTools

    agent: Agent[Deps, Out] = Agent(output_type=PromptedOutput(Out), deps_type=Deps, capabilities=[PrepareTools(wind_down)])

    @agent.tool
    async def search(ctx: RunContext[Deps], q: str) -> str:
        ctx.deps.calls += 1
        return "result"

    return agent


@pytest.mark.parametrize("cap", [4, 8, 10])
async def test_greedy_agent_finalizes_under_cap(cap: int) -> None:
    deps = Deps()
    res = await run_agent(make_agent(), "go", model=model(), render=lambda o: o.answer,
                          deps=deps, limits=UsageLimits(request_limit=cap))
    assert res.output.answer == "done"
    assert deps.request_cap == cap
    assert deps.calls == cap - FINAL_REQUESTS  # every request before the wind-down used a tool


async def test_without_a_cap_tools_stay_on() -> None:
    from pydantic_ai.exceptions import UsageLimitExceeded

    agent = make_agent()
    with pytest.raises(UsageLimitExceeded):  # no request_cap on deps -> no wind-down (old behaviour)
        await agent.run("go", model=FunctionModel(greedy), deps=Deps(), usage_limits=UsageLimits(request_limit=3))


async def test_final_requests_tell_the_model_tools_are_gone() -> None:
    from pydantic_ai.capabilities import PrepareTools
    from pydantic_ai.messages import ModelRequest

    from livegraph.agents.base import FINAL_NOTE, final_note

    seen: list[tuple[bool, bool]] = []  # (tools offered, note present) per request

    async def stream(messages: list[ModelMessage], info: AgentInfo):
        req = next(m for m in reversed(messages) if isinstance(m, ModelRequest))
        seen.append((bool(info.function_tools), FINAL_NOTE.strip() in (req.instructions or "")))
        async for part in greedy_stream(messages, info):
            yield part

    from pydantic_ai import PromptedOutput

    agent: Agent[Deps, Out] = Agent(output_type=PromptedOutput(Out), deps_type=Deps, instructions=["Base.", final_note],
                                    capabilities=[PrepareTools(wind_down)])

    @agent.tool
    async def search(ctx: RunContext[Deps], q: str) -> str:
        return "result"

    await run_agent(agent, "go", model=FunctionModel(greedy, stream_function=stream), render=lambda o: o.answer,
                    deps=Deps(), limits=UsageLimits(request_limit=5))
    assert seen == [(True, False), (True, False), (True, False), (False, True)]


def test_validator_repairs_instead_of_retrying_on_the_last_request() -> None:
    from types import SimpleNamespace

    from pydantic_ai import ModelRetry

    from livegraph.agents.validators import _retry_or

    repaired: list[bool] = []

    def ctx(requests: int, cap: int | None) -> SimpleNamespace:
        return SimpleNamespace(retry=0, max_retries=2, usage=SimpleNamespace(requests=requests),
                               deps=SimpleNamespace(request_cap=cap))

    with pytest.raises(ModelRetry):  # requests left: bounce it back to the model
        _retry_or(ctx(8, 10), ["bad id"], lambda: repaired.append(True))
    _retry_or(ctx(10, 10), ["bad id"], lambda: repaired.append(True))  # last allowed request: repair
    assert repaired == [True]
    with pytest.raises(ModelRetry):  # no cap known: old behaviour
        _retry_or(ctx(50, None), ["bad id"], lambda: None)
