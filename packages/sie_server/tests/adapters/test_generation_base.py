from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sie_server.adapters._generation_base import (
    GenerationChunk,
    ToolCallDelta,
    collect_generation,
    reasoning_starts_in_prompt,
    suppress_thinking_blocks,
)

_GEMMA_OPEN = "<" + "|channel" + ">" + "thought\n"
_GEMMA_CLOSE = "<" + "channel|" + ">"


async def _chunks(*chunks: GenerationChunk) -> AsyncIterator[GenerationChunk]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_suppress_thinking_blocks_handles_delimiters_split_across_chunks() -> None:
    raw = "<think>private chain of thought</think>Visible answer"
    source = _chunks(
        *(GenerationChunk(text_delta=char, is_first=index == 0) for index, char in enumerate(raw)),
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=11,
        ),
    )

    normalized = [chunk async for chunk in suppress_thinking_blocks(source)]

    assert "".join(chunk.text_delta for chunk in normalized) == "Visible answer"
    assert sum(chunk.is_first for chunk in normalized) == 1
    first_visible = next(chunk for chunk in normalized if chunk.text_delta)
    assert first_visible.is_first is True
    terminal = normalized[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "stop"
    assert terminal.prompt_tokens == 3
    assert terminal.completion_tokens == 11


@pytest.mark.asyncio
async def test_suppress_gemma_thinking_handles_boundaries_split_across_chunks() -> None:
    raw = _GEMMA_OPEN + "private chain of thought" + _GEMMA_CLOSE + "Visible answer"
    source = _chunks(
        *(GenerationChunk(text_delta=char, is_first=index == 0) for index, char in enumerate(raw)),
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=13,
        ),
    )

    normalized = [chunk async for chunk in suppress_thinking_blocks(source, reasoning_format="gemma4")]

    assert "".join(chunk.text_delta for chunk in normalized) == "Visible answer"
    assert sum(chunk.is_first for chunk in normalized) == 1
    terminal = normalized[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "stop"
    assert terminal.prompt_tokens == 5
    assert terminal.completion_tokens == 13


@pytest.mark.asyncio
async def test_suppress_gemma_thinking_can_start_inside_prompt_seeded_reasoning() -> None:
    prompt = "prefix" + _GEMMA_OPEN.removesuffix("thought\n")
    source = _chunks(
        GenerationChunk(text_delta="thought\nprivate reasoning" + _GEMMA_CLOSE[:5], is_first=True),
        GenerationChunk(text_delta=_GEMMA_CLOSE[5:] + "Visible answer"),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    )

    normalized = [
        chunk
        async for chunk in suppress_thinking_blocks(
            source,
            start_inside=reasoning_starts_in_prompt(prompt, "gemma4"),
            reasoning_format="gemma4",
        )
    ]

    assert "".join(chunk.text_delta for chunk in normalized) == "Visible answer"


@pytest.mark.asyncio
async def test_suppress_thinking_blocks_can_start_inside_prompt_seeded_reasoning() -> None:
    source = _chunks(
        GenerationChunk(text_delta="private reasoning</thi", is_first=True),
        GenerationChunk(text_delta="nk>Visible answer"),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    )

    normalized = [chunk async for chunk in suppress_thinking_blocks(source, start_inside=True)]

    assert "".join(chunk.text_delta for chunk in normalized) == "Visible answer"


@pytest.mark.asyncio
async def test_suppress_thinking_blocks_preserves_normal_text_and_logprobs() -> None:
    logprobs = (
        {
            "token": "hello",
            "logprob": -0.1,
            "bytes": list(b"hello"),
            "top_logprobs": [],
        },
    )
    ordinary = GenerationChunk(text_delta="hello", is_first=True, logprobs=logprobs)
    terminal = GenerationChunk(text_delta="", done=True, finish_reason="stop")

    normalized = [chunk async for chunk in suppress_thinking_blocks(_chunks(ordinary, terminal))]

    assert normalized == [ordinary, terminal]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["answer <", "answer </", "answer <thi"])
async def test_suppress_thinking_blocks_flushes_partial_marker_prefix_at_end(text: str) -> None:
    source = _chunks(
        GenerationChunk(text_delta=text, is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    )

    normalized = [chunk async for chunk in suppress_thinking_blocks(source)]

    assert "".join(chunk.text_delta for chunk in normalized) == text
    assert normalized[-1].done is True


@pytest.mark.asyncio
async def test_suppress_thinking_blocks_hides_truncated_reasoning_and_its_logprobs() -> None:
    reasoning_logprobs = (
        {
            "token": "private",
            "logprob": -0.1,
            "bytes": list(b"private"),
            "top_logprobs": [],
        },
    )
    source = _chunks(
        GenerationChunk(text_delta="<think>private", is_first=True, logprobs=reasoning_logprobs),
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="length",
            completion_tokens=2,
            logprobs=reasoning_logprobs,
        ),
    )

    normalized = [chunk async for chunk in suppress_thinking_blocks(source)]

    assert len(normalized) == 1
    assert normalized[0].done is True
    assert normalized[0].finish_reason == "length"
    assert normalized[0].completion_tokens == 2
    assert normalized[0].text_delta == ""
    assert normalized[0].logprobs is None


@pytest.mark.asyncio
async def test_suppress_thinking_blocks_normalizes_non_streaming_candidates() -> None:
    candidate_logprobs = [{"token": "<think>", "logprob": -0.2}]
    terminal = GenerationChunk(
        text_delta="",
        done=True,
        finish_reason="stop",
        candidates=(
            {
                "text": "<think>secret</think>answer one",
                "finish_reason": "stop",
                "logprobs": candidate_logprobs,
            },
            {
                "text": "answer two",
                "finish_reason": "stop",
                "logprobs": None,
            },
            {
                "text": "answer <thi",
                "finish_reason": "stop",
                "logprobs": [{"token": "<thi", "logprob": -0.3}],
            },
        ),
    )

    [normalized] = [chunk async for chunk in suppress_thinking_blocks(_chunks(terminal))]

    assert normalized.candidates is not None
    assert normalized.candidates[0]["text"] == "answer one"
    assert normalized.candidates[0]["logprobs"] is None
    assert normalized.candidates[1]["text"] == "answer two"
    assert normalized.candidates[2]["text"] == "answer <thi"
    assert normalized.candidates[2]["logprobs"] == [{"token": "<thi", "logprob": -0.3}]


@pytest.mark.asyncio
async def test_suppress_thinking_blocks_normalizes_prompt_seeded_reasoning_candidate() -> None:
    terminal = GenerationChunk(
        text_delta="",
        done=True,
        finish_reason="stop",
        candidates=(
            {
                "text": "private reasoning</think>Visible answer",
                "finish_reason": "stop",
                "logprobs": [{"token": "private", "logprob": -0.2}],
            },
        ),
    )

    [normalized] = [chunk async for chunk in suppress_thinking_blocks(_chunks(terminal), start_inside=True)]

    assert normalized.candidates is not None
    assert normalized.candidates[0]["text"] == "Visible answer"
    assert normalized.candidates[0]["logprobs"] is None


@pytest.mark.asyncio
async def test_suppress_gemma_thinking_normalizes_blocking_candidate() -> None:
    terminal = GenerationChunk(
        text_delta="",
        done=True,
        finish_reason="stop",
        candidates=(
            {
                "text": _GEMMA_OPEN + "private reasoning" + _GEMMA_CLOSE + "Visible answer",
                "finish_reason": "stop",
                "logprobs": [{"token": "private", "logprob": -0.2}],
            },
        ),
    )

    [normalized] = [chunk async for chunk in suppress_thinking_blocks(_chunks(terminal), reasoning_format="gemma4")]

    assert normalized.candidates is not None
    assert normalized.candidates[0]["text"] == "Visible answer"
    assert normalized.candidates[0]["logprobs"] is None


def test_reasoning_starts_in_prompt_uses_family_specific_unmatched_boundary() -> None:
    assert reasoning_starts_in_prompt("prefix" + _GEMMA_OPEN, "gemma4") is True
    assert reasoning_starts_in_prompt("prefix" + _GEMMA_OPEN.removesuffix("thought\n"), "gemma4") is True
    assert reasoning_starts_in_prompt("prefix" + _GEMMA_OPEN.removesuffix("thought\n") + "final\n", "gemma4") is False
    assert reasoning_starts_in_prompt("prefix" + _GEMMA_OPEN + "done" + _GEMMA_CLOSE, "gemma4") is False
    assert reasoning_starts_in_prompt("ordinary prompt", "gemma4") is False


@pytest.mark.asyncio
async def test_suppress_thinking_blocks_closes_upstream_on_client_cancel() -> None:
    closed = False

    async def source() -> AsyncIterator[GenerationChunk]:
        nonlocal closed
        try:
            yield GenerationChunk(text_delta="visible", is_first=True)
            yield GenerationChunk(text_delta="still running")
        finally:
            closed = True

    normalized = suppress_thinking_blocks(source())
    first = await normalized.__anext__()
    assert first.text_delta == "visible"
    await normalized.aclose()

    assert closed is True


@pytest.mark.asyncio
async def test_reasoning_consuming_the_whole_budget_stamps_empty_output_error() -> None:
    """#3104/#3136: budget exhausted inside a think block -> only whitespace is
    visible and the terminal chunk must carry the typed empty-output error.
    """
    source = _chunks(
        GenerationChunk(text_delta="\n\n", is_first=True),
        GenerationChunk(text_delta="<think>"),
        GenerationChunk(text_delta="private reasoning that never closes"),
        GenerationChunk(text_delta="", done=True, finish_reason="length", completion_tokens=32),
    )

    normalized = [chunk async for chunk in suppress_thinking_blocks(source)]

    assert "".join(chunk.text_delta for chunk in normalized) == "\n\n"
    terminal = normalized[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "length"
    assert terminal.error_code == "empty_model_output"
    assert terminal.error_message is not None
    assert "reasoning" in terminal.error_message


@pytest.mark.asyncio
async def test_whitespace_only_output_without_reasoning_stamps_empty_output_error() -> None:
    source = _chunks(
        GenerationChunk(text_delta="\n\n", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", completion_tokens=2),
    )

    normalized = [chunk async for chunk in suppress_thinking_blocks(source)]

    terminal = normalized[-1]
    assert terminal.error_code == "empty_model_output"
    assert terminal.error_message is not None
    assert "reasoning" not in terminal.error_message


@pytest.mark.asyncio
async def test_visible_answer_is_not_stamped() -> None:
    source = _chunks(
        GenerationChunk(text_delta="<think>hidden</think>", is_first=True),
        GenerationChunk(text_delta="Visible answer"),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    )

    normalized = [chunk async for chunk in suppress_thinking_blocks(source)]

    assert normalized[-1].error_code is None
    assert "".join(chunk.text_delta for chunk in normalized) == "Visible answer"


@pytest.mark.asyncio
async def test_tool_call_only_stream_is_not_stamped() -> None:
    """A tool-call response legitimately has no visible prose."""
    source = _chunks(
        GenerationChunk(
            text_delta="",
            tool_call_delta=ToolCallDelta(index=0, id="call_1", function_name="lookup"),
        ),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    )

    normalized = [chunk async for chunk in suppress_thinking_blocks(source)]

    assert normalized[-1].error_code is None


@pytest.mark.asyncio
async def test_cancelled_stream_is_not_stamped() -> None:
    source = _chunks(
        GenerationChunk(text_delta="<think>partial", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="cancelled"),
    )

    normalized = [chunk async for chunk in suppress_thinking_blocks(source)]

    assert normalized[-1].error_code is None


@pytest.mark.asyncio
async def test_candidates_with_visible_text_are_not_stamped() -> None:
    """Non-streaming n>1: visible candidate text counts as usable output."""
    terminal = GenerationChunk(
        text_delta="",
        done=True,
        finish_reason="stop",
        candidates=(
            {"index": 0, "text": "<think>hidden</think>Answer A", "finish_reason": "stop"},
            {"index": 1, "text": "Answer B", "finish_reason": "stop"},
        ),
    )

    [normalized] = [chunk async for chunk in suppress_thinking_blocks(_chunks(terminal))]

    assert normalized.error_code is None


@pytest.mark.asyncio
async def test_candidates_all_reasoning_are_stamped() -> None:
    """Non-streaming n>1 where every candidate was reasoning-only."""
    terminal = GenerationChunk(
        text_delta="",
        done=True,
        finish_reason="length",
        candidates=(
            {"index": 0, "text": "<think>truncated reasoning", "finish_reason": "length"},
            {"index": 1, "text": "<think>more truncated reasoning", "finish_reason": "length"},
        ),
    )

    [normalized] = [chunk async for chunk in suppress_thinking_blocks(_chunks(terminal))]

    assert normalized.error_code == "empty_model_output"
    assert normalized.error_message is not None
    assert "reasoning" in normalized.error_message


@pytest.mark.asyncio
async def test_collect_generation_carries_terminal_error_fields() -> None:
    """#3104/#3136: buffered aggregation must not drop the terminal chunk's
    typed error — ``empty_model_output`` keeps a ``stop``/``length``
    finish_reason, so the fields are the only failure signal.
    """
    source = _chunks(
        GenerationChunk(text_delta="\n\n", is_first=True),
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="length",
            completion_tokens=2,
            error_code="empty_model_output",
            error_message="model produced no visible output text",
        ),
    )

    result = await collect_generation(source)

    assert result.finish_reason == "length"
    assert result.error_code == "empty_model_output"
    assert result.error_message == "model produced no visible output text"


@pytest.mark.asyncio
async def test_collect_generation_defaults_error_fields_to_none_on_success() -> None:
    source = _chunks(
        GenerationChunk(text_delta="Visible answer", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=2),
    )

    result = await collect_generation(source)

    assert result.text == "Visible answer"
    assert result.error_code is None
    assert result.error_message is None
