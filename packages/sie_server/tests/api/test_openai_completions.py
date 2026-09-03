"""Regression tests for direct ``/v1/completions`` generation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters._generation_base import FinishReason, GenerationAdapter, GenerationChunk
from sie_server.adapters._spec import AdapterSpec
from sie_server.api.openai_completions import router
from sie_server.config.model import AdapterOptions, GenerateTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.registry import ModelRegistry
from sie_server.types.grammar import GrammarSpec
from sie_server.types.inputs import ImageInput

_GEMMA_OPEN = "<" + "|channel" + ">" + "thought\n"
_GEMMA_CLOSE = "<" + "channel|" + ">"


class _FakeCompletionAdapter(GenerationAdapter):
    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self) -> None:
        self.last_call: dict[str, object] | None = None
        self.text_chunks = ["hello", " world"]
        self.finish_reason: FinishReason = "stop"

    def load(self, device: str) -> None:  # pragma: no cover - registry is mocked loaded
        _ = device

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int | None = None,
        grammar: GrammarSpec | None = None,
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        images: list[ImageInput] | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        self.last_call = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_k": top_k,
            "min_new_tokens": min_new_tokens,
            "seed": seed,
        }
        _ = repetition_penalty, grammar, logit_bias, logprobs, top_logprobs, images
        for index, text in enumerate(self.text_chunks):
            yield GenerationChunk(text_delta=text, is_first=index == 0)
        yield GenerationChunk(
            text_delta="",
            done=True,
            finish_reason=self.finish_reason,
            prompt_tokens=3,
            completion_tokens=2,
        )


def _config(model_id: str = "Qwen/Qwen3-4B-Instruct") -> ModelConfig:
    return ModelConfig(
        sie_id=model_id,
        hf_id=model_id,
        tasks=Tasks(generate=GenerateTask(context_length=32768, max_output_tokens=64)),
        profiles={
            "default": ProfileConfig(
                adapter_path="test:FakeCompletionAdapter",
                max_batch_tokens=8192,
                kv_budget_tokens=4096,
                adapter_options=AdapterOptions(
                    loadtime={"reasoning_parser": "gemma4" if model_id.startswith("google/gemma") else "qwen3"},
                    runtime={
                        "default_sampling": {
                            "temperature": 0.25,
                            "top_p": 0.8,
                            "frequency_penalty": 0.75,
                            "top_k": 12,
                            "min_new_tokens": 2,
                            "seed": 23,
                        },
                        "stop_tokens": ["</s>"],
                    },
                ),
            )
        },
    )


@pytest.fixture
def adapter() -> _FakeCompletionAdapter:
    return _FakeCompletionAdapter()


@pytest.fixture
def registry(adapter: _FakeCompletionAdapter) -> MagicMock:
    reg = MagicMock(spec=ModelRegistry)
    reg.has_model.return_value = True
    reg.is_loaded.return_value = True
    reg.is_loading.return_value = False
    reg.is_unloading.return_value = False
    reg.is_failed.return_value = False
    reg.get_failure.return_value = None
    reg.get_config.return_value = _config()
    reg.get.return_value = adapter
    reg.device = "cpu"
    return reg


@pytest.fixture
def client(registry: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.registry = registry
    return TestClient(app)


def test_blocking_completion_uses_direct_adapter_and_openai_shape(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "prompt": ["Continue this"],
            "max_tokens": 8,
            "frequency_penalty": 0.5,
            "seed": -1,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"].startswith("cmpl-")
    assert body["object"] == "text_completion"
    assert body["model"] == "Qwen/Qwen3-4B-Instruct"
    assert body["choices"] == [{"text": "hello world", "index": 0, "finish_reason": "stop"}]
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert body["system_fingerprint"].startswith("fp_")
    assert adapter.last_call == {
        "prompt": "Continue this",
        "max_new_tokens": 8,
        "temperature": 0.25,
        "top_p": 0.8,
        "stop": ["</s>"],
        "frequency_penalty": 0.5,
        "presence_penalty": None,
        "top_k": 12,
        "min_new_tokens": 2,
        "seed": -1,
    }


def test_completion_uses_profile_frequency_penalty_and_seed_defaults(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "prompt": "Continue this",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200, response.text
    assert adapter.last_call is not None
    assert adapter.last_call["frequency_penalty"] == 0.75
    assert adapter.last_call["seed"] == 23


def test_completion_request_frequency_penalty_and_seed_override_profile_defaults(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "prompt": "Continue this",
            "max_tokens": 8,
            "frequency_penalty": -0.5,
            "seed": -1,
        },
    )

    assert response.status_code == 200, response.text
    assert adapter.last_call is not None
    assert adapter.last_call["frequency_penalty"] == -0.5
    assert adapter.last_call["seed"] == -1


def test_streaming_completion_emits_openai_chunks_usage_and_done(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "prompt": "Continue this",
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200, response.text
    payloads = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert payloads[-1] == "[DONE]"
    events = [json.loads(payload) for payload in payloads[:-1]]
    assert [event["choices"][0]["text"] for event in events[:-1]] == ["hello", " world", ""]
    assert events[0]["choices"][0]["finish_reason"] is None
    assert events[2]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["choices"] == []
    assert events[-1]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert len({event["id"] for event in events}) == 1


def test_streaming_completion_surfaces_cancelled_terminal_as_error(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
) -> None:
    adapter.finish_reason = "cancelled"

    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "prompt": "Continue this",
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200, response.text
    payloads = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert payloads[-1] == "[DONE]"
    events = [json.loads(payload) for payload in payloads[:-1]]
    terminal = events[-1]
    assert terminal["choices"] == [{"text": "", "index": 0, "finish_reason": None}]
    assert terminal["error"] == {
        "message": "generation was cancelled before completion",
        "type": "server_error",
        "param": None,
        "code": "generation_cancelled",
    }
    assert all("usage" not in event for event in events)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enable_thinking", [False, True])
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3.5-4B", "google/gemma-4-E2B-it"])
def test_completion_hides_reasoning_for_every_resolved_profile(
    client: TestClient,
    registry: MagicMock,
    adapter: _FakeCompletionAdapter,
    stream: bool,
    enable_thinking: bool,
    model_id: str,
) -> None:
    config = _config(model_id)
    assert config.tasks.generate is not None
    config.tasks.generate.chat_template_kwargs = {"enable_thinking": enable_thinking}
    registry.get_config.return_value = config
    if model_id.startswith("google/gemma"):
        raw = _GEMMA_OPEN + "secret" + _GEMMA_CLOSE + "answer"
        adapter.text_chunks = [raw[:7], raw[7:19], raw[19:]]
    else:
        adapter.text_chunks = ["<thi", "nk>secret</th", "ink>answer"]

    response = client.post(
        "/v1/completions",
        json={
            "model": model_id,
            "prompt": "Continue this",
            "max_tokens": 8,
            "stream": stream,
        },
    )

    assert response.status_code == 200, response.text
    assert "secret" not in response.text
    assert "answer" in response.text


@pytest.mark.parametrize(
    ("body", "param", "code"),
    [
        ({"model": "m", "prompt": "x", "tools": []}, "tools", "unsupported_field"),
        ({"model": "m", "prompt": ["x", "y"]}, "prompt", "unsupported_field"),
        ({"model": "m", "prompt": "x", "n": 2}, "n", "unsupported_field"),
        ({"model": "m", "prompt": "x", "max_tokens": True}, "max_tokens", "invalid_request"),
        ({"model": "m", "prompt": "x", "temperature": 1e100}, "temperature", "invalid_request"),
        ({"model": "m", "prompt": "x", "top_p": 0}, "top_p", "invalid_request"),
        ({"model": "m", "prompt": "x", "stop": ""}, "stop", "invalid_request"),
        ({"model": "m", "prompt": "x", "stop": ["ok", ""]}, "stop", "invalid_request"),
    ],
)
def test_invalid_requests_fail_before_registry_lookup(
    client: TestClient,
    registry: MagicMock,
    body: dict[str, object],
    param: str,
    code: str,
) -> None:
    response = client.post("/v1/completions", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    assert response.json()["error"]["code"] == code
    registry.has_model.assert_not_called()


def test_model_output_cap_rejects_before_load(client: TestClient, registry: MagicMock) -> None:
    registry.is_loaded.return_value = False

    response = client.post(
        "/v1/completions",
        json={"model": "Qwen/Qwen3-4B-Instruct", "prompt": "x", "max_tokens": 65},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_exceeded"
    registry.start_load_async.assert_not_called()


def test_model_state_errors_use_openai_error_envelope(client: TestClient, registry: MagicMock) -> None:
    registry.has_model.return_value = False

    response = client.post(
        "/v1/completions",
        json={"model": "missing/model", "prompt": "x"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "Model 'missing/model' not found",
            "type": "invalid_request_error",
            "param": None,
            "code": "MODEL_NOT_FOUND",
        }
    }
