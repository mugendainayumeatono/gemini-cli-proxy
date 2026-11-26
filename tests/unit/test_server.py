import json
import logging
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from gemini_cli_proxy.acp_client import ACPError
from gemini_cli_proxy.models import (
    CompletionResult,
    CompletionStreamChunk,
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from gemini_cli_proxy.server import app
from gemini_cli_proxy.config import config


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_models_endpoint_auth(client, caplog):
    # Missing API key
    with caplog.at_level(logging.WARNING, logger="gemini_cli_proxy"):
        response = client.get("/v1/models")
        assert response.status_code == 401
        assert "Authentication failed: API key missing" in caplog.text

    # With API key
    response = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert "data" in response.json()


def test_chat_completions_cli_error_logs_and_returns_502(client, caplog):
    with patch(
        "gemini_cli_proxy.gemini_client.gemini_client.chat_completion",
        AsyncMock(side_effect=RuntimeError("agy execution failed (exit code 1 (failure)): some error"))
    ):
        with caplog.at_level(logging.INFO, logger="gemini_cli_proxy"):
            payload = {
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "Hello"}]
            }
            response = client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer test-key"}
            )
            assert response.status_code == 502
            data = response.json()
            assert "error" in data
            assert "exit code 1 (failure)" in data["error"]["message"]
            assert "Chat completion failed: Gemini CLI execution error" in caplog.text


def test_lifespan_fetch_models_success():
    mock_stdout = b"gemini-3.8-flash-high\tGemini 3.8 Flash (High)\nclaude-sonnet-4-6\tClaude Sonnet 4.6\n"
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (mock_stdout, b"Fetching available models...\n")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
        with TestClient(app):
            assert "gemini-3.8-flash-high" in config.supported_models
            assert "claude-sonnet-4-6" in config.supported_models


def test_lifespan_fetch_models_failure_fallback(caplog):
    original_models = list(config.supported_models)
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"", b"Command error")
    mock_process.returncode = 1

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
        with caplog.at_level(logging.WARNING, logger="gemini_cli_proxy"):
            with TestClient(app):
                assert len(config.supported_models) > 0
                assert "Failed to fetch models" in caplog.text


def test_lifespan_acp_mode_starts_and_stops_client(monkeypatch):
    monkeypatch.setattr(config, "acp_mode", True)
    mock_stdout = b"gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n"
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (mock_stdout, b"")
    mock_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)), \
         patch("gemini_cli_proxy.server.acp_client.start", AsyncMock()) as mock_start, \
         patch("gemini_cli_proxy.server.acp_client.stop", AsyncMock()) as mock_stop:
        with TestClient(app):
            mock_start.assert_awaited_once()
        mock_stop.assert_awaited_once()


def test_chat_completions_acp_error_returns_502(client, caplog):
    with patch(
        "gemini_cli_proxy.gemini_client.gemini_client.chat_completion",
        AsyncMock(side_effect=ACPError("Permission denied by policy", code="acp_permission_required"))
    ):
        with caplog.at_level(logging.ERROR, logger="gemini_cli_proxy"):
            payload = {
                "model": "gemini-3.8-flash-low",
                "messages": [{"role": "user", "content": "Hello"}]
            }
            response = client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer test-key"}
            )
            assert response.status_code == 502
            data = response.json()
            assert data["error"]["type"] == "acp_error"
            assert data["error"]["code"] == "acp_permission_required"
            assert "Permission denied by policy" in data["error"]["message"]
            assert "acp_permission_required" in caplog.text


def test_chat_completions_stream_acp_error_emits_sse_error(client, caplog):
    async def failing_stream(*args, **kwargs):
        raise ACPError("Permission denied in stream", code="acp_permission_required")
        yield  # pragma: no cover

    with patch(
        "gemini_cli_proxy.gemini_client.gemini_client.chat_completion_stream",
        side_effect=failing_stream
    ):
        with caplog.at_level(logging.ERROR, logger="gemini_cli_proxy"):
            payload = {
                "model": "gemini-3.8-flash-low",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
            response = client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer test-key"}
            )
            assert response.status_code == 200
            body = response.text
            assert '"code": "acp_permission_required"' in body
            assert '"type": "acp_error"' in body
            assert "data: [DONE]" in body
            assert "acp_permission_required" in caplog.text


def _parse_sse_events(body: str):
    """Parse SSE data lines into JSON dicts and raw markers."""
    events = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))
    return events


def test_chat_completions_with_reasoning_and_usage(client):
    usage = CompletionUsage(
        prompt_tokens=100,
        completion_tokens=45,
        total_tokens=145,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=20),
        prompt_tokens_details=PromptTokensDetails(cached_tokens=60),
    )
    mock_result = CompletionResult(
        content="The answer is 42.",
        reasoning_content="Let me think step by step.",
        usage=usage,
    )

    with patch(
        "gemini_cli_proxy.gemini_client.gemini_client.chat_completion",
        AsyncMock(return_value=mock_result),
    ):
        payload = {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "What is the answer?"}],
        }
        response = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
        data = response.json()
        message = data["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert message["content"] == "The answer is 42."
        assert message["reasoning_content"] == "Let me think step by step."
        assert data["usage"] == {
            "prompt_tokens": 100,
            "completion_tokens": 45,
            "total_tokens": 145,
            "completion_tokens_details": {"reasoning_tokens": 20},
            "prompt_tokens_details": {"cached_tokens": 60},
        }


def test_chat_completions_omits_none_reasoning_and_usage(client):
    mock_result = CompletionResult(
        content="Plain reply.",
        reasoning_content=None,
        usage=None,
    )

    with patch(
        "gemini_cli_proxy.gemini_client.gemini_client.chat_completion",
        AsyncMock(return_value=mock_result),
    ):
        payload = {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        response = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
        data = response.json()
        message = data["choices"][0]["message"]
        assert message["content"] == "Plain reply."
        assert "reasoning_content" not in message
        assert "usage" not in data


def test_chat_completions_multiturn_with_assistant_reasoning_content(client):
    mock_result = CompletionResult(content="Follow-up answer.")

    with patch(
        "gemini_cli_proxy.gemini_client.gemini_client.chat_completion",
        AsyncMock(return_value=mock_result),
    ) as mock_chat:
        payload = {
            "model": "gemini-2.5-flash",
            "messages": [
                {"role": "user", "content": "First question"},
                {
                    "role": "assistant",
                    "content": "First answer",
                    "reasoning_content": "Previous reasoning trace",
                },
                {"role": "user", "content": "Second question"},
            ],
        }
        response = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Follow-up answer."
        passed_messages = mock_chat.call_args.kwargs["messages"]
        assert passed_messages[1].reasoning_content == "Previous reasoning trace"


@pytest.mark.parametrize(
    "stream_options,expected_mode",
    [
        (None, "attached_to_stop"),
        ({"include_usage": True}, "separate_chunk"),
        ({"include_usage": False}, "omitted"),
    ],
)
def test_chat_completions_stream_reasoning_and_usage(client, stream_options, expected_mode):
    usage = CompletionUsage(
        prompt_tokens=30,
        completion_tokens=15,
        total_tokens=45,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=5),
        prompt_tokens_details=PromptTokensDetails(cached_tokens=10),
    )

    async def fake_stream(*args, **kwargs):
        yield CompletionStreamChunk(reasoning_content="Thinking...")
        yield CompletionStreamChunk(content="Hello ")
        yield CompletionStreamChunk(content="world")
        yield CompletionStreamChunk(usage=usage)

    with patch(
        "gemini_cli_proxy.gemini_client.gemini_client.chat_completion_stream",
        side_effect=fake_stream,
    ):
        payload = {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        if stream_options is not None:
            payload["stream_options"] = stream_options

        response = client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
        events = _parse_sse_events(response.text)
        assert events[-1] == "[DONE]"

        chunks = events[:-1]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        assert chunks[0]["choices"][0]["finish_reason"] is None
        assert "usage" not in chunks[0]

        assert chunks[1]["choices"][0]["delta"] == {"reasoning_content": "Thinking..."}
        assert chunks[1]["choices"][0]["finish_reason"] is None
        assert "usage" not in chunks[1]

        assert chunks[2]["choices"][0]["delta"] == {"content": "Hello "}
        assert chunks[2]["choices"][0]["finish_reason"] is None
        assert "usage" not in chunks[2]

        assert chunks[3]["choices"][0]["delta"] == {"content": "world"}
        assert chunks[3]["choices"][0]["finish_reason"] is None
        assert "usage" not in chunks[3]

        expected_usage_dict = {
            "prompt_tokens": 30,
            "completion_tokens": 15,
            "total_tokens": 45,
            "completion_tokens_details": {"reasoning_tokens": 5},
            "prompt_tokens_details": {"cached_tokens": 10},
        }

        if expected_mode == "attached_to_stop":
            assert len(chunks) == 5
            stop_chunk = chunks[4]
            assert stop_chunk["choices"][0]["finish_reason"] == "stop"
            assert stop_chunk["usage"] == expected_usage_dict
        elif expected_mode == "separate_chunk":
            assert len(chunks) == 6
            stop_chunk = chunks[4]
            usage_chunk = chunks[5]
            assert stop_chunk["choices"][0]["finish_reason"] == "stop"
            assert "usage" not in stop_chunk
            assert usage_chunk["choices"] == []
            assert usage_chunk["usage"] == expected_usage_dict
        elif expected_mode == "omitted":
            assert len(chunks) == 5
            stop_chunk = chunks[4]
            assert stop_chunk["choices"][0]["finish_reason"] == "stop"
            assert "usage" not in stop_chunk


