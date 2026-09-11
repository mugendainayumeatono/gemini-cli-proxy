import logging
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

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

