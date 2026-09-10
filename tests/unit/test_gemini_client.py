import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from gemini_cli_proxy.gemini_client import GeminiClient, format_exit_status
from gemini_cli_proxy.models import ChatMessage


def test_format_exit_status():
    assert format_exit_status(0) == "exit code 0 (success)"
    assert format_exit_status(1) == "exit code 1 (failure)"
    assert format_exit_status(2) == "exit code 2 (failure)"
    assert "signal 15" in format_exit_status(-15)
    assert format_exit_status(None) == "still running / unknown"


def test_simplify_error_message():
    client = GeminiClient()
    
    # Test rate limit detection
    rate_limit_err = '{"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota)."}}'
    assert "rate limit exceeded" in client._simplify_error_message(rate_limit_err)
    
    # Test auth detection
    auth_err = "Error: Unauthenticated. OAuth credentials missing."
    assert "authentication failed" in client._simplify_error_message(auth_err)
    
    # Test model error detection
    model_err = "error: invalid model selection: not recognized as a known model"
    assert "model error" in client._simplify_error_message(model_err)
    
    # Test unknown error
    assert client._simplify_error_message("some random fatal error") is None
    assert client._simplify_error_message("") is None


@pytest.mark.asyncio
async def test_execute_gemini_command_success(caplog):
    client = GeminiClient()
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"Hello world from agy", b""))

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
        with caplog.at_level(logging.INFO, logger="gemini_cli_proxy"):
            messages = [ChatMessage(role="user", content="Hi")]
            response = await client.chat_completion(messages=messages, model="gemini-2.5-flash")
            
            assert response == "Hello world from agy"
            assert "Invoking agy command (sync mode)" in caplog.text
            assert "PID: 12345" in caplog.text
            assert "completed successfully with exit code 0 (success)" in caplog.text


@pytest.mark.asyncio
async def test_execute_gemini_command_exit_code_0_with_stderr(caplog):
    client = GeminiClient()
    mock_process = MagicMock()
    mock_process.pid = 12346
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"Hello world", b"Warning: Deprecated model version"))

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
        with caplog.at_level(logging.INFO, logger="gemini_cli_proxy"):
            messages = [ChatMessage(role="user", content="Hi")]
            response = await client.chat_completion(messages=messages, model="gemini-2.5-flash")
            
            assert response == "Hello world"
            assert "completed successfully with exit code 0 (success)" in caplog.text
            assert "exited with code 0 but produced stderr output" in caplog.text
            assert "Warning: Deprecated model version" in caplog.text


@pytest.mark.asyncio
async def test_execute_gemini_command_failure_records_exit_code_and_stderr(caplog):
    client = GeminiClient()
    mock_process = MagicMock()
    mock_process.pid = 12347
    mock_process.returncode = 2
    mock_process.communicate = AsyncMock(return_value=(b"", b"flags provided but not defined: -invalid-flag"))

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
        with caplog.at_level(logging.INFO, logger="gemini_cli_proxy"):
            messages = [ChatMessage(role="user", content="Hi")]
            with pytest.raises(RuntimeError) as exc_info:
                await client.chat_completion(messages=messages, model="gemini-2.5-flash")
            
            assert "exit code 2 (failure)" in str(exc_info.value)
            assert "command failed with exit code 2 (failure)" in caplog.text
            assert "flags provided but not defined: -invalid-flag" in caplog.text


@pytest.mark.asyncio
async def test_execute_gemini_command_failure_with_simplified_message(caplog):
    client = GeminiClient()
    mock_process = MagicMock()
    mock_process.pid = 12348
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b"Error: Resource exhausted (code: 429)"))

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
        with caplog.at_level(logging.INFO, logger="gemini_cli_proxy"):
            messages = [ChatMessage(role="user", content="Hi")]
            with pytest.raises(RuntimeError) as exc_info:
                await client.chat_completion(messages=messages, model="gemini-2.5-flash")
            
            # Error should include both exit code and simplified explanation
            assert "exit code 1 (failure)" in str(exc_info.value)
            assert "rate limit exceeded" in str(exc_info.value)
            # Log should contain both raw error, exit code, and recognized reason
            assert "command failed with exit code 1 (failure)" in caplog.text
            assert "Resource exhausted" in caplog.text
            assert "rate limit exceeded" in caplog.text


@pytest.mark.asyncio
async def test_execute_gemini_command_timeout(caplog):
    client = GeminiClient()
    mock_process = MagicMock()
    mock_process.pid = 12349
    mock_process.returncode = None
    mock_process.terminate = MagicMock()
    
    async def mock_wait():
        mock_process.returncode = -15
        return -15
        
    mock_process.wait = AsyncMock(side_effect=mock_wait)

    mock_process.communicate = AsyncMock()

    async def mock_wait_for(coro, timeout=None):
        try:
            # Cleanly close the inner coroutine before raising
            coro.close()
        except Exception:
            pass
        raise asyncio.TimeoutError()

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            with caplog.at_level(logging.INFO, logger="gemini_cli_proxy"):
                messages = [ChatMessage(role="user", content="Hi")]
                with pytest.raises(RuntimeError) as exc_info:
                    await client.chat_completion(messages=messages, model="gemini-2.5-flash")
                
                assert "execution timeout" in str(exc_info.value)
                assert "command execution timeout" in caplog.text
                assert "PID: 12349" in caplog.text
                assert mock_process.terminate.called


@pytest.mark.asyncio
async def test_execute_gemini_command_stream_success(caplog):
    client = GeminiClient()
    mock_process = MagicMock()
    mock_process.pid = 23456
    mock_process.returncode = 0
    mock_process.wait = AsyncMock(return_value=0)
    
    # Mock stdout stream
    mock_stdout = MagicMock()
    mock_stdout.read = AsyncMock(side_effect=[b"Hello ", b"streaming ", b"world", b""])
    mock_process.stdout = mock_stdout
    
    # Mock stderr stream
    mock_stderr = MagicMock()
    mock_stderr.read = AsyncMock(return_value=b"")
    mock_process.stderr = mock_stderr

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
        with caplog.at_level(logging.INFO, logger="gemini_cli_proxy"):
            messages = [ChatMessage(role="user", content="Hi")]
            chunks = []
            async for chunk in client.chat_completion_stream(messages=messages, model="gemini-2.5-flash"):
                chunks.append(chunk)
            
            assert "".join(chunks) == "Hello streaming world"
            assert "Invoking agy command (stream mode)" in caplog.text
            assert "PID: 23456" in caplog.text
            assert "stream command completed successfully with exit code 0 (success)" in caplog.text
            assert "chunks: 3" in caplog.text


@pytest.mark.asyncio
async def test_execute_gemini_command_stream_failure_records_exit_code(caplog):
    client = GeminiClient()
    mock_process = MagicMock()
    mock_process.pid = 23457
    mock_process.returncode = 1
    mock_process.wait = AsyncMock(return_value=1)
    
    # Empty stdout
    mock_stdout = MagicMock()
    mock_stdout.read = AsyncMock(return_value=b"")
    mock_process.stdout = mock_stdout
    
    # Stderr with error details
    mock_stderr = MagicMock()
    mock_stderr.read = AsyncMock(return_value=b"fatal: server returned status 503")
    mock_process.stderr = mock_stderr

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
        with caplog.at_level(logging.INFO, logger="gemini_cli_proxy"):
            messages = [ChatMessage(role="user", content="Hi")]
            with pytest.raises(RuntimeError) as exc_info:
                async for _ in client.chat_completion_stream(messages=messages, model="gemini-2.5-flash"):
                    pass
            
            assert "exit code 1 (failure)" in str(exc_info.value)
            assert "stream command failed with exit code 1 (failure)" in caplog.text
            assert "fatal: server returned status 503" in caplog.text
