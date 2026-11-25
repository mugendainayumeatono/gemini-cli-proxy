import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp import PROTOCOL_VERSION
from acp.connection import StreamDirection, StreamEvent
from acp.schema import NewSessionResponse, PromptResponse
from click.testing import CliRunner

from gemini_cli_proxy.acp_client import ACPClient
from gemini_cli_proxy.audit import AuditLogger, record_audit
from gemini_cli_proxy.cli import main as cli_main
from gemini_cli_proxy.config import Config, config
from gemini_cli_proxy.gemini_client import GeminiClient
from gemini_cli_proxy.models import ChatMessage


@pytest.fixture(autouse=True)
def _restore_global_config(monkeypatch, tmp_path):
    """Restore global config and audit env vars around every test."""
    orig_audit_enabled = config.audit_enabled
    orig_audit_file = config.audit_file
    orig_acp_mode = config.acp_mode
    orig_host = config.host
    orig_port = config.port
    orig_log_level = config.log_level
    orig_rate_limit = config.rate_limit
    orig_max_concurrency = config.max_concurrency
    orig_timeout = config.timeout
    orig_gemini_command = config.gemini_command
    orig_debug = config.debug
    orig_api_key = config.proxy_api_key

    config.audit_enabled = False
    config.audit_file = str(tmp_path / "audit.jsonl")
    config.acp_mode = False

    yield

    config.audit_enabled = orig_audit_enabled
    config.audit_file = orig_audit_file
    config.acp_mode = orig_acp_mode
    config.host = orig_host
    config.port = orig_port
    config.log_level = orig_log_level
    config.rate_limit = orig_rate_limit
    config.max_concurrency = orig_max_concurrency
    config.timeout = orig_timeout
    config.gemini_command = orig_gemini_command
    config.debug = orig_debug
    config.proxy_api_key = orig_api_key


class TestAuditConfigAndCLI:
    def test_config_defaults_disabled(self, monkeypatch):
        for var in ("GEMINI_AUDIT", "PROXY_AUDIT", "GEMINI_AUDIT_FILE", "PROXY_AUDIT_FILE"):
            monkeypatch.delenv(var, raising=False)
        with patch("os.path.isdir", return_value=False):
            cfg = Config()
        assert cfg.audit_enabled is False
        assert cfg.audit_file == "logs/audit.jsonl"

    def test_config_docker_logs_dir_default(self, monkeypatch):
        for var in ("GEMINI_AUDIT", "PROXY_AUDIT", "GEMINI_AUDIT_FILE", "PROXY_AUDIT_FILE"):
            monkeypatch.delenv(var, raising=False)
        with patch("os.path.isdir", side_effect=lambda p: p == "/app/logs"):
            cfg = Config()
        assert cfg.audit_enabled is False
        assert cfg.audit_file == "/app/logs/audit.jsonl"

    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
    def test_config_env_enabled(self, monkeypatch, val):
        monkeypatch.setenv("GEMINI_AUDIT", val)
        monkeypatch.setenv("GEMINI_AUDIT_FILE", "/tmp/custom-audit.jsonl")
        cfg = Config()
        assert cfg.audit_enabled is True
        assert cfg.audit_file == "/tmp/custom-audit.jsonl"

    def test_config_fallback_env_vars(self, monkeypatch):
        monkeypatch.delenv("GEMINI_AUDIT", raising=False)
        monkeypatch.delenv("GEMINI_AUDIT_FILE", raising=False)
        monkeypatch.setenv("PROXY_AUDIT", "true")
        monkeypatch.setenv("PROXY_AUDIT_FILE", "/tmp/proxy-audit.jsonl")
        cfg = Config()
        assert cfg.audit_enabled is True
        assert cfg.audit_file == "/tmp/proxy-audit.jsonl"

    def test_cli_default_disabled(self, monkeypatch):
        for var in ("GEMINI_AUDIT", "PROXY_AUDIT", "GEMINI_AUDIT_FILE", "PROXY_AUDIT_FILE"):
            monkeypatch.delenv(var, raising=False)
        runner = CliRunner()
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(cli_main, [])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        assert config.audit_enabled is False
        assert os.environ.get("GEMINI_AUDIT") == "False"

    def test_cli_audit_flags(self, monkeypatch, tmp_path):
        for var in ("GEMINI_AUDIT", "PROXY_AUDIT", "GEMINI_AUDIT_FILE", "PROXY_AUDIT_FILE"):
            monkeypatch.delenv(var, raising=False)
        audit_path = str(tmp_path / "cli_audit.jsonl")
        runner = CliRunner()
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(cli_main, ["--audit", "--audit-file", audit_path])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        assert config.audit_enabled is True
        assert config.audit_file == audit_path
        assert os.environ.get("GEMINI_AUDIT") == "True"
        assert os.environ.get("GEMINI_AUDIT_FILE") == audit_path

    def test_cli_audit_envvars(self, monkeypatch, tmp_path):
        audit_path = str(tmp_path / "env_audit.jsonl")
        monkeypatch.setenv("GEMINI_AUDIT", "true")
        monkeypatch.setenv("GEMINI_AUDIT_FILE", audit_path)
        runner = CliRunner()
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(cli_main, [])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        assert config.audit_enabled is True
        assert config.audit_file == audit_path


class TestRecordAudit:
    def test_disabled_writes_nothing(self, tmp_path):
        audit_path = tmp_path / "disabled" / "audit.jsonl"
        config.audit_enabled = False
        config.audit_file = str(audit_path)

        record_audit("input", {"hello": "world"}, mode="acp", pid=100, session_id="s1")
        assert not audit_path.exists()

    def test_enabled_writes_utf8_jsonl_without_ascii_escaping(self, tmp_path):
        audit_path = tmp_path / "nested" / "audit.jsonl"
        config.audit_enabled = True
        config.audit_file = str(audit_path)

        payload = {"prompt": "你好世界-audit", "nested": {"ok": True}}
        record_audit("input", payload, mode="acp", pid=42, session_id="sess-zh")
        record_audit("output", {"reply": "收到-你好"})

        assert audit_path.exists()
        raw_text = audit_path.read_text(encoding="utf-8")
        assert "你好世界-audit" in raw_text
        assert "收到-你好" in raw_text
        assert "\\u4f60\\u597d" not in raw_text

        lines = raw_text.strip().splitlines()
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["direction"] == "input"
        assert first["mode"] == "acp"
        assert first["pid"] == 42
        assert first["session_id"] == "sess-zh"
        assert first["data"] == payload
        assert "timestamp" in first

        second = json.loads(lines[1])
        assert second["direction"] == "output"
        assert second["mode"] == "acp"
        assert "pid" not in second
        assert "session_id" not in second
        assert second["data"] == {"reply": "收到-你好"}

    def test_io_error_does_not_raise(self, tmp_path, caplog):
        # Point audit_file at a directory path to trigger an IsADirectoryError on open()
        config.audit_enabled = True
        config.audit_file = str(tmp_path)

        logger_instance = AuditLogger()
        logger_instance.record("input", {"x": 1})
        assert "Failed to write audit log" in caplog.text or len(caplog.records) >= 1


class TestACPClientAudit:
    @staticmethod
    def _make_fake_process(pid: int = 4242) -> MagicMock:
        proc = MagicMock()
        proc.pid = pid
        proc.returncode = None
        proc.stdin = MagicMock()
        proc.stdin.close = MagicMock()
        proc.stdin.wait_closed = AsyncMock()
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()
        proc.stderr.readline = AsyncMock(return_value=b"")
        proc.terminate = MagicMock()
        proc.kill = MagicMock()

        async def _wait():
            proc.returncode = -15
            return -15

        proc.wait = AsyncMock(side_effect=_wait)
        return proc

    @staticmethod
    def _make_fake_conn() -> MagicMock:
        conn = MagicMock()
        conn._conn = MagicMock()
        conn._conn.send_request = AsyncMock(return_value={})
        conn.initialize = AsyncMock(
            return_value=MagicMock(
                protocol_version=PROTOCOL_VERSION,
                agent_info=MagicMock(name="Antigravity ACP Server", version="1.0.0"),
            )
        )
        conn.new_session = AsyncMock(return_value=NewSessionResponse(session_id="sess-1"))
        conn.close = AsyncMock(return_value=None)
        conn.prompt = AsyncMock(return_value=PromptResponse(stop_reason="end_turn"))
        return conn

    @pytest.mark.asyncio
    async def test_acp_disabled_passes_no_observers(self):
        config.audit_enabled = False
        client = ACPClient()
        proc = self._make_fake_process(pid=7001)
        conn = self._make_fake_conn()

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            with patch("gemini_cli_proxy.acp_client.acp.connect_to_agent", return_value=conn) as connect_mock:
                await client.start()
                connect_mock.assert_called_once_with(client._handler, proc.stdin, proc.stdout)
                assert connect_mock.call_args.kwargs == {}
                await client.stop()

    @pytest.mark.asyncio
    async def test_acp_enabled_passes_observer_and_records_events(self, tmp_path):
        audit_path = tmp_path / "acp_audit.jsonl"
        config.audit_enabled = True
        config.audit_file = str(audit_path)

        client = ACPClient()
        proc = self._make_fake_process(pid=7002)
        conn = self._make_fake_conn()

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            with patch("gemini_cli_proxy.acp_client.acp.connect_to_agent", return_value=conn) as connect_mock:
                await client.start()

                assert connect_mock.call_count == 1
                assert connect_mock.call_args.args == (client._handler, proc.stdin, proc.stdout)
                observers = connect_mock.call_args.kwargs.get("observers")
                assert isinstance(observers, list) and len(observers) == 1
                observer = observers[0]

                outgoing_msg = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/prompt",
                    "params": {"sessionId": "sess-acp-1", "prompt": [{"type": "text", "text": "你好ACP"}]},
                }
                incoming_msg = {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "sess-acp-1",
                        "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "收到ACP"}},
                    },
                }
                observer(StreamEvent(StreamDirection.OUTGOING, outgoing_msg))
                observer(StreamEvent(StreamDirection.INCOMING, incoming_msg))

                await client.stop()

        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]

        assert records[0]["direction"] == "input"
        assert records[0]["mode"] == "acp"
        assert records[0]["pid"] == 7002
        assert records[0]["session_id"] == "sess-acp-1"
        assert records[0]["data"] == outgoing_msg

        assert records[1]["direction"] == "output"
        assert records[1]["mode"] == "acp"
        assert records[1]["pid"] == 7002
        assert records[1]["session_id"] == "sess-acp-1"
        assert records[1]["data"] == incoming_msg


class TestGeminiClientAudit:
    @pytest.mark.asyncio
    async def test_cli_sync_records_input_and_output(self, tmp_path):
        audit_path = tmp_path / "cli_sync_audit.jsonl"
        config.audit_enabled = True
        config.audit_file = str(audit_path)
        config.acp_mode = False

        client = GeminiClient()
        mock_process = MagicMock()
        mock_process.pid = 8001
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'{"answer": "\xe4\xbd\xa0\xe5\xa5\xbd-sync"}', b""))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
            resp = await client.chat_completion(
                messages=[ChatMessage(role="user", content="测试sync")],
                model="gemini-3.8-flash-low",
            )
        assert "你好-sync" in resp

        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        inp = json.loads(lines[0])
        out = json.loads(lines[1])

        assert inp["direction"] == "input"
        assert inp["mode"] == "cli_sync"
        assert inp["pid"] == 8001
        assert inp["data"]["model"] == "gemini-3.8-flash-low"
        assert "测试sync" in inp["data"]["prompt"]

        assert out["direction"] == "output"
        assert out["mode"] == "cli_sync"
        assert out["pid"] == 8001
        assert out["data"]["exit_code"] == 0
        assert out["data"]["stdout"] == {"answer": "你好-sync"}

    @pytest.mark.asyncio
    async def test_cli_stream_records_input_and_output(self, tmp_path):
        audit_path = tmp_path / "cli_stream_audit.jsonl"
        config.audit_enabled = True
        config.audit_file = str(audit_path)
        config.acp_mode = False

        client = GeminiClient()
        mock_process = MagicMock()
        mock_process.pid = 8002
        mock_process.returncode = 0
        mock_process.wait = AsyncMock(return_value=0)

        mock_stdout = MagicMock()
        mock_stdout.readline = AsyncMock(
            side_effect=[
                b'{"event":"init","conversation_id":"8002"}\n',
                '{"event":"step_update","step_update":{"step_type":"agent_response","text_delta":"你好stream"}}\n'.encode(
                    "utf-8"
                ),
                b'{"event":"result","result":{"status":"SUCCESS"}}\n',
                b"",
            ]
        )
        mock_process.stdout = mock_stdout
        mock_stderr = MagicMock()
        mock_stderr.read = AsyncMock(return_value=b"")
        mock_process.stderr = mock_stderr

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_process)):
            chunks = []
            async for chunk in client.chat_completion_stream(
                messages=[ChatMessage(role="user", content="测试stream")],
                model="gemini-3.8-flash-low",
            ):
                chunks.append(chunk)

        assert "".join(chunks) == "你好stream"

        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4
        records = [json.loads(line) for line in lines]

        assert records[0]["direction"] == "input"
        assert records[0]["mode"] == "cli_stream"
        assert records[0]["pid"] == 8002
        assert records[0]["data"]["model"] == "gemini-3.8-flash-low"
        assert "测试stream" in records[0]["data"]["prompt"]

        for rec in records[1:]:
            assert rec["direction"] == "output"
            assert rec["mode"] == "cli_stream"
            assert rec["pid"] == 8002

        assert records[1]["data"] == {"event": "init", "conversation_id": "8002"}
        assert records[2]["data"]["step_update"]["text_delta"] == "你好stream"
        assert records[3]["data"] == {"event": "result", "result": {"status": "SUCCESS"}}
