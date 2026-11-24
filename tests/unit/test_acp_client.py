import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp import PROTOCOL_VERSION, RequestError
from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
from acp.schema import (
    AgentMessageChunk,
    AgentPlanContentUpdate,
    AgentPlanRemovedUpdate,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AvailableCommandsUpdate,
    ClientCapabilities,
    ConfigOptionUpdate,
    CurrentModeUpdate,
    DeclineElicitationResponse,
    DeniedOutcome,
    ElicitationFormSessionMode,
    ImageContentBlock,
    Implementation,
    NewSessionResponse,
    PermissionOption,
    PlanUpdateItems,
    PromptResponse,
    RequestPermissionResponse,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionInfoUpdate,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    Usage,
    UsageUpdate,
    UserMessageChunk,
)

from gemini_cli_proxy import __version__
from gemini_cli_proxy.acp_client import ACPClient, ACPError, _build_prompt_blocks
from gemini_cli_proxy.config import config
from gemini_cli_proxy.models import ChatContentPart, ChatMessage


def _make_fake_process(pid: int = 4242, returncode: int | None = None) -> MagicMock:
    proc = MagicMock()
    proc.pid = pid
    proc.returncode = returncode
    proc.stdin = MagicMock()
    proc.stdin.close = MagicMock()
    proc.stdin.wait_closed = AsyncMock()
    proc.stdout = MagicMock()
    proc.stderr = MagicMock()
    proc.stderr.readline = AsyncMock(return_value=b"")
    proc.terminate = MagicMock()
    proc.kill = MagicMock()

    async def _wait():
        if proc.returncode is None:
            proc.returncode = -15
        return proc.returncode

    proc.wait = AsyncMock(side_effect=_wait)
    return proc


def _make_fake_conn(session_id: str = "sess-1", config_options=None, models=None) -> MagicMock:
    conn = MagicMock()
    conn._conn = MagicMock()
    conn._conn.send_request = AsyncMock(return_value={})
    conn.initialize = AsyncMock(
        return_value=SimpleNamespace(
            protocol_version=PROTOCOL_VERSION,
            agent_info=SimpleNamespace(name="Antigravity ACP Server", version="1.5.16"),
        )
    )
    session = NewSessionResponse(
        session_id=session_id,
        config_options=config_options,
    )
    if models is not None:
        # Simulate a session response carrying a .models attribute
        session = SimpleNamespace(
            session_id=session_id,
            config_options=config_options,
            models=models,
        )
    conn.new_session = AsyncMock(return_value=session)
    conn.set_config_option = AsyncMock(return_value=SimpleNamespace(config_options=[]))
    conn.cancel = AsyncMock(return_value=None)
    conn.close_session = AsyncMock(return_value=None)
    conn.close = AsyncMock(return_value=None)
    conn.prompt = AsyncMock(return_value=PromptResponse(stop_reason="end_turn"))
    return conn


@pytest.mark.asyncio
async def test_start_and_stop_lifecycle(caplog):
    client = ACPClient()
    proc = _make_fake_process(pid=9001)
    proc.stderr.readline = AsyncMock(side_effect=[b"debug stderr line\n", b""])
    conn = _make_fake_conn()

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as exec_mock:
        with patch("gemini_cli_proxy.acp_client.acp.connect_to_agent", return_value=conn) as connect_mock:
            with caplog.at_level(logging.DEBUG, logger="gemini_cli_proxy"):
                await client.start()

                exec_mock.assert_awaited_once()
                args, kwargs = exec_mock.call_args
                assert args == (config.gemini_command, "acp")
                assert kwargs["limit"] == DEFAULT_STDIO_BUFFER_LIMIT_BYTES
                assert kwargs["stdin"] == asyncio.subprocess.PIPE
                assert kwargs["stdout"] == asyncio.subprocess.PIPE
                assert kwargs["stderr"] == asyncio.subprocess.PIPE

                connect_mock.assert_called_once_with(client._handler, proc.stdin, proc.stdout)
                conn.initialize.assert_awaited_once()
                init_kwargs = conn.initialize.call_args.kwargs
                assert init_kwargs["protocol_version"] == PROTOCOL_VERSION
                assert isinstance(init_kwargs["client_capabilities"], ClientCapabilities)
                assert init_kwargs["client_info"] == Implementation(
                    name="gemini-cli-proxy", version=__version__
                )
                await asyncio.sleep(0)
                assert "debug stderr line" in caplog.text

                await client.stop()
                conn.close.assert_awaited_once()
                proc.stdin.close.assert_called_once()
                proc.terminate.assert_called_once()
                assert "ACP agent process (PID: 9001) stopped" in caplog.text
                assert "terminated by signal 15" in caplog.text


@pytest.mark.asyncio
async def test_start_executable_not_found_raises_protocol_error():
    client = ACPClient()
    with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError("no agy"))):
        with pytest.raises(ACPError) as exc_info:
            await client.start()
        assert exc_info.value.code == "acp_protocol_error"


@pytest.mark.asyncio
async def test_start_initialize_timeout_cleans_up_process():
    client = ACPClient()
    proc = _make_fake_process(pid=9002)
    conn = _make_fake_conn()
    conn.initialize = AsyncMock(side_effect=asyncio.TimeoutError())

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with patch("gemini_cli_proxy.acp_client.acp.connect_to_agent", return_value=conn):
            with pytest.raises(ACPError) as exc_info:
                await client.start()
            assert exc_info.value.code == "acp_protocol_error"
            proc.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_auto_restart_when_process_exited(caplog):
    client = ACPClient()
    dead_proc = _make_fake_process(pid=1111, returncode=1)
    fresh_proc = _make_fake_process(pid=2222, returncode=None)
    old_conn = _make_fake_conn()
    fresh_conn = _make_fake_conn()

    client._process = dead_proc
    client._conn = old_conn

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=fresh_proc)):
        with patch("gemini_cli_proxy.acp_client.acp.connect_to_agent", return_value=fresh_conn):
            with caplog.at_level(logging.WARNING, logger="gemini_cli_proxy"):
                conn = await client._ensure_connection()
                assert conn is fresh_conn
                assert client._process is fresh_proc
                assert "exited unexpectedly" in caplog.text
                assert "restarting" in caplog.text
    await client.stop()


@pytest.mark.asyncio
async def test_chat_completion_sync_and_stream_with_benign_updates():
    client = ACPClient()
    proc = _make_fake_process()
    conn = _make_fake_conn(session_id="sess-text")
    client._process = proc
    client._conn = conn

    async def fake_prompt(session_id: str, prompt):
        await client._handler.session_update(
            session_id, AvailableCommandsUpdate(session_update="available_commands_update", available_commands=[])
        )
        await client._handler.session_update(
            session_id, UsageUpdate(session_update="usage_update", size=100, used=10)
        )
        await client._handler.session_update(
            session_id, SessionInfoUpdate(session_update="session_info_update", title="Test")
        )
        await client._handler.session_update(
            session_id, ConfigOptionUpdate(session_update="config_option_update", config_options=[])
        )
        await client._handler.session_update(
            session_id, CurrentModeUpdate(session_update="current_mode_update", current_mode_id="default")
        )
        await client._handler.session_update(
            session_id,
            UserMessageChunk(
                session_update="user_message_chunk",
                content=TextContentBlock(type="text", text="echo"),
            ),
        )
        await client._handler.session_update(
            session_id,
            AgentThoughtChunk(
                session_update="agent_thought_chunk",
                content=TextContentBlock(type="text", text="internal reasoning"),
            ),
        )
        await client._handler.session_update(
            session_id,
            AgentMessageChunk(
                session_update="agent_message_chunk",
                content=TextContentBlock(type="text", text="Hello "),
            ),
        )
        await client._handler.session_update(
            session_id,
            AgentMessageChunk(
                session_update="agent_message_chunk",
                content=TextContentBlock(type="text", text="ACP!"),
            ),
        )
        return PromptResponse(stop_reason="end_turn", usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3))

    conn.prompt = AsyncMock(side_effect=fake_prompt)

    messages = [ChatMessage(role="user", content="Hi")]
    chunks = []
    async for chunk in client.chat_completion_stream(messages, model="gemini-3.8-flash-low"):
        chunks.append(chunk)
    assert chunks == ["Hello ", "ACP!"]

    full = await client.chat_completion(messages, model="gemini-3.8-flash-low")
    assert full == "Hello ACP!"
    assert conn.close_session.await_count == 2


@pytest.mark.asyncio
async def test_session_model_and_effort_configuration(monkeypatch):
    client = ACPClient()
    client._process = _make_fake_process()

    # Case 1: session.models carries available_models
    conn_models = _make_fake_conn(
        session_id="sess-m1",
        models=SimpleNamespace(available_models=[SimpleNamespace(model_id="gemini-3.8-flash-low")]),
    )
    client._conn = conn_models
    async def ok_prompt(session_id: str, prompt):
        await client._handler.session_update(
            session_id,
            AgentMessageChunk(session_update="agent_message_chunk", content=TextContentBlock(type="text", text="ok")),
        )
        return PromptResponse(stop_reason="end_turn")
    conn_models.prompt = AsyncMock(side_effect=ok_prompt)
    await client.chat_completion([ChatMessage(role="user", content="hi")], model="gemini-3.8-flash-low")
    conn_models._conn.send_request.assert_awaited_once_with(
        "session/set_model",
        {"sessionId": "sess-m1", "modelId": "gemini-3.8-flash-low"},
    )

    # Case 2: config_options base model + config.effort when model has no suffix
    monkeypatch.setattr(config, "effort", "medium")
    config_options = [
        SessionConfigOptionSelect(
            id="model",
            name="Model",
            category="model",
            type="select",
            current_value="gemini-4-neon",
            options=[SessionConfigSelectOption(value="gemini-3.8-flash", name="Gemini 3.8 Flash")],
        ),
        SessionConfigOptionSelect(
            id="thought_level",
            name="Thinking Level",
            category="thought_level",
            type="select",
            current_value="high",
            options=[
                SessionConfigSelectOption(value="low", name="Low"),
                SessionConfigSelectOption(value="medium", name="Medium"),
            ],
        ),
    ]
    conn_cfg = _make_fake_conn(session_id="sess-m2", config_options=config_options)
    conn_cfg.prompt = AsyncMock(side_effect=ok_prompt)
    client._conn = conn_cfg
    await client.chat_completion([ChatMessage(role="user", content="hi")], model="gemini-3.8-flash")
    assert conn_cfg.set_config_option.await_count == 2
    conn_cfg.set_config_option.assert_any_await(
        config_id="model", session_id="sess-m2", value="gemini-3.8-flash"
    )
    conn_cfg.set_config_option.assert_any_await(
        config_id="thought_level", session_id="sess-m2", value="medium"
    )


def test_build_prompt_blocks_with_text_and_base64_image():
    messages = [
        ChatMessage(role="system", content="Be concise."),
        ChatMessage(
            role="user",
            content=[
                ChatContentPart(type="text", text="What is in this image?"),
                ChatContentPart(
                    type="image_url",
                    image_url={"url": "data:image/png;base64,iVBORw0KGgo="},
                ),
                ChatContentPart(type="text", text="Answer in one word."),
            ],
        ),
    ]
    blocks = _build_prompt_blocks(messages)
    assert len(blocks) == 3
    assert isinstance(blocks[0], TextContentBlock)
    assert blocks[0].text == "System: Be concise.\nUser: What is in this image?"
    assert isinstance(blocks[1], ImageContentBlock)
    assert blocks[1].mime_type == "image/png"
    assert blocks[1].data == "iVBORw0KGgo="
    assert isinstance(blocks[2], TextContentBlock)
    assert blocks[2].text == "Answer in one word."


@pytest.mark.asyncio
async def test_request_permission_aborts_with_acp_permission_required(caplog):
    client = ACPClient()
    client._process = _make_fake_process()
    conn = _make_fake_conn(session_id="sess-perm")
    client._conn = conn

    perm_responses = []

    async def perm_prompt(session_id: str, prompt):
        resp = await client._handler.request_permission(
            session_id=session_id,
            tool_call=ToolCallUpdate(tool_call_id="tc-1", title="Run bash"),
            options=[PermissionOption(option_id="allow", name="Allow", kind="allow_once")],
        )
        perm_responses.append(resp)
        await asyncio.sleep(0.05)
        return PromptResponse(stop_reason="cancelled")

    conn.prompt = AsyncMock(side_effect=perm_prompt)

    with caplog.at_level(logging.ERROR, logger="gemini_cli_proxy"):
        with pytest.raises(ACPError) as exc_info:
            await client.chat_completion([ChatMessage(role="user", content="run ls")], model="m")

    assert exc_info.value.code == "acp_permission_required"
    assert perm_responses == [RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))]
    conn.cancel.assert_awaited_once_with(session_id="sess-perm")
    conn.close_session.assert_awaited_once_with(session_id="sess-perm")
    assert "ACP permission request denied" in caplog.text


@pytest.mark.asyncio
async def test_tool_call_start_and_progress_raise_acp_tool_call_not_supported():
    for update_obj in (
        ToolCallStart(session_update="tool_call", tool_call_id="tc-1", title="Read file"),
        ToolCallProgress(session_update="tool_call_update", tool_call_id="tc-2", title="Edit file"),
    ):
        client = ACPClient()
        client._process = _make_fake_process()
        conn = _make_fake_conn(session_id="sess-tool")
        client._conn = conn

        async def tool_prompt(session_id: str, prompt, _upd=update_obj):
            await client._handler.session_update(session_id, _upd)
            return PromptResponse(stop_reason="cancelled")

        conn.prompt = AsyncMock(side_effect=tool_prompt)

        with pytest.raises(ACPError) as exc_info:
            await client.chat_completion([ChatMessage(role="user", content="use tool")], model="m")

        assert exc_info.value.code == "acp_tool_call_not_supported"
        conn.cancel.assert_awaited_once_with(session_id="sess-tool")
        conn.close_session.assert_awaited_once_with(session_id="sess-tool")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name,kwargs",
    [
        ("write_text_file", {"session_id": "sess-cap", "path": "a.txt", "content": "x"}),
        ("read_text_file", {"session_id": "sess-cap", "path": "a.txt"}),
        ("create_terminal", {"session_id": "sess-cap", "command": "ls"}),
        ("terminal_output", {"session_id": "sess-cap", "terminal_id": "t1"}),
        ("release_terminal", {"session_id": "sess-cap", "terminal_id": "t1"}),
        ("wait_for_terminal_exit", {"session_id": "sess-cap", "terminal_id": "t1"}),
        ("kill_terminal", {"session_id": "sess-cap", "terminal_id": "t1"}),
        ("connect_mcp", {"session_id": "sess-cap"}),
        ("mcp_message", {"session_id": "sess-cap"}),
        ("ext_method", {"method": "custom/op", "params": {"sessionId": "sess-cap"}}),
    ],
)
async def test_client_capability_callbacks_raise_and_abort(method_name, kwargs):
    client = ACPClient()
    client._process = _make_fake_process()
    conn = _make_fake_conn(session_id="sess-cap")
    client._conn = conn

    async def cap_prompt(session_id: str, prompt):
        with pytest.raises(RequestError) as rpc_err:
            await getattr(client._handler, method_name)(**kwargs)
        assert rpc_err.value.code == -32601
        return PromptResponse(stop_reason="cancelled")

    conn.prompt = AsyncMock(side_effect=cap_prompt)

    with pytest.raises(ACPError) as exc_info:
        await client.chat_completion([ChatMessage(role="user", content="cap")], model="m")

    assert exc_info.value.code == "acp_client_capability_not_supported"
    conn.cancel.assert_awaited_once_with(session_id="sess-cap")
    conn.close_session.assert_awaited_once_with(session_id="sess-cap")


@pytest.mark.asyncio
async def test_create_elicitation_declines_and_aborts():
    client = ACPClient()
    client._process = _make_fake_process()
    conn = _make_fake_conn(session_id="sess-elic")
    client._conn = conn

    elic_responses = []

    async def elic_prompt(session_id: str, prompt):
        mode = ElicitationFormSessionMode(
            mode="form",
            session_id=session_id,
            tool_call_id="tc-1",
            requested_schema={"type": "object", "properties": {}},
        )
        resp = await client._handler.create_elicitation("Provide input", mode)
        elic_responses.append(resp)
        return PromptResponse(stop_reason="cancelled")

    conn.prompt = AsyncMock(side_effect=elic_prompt)

    with pytest.raises(ACPError) as exc_info:
        await client.chat_completion([ChatMessage(role="user", content="ask")], model="m")

    assert exc_info.value.code == "acp_client_capability_not_supported"
    assert elic_responses == [DeclineElicitationResponse(action="decline")]
    conn.cancel.assert_awaited_once_with(session_id="sess-elic")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_update",
    [
        AgentMessageChunk(
            session_update="agent_message_chunk",
            content=ImageContentBlock(type="image", data="AAAA", mime_type="image/png"),
        ),
        AgentPlanUpdate(session_update="plan", entries=[]),
        AgentPlanContentUpdate(
            session_update="plan_update",
            plan=PlanUpdateItems(plan_id="p1", type="items", entries=[]),
        ),
        AgentPlanRemovedUpdate(session_update="plan_removed", plan_id="p1"),
        SimpleNamespace(session_update="unknown_custom_update"),
    ],
)
async def test_non_text_and_plan_updates_raise_acp_non_text_content(bad_update):
    client = ACPClient()
    client._process = _make_fake_process()
    conn = _make_fake_conn(session_id="sess-nontext")
    client._conn = conn

    async def nontext_prompt(session_id: str, prompt):
        await client._handler.session_update(session_id, bad_update)
        return PromptResponse(stop_reason="cancelled")

    conn.prompt = AsyncMock(side_effect=nontext_prompt)

    with pytest.raises(ACPError) as exc_info:
        await client.chat_completion([ChatMessage(role="user", content="img")], model="m")

    assert exc_info.value.code == "acp_non_text_content"
    conn.cancel.assert_awaited_once_with(session_id="sess-nontext")
    conn.close_session.assert_awaited_once_with(session_id="sess-nontext")


@pytest.mark.asyncio
async def test_empty_response_raises_acp_empty_response():
    client = ACPClient()
    client._process = _make_fake_process()
    conn = _make_fake_conn(session_id="sess-empty")
    client._conn = conn

    async def empty_prompt(session_id: str, prompt):
        await client._handler.session_update(
            session_id,
            AgentThoughtChunk(
                session_update="agent_thought_chunk",
                content=TextContentBlock(type="text", text="only thought"),
            ),
        )
        return PromptResponse(stop_reason="end_turn")

    conn.prompt = AsyncMock(side_effect=empty_prompt)

    with pytest.raises(ACPError) as exc_info:
        await client.chat_completion([ChatMessage(role="user", content="hi")], model="m")

    assert exc_info.value.code == "acp_empty_response"
    conn.close_session.assert_awaited_once_with(session_id="sess-empty")


@pytest.mark.asyncio
async def test_request_error_from_prompt_raises_acp_protocol_error():
    client = ACPClient()
    client._process = _make_fake_process()
    conn = _make_fake_conn(session_id="sess-err")
    client._conn = conn
    conn.prompt = AsyncMock(side_effect=RequestError.internal_error({"reason": "boom"}))

    with pytest.raises(ACPError) as exc_info:
        await client.chat_completion([ChatMessage(role="user", content="hi")], model="m")

    assert exc_info.value.code == "acp_protocol_error"
    conn.close_session.assert_awaited_once_with(session_id="sess-err")


@pytest.mark.asyncio
async def test_turn_timeout_cancels_and_closes_session(monkeypatch):
    monkeypatch.setattr(config, "timeout", 0.05)
    client = ACPClient()
    client._process = _make_fake_process()
    conn = _make_fake_conn(session_id="sess-timeout")
    client._conn = conn

    async def slow_prompt(session_id: str, prompt):
        await asyncio.sleep(5.0)
        return PromptResponse(stop_reason="end_turn")

    conn.prompt = AsyncMock(side_effect=slow_prompt)

    with pytest.raises(asyncio.TimeoutError):
        await client.chat_completion([ChatMessage(role="user", content="slow")], model="m")

    conn.cancel.assert_awaited_once_with(session_id="sess-timeout")
    conn.close_session.assert_awaited_once_with(session_id="sess-timeout")


@pytest.mark.asyncio
async def test_stream_cancellation_cancels_and_closes_session():
    client = ACPClient()
    client._process = _make_fake_process()
    conn = _make_fake_conn(session_id="sess-cancel")
    client._conn = conn

    async def hanging_prompt(session_id: str, prompt):
        await client._handler.session_update(
            session_id,
            AgentMessageChunk(
                session_update="agent_message_chunk",
                content=TextContentBlock(type="text", text="first"),
            ),
        )
        await asyncio.sleep(5.0)
        return PromptResponse(stop_reason="end_turn")

    conn.prompt = AsyncMock(side_effect=hanging_prompt)

    async def run_stream_and_cancel():
        gen = client.chat_completion_stream([ChatMessage(role="user", content="hi")], model="m")
        first = await gen.__anext__()
        assert first == "first"
        task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await gen.aclose()

    await run_stream_and_cancel()
    conn.cancel.assert_awaited_once_with(session_id="sess-cancel")
    conn.close_session.assert_awaited_once_with(session_id="sess-cancel")


@pytest.mark.asyncio
async def test_concurrent_sessions_isolated_by_session_id():
    client = ACPClient()
    client._process = _make_fake_process()
    conn = _make_fake_conn()
    client._conn = conn

    counter = 0

    async def new_session(**kwargs):
        nonlocal counter
        counter += 1
        return NewSessionResponse(session_id=f"sess-{counter}")

    async def prompt(session_id: str, prompt):
        if session_id == "sess-1":
            await asyncio.sleep(0.02)
            await client._handler.session_update(
                session_id,
                AgentMessageChunk(session_update="agent_message_chunk", content=TextContentBlock(type="text", text="one")),
            )
        else:
            await client._handler.session_update(
                session_id,
                AgentMessageChunk(session_update="agent_message_chunk", content=TextContentBlock(type="text", text="two")),
            )
        return PromptResponse(stop_reason="end_turn")

    conn.new_session = AsyncMock(side_effect=new_session)
    conn.prompt = AsyncMock(side_effect=prompt)

    res1, res2 = await asyncio.gather(
        client.chat_completion([ChatMessage(role="user", content="1")], model="m"),
        client.chat_completion([ChatMessage(role="user", content="2")], model="m"),
    )
    assert res1 == "one"
    assert res2 == "two"


@pytest.mark.asyncio
async def test_stream_raises_acp_error_mid_stream():
    client = ACPClient()
    client._process = _make_fake_process()
    conn = _make_fake_conn(session_id="sess-mid")
    client._conn = conn

    async def mid_stream_prompt(session_id: str, prompt):
        await client._handler.session_update(
            session_id,
            AgentMessageChunk(
                session_update="agent_message_chunk",
                content=TextContentBlock(type="text", text="partial "),
            ),
        )
        await client._handler.session_update(
            session_id,
            ToolCallStart(session_update="tool_call", tool_call_id="tc-mid", title="Run tool"),
        )
        return PromptResponse(stop_reason="cancelled")

    conn.prompt = AsyncMock(side_effect=mid_stream_prompt)

    received = []
    with pytest.raises(ACPError) as exc_info:
        async for chunk in client.chat_completion_stream(
            [ChatMessage(role="user", content="go")], model="m"
        ):
            received.append(chunk)

    assert received == ["partial "]
    assert exc_info.value.code == "acp_tool_call_not_supported"
    conn.cancel.assert_awaited_once_with(session_id="sess-mid")
    conn.close_session.assert_awaited_once_with(session_id="sess-mid")
