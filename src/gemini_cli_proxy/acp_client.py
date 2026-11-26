"""
ACP (Agent Client Protocol) client module.

Manages a persistent background ``agy acp`` subprocess using the official
``agent-client-protocol`` (``acp``) SDK and translates OpenAI-style chat
messages into isolated ACP sessions.
"""

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple

import acp
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
    ImageContentBlock,
    Implementation,
    PermissionOption,
    RequestPermissionResponse,
    SessionConfigSelectGroup,
    SessionInfoUpdate,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UsageUpdate,
    UserMessageChunk,
)

from . import __version__
from .audit import record_audit
from .config import config
from .gemini_client import format_exit_status
from .models import (
    ChatMessage,
    CompletionResult,
    CompletionStreamChunk,
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)

logger = logging.getLogger("gemini_cli_proxy")

_EFFORT_SUFFIXES = ("-xhigh", "-medium", "-high", "-low", "-max")

_BENIGN_UPDATE_TYPES = tuple(
    cls
    for cls in (
        AvailableCommandsUpdate,
        UsageUpdate,
        SessionInfoUpdate,
        ConfigOptionUpdate,
        CurrentModeUpdate,
        UserMessageChunk,
        getattr(acp.schema, "SessionUpdateNotice", None),
        getattr(acp.schema, "SessionUpdateCompactionUpdate", None),
        getattr(acp.schema, "SessionUpdateCompactionSummaryChunk", None),
    )
    if cls is not None
)

_BENIGN_UPDATE_NAMES = {
    "AvailableCommandsUpdate",
    "UsageUpdate",
    "SessionInfoUpdate",
    "ConfigOptionUpdate",
    "CurrentModeUpdate",
    "UserMessageChunk",
    "SessionUpdateNotice",
    "SessionUpdateCompactionUpdate",
    "SessionUpdateCompactionSummaryChunk",
}

_PLAN_UPDATE_TYPES = (
    AgentPlanUpdate,
    AgentPlanContentUpdate,
    AgentPlanRemovedUpdate,
)

_TOOL_CALL_UPDATE_TYPES = (
    ToolCallStart,
    ToolCallProgress,
)


class ACPError(RuntimeError):
    """Error raised when an ACP session or protocol interaction fails."""

    def __init__(self, message: str, code: str = "acp_error"):
        super().__init__(message)
        self.code = code


@dataclass
class _SessionTurnState:
    """Per-session turn state isolating concurrent turns on a shared connection."""

    session_id: str
    queue: asyncio.Queue[Optional[Any]] = field(default_factory=asyncio.Queue)
    abort_error: Optional[ACPError] = None
    saw_text: bool = False
    cancel_sent: bool = False
    usage: Optional[CompletionUsage] = None


def _convert_acp_usage(raw_usage: Any) -> Optional[CompletionUsage]:
    """Convert an ACP PromptResponse usage object or dict into OpenAI-style CompletionUsage."""
    if raw_usage is None:
        return None

    def _pick(snake: str, camel: str) -> Any:
        if isinstance(raw_usage, dict):
            val = raw_usage.get(snake)
            return val if val is not None else raw_usage.get(camel)
        val = getattr(raw_usage, snake, None)
        return val if val is not None else getattr(raw_usage, camel, None)

    input_tokens = _pick("input_tokens", "inputTokens")
    output_tokens = _pick("output_tokens", "outputTokens")
    if input_tokens is None or output_tokens is None:
        return None

    total_tokens = _pick("total_tokens", "totalTokens")
    thought_tokens = _pick("thought_tokens", "thoughtTokens")
    cached_read_tokens = _pick("cached_read_tokens", "cachedReadTokens")

    resolved_input = int(input_tokens)
    resolved_output = int(output_tokens)
    resolved_total = (
        int(total_tokens) if total_tokens is not None else (resolved_input + resolved_output)
    )

    completion_details = (
        CompletionTokensDetails(reasoning_tokens=int(thought_tokens))
        if thought_tokens is not None
        else None
    )
    prompt_details = (
        PromptTokensDetails(cached_tokens=int(cached_read_tokens))
        if cached_read_tokens is not None
        else None
    )
    return CompletionUsage(
        prompt_tokens=resolved_input,
        completion_tokens=resolved_output,
        total_tokens=resolved_total,
        completion_tokens_details=completion_details,
        prompt_tokens_details=prompt_details,
    )


def _strip_effort_suffix(model: str) -> Tuple[str, Optional[str]]:
    """Split trailing reasoning effort suffix (e.g. ``-low``) from a model name."""
    for suffix in _EFFORT_SUFFIXES:
        if model.endswith(suffix) and len(model) > len(suffix):
            return model[: -len(suffix)], suffix[1:]
    return model, None


def _extract_option_values(options: Any) -> Set[str]:
    """Extract selectable option values from flat or grouped config options."""
    values: Set[str] = set()
    if not options:
        return values
    for item in options:
        if isinstance(item, SessionConfigSelectGroup) or hasattr(item, "options"):
            sub_options = getattr(item, "options", None)
            if sub_options is None and isinstance(item, dict):
                sub_options = item.get("options")
            values.update(_extract_option_values(sub_options))
        else:
            val = getattr(item, "value", None)
            if val is None and isinstance(item, dict):
                val = item.get("value")
            if isinstance(val, str):
                values.add(val)
    return values


def _extract_available_model_ids(session: Any) -> Set[str]:
    """Extract model IDs from ``session.models`` if present on the session response."""
    models_state = getattr(session, "models", None)
    if models_state is None:
        return set()
    if isinstance(models_state, dict):
        raw_models = models_state.get("available_models") or models_state.get("availableModels") or []
    else:
        raw_models = (
            getattr(models_state, "available_models", None)
            or getattr(models_state, "availableModels", None)
            or []
        )
    model_ids: Set[str] = set()
    for item in raw_models:
        if isinstance(item, str):
            model_ids.add(item)
        elif isinstance(item, dict):
            mid = item.get("model_id") or item.get("modelId") or item.get("id")
            if isinstance(mid, str):
                model_ids.add(mid)
        else:
            mid = getattr(item, "model_id", None) or getattr(item, "modelId", None) or getattr(item, "id", None)
            if isinstance(mid, str):
                model_ids.add(mid)
    return model_ids


def _build_prompt_blocks(messages: List[ChatMessage]) -> List[TextContentBlock | ImageContentBlock]:
    """Convert OpenAI chat messages into ACP prompt content blocks."""
    role_labels = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    blocks: List[TextContentBlock | ImageContentBlock] = []
    text_lines: List[str] = []

    def flush_text() -> None:
        if text_lines:
            combined = "\n".join(text_lines)
            if combined:
                blocks.append(TextContentBlock(type="text", text=combined))
            text_lines.clear()

    for message in messages:
        label = role_labels.get(message.role, message.role.capitalize())
        if isinstance(message.content, str):
            text_lines.append(f"{label}: {message.content}")
            continue

        inline_segments: List[str] = []
        emitted_prefix = False

        def flush_inline() -> None:
            nonlocal emitted_prefix
            if inline_segments:
                body = " ".join(inline_segments)
                text_lines.append(f"{label}: {body}" if not emitted_prefix else body)
                inline_segments.clear()
                emitted_prefix = True

        for part in message.content:
            if part.type == "text" and part.text:
                inline_segments.append(part.text)
            elif part.type == "image_url" and part.image_url:
                url = part.image_url.get("url", "")
                if url.startswith("data:") and "," in url:
                    header, b64_data = url.split(",", 1)
                    mime_section = header.split(";")[0]
                    if ":" in mime_section and b64_data:
                        mime_type = mime_section.split(":", 1)[1].strip()
                        if mime_type:
                            if not emitted_prefix:
                                if inline_segments:
                                    flush_inline()
                                else:
                                    text_lines.append(f"{label}:")
                                    emitted_prefix = True
                            else:
                                flush_inline()
                            flush_text()
                            blocks.append(
                                ImageContentBlock(type="image", data=b64_data, mime_type=mime_type)
                            )
                            continue
                inline_segments.append(f"<image_url>{url}</image_url>")

        if inline_segments:
            flush_inline()
        elif not emitted_prefix:
            text_lines.append(f"{label}:")

    flush_text()
    if not blocks:
        blocks.append(TextContentBlock(type="text", text=""))
    return blocks


class _ACPClientHandler(acp.Client):
    """Official ``acp.Client`` implementation enforcing plain-text-only ACP turns."""

    def __init__(self, client: "ACPClient"):
        self._client = client

    async def _reject_capability(
        self,
        cap_name: str,
        rpc_method: str,
        session_id: Optional[str],
        detail: str = "",
    ) -> Any:
        suffix = f" ({detail})" if detail else ""
        logger.error(
            f"ACP client capability '{cap_name}' is not supported "
            f"(session_id={session_id or 'unknown'}){suffix}"
        )
        err = ACPError(
            f"ACP client capability '{cap_name}' is not supported",
            code="acp_client_capability_not_supported",
        )
        await self._client._abort_turn_or_active(session_id, err)
        raise RequestError.method_not_found(rpc_method)

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        turn = self._client._turns.get(session_id)
        if turn is None:
            logger.debug(
                f"Ignoring ACP session_update for unregistered session {session_id}: "
                f"{type(update).__name__}"
            )
            return

        if turn.abort_error is not None:
            logger.debug(
                f"Ignoring ACP session_update for already-aborted session {session_id}: "
                f"{type(update).__name__}"
            )
            return

        if isinstance(update, AgentMessageChunk) or (
            getattr(update, "session_update", None) == "agent_message_chunk"
        ):
            content = getattr(update, "content", None)
            if isinstance(content, TextContentBlock) or getattr(content, "type", None) == "text":
                text = getattr(content, "text", "")
                if text:
                    turn.saw_text = True
                    await turn.queue.put(CompletionStreamChunk(content=text))
                return

            content_desc = getattr(content, "type", None) or type(content).__name__
            logger.error(
                f"ACP non-text content received in session {session_id}: content={content_desc}"
            )
            err = ACPError(
                f"ACP non-text content is not supported: {content_desc}",
                code="acp_non_text_content",
            )
            await self._client._abort_turn(session_id, err)
            return

        if isinstance(update, AgentThoughtChunk) or (
            getattr(update, "session_update", None) == "agent_thought_chunk"
        ):
            content = getattr(update, "content", None)
            if isinstance(content, TextContentBlock) or getattr(content, "type", None) == "text":
                text = getattr(content, "text", "")
                if text:
                    await turn.queue.put(CompletionStreamChunk(reasoning_content=text))
                return

            content_desc = getattr(content, "type", None) or type(content).__name__
            logger.debug(
                f"Ignoring non-text ACP thought content in session {session_id}: "
                f"content={content_desc}"
            )
            return

        if isinstance(update, _TOOL_CALL_UPDATE_TYPES):
            title = getattr(update, "title", None)
            tool_call_id = getattr(update, "tool_call_id", None)
            logger.error(
                f"ACP tool call not supported in session {session_id}: "
                f"type={type(update).__name__}, tool_call_id={tool_call_id}, title={title!r}"
            )
            err = ACPError(
                f"ACP tool calls are not supported ({title or type(update).__name__})",
                code="acp_tool_call_not_supported",
            )
            await self._client._abort_turn(session_id, err)
            return

        if isinstance(update, _PLAN_UPDATE_TYPES):
            logger.error(
                f"ACP plan update not supported in session {session_id}: {type(update).__name__}"
            )
            err = ACPError(
                f"ACP plan updates are not supported: {type(update).__name__}",
                code="acp_non_text_content",
            )
            await self._client._abort_turn(session_id, err)
            return

        if isinstance(update, _BENIGN_UPDATE_TYPES) or type(update).__name__ in _BENIGN_UPDATE_NAMES:
            logger.debug(
                f"Ignoring ACP session update in session {session_id}: {type(update).__name__}"
            )
            return

        logger.error(
            f"ACP unrecognized non-metadata update in session {session_id}: "
            f"{type(update).__name__}"
        )
        err = ACPError(
            f"ACP unsupported session update: {type(update).__name__}",
            code="acp_non_text_content",
        )
        await self._client._abort_turn(session_id, err)

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: List[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        tool_title = getattr(tool_call, "title", None)
        option_names = [
            getattr(opt, "name", None) or getattr(opt, "option_id", None) or str(opt)
            for opt in (options or [])
        ]
        logger.error(
            f"ACP permission request denied in session {session_id}: "
            f"tool_call={tool_title!r}, options={option_names}"
        )
        err = ACPError(
            f"ACP permission request is not supported (tool: {tool_title or 'unknown'})",
            code="acp_permission_required",
        )
        await self._client._abort_turn(session_id, err)
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def write_text_file(
        self, session_id: str, path: str, content: str, **kwargs: Any
    ) -> Any:
        return await self._reject_capability(
            "write_text_file", "fs/write_text_file", session_id, f"path={path!r}"
        )

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        return await self._reject_capability(
            "read_text_file", "fs/read_text_file", session_id, f"path={path!r}"
        )

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[List[Any]] = None,
        cwd: Optional[str] = None,
        output_byte_limit: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        return await self._reject_capability(
            "create_terminal", "terminal/create", session_id, f"command={command!r}"
        )

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> Any:
        return await self._reject_capability(
            "terminal_output", "terminal/output", session_id, f"terminal_id={terminal_id!r}"
        )

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> Any:
        return await self._reject_capability(
            "release_terminal", "terminal/release", session_id, f"terminal_id={terminal_id!r}"
        )

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> Any:
        return await self._reject_capability(
            "wait_for_terminal_exit",
            "terminal/wait_for_exit",
            session_id,
            f"terminal_id={terminal_id!r}",
        )

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> Any:
        return await self._reject_capability(
            "kill_terminal", "terminal/kill", session_id, f"terminal_id={terminal_id!r}"
        )

    async def create_elicitation(
        self, message: str, mode: Any, **kwargs: Any
    ) -> DeclineElicitationResponse:
        session_id = getattr(mode, "session_id", None) or kwargs.get("session_id")
        logger.error(
            f"ACP client capability 'create_elicitation' is not supported "
            f"(session_id={session_id or 'unknown'}, message={message!r})"
        )
        err = ACPError(
            "ACP client capability 'create_elicitation' is not supported",
            code="acp_client_capability_not_supported",
        )
        await self._client._abort_turn_or_active(session_id, err)
        return DeclineElicitationResponse(action="decline")

    async def connect_mcp(self, *args: Any, **kwargs: Any) -> Any:
        session_id = kwargs.get("session_id")
        return await self._reject_capability("connect_mcp", "mcp/connect", session_id)

    async def mcp_message(self, *args: Any, **kwargs: Any) -> Any:
        session_id = kwargs.get("session_id")
        return await self._reject_capability("mcp_message", "mcp/message", session_id)

    async def ext_method(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        session_id = None
        if isinstance(params, dict):
            session_id = params.get("sessionId") or params.get("session_id")
        return await self._reject_capability(
            "ext_method", f"_{method}", session_id, f"method={method!r}"
        )


class ACPClient:
    """Persistent ACP client managing a background ``agy acp`` process."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._process: Optional[asyncio.subprocess.Process] = None
        self._conn: Optional[Any] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._handler = _ACPClientHandler(self)
        self._turns: Dict[str, _SessionTurnState] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_limit: Optional[int] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        limit = max(1, int(config.max_concurrency))
        if self._semaphore is None or self._semaphore_limit != limit:
            self._semaphore = asyncio.Semaphore(limit)
            self._semaphore_limit = limit
        return self._semaphore

    async def _drain_stderr(self, stderr: asyncio.StreamReader, pid: Optional[int]) -> None:
        cmd_name = config.gemini_command
        try:
            while True:
                line = await stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._stderr_tail.append(text)
                    logger.debug(f"[{cmd_name} acp PID:{pid} stderr] {text}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"ACP stderr reader stopped for PID {pid}: {e}")

    def _make_audit_observer(self, pid: Optional[int]) -> Callable[[Any], None]:
        """Create a stream observer that records raw ACP JSON-RPC payloads to the audit log."""

        def _observe(event: Any) -> None:
            raw_dir = getattr(event, "direction", None)
            direction_val = getattr(raw_dir, "value", raw_dir)
            direction = "input" if direction_val == "outgoing" else "output"
            message = getattr(event, "message", None)
            session_id: Optional[str] = None
            if isinstance(message, dict):
                params = message.get("params")
                result = message.get("result")
                if isinstance(params, dict):
                    sid = params.get("sessionId") or params.get("session_id")
                    if isinstance(sid, str):
                        session_id = sid
                if session_id is None and isinstance(result, dict):
                    sid = result.get("sessionId") or result.get("session_id")
                    if isinstance(sid, str):
                        session_id = sid
            record_audit(direction, message, mode="acp", pid=pid, session_id=session_id)

        return _observe

    async def _start_unlocked(self) -> None:
        if self._conn is not None and self._process is not None and self._process.returncode is None:
            return

        if self._process is not None or self._conn is not None:
            await self._stop_unlocked()

        cmd_name = config.gemini_command
        self._stderr_tail.clear()
        logger.info(f"Starting persistent ACP agent process: {cmd_name} acp")

        try:
            proc = await asyncio.create_subprocess_exec(
                cmd_name,
                "acp",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
                cwd=os.getcwd(),
                limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES,
            )
        except FileNotFoundError as e:
            logger.error(f"{cmd_name} executable not found in PATH when starting ACP mode: {e}")
            raise ACPError(
                f"{cmd_name} executable not found: ensure {cmd_name} is installed and available in PATH",
                code="acp_protocol_error",
            ) from e
        except PermissionError as e:
            logger.error(f"Permission denied when starting '{cmd_name} acp': {e}")
            raise ACPError(
                f"Permission denied executing {cmd_name} acp: check executable permissions",
                code="acp_protocol_error",
            ) from e
        except Exception as e:
            logger.error(f"Failed to spawn '{cmd_name} acp': {e}", exc_info=True)
            raise ACPError(
                f"Failed to start {cmd_name} acp process: {e}",
                code="acp_protocol_error",
            ) from e

        self._process = proc
        if proc.stderr is not None:
            self._stderr_task = asyncio.create_task(self._drain_stderr(proc.stderr, proc.pid))

        try:
            if config.audit_enabled:
                conn = acp.connect_to_agent(
                    self._handler,
                    proc.stdin,
                    proc.stdout,
                    observers=[self._make_audit_observer(proc.pid)],
                )
            else:
                conn = acp.connect_to_agent(self._handler, proc.stdin, proc.stdout)
            self._conn = conn
            init_resp = await asyncio.wait_for(
                conn.initialize(
                    protocol_version=PROTOCOL_VERSION,
                    client_capabilities=ClientCapabilities(),
                    client_info=Implementation(name="gemini-cli-proxy", version=__version__),
                ),
                timeout=config.timeout,
            )
            agent_info = getattr(init_resp, "agent_info", None)
            agent_name = getattr(agent_info, "name", "unknown")
            agent_ver = getattr(agent_info, "version", "unknown")
            logger.info(
                f"ACP connection initialized (PID: {proc.pid}, protocol={getattr(init_resp, 'protocol_version', PROTOCOL_VERSION)}, "
                f"agent={agent_name} {agent_ver})"
            )
        except asyncio.TimeoutError as e:
            stderr_ctx = " | ".join(self._stderr_tail)
            logger.error(
                f"ACP initialize timed out after {config.timeout}s (PID: {proc.pid})"
                + (f"; stderr: {stderr_ctx}" if stderr_ctx else "")
            )
            await self._stop_unlocked()
            raise ACPError(
                f"ACP initialize timed out after {config.timeout}s",
                code="acp_protocol_error",
            ) from e
        except RequestError as e:
            stderr_ctx = " | ".join(self._stderr_tail)
            logger.error(
                f"ACP initialize failed with protocol error (PID: {proc.pid}): {e} (code={e.code})"
                + (f"; stderr: {stderr_ctx}" if stderr_ctx else "")
            )
            await self._stop_unlocked()
            raise ACPError(
                f"ACP initialize protocol error: {e}",
                code="acp_protocol_error",
            ) from e
        except Exception as e:
            stderr_ctx = " | ".join(self._stderr_tail)
            logger.error(
                f"ACP initialize failed unexpectedly (PID: {proc.pid}): {e}"
                + (f"; stderr: {stderr_ctx}" if stderr_ctx else ""),
                exc_info=True,
            )
            await self._stop_unlocked()
            raise ACPError(
                f"ACP initialize failed: {e}",
                code="acp_protocol_error",
            ) from e

    async def _stop_unlocked(self) -> None:
        conn = self._conn
        proc = self._process
        stderr_task = self._stderr_task

        self._conn = None
        self._process = None
        self._stderr_task = None

        # Unblock any in-flight turns so waiters do not hang on shutdown.
        stop_err = ACPError("ACP client stopped", code="acp_protocol_error")
        for turn in list(self._turns.values()):
            if turn.abort_error is None:
                turn.abort_error = stop_err
                turn.queue.put_nowait(stop_err)

        if conn is not None:
            try:
                await asyncio.wait_for(conn.close(), timeout=2.0)
            except Exception as e:
                logger.debug(f"Ignoring error closing ACP connection: {e}")

        if proc is not None:
            stdin = getattr(proc, "stdin", None)
            if stdin is not None:
                try:
                    stdin.close()
                    wait_closed = getattr(stdin, "wait_closed", None)
                    if callable(wait_closed):
                        await asyncio.wait_for(wait_closed(), timeout=1.0)
                except Exception:
                    pass

            if proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        proc.kill()
                        await proc.wait()
                except Exception as e:
                    logger.warning(f"Error terminating ACP process (PID: {proc.pid}): {e}")

            logger.info(
                f"ACP agent process (PID: {proc.pid}) stopped "
                f"(status: {format_exit_status(proc.returncode)})"
            )

        if stderr_task is not None:
            if not stderr_task.done():
                stderr_task.cancel()
            try:
                await stderr_task
            except (asyncio.CancelledError, Exception):
                pass

    async def start(self) -> None:
        """Start the persistent ``agy acp`` subprocess and initialize ACP."""
        async with self._lock:
            await self._start_unlocked()

    async def stop(self) -> None:
        """Gracefully close the ACP connection and terminate the subprocess."""
        async with self._lock:
            await self._stop_unlocked()

    async def _ensure_connection(self) -> Any:
        """Return an active ACP connection, restarting if the process has exited."""
        async with self._lock:
            if (
                self._conn is None
                or self._process is None
                or self._process.returncode is not None
            ):
                if self._process is not None and self._process.returncode is not None:
                    logger.warning(
                        f"ACP agent process (PID: {self._process.pid}) exited unexpectedly "
                        f"({format_exit_status(self._process.returncode)}); restarting"
                    )
                await self._start_unlocked()
            return self._conn

    async def _safe_cancel(self, conn: Any, session_id: str) -> None:
        if conn is None:
            return
        try:
            await asyncio.wait_for(conn.cancel(session_id=session_id), timeout=2.0)
        except Exception as e:
            logger.debug(f"Failed to send ACP session/cancel for {session_id}: {e}")

    async def _safe_close_session(self, conn: Any, session_id: str) -> None:
        if conn is None:
            return
        try:
            await asyncio.wait_for(conn.close_session(session_id=session_id), timeout=2.0)
        except Exception as e:
            logger.debug(f"Failed to close ACP session {session_id}: {e}")

    async def _abort_turn(self, session_id: str, err: ACPError) -> None:
        turn = self._turns.get(session_id)
        if turn is None:
            return
        if turn.abort_error is None:
            turn.abort_error = err
            turn.queue.put_nowait(err)
        if not turn.cancel_sent and self._conn is not None:
            turn.cancel_sent = True
            await self._safe_cancel(self._conn, session_id)

    async def _abort_turn_or_active(self, session_id: Optional[str], err: ACPError) -> None:
        if session_id and session_id in self._turns:
            await self._abort_turn(session_id, err)
            return
        for active_sid in list(self._turns.keys()):
            await self._abort_turn(active_sid, err)

    async def _configure_session(self, conn: Any, session: Any, model: str) -> None:
        session_id = session.session_id
        available_model_ids = _extract_available_model_ids(session)
        config_options = getattr(session, "config_options", None) or []

        model_opt = next((opt for opt in config_options if getattr(opt, "id", None) == "model"), None)
        thought_opt = next(
            (opt for opt in config_options if getattr(opt, "id", None) == "thought_level"), None
        )
        model_values = _extract_option_values(getattr(model_opt, "options", None)) if model_opt else set()
        thought_values = (
            _extract_option_values(getattr(thought_opt, "options", None)) if thought_opt else set()
        )
        base_model, model_suffix = _strip_effort_suffix(model)

        try:
            if model in available_model_ids:
                await conn._conn.send_request(
                    "session/set_model",
                    {"sessionId": session_id, "modelId": model},
                )
            elif model in model_values:
                await conn.set_config_option(config_id="model", session_id=session_id, value=model)
            elif model_suffix is not None and base_model in model_values:
                await conn.set_config_option(
                    config_id="model", session_id=session_id, value=base_model
                )
                if model_suffix in thought_values:
                    await conn.set_config_option(
                        config_id="thought_level", session_id=session_id, value=model_suffix
                    )
            else:
                logger.warning(
                    f"Requested model {model!r} did not match ACP session models/config_options "
                    f"(session_id={session_id}); using session default"
                )
        except Exception as e:
            logger.debug(
                f"Ignoring non-fatal ACP model configuration error for {model!r} "
                f"(session_id={session_id}): {e}"
            )

        effort = getattr(config, "effort", None)
        if effort and model_suffix is None and thought_opt is not None and effort in thought_values:
            try:
                await conn.set_config_option(
                    config_id="thought_level", session_id=session_id, value=effort
                )
            except Exception as e:
                logger.debug(
                    f"Ignoring non-fatal ACP thought_level configuration error for {effort!r} "
                    f"(session_id={session_id}): {e}"
                )

    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[CompletionStreamChunk, None]:
        """Execute a streaming ACP chat completion turn and yield completion chunks."""
        if temperature is not None:
            logger.debug(f"Ignoring temperature parameter in ACP mode: {temperature}")
        if max_tokens is not None:
            logger.debug(f"Ignoring max_tokens parameter in ACP mode: {max_tokens}")

        async with self._get_semaphore():
            conn = await self._ensure_connection()
            prompt_blocks = _build_prompt_blocks(messages)
            start_time = time.monotonic()

            try:
                session = await asyncio.wait_for(
                    conn.new_session(cwd=os.getcwd(), mcp_servers=[]),
                    timeout=config.timeout,
                )
            except asyncio.TimeoutError:
                logger.error(f"ACP session/new timed out after {config.timeout}s for model={model}")
                raise
            except RequestError as e:
                logger.error(f"ACP session/new protocol error for model={model}: {e}")
                raise ACPError(f"ACP session creation failed: {e}", code="acp_protocol_error") from e
            except Exception as e:
                logger.error(f"ACP session/new failed for model={model}: {e}", exc_info=True)
                raise ACPError(f"ACP session creation failed: {e}", code="acp_protocol_error") from e

            session_id = session.session_id
            logger.info(
                f"ACP turn started (session_id={session_id}, model={model}, blocks={len(prompt_blocks)})"
            )
            await self._configure_session(conn, session, model)

            turn = _SessionTurnState(session_id=session_id)
            self._turns[session_id] = turn

            prompt_task = asyncio.create_task(
                conn.prompt(session_id=session_id, prompt=prompt_blocks)
            )

            def _on_prompt_done(task: asyncio.Task[Any]) -> None:
                if task.cancelled():
                    turn.queue.put_nowait(None)
                    return
                exc = task.exception()
                if exc is not None:
                    if turn.abort_error is not None:
                        turn.queue.put_nowait(turn.abort_error)
                    elif isinstance(exc, ACPError):
                        turn.queue.put_nowait(exc)
                    elif isinstance(exc, RequestError):
                        logger.error(
                            f"ACP prompt RequestError in session {session_id}: "
                            f"{exc} (code={exc.code}, data={exc.data})"
                        )
                        turn.queue.put_nowait(
                            ACPError(f"ACP protocol error: {exc}", code="acp_protocol_error")
                        )
                    else:
                        logger.error(
                            f"ACP prompt failed in session {session_id}: {exc}", exc_info=True
                        )
                        turn.queue.put_nowait(
                            ACPError(f"ACP protocol error: {exc}", code="acp_protocol_error")
                        )
                else:
                    turn.usage = _convert_acp_usage(getattr(task.result(), "usage", None))
                    turn.queue.put_nowait(None)

            prompt_task.add_done_callback(_on_prompt_done)

            needs_cancel = False
            try:
                while True:
                    item = await asyncio.wait_for(turn.queue.get(), timeout=config.timeout)
                    if isinstance(item, ACPError):
                        raise turn.abort_error or item
                    if item is None:
                        if turn.abort_error is not None:
                            raise turn.abort_error
                        if not turn.saw_text:
                            logger.error(
                                f"ACP prompt completed with empty response (session_id={session_id})"
                            )
                            raise ACPError(
                                "ACP agent returned empty response",
                                code="acp_empty_response",
                            )
                        if turn.usage is not None:
                            yield CompletionStreamChunk(usage=turn.usage)
                        break
                    yield item

                duration = time.monotonic() - start_time
                logger.info(
                    f"ACP turn completed successfully (session_id={session_id}, model={model}, "
                    f"duration={duration:.2f}s)"
                )
            except (asyncio.CancelledError, GeneratorExit):
                needs_cancel = True
                logger.warning(
                    f"ACP turn cancelled / client disconnected (session_id={session_id}, model={model})"
                )
                raise
            except asyncio.TimeoutError:
                needs_cancel = True
                logger.error(
                    f"ACP turn timed out after {config.timeout}s (session_id={session_id}, model={model})"
                )
                raise
            except ACPError:
                needs_cancel = True
                raise
            finally:
                self._turns.pop(session_id, None)
                if not prompt_task.done():
                    needs_cancel = True
                    prompt_task.cancel()
                    try:
                        await asyncio.wait_for(prompt_task, timeout=1.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        pass

                async def _cleanup() -> None:
                    if (needs_cancel or turn.abort_error is not None) and not turn.cancel_sent:
                        turn.cancel_sent = True
                        await self._safe_cancel(conn, session_id)
                    await self._safe_close_session(conn, session_id)

                try:
                    await asyncio.shield(_cleanup())
                except BaseException:
                    pass

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Execute a synchronous ACP chat completion turn and return the aggregated result."""
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        usage: Optional[CompletionUsage] = None
        async for chunk in self.chat_completion_stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            if chunk.reasoning_content:
                reasoning_parts.append(chunk.reasoning_content)
            if chunk.content:
                content_parts.append(chunk.content)
            if chunk.usage is not None:
                usage = chunk.usage
        return CompletionResult(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts) or None,
            usage=usage,
        )


acp_client = ACPClient()
