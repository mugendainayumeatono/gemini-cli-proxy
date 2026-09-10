"""
Gemini client module

Handles interaction with Gemini CLI tool
"""

import asyncio
import logging
import os
import signal
import time
import uuid
import base64
from typing import List, Optional, AsyncGenerator, Tuple
from .models import ChatMessage
from .config import config

logger = logging.getLogger('gemini_cli_proxy')


def format_exit_status(returncode: Optional[int]) -> str:
    """Format process exit status into a readable string including signal name if terminated."""
    if returncode is None:
        return "still running / unknown"
    if returncode == 0:
        return "exit code 0 (success)"
    if returncode < 0:
        sig_num = -returncode
        try:
            sig_name = signal.Signals(sig_num).name
            return f"terminated by signal {sig_num} ({sig_name})"
        except ValueError:
            return f"terminated by signal {sig_num}"
    return f"exit code {returncode} (failure)"


class GeminiClient:
    """Gemini CLI client"""
    
    def __init__(self):
        self.semaphore = asyncio.Semaphore(config.max_concurrency)
    
    def _simplify_error_message(self, raw_error: str) -> Optional[str]:
        """
        Convert Gemini CLI error messages to more readable user-friendly messages
        
        Args:
            raw_error: Raw error message from Gemini CLI
            
        Returns:
            Simplified error message, or None if the error cannot be recognized
        """
        if not raw_error:
            return None
            
        lower_err = raw_error.lower()
        cmd_name = config.gemini_command
        
        # Check for rate limiting related keywords
        rate_limit_indicators = [
            "code\": 429",
            "code: 429",
            "status 429",
            "status: 429", 
            "ratelimitexceeded",
            "resource_exhausted",
            "resource exhausted",
            "quota exceeded",
            "quota metric",
            "requests per day",
            "requests per minute",
            "rate limit",
            "limit exceeded"
        ]
        
        if any(keyword in lower_err for keyword in rate_limit_indicators):
            return f"{cmd_name} rate limit exceeded. Please run `{cmd_name}` directly to check."
            
        # Authentication / Token error
        auth_indicators = [
            "unauthenticated",
            "invalid authentication",
            "oauth",
            "not logged in",
            "login required",
            "token expired",
            "credentials missing",
            "unauthorized"
        ]
        if any(keyword in lower_err for keyword in auth_indicators):
            return f"{cmd_name} authentication failed. Please run `{cmd_name}` to authenticate."
        
        # Model selection error
        if "invalid model selection" in lower_err or "not recognized as a known model" in lower_err:
            return f"{cmd_name} model error: selected model is not recognized by `{cmd_name}`."
        
        return None

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Execute chat completion request
        
        Args:
            messages: List of chat messages
            model: Model name to use
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Returns:
            Response text from Gemini CLI
            
        Raises:
            asyncio.TimeoutError: Timeout error
            subprocess.CalledProcessError: Command execution error
        """
        async with self.semaphore:
            return await self._execute_gemini_command(
                messages, model, temperature, max_tokens, **kwargs
            )
    
    async def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Execute streaming chat completion request
        
        Args:
            messages: List of chat messages
            model: Model name to use
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Yields:
            Response text chunks
        """
        async with self.semaphore:
            async for chunk in self._execute_gemini_command_stream(
                messages, model, temperature, max_tokens, **kwargs
            ):
                yield chunk
    
    async def _execute_gemini_command_stream(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Execute Gemini CLI command and yield output as it arrives
        
        Args:
            messages: List of chat messages
            model: Model name to use
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Yields:
            Command output chunks
        """
        prompt, temp_files = self._build_prompt_with_images(messages)
        cmd_name = config.gemini_command
        cmd_args = [cmd_name, "-p", prompt]
        
        # Real CLI doesn't support temperature and max_tokens parameters
        if temperature is not None:
            logger.debug(f"Ignoring temperature parameter: {temperature} ({cmd_name} doesn't support)")
        if max_tokens is not None:
            logger.debug(f"Ignoring max_tokens parameter: {max_tokens} ({cmd_name} doesn't support)")
        
        start_time = time.monotonic()
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(
            f"Invoking {cmd_name} command (stream mode): "
            f"model={model}, prompt_len={len(prompt)}, temp_files={len(temp_files)}"
        )
        logger.debug(f"Prompt preview: {prompt_preview}")
        logger.debug(f"Full command args: {' '.join(cmd_args[:2])} [prompt length: {len(prompt)} chars]")
        
        process = None
        stderr_task = None
        chunk_count = 0
        total_bytes = 0
        
        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=os.getcwd()
                )
            except FileNotFoundError as e:
                logger.error(f"{cmd_name} executable not found in PATH: {e}")
                raise RuntimeError(f"{cmd_name} executable not found: ensure {cmd_name} is installed and available in PATH") from e
            except PermissionError as e:
                logger.error(f"Permission denied when executing {cmd_name}: {e}")
                raise RuntimeError(f"Permission denied executing {cmd_name}: check executable permissions") from e

            logger.info(f"{cmd_name} stream process started (PID: {process.pid})")
            
            # Read stderr concurrently in background to avoid OS pipe deadlock
            stderr_task = asyncio.create_task(process.stderr.read())
            
            # Read from stdout incrementally
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        process.stdout.read(1024),
                        timeout=config.timeout
                    )
                    if not chunk:
                        break
                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield chunk.decode('utf-8', errors='replace')
                except asyncio.TimeoutError:
                    duration = time.monotonic() - start_time
                    logger.error(
                        f"{cmd_name} stream command timeout ({config.timeout}s without data) "
                        f"(PID: {process.pid}, elapsed: {duration:.2f}s, chunks: {chunk_count}, bytes: {total_bytes})"
                    )
                    if process and process.returncode is None:
                        try:
                            process.terminate()
                            try:
                                await asyncio.wait_for(process.wait(), timeout=2.0)
                            except asyncio.TimeoutError:
                                process.kill()
                                await process.wait()
                            logger.info(
                                f"{cmd_name} stream process (PID: {process.pid}) terminated after timeout "
                                f"(status: {format_exit_status(process.returncode)})"
                            )
                        except Exception as term_err:
                            logger.warning(f"Error terminating {cmd_name} stream process (PID: {process.pid}): {term_err}")
                    raise RuntimeError(f"{cmd_name} execution timeout ({config.timeout} seconds)")
            
            # Wait for command execution to complete
            await process.wait()
            duration = time.monotonic() - start_time
            exit_code = process.returncode
            status_str = format_exit_status(exit_code)
            
            stderr_bytes = await stderr_task if stderr_task else b""
            stderr_text = stderr_bytes.decode('utf-8', errors='replace').strip() if stderr_bytes else ""
            
            # Check return code
            if exit_code != 0:
                error_msg = stderr_text if stderr_text else f"Process exited with {status_str}"
                simplified_msg = self._simplify_error_message(error_msg)
                
                logger.error(
                    f"{cmd_name} stream command failed with {status_str} "
                    f"(PID: {process.pid}, duration: {duration:.2f}s, chunks: {chunk_count}, bytes: {total_bytes})\n"
                    f"Stderr: {stderr_text or '<empty>'}"
                )
                if simplified_msg:
                    logger.info(f"{cmd_name} stream error recognized: {simplified_msg} [raw {status_str}]")
                    raise RuntimeError(f"{cmd_name} execution failed ({status_str}): {simplified_msg}")
                else:
                    raise RuntimeError(f"{cmd_name} execution failed ({status_str}): {error_msg}")
            
            # exit_code == 0
            logger.info(
                f"{cmd_name} stream command completed successfully with {status_str} "
                f"(PID: {process.pid}, duration: {duration:.2f}s, chunks: {chunk_count}, bytes: {total_bytes})"
            )
            if stderr_text:
                logger.warning(
                    f"{cmd_name} stream process (PID: {process.pid}) exited with code 0 but produced stderr output: {stderr_text}"
                )
            
        except RuntimeError:
            raise
        except Exception as e:
            duration = time.monotonic() - start_time
            pid = process.pid if process else "unknown"
            logger.error(f"Error executing {cmd_name} stream command (PID: {pid}, elapsed: {duration:.2f}s): {e}", exc_info=True)
            raise RuntimeError(f"Error executing {cmd_name} command: {str(e)}") from e
        finally:
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            # Clean up process if still running
            if process and process.returncode is None:
                try:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                    logger.info(f"{cmd_name} stream process (PID: {process.pid}) terminated in cleanup (status: {format_exit_status(process.returncode)})")
                except Exception:
                    pass
                    
            # Clean up temporary files (skip in debug mode)
            if not config.debug:
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file {temp_file}: {e}")

    async def _execute_gemini_command(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Execute Gemini CLI command
        
        Args:
            messages: List of chat messages
            model: Model name to use
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            **kwargs: Other parameters
            
        Returns:
            Command output result
        """
        prompt, temp_files = self._build_prompt_with_images(messages)
        cmd_name = config.gemini_command
        cmd_args = [cmd_name, "-p", prompt]
        
        # Real CLI doesn't support temperature and max_tokens parameters
        if temperature is not None:
            logger.debug(f"Ignoring temperature parameter: {temperature} ({cmd_name} doesn't support)")
        if max_tokens is not None:
            logger.debug(f"Ignoring max_tokens parameter: {max_tokens} ({cmd_name} doesn't support)")
        
        start_time = time.monotonic()
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(
            f"Invoking {cmd_name} command (sync mode): "
            f"model={model}, prompt_len={len(prompt)}, temp_files={len(temp_files)}"
        )
        logger.debug(f"Prompt preview: {prompt_preview}")
        logger.debug(f"Full command args: {' '.join(cmd_args[:2])} [prompt length: {len(prompt)} chars]")
        
        process = None
        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=os.getcwd()
                )
            except FileNotFoundError as e:
                logger.error(f"{cmd_name} executable not found in PATH: {e}")
                raise RuntimeError(f"{cmd_name} executable not found: ensure {cmd_name} is installed and available in PATH") from e
            except PermissionError as e:
                logger.error(f"Permission denied when executing {cmd_name}: {e}")
                raise RuntimeError(f"Permission denied executing {cmd_name}: check executable permissions") from e

            logger.info(f"{cmd_name} process started (PID: {process.pid})")
            
            # Wait for command execution to complete with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout
            )
            duration = time.monotonic() - start_time
            exit_code = process.returncode
            status_str = format_exit_status(exit_code)
            
            stderr_text = stderr.decode('utf-8', errors='replace').strip() if stderr else ""
            stdout_text = stdout.decode('utf-8', errors='replace').strip() if stdout else ""
            
            # Check return code
            if exit_code != 0:
                error_msg = stderr_text if stderr_text else stdout_text
                simplified_msg = self._simplify_error_message(error_msg)
                
                logger.error(
                    f"{cmd_name} command failed with {status_str} (PID: {process.pid}, duration: {duration:.2f}s).\n"
                    f"Stderr: {stderr_text or '<empty>'}\n"
                    + (f"Stdout: {stdout_text}" if stdout_text else "")
                )
                if simplified_msg:
                    logger.info(f"{cmd_name} error recognized: {simplified_msg} [raw {status_str}]")
                    raise RuntimeError(f"{cmd_name} execution failed ({status_str}): {simplified_msg}")
                else:
                    raise RuntimeError(f"{cmd_name} execution failed ({status_str}): {error_msg}")
            
            # exit_code == 0
            logger.info(
                f"{cmd_name} command completed successfully with {status_str} "
                f"(PID: {process.pid}, duration: {duration:.2f}s, output length: {len(stdout_text)} chars)"
            )
            if stderr_text:
                logger.warning(
                    f"{cmd_name} process (PID: {process.pid}) exited with code 0 but produced stderr output: {stderr_text}"
                )
            logger.debug(f"{cmd_name} response content: {stdout_text}")
            return stdout_text
            
        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            pid = process.pid if process else "unknown"
            logger.error(
                f"{cmd_name} command execution timeout ({config.timeout}s) (PID: {pid}, elapsed: {duration:.2f}s)"
            )
            if process and process.returncode is None:
                try:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                    logger.info(f"{cmd_name} process (PID: {pid}) terminated after timeout (status: {format_exit_status(process.returncode)})")
                except Exception as term_err:
                    logger.warning(f"Error terminating {cmd_name} process (PID: {pid}): {term_err}")
            raise RuntimeError(
                f"{cmd_name} execution timeout ({config.timeout} seconds), please retry later or check your network connection"
            ) from None
        except RuntimeError:
            # Re-raise already processed RuntimeError
            raise
        except Exception as e:
            duration = time.monotonic() - start_time
            pid = process.pid if process else "unknown"
            logger.error(f"Error executing {cmd_name} command (PID: {pid}, elapsed: {duration:.2f}s): {e}", exc_info=True)
            raise RuntimeError(f"Error executing {cmd_name} command: {str(e)}") from e
        finally:
            if process and process.returncode is None:
                try:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                    logger.info(f"{cmd_name} process (PID: {process.pid}) terminated in cleanup (status: {format_exit_status(process.returncode)})")
                except Exception:
                    pass
            # Clean up temporary files (skip in debug mode)
            if not config.debug:
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    def _build_prompt_with_images(self, messages: List[ChatMessage]) -> Tuple[str, List[str]]:
        """
        Build prompt text with image processing
        
        Args:
            messages: List of chat messages
            
        Returns:
            Tuple of (formatted prompt text, list of temporary file paths)
        """
        prompt_parts = []
        temp_files = []
        
        for i, message in enumerate(messages):
            if isinstance(message.content, str):
                # Simple string content
                if message.role == "system":
                    prompt_parts.append(f"System: {message.content}")
                elif message.role == "user":
                    prompt_parts.append(f"User: {message.content}")
                elif message.role == "assistant":
                    prompt_parts.append(f"Assistant: {message.content}")
            else:
                # List of content parts (vision support)
                content_parts = []
                
                for j, part in enumerate(message.content):
                    if part.type == "text" and part.text:
                        content_parts.append(part.text)
                    elif part.type == "image_url" and part.image_url:
                        url = part.image_url.get("url", "")
                        if url.startswith("data:"):
                            # Process base64 image
                            temp_file_path = self._save_base64_image(url)
                            temp_files.append(temp_file_path)
                            content_parts.append(f"@{temp_file_path}")
                        else:
                            # For regular URLs, we'll just pass them through for now
                            # TODO: Download and save remote images if needed
                            content_parts.append(f"<image_url>{url}</image_url>")
                
                combined_content = " ".join(content_parts)
                if message.role == "system":
                    prompt_parts.append(f"System: {combined_content}")
                elif message.role == "user":
                    prompt_parts.append(f"User: {combined_content}")
                elif message.role == "assistant":
                    prompt_parts.append(f"Assistant: {combined_content}")

        final_prompt = "\n".join(prompt_parts)
        logger.debug(f"Prompt sent to Gemini CLI: {final_prompt}")
        
        return final_prompt, temp_files
    
    def _save_base64_image(self, data_url: str) -> str:
        """
        Save base64 image data to temporary file
        
        Args:
            data_url: Data URL in format "data:image/type;base64,..."
            
        Returns:
            Path to temporary file
            
        Raises:
            ValueError: Invalid data URL format
        """
        try:
            # Parse data URL
            if not data_url.startswith("data:"):
                raise ValueError("Invalid data URL format")
            
            # Extract MIME type and base64 data
            header, data = data_url.split(",", 1)
            mime_info = header.split(";")[0].split(":")[1]  # e.g., "image/png"
            
            # Determine file extension
            if "png" in mime_info.lower():
                ext = ".png"
            elif "jpeg" in mime_info.lower() or "jpg" in mime_info.lower():
                ext = ".jpg"
            elif "gif" in mime_info.lower():
                ext = ".gif"
            elif "webp" in mime_info.lower():
                ext = ".webp"
            else:
                ext = ".png"  # Default to PNG
            
            # Decode base64 data
            image_data = base64.b64decode(data)
            
            # Create .gemini-cli-proxy directory in project root
            temp_dir = ".gemini-cli-proxy"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create temporary file with simplified name
            filename = f"{uuid.uuid4().hex[:8]}{ext}"
            temp_file_path = os.path.join(temp_dir, filename)
            
            # Write image data
            with open(temp_file_path, 'wb') as f:
                f.write(image_data)
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Error saving base64 image: {e}")
            raise ValueError(f"Failed to save base64 image: {e}") from e

    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """
        Build prompt text (legacy method, kept for compatibility)
        
        Args:
            messages: List of chat messages
            
        Returns:
            Formatted prompt text
        """
        prompt, _ = self._build_prompt_with_images(messages)
        return prompt


# Global client instance
gemini_client = GeminiClient() 