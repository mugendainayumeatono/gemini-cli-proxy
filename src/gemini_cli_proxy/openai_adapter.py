"""
OpenAI adapter module

Handles format conversion and compatibility
"""

import time
import uuid
import logging
import json
import asyncio
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatMessage
)
from .gemini_client import gemini_client

logger = logging.getLogger('gemini_cli_proxy')


class OpenAIAdapter:
    """OpenAI format adapter"""
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Handle chat completion request (non-streaming)
        
        Args:
            request: OpenAI format chat completion request
            
        Returns:
            OpenAI format chat completion response
        """
        start_time = time.monotonic()
        logger.info(f"Processing chat completion request: model={request.model}, messages={len(request.messages)}")
        
        try:
            # Call Gemini CLI
            response_text = await gemini_client.chat_completion(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            duration = time.monotonic() - start_time
            
            # Build OpenAI format response
            response = ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=response_text
                        ),
                        finish_reason="stop"
                    )
                ]
            )
            
            logger.info(
                f"Chat completion request processed successfully: "
                f"model={request.model}, duration={duration:.2f}s, response length={len(response_text)}"
            )
            return response
            
        except (asyncio.CancelledError, GeneratorExit):
            duration = time.monotonic() - start_time
            logger.warning(
                f"Chat completion client disconnected / request cancelled: model={request.model}, duration={duration:.2f}s"
            )
            raise
        except Exception as e:
            duration = time.monotonic() - start_time
            logger.error(
                f"Error processing chat completion request: model={request.model}, duration={duration:.2f}s: {e}"
            )
            raise
    
    async def chat_completion_stream(self, request: ChatCompletionRequest) -> StreamingResponse:
        """
        Handle streaming chat completion request
        
        Args:
            request: OpenAI format chat completion request
            
        Returns:
            Streaming response
        """
        logger.info(f"Processing streaming chat completion request: model={request.model}, messages={len(request.messages)}")
        
        async def generate_stream():
            """Generate streaming response data"""
            stream_start_time = time.monotonic()
            chunk_count = 0
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created_time = int(time.time())
            
            # Send initial role chunk for better compatibility
            initial_response = ChatCompletionStreamResponse(
                id=completion_id,
                created=created_time,
                model=request.model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta={"role": "assistant", "content": ""},
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {initial_response.model_dump_json()}\n\n"
            
            try:
                # Get streaming data generator
                stream_generator = gemini_client.chat_completion_stream(
                    messages=request.messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                # Send data chunks one by one
                async for chunk in stream_generator:
                    if not chunk:
                        continue
                    chunk_count += 1
                        
                    stream_response = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=created_time,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"content": chunk},
                                finish_reason=None
                            )
                        ]
                    )
                    
                    # Send data chunk
                    yield f"data: {stream_response.model_dump_json()}\n\n"
                
                # Send end marker
                final_response = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={},
                            finish_reason="stop"
                        )
                    ]
                )
                yield f"data: {final_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                
                stream_duration = time.monotonic() - stream_start_time
                logger.info(
                    f"Streaming chat completion request processed successfully: "
                    f"model={request.model}, duration={stream_duration:.2f}s, chunks={chunk_count}"
                )
                
            except (asyncio.CancelledError, GeneratorExit):
                stream_duration = time.monotonic() - stream_start_time
                logger.warning(
                    f"Streaming chat completion client disconnected / request cancelled for model={request.model} "
                    f"(elapsed: {stream_duration:.2f}s, chunks={chunk_count})"
                )
                raise
            except Exception as e:
                stream_duration = time.monotonic() - stream_start_time
                logger.error(
                    f"Error processing streaming chat completion request for model={request.model} "
                    f"(elapsed: {stream_duration:.2f}s, chunks={chunk_count}): {e}"
                )
                # Send error information in OpenAI format
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "internal_error",
                        "code": "500"
                    }
                }
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )


# Global adapter instance
openai_adapter = OpenAIAdapter() 