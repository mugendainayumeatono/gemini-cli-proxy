"""
Data model definition module

Uses Pydantic to define OpenAI format request and response models
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer
import time
import uuid


class CompletionTokensDetails(BaseModel):
    """Detailed breakdown of completion tokens"""
    reasoning_tokens: int = 0


class PromptTokensDetails(BaseModel):
    """Detailed breakdown of prompt tokens"""
    cached_tokens: int = 0


class CompletionUsage(BaseModel):
    """Token usage statistics for a completion"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[CompletionTokensDetails] = None
    prompt_tokens_details: Optional[PromptTokensDetails] = None

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        data = handler(self)
        if data.get("completion_tokens_details") is None:
            data.pop("completion_tokens_details", None)
        if data.get("prompt_tokens_details") is None:
            data.pop("prompt_tokens_details", None)
        return data


@dataclass
class CompletionStreamChunk:
    """Internal streaming chunk yielded by backend clients"""
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    usage: Optional[CompletionUsage] = None


@dataclass
class CompletionResult:
    """Internal non-streaming completion result returned by backend clients"""
    content: str
    reasoning_content: Optional[str] = None
    usage: Optional[CompletionUsage] = None


class ChatContentPart(BaseModel):
    """Chat content part model for vision support"""
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None  # {"url": "..."}


class ChatMessage(BaseModel):
    """Chat message model"""
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ChatContentPart]]
    reasoning_content: Optional[str] = None

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        data = handler(self)
        if not data.get("reasoning_content"):
            data.pop("reasoning_content", None)
        return data


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice model"""
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"] = "stop"


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[CompletionUsage] = None

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        data = handler(self)
        if data.get("usage") is None:
            data.pop("usage", None)
        return data


class ChatCompletionStreamChoice(BaseModel):
    """Streaming chat completion choice model"""
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class ChatCompletionStreamResponse(BaseModel):
    """Streaming chat completion response model"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
    usage: Optional[CompletionUsage] = None

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        data = handler(self)
        if data.get("usage") is None:
            data.pop("usage", None)
        return data


class ErrorDetail(BaseModel):
    """Error detail model"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: ErrorDetail


class ModelInfo(BaseModel):
    """Model information model"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "gemini-cli-proxy"


class ModelsResponse(BaseModel):
    """Models list response model"""
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = "ok"
    version: str
    timestamp: int = Field(default_factory=lambda: int(time.time())) 