"""
FastAPI server module

Implements HTTP service and API endpoints
"""

import asyncio
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from . import __version__
from .config import config
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    ErrorDetail,
    HealthResponse,
    ModelsResponse,
    ModelInfo
)
from .openai_adapter import openai_adapter

# Configure logging (will be updated when config is set)
logging.basicConfig(
    level=logging.INFO,  # Default level, will be updated in CLI
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gemini_cli_proxy')

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Ensure logging level is applied after uvicorn starts
    logging.getLogger('gemini_cli_proxy').setLevel(getattr(logging, config.log_level.upper()))
    
    logger.info(f"Starting Gemini CLI Proxy v{__version__}")
    logger.info(
        f"Configuration: command={config.gemini_command}, host={config.host}, port={config.port}, "
        f"rate_limit={config.rate_limit}/min, concurrency={config.max_concurrency}, timeout={config.timeout}s"
    )
    logger.debug(f"Debug logging is enabled (log_level={config.log_level})")
    
    # Fetch supported models from CLI tool (e.g. agy models)
    cmd_name = config.gemini_command
    try:
        logger.info(f"Attempting to fetch supported models from '{cmd_name} models'...")
        process = await asyncio.create_subprocess_exec(
            cmd_name,
            "models",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=45.0)
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            raise

        if process.returncode == 0 and stdout:
            fetched_models = []
            for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if not line or "Fetching" in line:
                    continue
                parts = line.split()
                if parts:
                    fetched_models.append(parts[0])

            if fetched_models:
                config.supported_models = fetched_models
                logger.info(f"Successfully updated supported models from {cmd_name}: {len(fetched_models)} models loaded")
            else:
                logger.warning(f"'{cmd_name} models' returned an empty model list, falling back to hardcoded models")
        else:
            stderr_text = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
            logger.warning(
                f"Failed to fetch models from '{cmd_name} models' (exit code: {process.returncode}), using hardcoded models. "
                f"Stderr: {stderr_text or '<empty>'}"
            )
    except asyncio.TimeoutError:
        logger.warning(f"Timed out fetching models from '{cmd_name} models' (timeout: 45s), using hardcoded models")
    except FileNotFoundError:
        logger.warning(f"Command '{cmd_name}' not found in PATH, using hardcoded models")
    except Exception as e:
        logger.error(f"Failed to fetch or parse models from '{cmd_name} models', using hardcoded models. Reason: {e}")
        
    yield
    logger.info("Shutting down Gemini CLI Proxy")


# Create FastAPI application
app = FastAPI(
    title="Gemini CLI Proxy",
    description="OpenAI-compatible API wrapper for Gemini CLI",
    version=__version__,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Security
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from Authorization header"""
    if not credentials or not credentials.credentials:
        logger.warning("Authentication failed: API key missing in Authorization header")
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="API key is missing",
                    type="invalid_request_error",
                    code="missing_api_key"
                )
            ).model_dump()
        )
    
    token = credentials.credentials
    if config.proxy_api_key:
        if token != config.proxy_api_key:
            logger.warning("Authentication failed: Invalid API key provided")
            raise HTTPException(
                status_code=401,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message="Invalid API key",
                        type="invalid_request_error",
                        code="invalid_api_key"
                    )
                ).model_dump()
            )
    return token


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTPException to return OpenAI-compatible error format"""
    # If detail is already an ErrorResponse dict, return it directly
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    # Otherwise wrap it in an OpenAI-compatible format
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                message=str(exc.detail),
                type="invalid_request_error",
                code=str(exc.status_code)
            )
        ).model_dump()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception occurred while processing request: {exc}")
    logger.error(f"Exception details: {traceback.format_exc()}")
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            message="Internal server error",
            type="internal_error",
            code="500"
        )
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(version=__version__)


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models"""
    models = [
        ModelInfo(id=model_id) for model_id in config.supported_models
    ]
    
    return ModelsResponse(data=models)


@app.post("/v1/chat/completions")
@limiter.limit(f"{config.rate_limit}/minute")
async def chat_completions(
    chat_request: ChatCompletionRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Chat completion endpoint
    
    Implements OpenAI-compatible chat completion API
    """
    logger.info(f"Received chat completion request: model={chat_request.model}, stream={chat_request.stream}, messages={len(chat_request.messages)}")
    
    try:
        # Handle streaming request
        if chat_request.stream:
            return await openai_adapter.chat_completion_stream(chat_request)
        
        # Handle non-streaming request
        response = await openai_adapter.chat_completion(chat_request)
        return response
        
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error(f"Chat completion failed: Gemini CLI command execution timeout ({config.timeout}s) for model={chat_request.model}")
        raise HTTPException(
            status_code=504,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Gemini CLI command execution timeout",
                    type="bad_gateway",
                    code="504"
                )
            ).model_dump()
        )
    except RuntimeError as e:
        logger.error(f"Chat completion failed: Gemini CLI execution error for model={chat_request.model}: {e}")
        raise HTTPException(
            status_code=502,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=str(e),
                    type="bad_gateway",
                    code="502"
                )
            ).model_dump()
        )
    except Exception as e:
        logger.error(f"Unexpected error processing chat completion request for model={chat_request.model}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Internal server error",
                    type="internal_error",
                    code="500"
                )
            ).model_dump()
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port) 