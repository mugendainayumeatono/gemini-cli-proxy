"""
CLI entry module

Uses Click to handle command line arguments and application startup
"""

import click
import uvicorn


@click.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Server host address"
)
@click.option(
    "--port",
    default=8765,
    type=int,
    help="Server port"
)
@click.option(
    "--rate-limit",
    default=60,
    type=int,
    help="Maximum requests per minute"
)
@click.option(
    "--max-concurrency",
    default=4,
    type=int,
    help="Maximum concurrent subprocesses"
)
@click.option(
    "--timeout",
    default=120.0,
    type=float,
    help="Gemini CLI command timeout in seconds"
)
@click.option(
    "--api-key",
    envvar="PROXY_API_KEY",
    help="API key required for client requests. If not set, any non-empty key in Authorization header is accepted."
)
@click.option(
    "--gemini-command",
    envvar="GEMINI_COMMAND",
    default="agy",
    help="CLI command path to execute (default: agy)"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode"
)
def main(
    host: str,
    port: int,
    rate_limit: int,
    max_concurrency: int,
    timeout: float,
    api_key: str,
    gemini_command: str,
    debug: bool
):
    """Start Gemini CLI Proxy server"""
    
    # Set configuration
    import os
    from .config import config
    
    # Set environment variable for reload mode
    os.environ['GEMINI_CLI_PROXY_DEBUG'] = str(debug)
    if api_key:
        os.environ['PROXY_API_KEY'] = api_key
    os.environ['PROXY_HOST'] = host
    os.environ['PROXY_PORT'] = str(port)
    os.environ['PROXY_RATE_LIMIT'] = str(rate_limit)
    os.environ['PROXY_MAX_CONCURRENCY'] = str(max_concurrency)
    os.environ['GEMINI_TIMEOUT'] = str(timeout)
    os.environ['GEMINI_COMMAND'] = gemini_command
    
    config.host = host
    config.port = port
    config.log_level = "debug" if debug else "info"
    config.rate_limit = rate_limit
    config.max_concurrency = max_concurrency
    config.timeout = timeout
    config.gemini_command = gemini_command
    config.debug = debug
    config.proxy_api_key = api_key
    
    # Update logging level based on configuration
    import logging
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
    # Also set level for all gemini_cli_proxy loggers
    logging.getLogger('gemini_cli_proxy').setLevel(getattr(logging, config.log_level.upper()))
    
    # Start server
    uvicorn.run(
        "gemini_cli_proxy.server:app",
        host=host,
        port=port,
        log_level=config.log_level,
        reload=debug
    )


if __name__ == "__main__":
    main() 