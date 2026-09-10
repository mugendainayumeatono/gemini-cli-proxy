"""
Configuration management module

Manages application configuration
"""


from typing import Optional

class Config:
    """Application configuration class"""
    
    def __init__(self):
        import os
        
        # Server configuration
        self.host: str = os.environ.get('PROXY_HOST', "127.0.0.1")
        self.port: int = int(os.environ.get('PROXY_PORT', '8765'))
        # Read from environment variable if available (for reload mode)
        self.debug: bool = os.environ.get('GEMINI_CLI_PROXY_DEBUG', 'false').lower() == 'true'
        self.log_level: str = "debug" if self.debug else "info"
        
        # Gemini CLI configuration
        self.gemini_command: str = os.environ.get('GEMINI_COMMAND', 'agy')  # Gemini CLI command path
        self.timeout: float = float(os.environ.get('GEMINI_TIMEOUT', '120.0'))
        
        # Policy configuration
        self.use_no_tools_policy: bool = os.environ.get('USE_NO_TOOLS_POLICY', 'false').lower() == 'true'
        self.policy_path: str = os.environ.get('GEMINI_POLICY_PATH', '/app/policies/no-tools.json')
        
        # Limit configuration
        self.rate_limit: int = int(os.environ.get('PROXY_RATE_LIMIT', '60'))  # Requests per minute
        self.max_concurrency: int = int(os.environ.get('PROXY_MAX_CONCURRENCY', '4'))  # Maximum concurrent subprocesses
        
        # Proxy API Key for client authentication
        self.proxy_api_key: Optional[str] = os.environ.get('PROXY_API_KEY')
        
        # API Key for fetching models
        self.api_key: Optional[str] = os.environ.get('GEMINI_API_KEY')
        
        # Supported models list
        self.supported_models: list = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ]


# Global configuration instance
config = Config() 