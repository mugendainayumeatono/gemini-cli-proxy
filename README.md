# Gemini CLI Proxy

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Mentioned in Awesome Gemini CLI](https://awesome.re/mentioned-badge.svg)](https://github.com/Piebald-AI/awesome-gemini-cli)

Wrap Gemini CLI as an OpenAI-compatible API service, allowing you to enjoy the free Gemini 2.5 Pro model through API!

[English](./README.md) | [简体中文](./README_zh.md)

## ✨ Features

- 🔌 **OpenAI API Compatible**: Implements `/v1/chat/completions` endpoint
- 🚀 **Quick Setup**: Zero-config run with `uvx`
- ⚡ **High Performance**: Built on FastAPI + asyncio with concurrent request support

If you want to know the principle of this tool, you can read [my blog post](https://www.nettee.io/blog/gemini-cli-proxy) (in Chinese).

## 🚀 Quick Start

### Network Configuration

Since Gemini needs to access Google services, you may need to configure terminal proxy in certain network environments:

```bash
# Configure proxy (adjust according to your proxy server)
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890  
export all_proxy=socks5://127.0.0.1:7890
```

### Install Antigravity CLI (agy)

The proxy now relies on `antigravity-cli` (`agy`) as its underlying engine. Make sure `agy` is installed in your environment.

You need to start it once first for login and initial configuration.

After configuration is complete, please confirm you can successfully run the following command:

```bash
agy -p "Hello, Gemini"
```

## 🐳 Docker Deployment

The proxy provides Docker deployment with a robust interactive authorization flow.

1. **Build the image**:
   ```bash
   ./build.sh
   ```
2. **Start the service**:
   ```bash
   docker compose up -d
   ```
3. **Interactive Authorization**:
   Since `agy` requires an initial interactive login, the container will run in a waiting state if not authorized. Use the provided script to enter the container and complete the login:
   ```bash
   ./enter.sh
   # Once inside the container's bash shell, run:
   agy
   ```
   After finishing the login, restart the container (`docker compose restart`).

### Start Gemini CLI Proxy

Method 1: Direct startup
```bash
uvx gemini-cli-proxy
```

Method 2: Clone this repository and run:
```bash
uv run gemini-cli-proxy
```

Gemini CLI Proxy listens on port `8765` by default. You can customize the startup port with the `--port` parameter.

After startup, test the service with curl:

```bash
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gemini-2.5-pro",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Usage Examples

#### OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8765/v1',
    api_key='your-api-key'  # Must match the server configuration
)

response = client.chat.completions.create(
    model='gemini-2.5-pro',
    messages=[
        {'role': 'user', 'content': 'Hello!'}
    ],
)

print(response.choices[0].message.content)
```

#### Cherry Studio

Add Model Provider in Cherry Studio settings:
- Provider Type: OpenAI
- API Host: `http://localhost:8765`
- API Key: Any string works
- Model Name: `gemini-3.8-flash-high`, `gemini-3.7-flash-high`, or any model returned by `/v1/models`

![Cherry Studio Config 1](./img/cherry-studio-1.jpg)

![Cherry Studio Config 2](./img/cherry-studio-2.jpg)

## ⚙️ Configuration Options

### Environment Variables

- `PROXY_API_KEY`: API key for client authentication.
- `GEMINI_COMMAND`: CLI command path to execute (default: `agy`).

*Note: During startup, the proxy automatically executes `agy models` to dynamically discover supported models for the `/v1/models` endpoint without needing any external Gemini API keys.*

### Command Line Arguments

View command line parameters:

```bash
gemini-cli-proxy --help
```

Available options:
- `--host`: Server host address (default: 127.0.0.1)
- `--port`: Server port (default: 8765)
- `--api-key`: API key required for client requests. (Can also be set via `PROXY_API_KEY` env var)
- `--rate-limit`: Max requests per minute (default: 60)
- `--max-concurrency`: Max concurrent subprocesses (default: 4)
- `--timeout`: CLI command timeout in seconds (default: 30.0)
- `--debug`: Enable debug mode (enables debug logging and file watching)

## ❓ FAQ

### Q: Why do requests keep timing out?

A: This is usually a network connectivity issue. Gemini needs to access Google services, which may require proxy configuration in certain regions:

```bash
# Configure proxy (adjust according to your proxy server)
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890

# Then start the service
uvx gemini-cli-proxy
```

## 📄 License

MIT License

## 🤝 Contributing

Issues and Pull Requests are welcome!
