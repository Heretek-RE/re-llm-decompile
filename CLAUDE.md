# re-llm-decompile

MCP server exposing LLM4Decompile / vLLM / Ollama OpenAI-compatible endpoints for AI decompilation.

Version: 0.1.0 | License: MIT

## Structure

```
re-llm-decompile/
  pyproject.toml                    # build config (setuptools, mcp[cli] + deps)
  src/re_llm_decompile/
    __init__.py
    __main__.py                     # entry: from server import main; main()
    server.py                       # FastMCP app with @mcp.tool() functions
  README.md
  LICENSE
  SECURITY.md


```

## Build

```bash
pip install -e .                    # install with deps
re-llm-decompile                         # start MCP server on stdio
```



## Tools

This server exposes these MCP tools: `check_endpoint,decompile_function,explain_function,rename_variables,summarize_binary`

## Usage (standalone)

Register this server in your `.mcp.json`:

```json
{
  "mcpServers": {
    "re-llm-decompile": {
      "command": "uv",
      "args": ["--directory", "/path/to/re-llm-decompile", "run", "re-llm-decompile"]
    }
  }
}
```

Or use via the [RE-AI agent-space](https://github.com/Heretek-RE/RE-AI): `./install.sh` clones all servers at pinned versions.
