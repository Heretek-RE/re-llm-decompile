# re-llm-decompile

MCP server that exposes an **AI decompiler** via the OpenAI-compatible `/v1/chat/completions` API. Works with:

- **[LLM4Decompile](https://github.com/albertan017/LLM4Decompile)** (the `End` or `Ref` variants, or the latest SK²Decompile) served via vLLM
- **[Ollama](https://ollama.com/)** running any code model (CodeLlama, DeepSeek-Coder, Qwen2.5-Coder, etc.)
- Any other OpenAI-compatible endpoint

The server does **not** read files itself. The caller (Claude Code) is expected to obtain the disassembly with `re-rizin.disassemble_function` and pass it in. The server's job is to take the disassembly, optionally the raw bytes, and produce C-like pseudocode.

## Tools

| Tool | What it does |
|---|---|
| `check_endpoint` | Hit `/v1/models`, return the list of available models |
| `decompile_function` | Send disassembly to the LLM, return C-like pseudocode |
| `explain_function` | Have the LLM explain disassembly (no rewrite) |
| `rename_variables` | Have the LLM propose better names for compiler-generated symbols |
| `summarize_binary` | Whole-binary summary from strings + imports + entry-point disasm |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `LLM_DECOMPILE_ENDPOINT` | `http://localhost:11434/v1` | OpenAI-compatible base URL |
| `LLM_DECOMPILE_MODEL` | `llm4decompile` | Model name to request |
| `LLM_DECOMPILE_API_KEY` | (empty) | API key (use `sk-...` for OpenAI; empty for Ollama) |

## Install

This server is part of the RE-AI plugin. The plugin's `install.sh` / `install.bat` installs it as part of the standard flow.

To install standalone:

```bash
pip install -e ./servers/re-llm-decompile
```

## Run

```bash
re-llm-decompile                  # stdio transport (default for MCP)
python -m re_llm_decompile        # equivalent
```

## Usage pattern (in Claude Code)

```
1. "Decompile main in /bin/ls"
2. Claude calls re-rizin.analyze_function  →  function list
3. Claude calls re-rizin.disassemble_function(name="main")  →  asm
4. Claude calls re-llm-decompile.decompile_function(asm=..., arch="x86_64")
5. Claude returns the C-like pseudocode to the user
```

## Choosing a model

- **LLM4Decompile 22B (Ref):** best quality for Linux x86_64 binaries, requires ~44GB VRAM (or AWQ/GPTQ quantizations).
- **LLM4Decompile 6.7B (Ref):** a good middle ground, ~14GB VRAM.
- **Ollama + Qwen2.5-Coder 7B:** reasonable general-purpose code model. Quality is lower than LLM4Decompile for pure binary decompilation but it explains disassembly well.
- **Claude / GPT (via this server):** not recommended — the prompt is tuned for open decompilation models. If you want to use Claude, call it directly through Claude Code rather than going through this server.

## Deprecation note

The v1 `re-ai` repo did not have this server — it tried to decompile with pefile+capstone+llm prompts in its own agent loop. That is exactly the kind of thing Claude Code is good at. This server exists to give Claude Code a clean decompilation handle.
