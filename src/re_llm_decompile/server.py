"""MCP server entry point for re-llm-decompile.

Exposes AI decompilation tools to Claude Code via the Model
Context Protocol stdio transport. Uses the OpenAI-compatible chat-completions
API so it works with vLLM, Ollama, and any compatible endpoint.

Cycle 1 / T1.2 fix: model auto-fallback. The configured
``LLM_DECOMPILE_MODEL`` env var is preferred; if absent or unknown
to the endpoint, ``client.resolve_model()`` queries ``/v1/models`` and
picks the first non-embed model. The ``check_endpoint`` tool exposes
the resolution state (``configured`` / ``resolved`` / ``fallback_used``)
so the agent can see when a fallback kicked in. Decompile calls
return a ``WARNING`` (not ``ERROR``) when a fallback was used, so the
agent can decide whether to retry with a different model.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from re_llm_decompile import client

logger = logging.getLogger("re_llm_decompile")
logger.setLevel(logging.INFO)

mcp = FastMCP("re-llm-decompile")


# ── Health ──────────────────────────────────────────────────────────────


@mcp.tool()
def check_endpoint() -> dict:
    """Return endpoint health + the model-resolution state.

    The ``resolution`` block reports:
      - ``configured``: the raw env-var value (or the default)
      - ``available``: list of model names the endpoint serves
      - ``resolved``: the model the next chat() call will use
      - ``fallback_used``: True if ``resolved`` != ``configured``
      - ``endpoint_reachable``: True if /v1/models returned 2xx

    A4 fix (v2.8.0): the status field is now truthful with respect to
    the resolved model. Returns ``WARN`` (not ``OK``) when the resolved
    model is cloud-backed (e.g. ``deepseek-v4-flash:cloud``) because
    such models are observed to return HTTP 403 on programmatic chat
    completions — confirmed in stress testing across multiple representative targets.
    The ``warning`` field carries a remediation hint. Set
    ``LLM_DECOMPILE_DISABLE_CLOUD=1`` to skip cloud models in
    auto-fallback, or pin ``LLM_DECOMPILE_MODEL`` to a local model.

    Use this to confirm the endpoint is reachable and to see when the
    LLM_DECOMPILE_MODEL env var is out of sync with the endpoint.
    """
    try:
        models = client.list_models()
    except Exception as exc:  # noqa: BLE001
        return {
            "endpoint": client.get_endpoint(),
            "status": "ERROR",
            "error": str(exc),
        }
    resolution = client.get_resolution_state()
    resolved = resolution.get("resolved") or ""
    resolved_is_cloud = client.is_cloud_model(resolved)
    status = "WARN" if resolved_is_cloud else "OK"
    result: dict = {
        "endpoint": client.get_endpoint(),
        "status": status,
        "available_models": models,
        "resolution": resolution,
        "is_cloud_model": resolved_is_cloud,
    }
    if resolved_is_cloud:
        result["warning"] = (
            f"resolved model {resolved!r} appears to be cloud-backed; "
            "cloud bridges commonly return HTTP 403 on programmatic "
            "chat completions. Either set LLM_DECOMPILE_DISABLE_CLOUD=1 "
            "to skip cloud models in auto-fallback, or pin "
            "LLM_DECOMPILE_MODEL to a known local model "
            "(e.g. llama3.2:3b, llm4decompile). The agent may also "
            "fall back to the Tier 1.5 path in skills/re-decompile/SKILL.md "
            "(do the decompile reasoning yourself with the disasm as context)."
        )
    return result


# ── Decompilation ──────────────────────────────────────────────────────


DECOMPILE_SYSTEM = (
    "You are a decompiler. Convert the assembly listing into clean, "
    "idiomatic C pseudocode. Preserve calling conventions, recover "
    "stack variables, and name arguments when their purpose is clear. "
    "Do NOT invent behavior that is not visible in the disassembly. "
    "If a value is opaque, declare it as a placeholder variable. "
    "Output only the C code, no commentary."
)

EXPLAIN_SYSTEM = (
    "You are a senior reverse engineer explaining a function to a "
    "junior. Describe the high-level purpose, identify any interesting "
    "patterns (anti-debug, obfuscation, crypto, syscalls), and call out "
    "any IOCs (URLs, file paths, registry keys) you see referenced. "
    "Be concise but thorough."
)

RENAME_SYSTEM = (
    "You rename compiler-generated symbols in decompiled C code. "
    "Use a single, descriptive snake_case name per symbol. "
    "Return the entire source with the renames applied. "
    "Do not change behavior or control flow."
)


def _model_block() -> dict:
    """Return the {model, fallback_used, configured} block for chat responses."""
    state = client.get_resolution_state()
    return {
        "model": state.get("resolved") or client.get_model(),
        "configured": state.get("configured"),
        "fallback_used": bool(state.get("fallback_used")),
        "resolution_state": state,
    }


@mcp.tool()
def decompile_function(
    disasm: str,
    arch: str = "x86_64",
    calling_conv: str = "SystemV",
    context: str = "",
    temperature: float = 0.0,
) -> dict:
    """Decompile a disassembled function to C-like pseudocode.

    Args:
        disasm: the disassembly text (one instruction per line, ideally)
        arch: target architecture — x86_64, x86, aarch64, arm, mips, ppc, riscv
        calling_conv: calling convention hint (SystemV, MS_x64, AAPCS, ...)
        context: optional caller-provided context (e.g. "this is the
            HTTP request handler for POST /login")
        temperature: 0.0 = deterministic; raise to 0.2-0.4 for more
            "creative" naming

    Returns a dict with ``code`` (the C pseudocode) and ``model``. If
    the auto-fallback picked a different model than the configured
    one, ``fallback_used`` is True and a ``warning`` field surfaces a
    short explanation.
    """
    user = f"Architecture: {arch}\nCalling convention: {calling_conv}\n"
    if context:
        user += f"Context: {context}\n"
    user += f"\n```asm\n{disasm}\n```\n"
    code = client.chat(
        DECOMPILE_SYSTEM, user, temperature=temperature, max_tokens=2048
    )
    block = _model_block()
    result: dict = {"code": code, **block}
    if block["fallback_used"]:
        result["warning"] = (
            f"Configured model {block['configured']!r} is not available on the endpoint; "
            f"auto-fell back to {block['model']!r}. "
            "Set LLM_DECOMPILE_MODEL to one of the available_models to silence this."
        )
    return result


@mcp.tool()
def explain_function(disasm: str, arch: str = "x86_64", context: str = "") -> dict:
    """Explain what a function does, in natural language, without rewriting it.

    Useful for triage: get a fast read of what a function does before
    deciding whether to spend a decompile call on it.
    """
    user = f"Architecture: {arch}\n"
    if context:
        user += f"Context: {context}\n"
    user += f"\n```asm\n{disasm}\n```\n"
    explanation = client.chat(
        EXPLAIN_SYSTEM, user, temperature=0.0, max_tokens=1024
    )
    block = _model_block()
    result = {"explanation": explanation, **block}
    if block["fallback_used"]:
        result["warning"] = (
            f"Configured model {block['configured']!r} is not available on the endpoint; "
            f"auto-fell back to {block['model']!r}."
        )
    return result


@mcp.tool()
def rename_variables(decompiled: str) -> dict:
    """Propose better names for compiler-generated symbols in decompiled C."""
    user = (
        "Rename compiler-generated variables in this decompiled code:\n\n"
        f"```c\n{decompiled}\n```\n"
    )
    renamed = client.chat(
        RENAME_SYSTEM, user, temperature=0.0, max_tokens=2048
    )
    block = _model_block()
    result = {"code": renamed, **block}
    if block["fallback_used"]:
        result["warning"] = (
            f"Configured model {block['configured']!r} is not available on the endpoint; "
            f"auto-fell back to {block['model']!r}."
        )
    return result


@mcp.tool()
def summarize_binary(
    strings: list[str] | None = None,
    imports: list[str] | None = None,
    entrypoint_disasm: str = "",
    format: str = "PE",
) -> dict:
    """Generate a one-paragraph summary of a binary from lightweight features.

    Args:
        strings: list of interesting strings (URLs, paths, registry keys)
        imports: list of imported library:fn pairs
        entrypoint_disasm: disassembly of the entry point (first ~30 insns)
        format: PE | ELF | MACHO | DEX

    Returns a dict with ``summary`` (one paragraph) and ``category_guess``
    (e.g. "downloader", "keylogger", "installer", "library", "unclear").
    """
    sys = (
        "You classify binaries from lightweight static features. "
        "Return a one-paragraph summary and a category guess. "
        "Categories: downloader, dropper, keylogger, ransomware, "
        "backdoor, rootkit, stealer, loader, installer, library, "
        "driver, game, app, or unclear."
    )
    user = f"Format: {format}\n"
    if imports:
        user += "Imports (top 50):\n" + "\n".join(imports[:50]) + "\n"
    if strings:
        user += "Interesting strings (top 50):\n" + "\n".join(strings[:50]) + "\n"
    if entrypoint_disasm:
        user += f"\nEntry-point disassembly:\n```asm\n{entrypoint_disasm}\n```\n"
    text = client.chat(sys, user, temperature=0.0, max_tokens=512)
    # Parse into summary + category if possible
    summary = text
    category = "unclear"
    for line in text.splitlines():
        if "category" in line.lower() and ":" in line:
            category = line.split(":", 1)[1].strip().lower().split()[0]
    block = _model_block()
    result = {"summary": summary, "category_guess": category, **block}
    if block["fallback_used"]:
        result["warning"] = (
            f"Configured model {block['configured']!r} is not available on the endpoint; "
            f"auto-fell back to {block['model']!r}."
        )
    return result


# ── Entrypoint ─────────────────────────────────────────────────────────


def main() -> None:
    """Run the MCP server over stdio (the standard Claude Code transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
