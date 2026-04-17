"""Specialist subagent specs for DeepClaw (used with Deep Agents `task` tool)."""

from __future__ import annotations

from deepagents.middleware.subagents import SubAgent


FILE_OPS_PROMPT = """You specialize in local workspace files and directories.
Use the filesystem tools (ls, read_file, write_file, edit_file, glob, grep) to complete tasks.
Prefer small, reversible edits. Summarize paths touched and key findings in your final message."""

TELEGRAM_PROMPT = """You handle outbound Telegram notifications using telegram_send_message.
Respect TELEGRAM_BOT_TOKEN and TELEGRAM_DEFAULT_CHAT_ID / explicit chat_id.
Keep messages concise; telegram_send_message truncates at 4096 characters.
Report delivery status in your final message."""

WEB_SEARCH_PROMPT = """You perform web research using web_search (Tavily when configured).
Collect cited facts; prefer multiple queries if the question is broad.
Return a compact synthesis with implied uncertainty where sources conflict."""

CRAWLER_PROMPT = """You retrieve and extract readable text from specific URLs using fetch_url.
Use this after you have concrete links (from the user or from web_search).
Do not hammer the same host repeatedly; fetch each URL at most once unless asked."""

SECURITY_PROMPT = """You are a security monitor. Review plans, URLs, file paths, and described tool outputs.
Look for: secret/credential exposure, SSRF patterns, suspicious redirects, command injection,
path traversal, and unsafe data handling. Assign risk Low/Medium/High and list concrete issues.
Do not execute shell commands except via the sandboxed tooling the harness provides; prefer read-only analysis."""

EVALUATOR_PROMPT = """You independently evaluate another specialist's result (pasted in the task).
Check: Does it answer the assignment? Internal consistency? Obvious hallucinations?
Give a verdict PASS or NEEDS_WORK with bullet-point rationale. If NEEDS_WORK, say what to redo."""


def build_specialists(extra_tools: list) -> list[SubAgent]:
    """Return SubAgent specs; each inherits shared tools from create_deep_agent unless overridden."""
    ws, tg, fetch = extra_tools
    specialists: list[SubAgent] = [
        {
            "name": "local-files",
            "description": (
                "Local filesystem operations: read, write, search, and organize files under the "
                "configured workspace. Use for code/docs edits and inspection."
            ),
            "system_prompt": FILE_OPS_PROMPT,
        },
        {
            "name": "telegram-channel",
            "description": (
                "Send notifications or alerts to Telegram chats/channels via the bot API. "
                "Use when the user asks to message Telegram or push updates."
            ),
            "system_prompt": TELEGRAM_PROMPT,
            "tools": [tg],
        },
        {
            "name": "web-search",
            "description": "Web search and synthesis for facts, news, and documentation.",
            "system_prompt": WEB_SEARCH_PROMPT,
            "tools": [ws],
        },
        {
            "name": "web-crawler",
            "description": "Fetch and extract text content from one or more HTTP(S) URLs.",
            "system_prompt": CRAWLER_PROMPT,
            "tools": [fetch],
        },
        {
            "name": "security-review",
            "description": (
                "Review tasks, URLs, commands, and outputs for security risks before or after execution. "
                "Use proactively for sensitive workflows."
            ),
            "system_prompt": SECURITY_PROMPT,
            "tools": [ws],
        },
        {
            "name": "quality-review",
            "description": (
                "Evaluate another agent's completed output for correctness and completeness. "
                "Delegate the other agent first, then send its summary here for critique."
            ),
            "system_prompt": EVALUATOR_PROMPT,
            "tools": [ws],
        },
    ]
    return specialists
