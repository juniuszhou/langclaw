"""Built-in tools for the agent."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.tools import tool

from langclaw.memory.sqlite import SqliteMemory
from langclaw.tools.a2a_client import a2a_send

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely. Use for any arithmetic.

    Args:
        expression: A mathematical expression, e.g. "2 + 2", "10 * 5", "sqrt(16)"

    Returns:
        The result of the calculation as a string.
    """
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression.replace("sqrt", "")):
            return "Error: Invalid characters in expression"
        # Safe eval - no imports
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_time(timezone: Optional[str] = None) -> str:
    """Get the current date and time.

    Args:
        timezone: Optional timezone name (e.g. UTC, America/New_York). Uses local if not specified.

    Returns:
        Current date and time as a string.
    """
    if timezone:
        try:
            from zoneinfo import ZoneInfo
            return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            pass
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information. Set TAVILY_API_KEY for real search via Tavily.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        Search results as text.
    """
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        search = TavilySearchResults(max_results=max_results)
        results = search.invoke({"query": query})
        if isinstance(results, list):
            return "\n\n".join(
                f"[{r.get('title', '')}]\n{r.get('content', str(r))}" for r in results
            )
        return str(results)
    except ImportError:
        return f"[Mock] Search results for: {query}\n(Install tavily-python and set TAVILY_API_KEY for real search)"
    except Exception as e:
        return f"Search error: {e}"


@tool
def read_file(path: str) -> str:
    """Read contents of a file from the local filesystem.

    Args:
        path: Path to the file to read.

    Returns:
        File contents as string.
    """
    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    if not p.is_file():
        return f"Error: Not a file: {path}"
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(path: str, content: str, append: bool = False) -> str:
    """Write content to a file on the local filesystem.

    Args:
        path: Path to the file to write.
        content: Content to write.
        append: If True, append to existing file; otherwise overwrite.

    Returns:
        Success or error message.
    """
    import pathlib
    p = pathlib.Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with p.open(mode, encoding="utf-8", errors="replace") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


# Sandboxed shell: allowlist of safe commands only
_SHELL_ALLOWED = frozenset({"ls", "pwd", "cat", "head", "tail", "wc", "echo", "date", "whoami", "env", "which"})


@tool
def shell(command: str, timeout_seconds: int = 30) -> str:
    """Run a shell command (sandboxed). Allowed: ls, pwd, cat, head, tail, wc, echo, date, whoami, env, which.

    Args:
        command: Shell command to run. Only safe read-only commands are permitted.
        timeout_seconds: Maximum execution time in seconds (default 30).

    Returns:
        Command output or error message.
    """
    import shlex
    import subprocess
    parts = shlex.split(command.strip()) if command.strip() else []
    if not parts:
        return "Error: Empty command"
    cmd = parts[0].lower()
    if cmd not in _SHELL_ALLOWED:
        return f"Error: Command '{cmd}' not allowed. Allowed: {', '.join(sorted(_SHELL_ALLOWED))}"
    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=None,
        )
        out = result.stdout or ""
        err = result.stderr or ""
        if result.returncode != 0:
            return f"Exit code {result.returncode}\nstdout: {out}\nstderr: {err}"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout_seconds}s"
    except Exception as e:
        return f"Error: {e}"


@tool
def telegram_send_message(message: str, chat_id: str = "") -> str:
    """Send a Telegram message with this project's bot (Telegram Bot HTTP API).

    Use when the user asks to notify someone on Telegram, push an update to a chat,
    or send text from a terminal/CLI session. This is separate from replying inside
    the Telegram app: replies are automatic; this tool actively sends a new message.

    Requires TELEGRAM_BOT_TOKEN (same as the Telegram channel). If chat_id is empty,
    TELEGRAM_DEFAULT_CHAT_ID is used—set it to your numeric chat id (shown by /start on the bot).

    Args:
        message: Plain text to send (max 4096 characters; longer text is truncated).
        chat_id: Numeric Telegram chat id (user, group, or channel id). Omit to use env default.

    Returns:
        Confirmation or error message from the API.
    """
    import os

    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        return "Error: TELEGRAM_BOT_TOKEN is not set."

    cid = (chat_id or os.getenv("TELEGRAM_DEFAULT_CHAT_ID") or "").strip()
    if not cid:
        return (
            "Error: chat_id is empty and TELEGRAM_DEFAULT_CHAT_ID is not set. "
            "Pass chat_id or set TELEGRAM_DEFAULT_CHAT_ID (see /start on the bot)."
        )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = json.dumps(
        {"chat_id": cid, "text": message[:4096]},
        ensure_ascii=False,
    ).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except HTTPError as e:
        try:
            err_body = e.read().decode()
        except Exception:
            err_body = str(e)
        return f"Error: Telegram API HTTP {e.code}: {err_body}"
    except URLError as e:
        return f"Error: could not reach Telegram: {e.reason}"
    except Exception as e:
        return f"Error: {e}"

    if not data.get("ok"):
        return f"Error: {data.get('description', data)}"
    return f"Message sent to chat_id {cid}."


@tool
def send_email(
    to: str,
    subject: str,
    body: str,
    smtp_host: Optional[str] = None,
    smtp_port: int = 587,
) -> str:
    """Send an email. Requires SMTP configured via SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD env vars.

    Args:
        to: Recipient email address.
        subject: Email subject.
        body: Email body text.
        smtp_host: SMTP server host (default from SMTP_HOST env).
        smtp_port: SMTP port (default 587).

    Returns:
        Success or error message.
    """
    import os
    smtp_host = smtp_host or os.getenv("SMTP_HOST")
    if not smtp_host:
        return "Error: SMTP not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD environment variables."
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    if not user or not password:
        return "Error: Set SMTP_USER and SMTP_PASSWORD for authentication."
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to], msg.as_string())
        return f"Email sent successfully to {to}"
    except Exception as e:
        return f"Error sending email: {e}"


# Simple in-memory calendar (persists per process; use file for production)
_CALENDAR_EVENTS: list[dict] = []


@tool
def calendar_add(title: str, start: str, end: Optional[str] = None, description: Optional[str] = None) -> str:
    """Add an event to the calendar. Times in ISO format (YYYY-MM-DD HH:MM or YYYY-MM-DD).

    Args:
        title: Event title.
        start: Start time (e.g. 2025-01-25 14:00).
        end: End time (optional).
        description: Event description (optional).

    Returns:
        Confirmation message.
    """
    event = {"title": title, "start": start, "end": end, "description": description}
    _CALENDAR_EVENTS.append(event)
    return f"Added event: {title} at {start}"


@tool
def calendar_list(limit: int = 10) -> str:
    """List upcoming calendar events.

    Args:
        limit: Maximum number of events to return (default 10).

    Returns:
        List of events as text.
    """
    if not _CALENDAR_EVENTS:
        return "No events in calendar."
    lines = []
    for i, e in enumerate(_CALENDAR_EVENTS[:limit]):
        lines.append(f"- {e['title']} @ {e['start']}" + (f" - {e['end']}" if e.get('end') else ""))
    return "\n".join(lines)


@tool
def memory_note_put(namespace: str, key: str, value: str, db_path: str = "langclaw.sqlite") -> str:
    """Store a long-term memory note (SQLite).

    Args:
        namespace: Logical namespace (e.g. user id, project id).
        key: Note key.
        value: Note value text.
        db_path: SQLite db path (default: langclaw.sqlite in CWD).

    Returns:
        Confirmation message.
    """
    mem = SqliteMemory(Path(db_path))
    mem.setup()
    mem.note_put(namespace, key, value)
    return f"Saved note ({namespace}/{key})."


@tool
def memory_note_search(namespace: str, query: str, limit: int = 5, db_path: str = "langclaw.sqlite") -> str:
    """Search long-term memory notes by keyword (SQLite LIKE search)."""
    mem = SqliteMemory(Path(db_path))
    mem.setup()
    hits = mem.note_search(namespace, query, limit=limit)
    if not hits:
        return "No matching notes."
    return "\n".join(f"- {h['key']}: {h['value']}" for h in hits)


BUILTIN_TOOLS = [
    calculator,
    get_current_time,
    a2a_send,
    web_search,
    read_file,
    write_file,
    shell,
    telegram_send_message,
    send_email,
    calendar_add,
    calendar_list,
    memory_note_put,
    memory_note_search,
]
