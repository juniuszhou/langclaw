import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from langclaw.tools.builtin import calculator, shell, telegram_send_message, write_file


def test_calculator_rejects_disallowed_characters():
    result = calculator.invoke({"expression": "__import__('os').system('echo hi')"})
    assert "Invalid characters" in result


def test_shell_rejects_unsafe_commands():
    result = shell.invoke({"command": "rm -rf /"})
    assert "not allowed" in result
    assert "Allowed:" in result


def test_telegram_send_message_requires_config(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_DEFAULT_CHAT_ID", raising=False)
    assert "TELEGRAM_BOT_TOKEN" in telegram_send_message.invoke(
        {"message": "hi", "chat_id": ""}
    )


def test_telegram_send_message_calls_api(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_DEFAULT_CHAT_ID", "999")

    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = json.dumps({"ok": True}).encode()

    with patch("langclaw.tools.builtin.urlopen", return_value=mock_resp):
        out = telegram_send_message.invoke({"message": "hello from cli", "chat_id": ""})

    assert "sent" in out.lower()
    mock_resp.read.assert_called()


def test_write_file_append_mode_appends_content(tmp_path: Path):
    target = tmp_path / "note.txt"
    first = write_file.invoke({"path": str(target), "content": "hello\n", "append": False})
    second = write_file.invoke({"path": str(target), "content": "world\n", "append": True})

    assert "Successfully wrote" in first
    assert "Successfully wrote" in second
    assert target.read_text(encoding="utf-8") == "hello\nworld\n"
