from pathlib import Path

from langclaw.tools.builtin import calculator, shell, write_file


def test_calculator_rejects_disallowed_characters():
    result = calculator.invoke({"expression": "__import__('os').system('echo hi')"})
    assert "Invalid characters" in result


def test_shell_rejects_unsafe_commands():
    result = shell.invoke({"command": "rm -rf /"})
    assert "not allowed" in result
    assert "Allowed:" in result


def test_write_file_append_mode_appends_content(tmp_path: Path):
    target = tmp_path / "note.txt"
    first = write_file.invoke({"path": str(target), "content": "hello\n", "append": False})
    second = write_file.invoke({"path": str(target), "content": "world\n", "append": True})

    assert "Successfully wrote" in first
    assert "Successfully wrote" in second
    assert target.read_text(encoding="utf-8") == "hello\nworld\n"
