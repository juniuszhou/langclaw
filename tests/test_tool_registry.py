from langchain_core.tools import tool

from langclaw.tools.registry import ToolRegistry, get_builtin_tools, get_tools


@tool
def custom_echo(text: str) -> str:
    """Return the same text."""
    return text


def test_get_builtin_tools_returns_subset_and_skips_unknown():
    names = ["calculator", "not-a-tool", "shell"]
    tools = get_builtin_tools(names)
    assert [t.name for t in tools] == ["calculator", "shell"]


def test_tool_registry_prefers_custom_tools():
    registry = ToolRegistry()
    registry.register(custom_echo)
    found = registry.get("custom_echo")
    assert found is not None
    assert found.name == "custom_echo"


def test_get_tools_without_names_returns_all_builtin():
    tools = get_tools()
    tool_names = {t.name for t in tools}
    assert "calculator" in tool_names
    assert "shell" in tool_names
