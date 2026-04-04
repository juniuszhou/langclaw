from langchain_core.tools import StructuredTool, tool

from langclaw.config.loader import A2APeerConfig
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
    assert "a2a_send" in tool_names


def test_a2a_peer_tools_registered_and_resolved_by_name():
    peers = [
        A2APeerConfig(
            tool_name="peer_integration_test",
            base_url="http://127.0.0.1:9",
            description="Test peer tool for registry resolution.",
        )
    ]
    tools = get_tools(
        tool_names=["calculator", "peer_integration_test"],
        a2a_peers=peers,
    )
    names = [t.name for t in tools]
    assert "calculator" in names
    assert "peer_integration_test" in names

    reg = ToolRegistry()
    p = reg.get("peer_integration_test")
    assert p is not None
    assert p.name == "peer_integration_test"


def test_tool_registry_register_a2a_peer_tool():
    def _echo(message: str, context_id: str = "") -> str:
        return message

    t = StructuredTool.from_function(
        name="manual_peer", description="manual", func=_echo
    )
    reg = ToolRegistry()
    reg.register_a2a_peer_tool(t)
    assert reg.get("manual_peer") is t
