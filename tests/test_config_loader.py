from pathlib import Path

from langclaw.config.loader import load_config


def test_load_config_returns_empty_when_file_missing(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.agents == {}


def test_load_config_parses_nested_agent_settings(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
agents:
  default:
    model: openai/gpt-4o-mini
    system_prompt: You are concise.
    tools: [calculator]
    skills: [debugger]
    mcp:
      server: npx
      args: ["-y", "@example/server"]
    rag:
      enabled: true
      sources: ["docs"]
      include_pdf: false
      k: 6
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_file)
    agent = cfg.agents["default"]
    assert agent.model == "openai/gpt-4o-mini"
    assert agent.tools == ["calculator"]
    assert agent.skills == ["debugger"]
    assert agent.mcp is not None
    assert agent.mcp.args == ["-y", "@example/server"]
    assert agent.rag is not None
    assert agent.rag.sources == ["docs"]
    assert agent.rag.k == 6
