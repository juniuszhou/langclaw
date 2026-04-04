"""Build Agent Card metadata for A2A discovery."""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

DEFAULT_SKILL_ID = "langclaw-default"


def build_langclaw_agent_card(
    *,
    public_url: str,
    name: str,
    description: str,
    version: str = "0.1.0",
) -> AgentCard:
    """Create an :class:`AgentCard` pointing at this server's JSON-RPC endpoint.

    ``public_url`` must be the absolute base URL peers use to reach this host
    (scheme + host + port), with no trailing slash. JSON-RPC is posted to ``/``.
    """
    base = public_url.rstrip("/")
    return AgentCard(
        name=name,
        description=description,
        version=version,
        url=base,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        skills=[
            AgentSkill(
                id=DEFAULT_SKILL_ID,
                name=name,
                description=description,
                tags=["assistant", "langclaw", "general"],
            )
        ],
        preferred_transport="JSONRPC",
        protocol_version="0.3.0",
    )
