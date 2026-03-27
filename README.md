# LangClaw

OpenCLaw-equivalent AI agent platform built on **LangGraph**. Phase 1: Core agent runtime with ReAct loop, model abstraction, tool registry, skills, and MCP.

## Phase 1 Features

- **LangGraph Agent**: StateGraph with ReAct loop (agent → tools → agent)
- **Model Abstraction**: OpenAI, Anthropic, Google Gemini, Ollama (config-driven)
- **Built-in Tools**: calculator, get_current_time, web_search, read_file, write_file, shell (sandboxed), send_email, calendar_add, calendar_list
- **Skills**: Load `SKILL.md` from `skills/<name>/` folders; inject instructions into system prompt
- **MCP Integration**: Load tools from MCP servers (e.g. firecrawl-mcp) via config
- **Checkpointer**: MemorySaver for conversation persistence
- **YAML Config**: Agent definition in config.yaml

## Quick Start

1. **Install** (from de-LLM root, or install langclaw):

   ```bash
   cd applications/langclaw
   pip install -e .
   ```

2. **Run Ollama** (for local model):

   ```bash
   ollama serve
   ollama pull llama3.2:3b
   ```

3. **Run agent**:
   ```bash
   python run.py
   # Or: python run.py default
   ```

## Telegram Channel

1. Install telegram extra:

```bash
cd applications/langclaw
pip install -e ".[telegram]"
```

2. Create a Telegram bot with BotFather and set:

```bash
export TELEGRAM_BOT_TOKEN="..."
```

3. Run the Telegram adapter:

```bash
source ../../.venv/bin/activate
python run_telegram.py default
```

Each Telegram `chat_id` maps to a stable LangClaw `thread_id` (`telegram:<chat_id>`), so memory persists per chat in `langclaw.sqlite`.

## Phase 2: Memory (SQLite)

LangClaw now persists **short-term conversation history** to `applications/langclaw/langclaw.sqlite` keyed by `thread_id`.

- **Resume a previous session**: set `LANGCLAW_THREAD_ID` before running:

```bash
export LANGCLAW_THREAD_ID="my-session-1"
python run.py
```

- **Demo of “history in DB as new input”**:

```bash
python demo_db_history.py
```

This script does:

- load previous messages from SQLite (`load_messages(thread_id)`)
- append them to the new user message
- invoke the agent
- write the new turn back to SQLite (`append_messages(thread_id, ...)`)

## Config

Edit `config.yaml`:

```yaml
agents:
  default:
    model: ollama/llama3.2:3b
    system_prompt: "You are a helpful assistant."
    tools:
      [
        calculator,
        get_current_time,
        read_file,
        write_file,
        shell,
        send_email,
        calendar_add,
        calendar_list,
      ]
    skills: [example]
    temperature: 0.7
    # Optional MCP tools (requires: pip install mcp langchain-mcp-adapters)
    # mcp:
    #   server: npx
    #   args: [firecrawl-mcp]
    #   env:
    #     FIRECRAWL_API_KEY: ${FIRECRAWL_API_KEY}
```

Model format: `provider/model` (e.g. `openai/gpt-4o`, `anthropic/claude-sonnet`, `ollama/llama3`).

### Skills

Add skills in `skills/<name>/SKILL.md`. Each SKILL.md is injected into the system prompt. Enable via `skills: [name1, name2]` in config.

### MCP

Install optional deps: `pip install langclaw[mcp]`. Configure MCP server in agent config to load additional tools (e.g. Firecrawl for web scraping).

## Project Structure

```bash
langclaw/
├── config.yaml
├── run.py
├── pyproject.toml
└── src/langclaw/
    ├── config/       # YAML loader
    ├── models/       # Model providers (OpenAI, Anthropic, Ollama, Google)
    ├── runtime/      # LangGraph agent graph
    └── tools/        # Registry + builtin tools
```

### Knowledge about　langgraph

the MCP is a tool
the RAG is a tool
retriever as a node, it will add the data from embedding db to system prompt

vector size and how to measure the similarity of content.

telegram service is saparate process, you just use the LLM invoke to response message.

### Test

```bash
uv run pytest
```
