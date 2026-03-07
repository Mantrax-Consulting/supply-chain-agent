# 🚢 Supply Chain Monitor Agent

> Workshop: Building an AI Agent from Scratch with LangChain

A practical, runnable agent that monitors supply chain disruptions using
real-time web search. Built to teach five enterprise-grade patterns:
Human-in-the-Loop, Model Flexibility, No Vendor Lock-In, Observability,
and Prompt Engineering & Agent Reliability.

---

## What This Agent Does

Given a query like _"Are there any port delays in Vancouver affecting lumber shipments?"_,
the agent will:

1. **Search** the web for current information using Google Custom Search
2. **Reason** about the risk using a ReAct loop (Thought → Action → Observation)
3. **Assess** severity: LOW / MEDIUM / HIGH
4. **Parse** the result into a structured object usable by downstream systems
5. **Ask a human** for approval before sending a HIGH risk alert
6. **Log every step** for observability and audit purposes

---

## Project Structure

```
supply-chain-agent/
├── agent/
│   ├── supply_chain_agent.py           ← Main agent (start here)
│   └── prompt_engineering.py          ← Prompt versions, output parsing, reliability
├── config/
│   └── config.py                      ← All settings in one place
├── notebooks/
│   ├── workshop_notebook.ipynb        ← Parts 1–8: core agent workshop
│   └── part9_prompt_engineering.ipynb ← Part 9: prompt engineering module
├── .env.example                       ← Copy to .env and fill in your keys
├── pyproject.toml                     ← Project dependencies (managed by uv)
├── uv.lock                            ← Pinned dependency versions (do not edit)
└── README.md                          ← This file
```

---

## API Keys You Will Need

Before setup, make sure you have accounts and keys for the following.

### 1. LLM Provider — choose one

| Provider | Sign up | Environment variable |
|---|---|---|
| OpenAI | https://platform.openai.com/api-keys | `OPENAI_API_KEY` |
| Anthropic | https://console.anthropic.com | `ANTHROPIC_API_KEY` |
| Google AI Studio | https://aistudio.google.com | `GOOGLE_API_KEY` |

### 2. Google Custom Search — required regardless of LLM provider

The agent uses Google Custom Search to fetch real-time web results. Without
it, the agent has no tools and can only answer from its training data —
which has a cutoff date and cannot find current disruptions.

You need two things:

**Search Engine ID:**
1. Go to https://programmablesearchengine.google.com
2. Click **Add** → give it any name → select **Search the entire web**
3. Copy the **Search engine ID** → paste into `GOOGLE_CSE_ID` in your `.env`

**Google API Key:**
1. Go to https://console.cloud.google.com
2. Enable the **Custom Search API**
3. Go to Credentials → Create API key
4. Paste into `GOOGLE_CSE_API_KEY` in your `.env`

Both are free up to 100 searches per day — sufficient for workshop use.

### 3. LangSmith — optional but recommended

LangSmith gives you a visual dashboard of every agent run — every thought,
tool call, token count, and result. Not required to run the agent, but
highly recommended for the observability section of the workshop.

1. Sign up free at https://smith.langchain.com
2. Go to Settings → Create an API key
3. Paste into `LANGCHAIN_API_KEY` in your `.env`
4. Set `LANGCHAIN_TRACING_V2=true`

---

## Quick Start

### Step 1 — Install uv

uv is a fast Python package manager. Install it once on your machine:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2 — Clone and set up

```bash
git clone https://github.com/your-org/supply-chain-agent.git
cd supply-chain-agent
uv sync
```

`uv sync` reads `uv.lock` and installs every dependency at the exact
pinned version. No virtual environment activation needed.

### Step 3 — Set up your API keys

```bash
cp .env.example .env
# Open .env in your editor and fill in your keys
```

See the API Keys section above for where to get each one.

### Step 4 — Confirm everything works

```bash
uv run python -c "from langchain.agents import AgentExecutor, create_react_agent; print('imports OK')"
```

### Step 5 — Run the agent

```bash
uv run python agent/supply_chain_agent.py
```

### Step 6 — Try the notebooks

```bash
uv run jupyter notebook
```

Then open `notebooks/workshop_notebook.ipynb` to start the guided exercises.

---

## Switching LLM Providers

Open `.env` and change one line:

```bash
# Use OpenAI
LLM_PROVIDER=openai

# Use Anthropic Claude
LLM_PROVIDER=anthropic

# Use Google Gemini
LLM_PROVIDER=google
```

Or pass it directly in code:

```python
run_supply_chain_monitor(
    query="Any port disruptions in Halifax?",
    provider="anthropic",   # ← change this
)
```

---

## Key Concepts

### Human-in-the-Loop (HITL)
The agent does NOT automatically send alerts. When it detects a HIGH risk,
it pauses and asks the operator for approval. This keeps humans in control
of high-stakes operational decisions.

In production, replace the `input()` call with a Slack message, an email,
or a dashboard button — the pattern stays the same.

### Model Flexibility
The `get_llm()` factory function in `supply_chain_agent.py` returns any
LangChain-compatible LLM. The agent is built against LangChain's
`BaseChatModel` interface, so it works with any provider without modification.

### No Vendor Lock-In
All provider-specific imports are isolated in `get_llm()`. The agent,
tools, and prompt reference zero provider names. Adding a new provider
(Mistral, Llama, etc.) requires adding one `elif` block.

### Observability
The `SupplyChainObservabilityHandler` inherits from `BaseCallbackHandler`
and logs every thought, tool call, and result. Connect LangSmith for a
visual dashboard of every run.

### Prompt Engineering & Agent Reliability
`agent/prompt_engineering.py` contains four prompt versions (V1–V4),
each fixing a specific real failure mode. The `parse_agent_output()`
function extracts the structured Final Answer into a `RiskAssessment`
object — making output usable by databases, Slack, and dashboards.
See `notebooks/part9_prompt_engineering.ipynb` for the guided exercises.

---

## Example Queries to Try

```python
# Port disruptions
"Are there any port strikes or delays in Canada right now?"

# Supplier risk
"Is there any news about financial difficulties at major automotive parts suppliers?"

# Trade route issues
"What is the current shipping situation through the Suez Canal?"

# Commodity shortages
"Are there any semiconductor shortages affecting North American manufacturers?"

# Weather and natural disasters
"Have any recent weather events disrupted shipping lanes in the Gulf of Mexico?"
```

---

## Workshop Notebooks

| Notebook | Contents |
|---|---|
| `workshop_notebook.ipynb` | Parts 1–8: tools, observability, model flexibility, HITL, full agent run |
| `part9_prompt_engineering.ipynb` | Part 9: prompt versions, output parsing, V1 vs V4 live comparison |

---

## License

MIT — use freely for learning and building.
