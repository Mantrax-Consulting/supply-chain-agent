# 🚢 Supply Chain Monitor Agent

> Workshop: Building an AI Agent from Scratch with LangChain

A practical, runnable agent that monitors supply chain disruptions using
real-time web search. Built to teach four enterprise-grade patterns:
Human-in-the-Loop, Model Flexibility, No Vendor Lock-In, and Observability.

---

## What This Agent Does

Given a query like _"Are there any port delays in Vancouver affecting lumber shipments?"_,
the agent will:

1. **Search** the web for current information using Google Search
2. **Reason** about the risk using a ReAct loop (Thought → Action → Observation)
3. **Assess** severity: LOW / MEDIUM / HIGH
4. **Ask a human** for approval before sending a HIGH risk alert
5. **Log every step** for observability and audit purposes

---

## Project Structure

```
supply-chain-agent/
├── agent/
│   └── supply_chain_agent.py   ← Main agent (start here)
├── config/
│   └── config.py               ← All settings in one place
├── notebooks/
│   └── workshop_notebook.ipynb ← Step-by-step guided exercises
├── .env.example                ← Copy to .env and fill in your keys
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## Quick Start (5 Minutes)

### Step 1 — Clone and install

```bash
git clone https://github.com/your-org/supply-chain-agent.git
cd supply-chain-agent
pip install -r requirements.txt
```

### Step 2 — Set up your API keys

```bash
cp .env.example .env
# Open .env in your editor and fill in:
#   - OPENAI_API_KEY (or ANTHROPIC_API_KEY or GOOGLE_API_KEY)
#   - GOOGLE_CSE_ID
#   - GOOGLE_CSE_API_KEY
```

See [Getting Your API Keys](#getting-your-api-keys) below for instructions.

### Step 3 — Run the agent

```bash
python agent/supply_chain_agent.py
```

### Step 4 — Try the notebook

```bash
jupyter notebook notebooks/workshop_notebook.ipynb
```

---

## Getting Your API Keys

### LLM (choose one)

| Provider | Where to get it | Environment variable |
|---|---|---|
| OpenAI | https://platform.openai.com/api-keys | `OPENAI_API_KEY` |
| Anthropic | https://console.anthropic.com | `ANTHROPIC_API_KEY` |
| Google AI | https://aistudio.google.com | `GOOGLE_API_KEY` |

### Google Custom Search (required for all providers)

1. Go to https://programmablesearchengine.google.com
2. Click **Add** → give it a name → select **Search the entire web**
3. Copy the **Search engine ID** → paste into `GOOGLE_CSE_ID`
4. Go to https://console.cloud.google.com
5. Enable the **Custom Search API**
6. Create an API key → paste into `GOOGLE_CSE_API_KEY`

### LangSmith Observability (optional — highly recommended)

1. Sign up free at https://smith.langchain.com
2. Go to Settings → Create an API key
3. Paste into `LANGCHAIN_API_KEY`
4. Set `LANGCHAIN_TRACING_V2=true`
5. After running the agent, visit https://smith.langchain.com to see traces

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

```python
# In agent/supply_chain_agent.py
if is_high_risk:
    approved = human_approval_gate(risk_summary=output, risk_level="HIGH")
```

In production, replace the `input()` call with a Slack message, an email,
or a dashboard button — the pattern stays the same.

### Model Flexibility
The `get_llm()` factory function returns any LangChain-compatible LLM.
The agent is built against LangChain's `BaseChatModel` interface,
so it works with any provider without modification.

### Observability
The `SupplyChainObservabilityHandler` inherits from `BaseCallbackHandler`
and logs every thought, tool call, and result. Connect LangSmith for a
visual dashboard of every run.

### No Vendor Lock-In
- All provider-specific imports are isolated in `get_llm()`
- The agent, tools, and prompt reference zero provider names
- Adding a new provider (Mistral, Llama, etc.) requires adding one elif block

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

## Workshop Exercises

See the notebook at `notebooks/workshop_notebook.ipynb` for guided exercises including:

- Adding a new search tool
- Adding token usage logging to the observability handler
- Modifying the HITL gate to handle different risk levels
- Swapping providers and comparing outputs
- Connecting the agent to a Slack webhook

---

## License

MIT — use freely for learning and building.
