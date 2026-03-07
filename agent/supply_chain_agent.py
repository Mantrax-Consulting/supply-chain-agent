"""
Supply Chain Disruption Monitor Agent
======================================
Workshop Use Case: A LangChain agent that monitors supply chain risks
by searching the web for news about suppliers, ports, and trade routes.

Key Concepts Demonstrated:
  - Human-in-the-Loop (HITL): Agent asks for approval before sending alerts
  - Model Flexibility: Swap LLMs with a single config change
  - Observability: Every step is logged via LangSmith callbacks
  - No Vendor Lock-In: Works with OpenAI, Anthropic, or any LangChain LLM
"""

import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

# Load .env file — must happen before any API clients are initialised
load_dotenv()

# Wire up LangSmith observability if configured
# These must be set as environment variables before LangChain is used
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "supply-chain-monitor")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ─── Step 1: Observability — Custom Callback Handler ──────────────────────────
# This runs automatically at every step of the agent's reasoning cycle.
# Think of it as a "flight recorder" for your agent.

class SupplyChainObservabilityHandler(BaseCallbackHandler):
    """Logs every thought, action, and result the agent produces."""

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n🧠 [THINK] Agent is reasoning...")

    def on_llm_end(self, response: LLMResult, **kwargs):
        text = response.generations[0][0].text if response.generations else ""
        if text:
            print(f"💭 [THOUGHT] {text[:200]}...")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        print(f"\n🔧 [ACTION] Using tool: {tool_name}")
        print(f"   Input: {input_str[:150]}")

    def on_tool_end(self, output, **kwargs):
        print(f"📄 [RESULT] {str(output)[:300]}...")

    def on_agent_action(self, action: AgentAction, **kwargs):
        print(f"\n⚡ [STEP] Action → {action.tool}: {action.tool_input[:100]}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        print(f"\n✅ [DONE] Agent finished.")
        print(f"   Final Answer: {finish.return_values.get('output', '')[:300]}")

    def on_chain_error(self, error, **kwargs):
        print(f"\n❌ [ERROR] {str(error)}")


# ─── Step 2: Model Flexibility — LLM Factory ──────────────────────────────────
# This function is the key to avoiding vendor lock-in.
# Change ONE line in config.py to switch between GPT-4, Claude, or Gemini.

def get_llm(provider: str = "openai", model_name: str = None, temperature: float = 0.0):
    """
    Factory function — returns any LangChain-compatible LLM.
    
    Supported providers (add more anytime):
      - "openai"    → Requires OPENAI_API_KEY
      - "anthropic" → Requires ANTHROPIC_API_KEY  
      - "google"    → Requires GOOGLE_API_KEY
    
    Workshop Note:
      The agent code below NEVER references a specific provider.
      It only receives an LLM object. This is the abstraction that
      prevents vendor lock-in — the agent doesn't know or care
      what model is running it.
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name or "claude-3-haiku-20240307",
            temperature=temperature,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-1.5-flash",
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            "Choose from: 'openai', 'anthropic', 'google'"
        )


# ─── Step 3: Tools — What the Agent Can Do ────────────────────────────────────

def build_tools() -> list:
    """
    Returns a list of tools available to the agent.
    
    Tools are how the agent interacts with the outside world.
    Each tool has:
      - name: what the agent calls it
      - func: the Python function to run
      - description: what the LLM reads to decide when to use it

    We use Tavily Search — built specifically for AI agents.
    It returns clean, relevant results without requiring a search engine ID.
    One API key is all you need: https://tavily.com
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not tavily_api_key:
        raise EnvironmentError(
            "Missing TAVILY_API_KEY in .env file."
            " Sign up free at https://tavily.com to get your key."
        )

    # TavilySearchResults returns structured results with title, url, and content
    # max_results=5 keeps token usage reasonable while giving enough context
    tavily = TavilySearchResults(
        max_results=5,
        tavily_api_key=tavily_api_key,
    )

    supply_chain_search = Tool(
        name="supply_chain_search",
        func=tavily.run,
        description=(
            "Search the web for current supply chain news, disruptions, "
            "port delays, supplier financial issues, trade route problems, "
            "or tariff changes. Input should be a specific search query. "
            "Use this tool to find real-time information about risks."
        ),
    )

    def check_supplier_risk(supplier_name: str) -> str:
        """Searches specifically for risk signals about a named supplier."""
        query = f"{supplier_name} bankruptcy risk financial trouble layoffs 2024 2025"
        return tavily.run(query)

    supplier_risk_tool = Tool(
        name="check_supplier_risk",
        func=check_supplier_risk,
        description=(
            "Check if a specific supplier or vendor is facing financial difficulties, "
            "bankruptcy, or operational risks. Input must be the company name."
        ),
    )

    def check_port_status(port_name: str) -> str:
        """Searches for port delays and congestion."""
        query = f"{port_name} port congestion delay strike closure today"
        return tavily.run(query)

    port_status_tool = Tool(
        name="check_port_status",
        func=check_port_status,
        description=(
            "Check the current status of a shipping port — delays, strikes, "
            "weather closures, or congestion. Input must be the port name and country."
        ),
    )

    return [supply_chain_search, supplier_risk_tool, port_status_tool]


# ─── Step 4: Human-in-the-Loop ────────────────────────────────────────────────
# Before the agent sends any alert, it must get human approval.
# This is the core HITL pattern — the agent pauses and asks a human.

def human_approval_gate(risk_summary: str, risk_level: str) -> bool:
    """
    Pauses execution and asks a human to approve or reject an alert.
    
    In production, this could:
      - Send a Slack message and wait for a reaction
      - Open a dashboard approval screen
      - Send an email with an approve/reject link
    
    For this workshop, we use simple console input.
    
    Returns:
      True  → Human approved, proceed with alert
      False → Human rejected, discard alert
    """
    print("\n" + "═" * 60)
    print("🔔  HUMAN APPROVAL REQUIRED — SUPPLY CHAIN ALERT")
    print("═" * 60)
    print(f"\n📊 Risk Level: {risk_level.upper()}")
    print(f"\n📋 Summary:\n{risk_summary}")
    print("\n" + "─" * 60)

    while True:
        response = input(
            "\n❓ Do you want to send this alert to the operations team?\n"
            "   Type 'yes' to approve or 'no' to discard: "
        ).strip().lower()

        if response in ("yes", "y"):
            print("✅  Alert approved. Sending to operations team...")
            return True
        elif response in ("no", "n"):
            print("🚫  Alert discarded by operator.")
            return False
        else:
            print("⚠️   Please type 'yes' or 'no'.")


# ─── Step 5: The Agent Prompt ─────────────────────────────────────────────────
# The ReAct prompt is the "brain" of the agent.
# ReAct = Reason + Act. The agent alternates between thinking and doing.

SUPPLY_CHAIN_PROMPT = PromptTemplate.from_template("""
You are a Supply Chain Risk Analyst AI. Your job is to monitor risks
that could disrupt our company's supply chain and logistics operations.

You have access to the following tools:
{tools}

Tool names available: {tool_names}

INSTRUCTIONS:
1. Analyse the query carefully
2. Search for relevant, current information using your tools
3. Identify concrete risks (NOT speculation)
4. Assess severity: LOW, MEDIUM, or HIGH
5. Provide a clear, concise summary with actionable recommendations

Use this exact format for EVERY response step:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: [exact input for the tool]
Observation: [what the tool returned]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to give a final answer.
Final Answer: [Your complete risk assessment with severity level and recommendations]

IMPORTANT RULES:
- Only report risks backed by evidence from your searches
- Always include a severity level: LOW / MEDIUM / HIGH
- Keep recommendations specific and actionable
- If no significant risk is found, say so clearly

Begin!

Question: {input}
{agent_scratchpad}
""")


# ─── Step 6: Build and Run the Agent ─────────────────────────────────────────

def build_agent(provider: str = "openai", model_name: str = None):
    """
    Assembles the full agent from its parts.
    
    This is the "assembly" function — it wires together:
      1. The LLM (swappable via provider argument)
      2. The tools (search capabilities)
      3. The prompt (agent's instructions)
      4. The observability handler (logging)
    """
    llm = get_llm(provider=provider, model_name=model_name)
    tools = build_tools()
    observability = SupplyChainObservabilityHandler()

    # create_react_agent creates the reasoning chain: Thought → Action → Observation
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=SUPPLY_CHAIN_PROMPT,
    )

    # AgentExecutor is the loop that keeps running until the agent says "Final Answer"
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[observability],
        verbose=True,          # Shows the full chain of thought
        max_iterations=6,      # Safety limit — prevents infinite loops
        handle_parsing_errors=True,  # Gracefully handles malformed LLM output
    )

    return executor


def run_supply_chain_monitor(
    query: str,
    provider: str = "openai",
    model_name: str = None,
    require_approval: bool = True,
):
    """
    Main entry point — runs the agent on a supply chain query.
    
    Args:
      query:            The risk question to investigate
      provider:         LLM provider ("openai", "anthropic", "google")
      model_name:       Specific model (optional — uses defaults if None)
      require_approval: If True, triggers Human-in-the-Loop before alerting
    
    Example:
      run_supply_chain_monitor(
          query="Are there any port delays in Vancouver affecting lumber imports?",
          provider="anthropic"
      )
    """
    print(f"\n{'═' * 60}")
    print(f"🚢  SUPPLY CHAIN MONITOR")
    print(f"    Provider: {provider} | Model: {model_name or 'default'}")
    print(f"{'═' * 60}")
    print(f"📝  Query: {query}\n")

    executor = build_agent(provider=provider, model_name=model_name)
    result = executor.invoke({"input": query})
    output = result.get("output", "No result returned.")

    # ── Human-in-the-Loop Gate ─────────────────────────────────────────────
    # We check the result for risk signals before deciding to alert.
    # In a real system, you'd parse severity from the structured output.

    risk_keywords = ["HIGH", "critical", "severe", "urgent", "immediate"]
    is_high_risk = any(word.lower() in output.lower() for word in risk_keywords)

    if require_approval and is_high_risk:
        # Pause — ask a human before sending the alert
        approved = human_approval_gate(
            risk_summary=output,
            risk_level="HIGH" if "HIGH" in output else "MEDIUM",
        )
        if approved:
            print("\n📧  [SIMULATED] Alert sent to: operations-team@yourcompany.com")
            print("    Slack notification sent to: #supply-chain-alerts")
    else:
        print("\n📊  Risk level does not require immediate escalation.")
        print("    Result logged to observability dashboard.")

    return output


# ─── Quick Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Workshop Exercise: Change "openai" to "anthropic" or "google" and re-run
    # Notice: the agent behaviour is identical — only the LLM changes.

    run_supply_chain_monitor(
        query=(
            "Search for any recent supply chain disruptions affecting electronics "
            "components from Southeast Asia, particularly semiconductor shortages "
            "or shipping delays from Taiwan or South Korea."
        ),
        provider="openai",         # ← Change this to switch models
        model_name="gpt-4o-mini",  # ← Or change this for a specific model
        require_approval=True,     # ← Set False to skip Human-in-the-Loop
    )
