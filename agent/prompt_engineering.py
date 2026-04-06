"""
Prompt Engineering & Agent Reliability
========================================
Workshop Module: Why prompts break agents, and how to fix them.

This file demonstrates five concrete prompt engineering techniques:
  1. Output Format Enforcement  — force structured, parseable responses
  2. Role + Constraint Framing  — give the agent a clear identity and rules
  3. Few-Shot Examples          — show the agent what good output looks like
  4. Failure Mode Handling      — tell the agent what to do when it can't find anything
  5. Prompt Versioning          — track prompt changes the same way you track code changes

Why this matters:
  Your agent's intelligence is NOT just the LLM you chose.
  It is 50% the quality of your prompt. A GPT-4o agent with a bad prompt
  will be outperformed by a GPT-4o-mini agent with a great prompt.
  This is the most cost-effective lever you have.
"""

from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
import re
import json
from dataclasses import dataclass, field
from typing import Optional
from .supply_chain_agent import get_llm, build_tools, SupplyChainObservabilityHandler


# ─── Technique 1: Output Format Enforcement ───────────────────────────────────
# The #1 cause of agent failures in production is the LLM deviating from
# the expected output format. The agent executor can't parse a free-form
# response — it needs Thought/Action/Action Input/Final Answer exactly.
#
# The fix: be brutally explicit about format in the prompt.
# Show the agent an example of a COMPLETE, correct response cycle.

PROMPT_V1_WEAK = PromptTemplate.from_template(
    """
You are a supply chain analyst. Use tools to find risks.

Tools: {tools}
Tool names: {tool_names}

Answer the question: {input}
{agent_scratchpad}
"""
)
# ❌ Problem: No format instructions. The LLM will sometimes just answer
# directly without using tools, or output a narrative instead of
# Thought/Action/Observation. This causes OutputParserException.


PROMPT_V2_FORMAT_ENFORCED = PromptTemplate.from_template(
    """
You are a Supply Chain Risk Analyst. Your ONLY job is to find supply chain risks.

Tools available:
{tools}

You MUST use EXACTLY this format for every step. Never deviate:

Thought: [what you are going to do and why]
Action: [must be one of: {tool_names}]
Action Input: [the exact string to pass to the tool]
Observation: [the tool result — do NOT write this yourself, it is filled in for you]

When you have enough information, end with:
Thought: I now have enough information for a final answer.
Final Answer: SEVERITY: [LOW|MEDIUM|HIGH]
SUMMARY: [2-3 sentence summary of the risk]
RECOMMENDATION: [1 specific action the operations team should take]
SOURCES: [what searches confirmed this]

Begin.

Question: {input}
{agent_scratchpad}
"""
)
# ✅ What changed:
#   - "You MUST use EXACTLY this format" — direct instruction, not suggestion
#   - Tool names listed inline in the Action line — agent knows its options
#   - Final Answer has a defined schema (SEVERITY / SUMMARY / RECOMMENDATION)
#   - "do NOT write this yourself" on Observation — prevents hallucinated results


# ─── Technique 2: Role + Constraint Framing ───────────────────────────────────
# Vague roles produce vague outputs. Specific roles with explicit constraints
# produce focused, reliable outputs. Think of this as giving the agent
# a job description, not just a job title.

PROMPT_V3_ROLE_CONSTRAINED = PromptTemplate.from_template(
    """
You are a Supply Chain Risk Analyst at a Canadian manufacturing company.
You monitor risks to our supplier network, shipping routes, and raw material sources.

YOUR CONSTRAINTS — follow these without exception:
1. Only report risks that are supported by evidence from your search results
2. Never speculate or infer risks not found in the search data
3. If your search returns no relevant results, say so explicitly — do not invent risks
4. Focus only on risks affecting: ports, shipping lanes, suppliers, tariffs, or raw materials
5. Ignore general business news unrelated to physical supply chain

Tools available:
{tools}

You MUST use EXACTLY this format — every step, no exceptions:

Thought: [your reasoning]
Action: [one of: {tool_names}]
Action Input: [search query]
Observation: [filled in automatically]

Final Answer: 
SEVERITY: [LOW | MEDIUM | HIGH]
SUMMARY: [evidence-based, 2-3 sentences only]
RECOMMENDATION: [one specific, actionable step]
CONFIDENCE: [HIGH if multiple sources confirm / LOW if only one source]

Question: {input}
{agent_scratchpad}
"""
)
# ✅ What changed from V2:
#   - Specific company context ("Canadian manufacturing company") anchors responses
#   - "Without exception" on constraints signals non-negotiable rules
#   - Explicit "if no results found" instruction prevents hallucination
#   - CONFIDENCE field added — teaches the agent to signal its own uncertainty


# ─── Technique 3: Few-Shot Examples ───────────────────────────────────────────
# Few-shot prompting shows the agent a complete, correct example before
# asking it to handle the real query. This is the single most reliable
# technique for getting consistent output format.
#
# Cost note: Few-shot examples add tokens. For GPT-4o-mini, the example
# below adds ~300 tokens ≈ $0.00015 per call. Worth it for reliability.

FEW_SHOT_EXAMPLE = """
--- EXAMPLE (study this format, then follow it for the real question) ---

Question: Are there any shipping delays at the Port of Vancouver?

Thought: I need to search for current information about Port of Vancouver delays.
Action: supply_chain_search
Action Input: Port of Vancouver shipping delays strikes 2025
Observation: [search results about port congestion and labour negotiations]

Thought: I found relevant results. I have enough to give a final answer.
Final Answer:
SEVERITY: MEDIUM
SUMMARY: The Port of Vancouver is experiencing 2-3 day delays due to equipment maintenance on Berths 4 and 5, according to Port Authority notices. No labour action is currently underway but union contract talks resume next month.
RECOMMENDATION: Notify tier-1 lumber suppliers to add 5-day buffer to delivery estimates for the next 3 weeks.
CONFIDENCE: HIGH — confirmed by Port Authority official notice and two freight broker reports.

--- END EXAMPLE — now answer the real question below ---
"""

PROMPT_V4_FEW_SHOT = PromptTemplate.from_template(
    """
You are a Supply Chain Risk Analyst at a Canadian manufacturing company.

YOUR CONSTRAINTS:
1. Only report risks supported by search evidence
2. Never speculate or invent risks
3. If no relevant results found, say "No current risk signals found" — do not fill gaps
4. Focus on: ports, shipping lanes, suppliers, tariffs, raw materials

Tools: {tools}
Available tool names: {tool_names}

"""
    + FEW_SHOT_EXAMPLE
    + """

You MUST use EXACTLY this format:
Thought: [reasoning]
Action: [one of: {tool_names}]
Action Input: [search query]
Observation: [filled in automatically]
...repeat...
Final Answer:
SEVERITY: [LOW | MEDIUM | HIGH]
SUMMARY: [2-3 sentences, evidence-based]
RECOMMENDATION: [one specific action]
CONFIDENCE: [HIGH | LOW]

Question: {input}
{agent_scratchpad}
"""
)


# ─── Technique 4: Parsing the Structured Output ───────────────────────────────
# Because our Final Answer now has a defined schema, we can parse it
# reliably into a Python dict. This is what makes the agent's output
# usable by downstream systems (databases, Slack, dashboards).


@dataclass
class RiskAssessment:
    """Structured output from the agent — safe to store and display."""

    severity: str = "UNKNOWN"
    summary: str = ""
    recommendation: str = ""
    confidence: str = "LOW"
    raw_output: str = ""
    parse_succeeded: bool = False


def parse_agent_output(raw_output: str) -> RiskAssessment:
    """
    Parses the agent's Final Answer into a RiskAssessment object.

    Workshop Note:
      This is defensive parsing — it handles partial matches and
      falls back gracefully when the LLM deviates from the format.
      Never assume the LLM will follow instructions perfectly every time.
    """
    result = RiskAssessment(raw_output=raw_output)

    try:
        # Extract SEVERITY
        sev_match = re.search(
            r"SEVERITY:\s*(LOW|MEDIUM|HIGH)", raw_output, re.IGNORECASE
        )
        if sev_match:
            result.severity = sev_match.group(1).upper()

        # Extract SUMMARY
        sum_match = re.search(
            r"SUMMARY:\s*(.+?)(?=RECOMMENDATION:|CONFIDENCE:|$)",
            raw_output,
            re.DOTALL | re.IGNORECASE,
        )
        if sum_match:
            result.summary = sum_match.group(1).strip()

        # Extract RECOMMENDATION
        rec_match = re.search(
            r"RECOMMENDATION:\s*(.+?)(?=CONFIDENCE:|$)",
            raw_output,
            re.DOTALL | re.IGNORECASE,
        )
        if rec_match:
            result.recommendation = rec_match.group(1).strip()

        # Extract CONFIDENCE
        conf_match = re.search(r"CONFIDENCE:\s*(HIGH|LOW)", raw_output, re.IGNORECASE)
        if conf_match:
            result.confidence = conf_match.group(1).upper()

        result.parse_succeeded = bool(sev_match and sum_match)

    except Exception as e:
        print(f"⚠️  Parse warning: {e} — falling back to raw output")

    return result


def format_assessment_for_display(assessment: RiskAssessment) -> str:
    """Formats a RiskAssessment for clean console or Slack output."""
    icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
    icon = icons.get(assessment.severity, "⚪")

    lines = [
        f"\n{'═' * 55}",
        f"{icon}  RISK ASSESSMENT — {assessment.severity}",
        f"{'═' * 55}",
        f"\n📋 SUMMARY:\n{assessment.summary}",
        f"\n✅ RECOMMENDATION:\n{assessment.recommendation}",
        f"\n🎯 CONFIDENCE: {assessment.confidence}",
        f"{'─' * 55}",
    ]
    return "\n".join(lines)


# ─── Technique 5: Prompt Versioning ───────────────────────────────────────────
# Prompts should be versioned exactly like code. When you change a prompt,
# you should know: what changed, when, why, and whether it improved outcomes.
# This is the difference between prompt engineering and prompt guessing.


@dataclass
class PromptVersion:
    """Tracks a prompt version with metadata."""

    version: str
    prompt: PromptTemplate
    description: str
    added_date: str
    known_failure_modes: list = field(default_factory=list)


PROMPT_REGISTRY = {
    "v1": PromptVersion(
        version="v1",
        prompt=PROMPT_V1_WEAK,
        description="Initial prompt — minimal instructions",
        added_date="2025-01-01",
        known_failure_modes=[
            "Often skips tools and answers directly from training data",
            "Output format inconsistent — hard to parse",
            "No failure handling when search returns nothing",
        ],
    ),
    "v2": PromptVersion(
        version="v2",
        prompt=PROMPT_V2_FORMAT_ENFORCED,
        description="Added strict format enforcement and structured Final Answer",
        added_date="2025-01-15",
        known_failure_modes=[
            "Can still speculate when search results are thin",
            "No confidence signalling",
        ],
    ),
    "v3": PromptVersion(
        version="v3",
        prompt=PROMPT_V3_ROLE_CONSTRAINED,
        description="Added role framing, constraints, confidence field",
        added_date="2025-02-01",
        known_failure_modes=[
            "Occasional format deviation on smaller models (Haiku, Flash)",
        ],
    ),
    "v4": PromptVersion(
        version="v4",
        prompt=PROMPT_V4_FEW_SHOT,
        description="Added few-shot example — most reliable format compliance",
        added_date="2025-02-15",
        known_failure_modes=[
            "Higher token cost due to example (~300 extra tokens per call)",
        ],
    ),
}


def get_prompt(version: str = "v4") -> PromptTemplate:
    """
    Returns a versioned prompt by key.
    Default is always the latest stable version.

    Workshop Exercise:
      Run the agent with version='v1' and version='v4' on the same query.
      Compare: output consistency, parse success rate, hallucination rate.
    """
    if version not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown prompt version '{version}'. Available: {list(PROMPT_REGISTRY.keys())}"
        )
    pv = PROMPT_REGISTRY[version]
    print(f"📝 Using prompt {pv.version}: {pv.description}")
    if pv.known_failure_modes:
        print(f"⚠️  Known failure modes: {'; '.join(pv.known_failure_modes)}")
    return pv.prompt


# ─── Demo: Run and Compare Prompt Versions ────────────────────────────────────


def run_with_prompt_version(
    query: str,
    prompt_version: str = "v4",
    provider: str = "openai",
    model_name: Optional[str] = None,
):
    """
    Runs the agent with a specific prompt version and returns a parsed assessment.

    Use this in the workshop to show how prompt version affects reliability.
    """

    prompt = get_prompt(version=prompt_version)
    llm = get_llm(provider=provider, model_name=model_name)
    tools = build_tools()
    handler = SupplyChainObservabilityHandler()

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=[handler],
        verbose=False,  # Keep output clean for the comparison demo
        max_iterations=6,
        handle_parsing_errors=True,
    )

    print(f"\n{'═' * 55}")
    print(f"🧪 PROMPT VERSION COMPARISON")
    print(f"   Version: {prompt_version} | Provider: {provider}")
    print(f"   Query: {query[:80]}")
    print(f"{'═' * 55}\n")

    result = executor.invoke({"input": query})
    raw_output = result.get("output", "")

    assessment = parse_agent_output(raw_output)

    print(format_assessment_for_display(assessment))
    print(f"\n🔍 Parse succeeded: {assessment.parse_succeeded}")

    return assessment


# ─── Quick Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Workshop Demo: Run the same query with v1 and v4 — compare results
    QUERY = "Are there any current disruptions at major Canadian ports?"

    print("\n" + "█" * 55)
    print("DEMO: Prompt V1 (weak) vs V4 (engineered)")
    print("█" * 55)

    # V1: Weak prompt — expect inconsistent output
    print("\n--- Running with V1 (weak prompt) ---")
    result_v1 = run_with_prompt_version(QUERY, prompt_version="v1")

    print("\n\n--- Running with V4 (engineered prompt) ---")
    result_v4 = run_with_prompt_version(QUERY, prompt_version="v4")

    # Side-by-side comparison
    print("\n\n" + "═" * 55)
    print("COMPARISON SUMMARY")
    print("═" * 55)
    print(f"{'Metric':<25} {'V1':<15} {'V4':<15}")
    print(f"{'─'*25} {'─'*15} {'─'*15}")
    print(
        f"{'Parse succeeded':<25} {str(result_v1.parse_succeeded):<15} {str(result_v4.parse_succeeded):<15}"
    )
    print(
        f"{'Severity extracted':<25} {result_v1.severity:<15} {result_v4.severity:<15}"
    )
    print(
        f"{'Confidence reported':<25} {result_v1.confidence:<15} {result_v4.confidence:<15}"
    )
