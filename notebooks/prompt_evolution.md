# Prompt Evolution — Supply Chain Agent Workshop

> **Key insight:** Your agent's intelligence is not just the LLM you chose.
> It is 50% the quality of your prompt. A well-prompted `gpt-4o-mini` will
> outperform a poorly-prompted `gpt-4o` — at a fraction of the cost.

---

## V1 — Minimal (the wrong way)

**Status:** ❌ Unreliable in production

```
You are a supply chain analyst. Use tools to find risks.

Tools: {tools}
Tool names: {tool_names}

Answer: {input}
{agent_scratchpad}
```

**What goes wrong:**
- LLM skips tools and answers from memory (stale data)
- Output format is inconsistent — AgentExecutor cannot parse it
- No handling when search returns nothing — agent invents risks

---

## V2 — Format Enforced

**Status:** ⚠️ Better, but still speculates

**What changed:** Added strict Thought/Action/Observation format and a defined Final Answer schema.

```
You MUST use EXACTLY this format — never deviate:

Thought: [reasoning]
Action: [one of: {tool_names}]
Action Input: [search query]
Observation: [filled in automatically — do not write this yourself]

Final Answer:
SEVERITY: [LOW | MEDIUM | HIGH]
SUMMARY: [2-3 sentence summary]
RECOMMENDATION: [one specific action]
SOURCES: [what searches confirmed this]
```

**Key additions:**
- `"You MUST use EXACTLY this format"` — direct instruction, not suggestion
- `"do NOT write this yourself"` on Observation — prevents hallucinated results
- Final Answer has a defined schema — output is now parseable

**Remaining failure modes:**
- Can still speculate when search results are thin
- No confidence signalling

---

## V3 — Role + Constraints

**Status:** ⚠️ Reliable on strong models, occasional drift on smaller ones

**What changed:** Added specific company context, non-negotiable constraints, and a CONFIDENCE field.

```
You are a Supply Chain Risk Analyst at a Canadian manufacturing company.

YOUR CONSTRAINTS — follow these without exception:
1. Only report risks supported by evidence from your search results
2. Never speculate or infer risks not found in the search data
3. If your search returns no relevant results, say so explicitly — do not invent risks
4. Focus only on: ports, shipping lanes, suppliers, tariffs, raw materials

Final Answer:
SEVERITY: [LOW | MEDIUM | HIGH]
SUMMARY: [evidence-based, 2-3 sentences]
RECOMMENDATION: [one specific, actionable step]
CONFIDENCE: [HIGH if multiple sources confirm / LOW if only one source]
```

**Key additions:**
- `"Canadian manufacturing company"` — anchors responses to a specific context
- `"without exception"` — signals non-negotiable rules to the LLM
- Explicit empty-result instruction — prevents hallucination
- `CONFIDENCE` field — teaches the agent to signal its own uncertainty

**Remaining failure modes:**
- Occasional format deviation on smaller models (Haiku, Flash)

---

## V4 — Few-Shot Example

**Status:** ✅ Most reliable — recommended for production

**What changed:** A complete worked example is embedded before the real question.

> **Why this works:** The LLM has seen a correct full Thought/Action/Observation
> cycle before it starts its own. It copies the pattern rather than inventing one.
>
> **Cost note:** Adds ~300 tokens per call ≈ $0.00015 on gpt-4o-mini. Worth it.

```
--- EXAMPLE (study this format, then follow it for the real question) ---

Question: Are there any shipping delays at the Port of Vancouver?

Thought: I need to search for current information about Port of Vancouver delays.
Action: supply_chain_search
Action Input: Port of Vancouver shipping delays strikes 2025
Observation: [search results about port congestion and labour negotiations]

Thought: I found relevant results. I have enough to give a final answer.
Final Answer:
SEVERITY: MEDIUM
SUMMARY: The Port of Vancouver is experiencing 2-3 day delays due to equipment
maintenance on Berths 4 and 5. No labour action is currently underway but union
contract talks resume next month.
RECOMMENDATION: Notify tier-1 lumber suppliers to add 5-day buffer to delivery
estimates for the next 3 weeks.
CONFIDENCE: HIGH — confirmed by Port Authority notice and two freight broker reports.

--- END EXAMPLE — now answer the real question below ---
```

**Key additions:**
- Complete worked example before the real question
- Highest format compliance across all model sizes
- Parseable output — `SEVERITY`, `SUMMARY`, `RECOMMENDATION`, `CONFIDENCE` all extractable with regex

**Remaining failure modes:**
- Higher token cost (~300 extra tokens per call)

---

## Notebook Prompt — Live Workshop Version

**Status:** ✅ Production-ready with date injection and stop rules

**What changed:** Today's date injected via Python f-string. Stop rules prevent infinite loops when search results are poor.

```python
from datetime import datetime
today = datetime.now().strftime("%B %Y")  # e.g. "April 2026"

PROMPT = PromptTemplate.from_template(f"""
You are a Supply Chain Risk Analyst. Today's date is {today}.

CRITICAL RULES — FOLLOW WITHOUT EXCEPTION:
1. You MUST call a tool at least once before giving a Final Answer
2. Never answer from memory — always search first
3. Always include the current month and year in your search queries
4. If search results are not relevant, try ONE different query then move on
5. Never repeat the same search query twice — if it failed once, it will fail again
6. After 3 tool calls, write your Final Answer using whatever you found

Tools available:
{{tools}}

Available tool names: {{tool_names}}

Thought: I need to search for current information about this.
Action: supply_chain_search
Action Input: [your search query including {today}]
Observation: [tool result — filled in automatically]
Thought: I now have enough information for a final answer.
Final Answer: SEVERITY: [LOW|MEDIUM|HIGH]
SUMMARY: [2-3 sentences based on search results]
RECOMMENDATION: [one specific action]

Question: {{input}}
{{agent_scratchpad}}
""")
```

**Key additions:**
- `today` injected at runtime — agent always searches with current month/year
- Rule 5: `"Never repeat the same query"` — prevents looping on failed searches
- Rule 6: `"After 3 tool calls, write your Final Answer"` — forces synthesis
- `max_iterations` reduced from 6 → 4

---

## Summary Comparison

| Version | Tools Used? | Output Parseable? | Hallucinates? | Token Cost |
|---------|-------------|-------------------|---------------|------------|
| V1 | Sometimes | No | Often | Lowest |
| V2 | Usually | Yes | Sometimes | Low |
| V3 | Yes | Yes | Rarely | Low |
| V4 | Yes | Yes | Very rarely | Medium (+300) |
| Notebook | Yes | Yes | No | Medium |

---

## The Rule to Remember

> **Vague prompt → vague agent.**
> Every failure mode in V1 is a missing instruction in the prompt.
> The fix is always to be more explicit — not to upgrade the model.
