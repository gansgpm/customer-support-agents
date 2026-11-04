"""Agent 2 – Knowledge Base Agent: retrieves the facts and builds the context pack for the Resolution Agent.

Tool
----
* ``search_kb`` – hybrid RAG (BM25 + FAISS vectors) over two collections:
  the company policy documents and the learned solutions fed back by the Log Agent.
"""
from __future__ import annotations

import json
import time

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

from agents.context import AgentContext, agent_name, division_brief

NAME = "knowledge_agent"
DESCRIPTION = ("Knowledge Base Agent – searches the product-line guides, general policies and verified past solutions "
               "and returns the relevant facts with source files.")

SYSTEM_MESSAGE = """You are the Knowledge Base Agent of Remya Industries Ltd's AI customer support desk.
Your single responsibility is to find the company facts needed to resolve the ticket and hand a clear context pack to the Resolution Agent. You never write to the customer.

search_kb searches two knowledge sources: (1) Remya Industries Ltd policy documents (the .md files) and (2) previously resolved tickets that were verified (learned solutions).
Remya Industries Ltd has several product lines and services. Each has its own guide (electronics.md, home_appliances.md, fashion_footwear.md, remya_fresh_grocery.md, remya_care_installation_repair.md, remya_shield_protection_plans.md, remya_plus_membership.md) on top of the general policies (shipping, refunds, account, billing, support). Passing product_line restricts the search to those guides plus the general policies, so one line's rules never leak into another line's answer.

Work in this order:
1. Read the Customer Query Agent's triage. Call search_kb with a focused English query built from the summary and intent (translate it to English if the customer wrote in another language; for a follow-up message use the triage's standalone_query), the category, and product_line = the triage's product_lines joined by commas (or "general"). If the ticket involves two product lines or services, make sure facts for each are found - search again for the second one if needed. If the results miss the point, search once more with different words.
2. Keep ONLY passages that actually answer this customer's issue. Copy each fact VERBATIM from the passages. Never add numbers, days, fees, steps or promises that are not in the passages. Every passage has a "ref" (S1, S2, ...): keep it with each fact so the reply can be checked against its source.
3. Answer with ONLY one JSON object inside a ```json block:

{"match_found": true or false,
 "sources": ["file.md"],
 "key_facts": [{"ref": "S1", "source": "file.md", "fact": "verbatim sentence from the passage"}],
 "similar_resolutions": [{"ticket_id": "TKT-0002", "note": "how a similar ticket was solved"}],
 "context_brief": "2-4 sentences for the Resolution Agent: what the customer needs, which policy applies, what the reply must include",
 "gaps": "what the documents do not cover (empty string if nothing is missing)"}

When a product-line guide and a general policy both apply, prefer the product-line guide (it is more specific) and mention both files in sources.
You decide match_found yourself from the passages. The search's own match_found is only a word-overlap estimate: it can miss a passage that answers the question in different words, and it can accept loosely related text. Set match_found to true when a passage actually answers what the customer asks, and to false when none does (for example the topic is never mentioned), whatever the estimate says. Learned solutions are supporting evidence only: when they disagree with a policy document, the policy document wins. Use 3 to 6 key_facts."""


def create_knowledge_agent(model_client: ChatCompletionClient, ctx: AgentContext,
                           division: str = "general") -> AssistantAgent:
    state = ctx.state

    async def search_kb(query: str, category: str = "", product_line: str = "") -> str:
        """Search Remya Industries Ltd's policy documents and verified past solutions, scoped to the ticket's product line(s) or service(s) (comma-separated ids, or 'general'). Returns passages with their source file, relevance and whether the knowledge base covers the question."""
        t0 = time.time()
        if not product_line.strip():  # default scope: this group's division + the Manager's secondary divisions
            product_line = ",".join([division] + state.manager_route.get("secondary_divisions", []))
        # Customers may write in Tamil, Hindi, ...: the Query Agent's English summary is searched as well
        query = query or state.ticket.complaint
        # the agent's English query is searched together with the customer's own words (fused with RRF), so a
        # query that drifts from what the customer wrote cannot hide the right passage
        own_words = () if state.ticket.complaint.strip() in query else (state.ticket.complaint,)
        result = ctx.store.search_kb(query, category=category, product_line=product_line,
                                     english_hint=state.triage.get("standalone_query") or state.triage.get("summary", ""),
                                     also=own_words)
        for p in result["policy_passages"] + result["resolved_ticket_matches"]:
            p["ref"] = state.assign_ref(p)  # S1, S2, ... - cited by the Resolution Agent and validated afterwards
        state.kb_results.append(result)
        state.tool_calls.append({"agent": agent_name(division, NAME), "tool": "search_kb",
                                 "args": {"query": query[:80], "category": category, "product_line": product_line},
                                 "seconds": round(time.time() - t0, 3),
                                 "sources": sorted({p["source"] for p in result["policy_passages"]})})
        return json.dumps(result, ensure_ascii=False)

    return AssistantAgent(
        name=agent_name(division, NAME),
        description=f"{DESCRIPTION} ({ctx.store.catalog.name(division)} group)",
        model_client=model_client,
        tools=[search_kb],
        system_message=SYSTEM_MESSAGE + division_brief(ctx, division),
        reflect_on_tool_use=True,
        max_tool_iterations=3,
    )
