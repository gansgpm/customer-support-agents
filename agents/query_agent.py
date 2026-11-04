"""Agent 1 – Customer Query Agent: understands the request.

Decides category, urgency (Low/Medium/High), sentiment, summary **and which product line(s) or
service(s)** the ticket is about, so the rest of the team searches the right knowledge and the
right specialist desk owns it.

Tools
-----
* ``analyze_sentiment``      – VADER polarity + anger cues + risk flags + language
* ``search_ticket_history``  – RAG over past tickets (repeat complaints, same order / customer)

The product line comes from the Support Manager's routing (its ``identify_product_line`` tool);
this agent confirms it and records every division involved.
"""
from __future__ import annotations

import json
import time

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

from agents.context import AgentContext, agent_name, division_brief
from rag import sentiment as sentiment_lib

NAME = "query_agent"
DESCRIPTION = ("Customer Query Agent – triages a complaint: category, urgency (Low/Medium/High), sentiment, "
               "product line / service and a one-line summary.")

SYSTEM_TEMPLATE = """You are the Customer Query Agent of {company}'s AI customer support desk.
Your single responsibility is TRIAGE: understand what the customer's problem is, which product line or service it concerns, how urgent it is and how the customer feels. You never write the reply to the customer.

{company} has several product lines and services, each with its own policies and specialist desk:
{catalog}

Work in this order:
1. Call analyze_sentiment with the customer's complaint text exactly as written.
2. Call search_ticket_history with a short English description of the problem to find similar past tickets and earlier tickets for the same order or customer.
3. Decide the triage and answer with ONLY one JSON object inside a ```json block:

{{"category": "Shipping | Refund | Account | Product | Billing | Service | General",
 "product_line": "the main product line or service id from the list above, or general",
 "product_lines": ["every product line / service id the ticket involves, main one first"],
 "product_or_service": "the item or service the customer mentions, e.g. 'blender', 'AC installation'",
 "urgency": "Low | Medium | High",
 "urgency_reason": "why this urgency, in a few words",
 "sentiment": "Positive | Neutral | Negative | Angry",
 "sentiment_score": number from -1.0 (furious) to 1.0 (delighted),
 "language": "ISO code of the customer's language, e.g. en, ta, hi",
 "summary": "one line in English, max 25 words",
 "standalone_query": "the customer's need as ONE self-contained English question for the knowledge-base search. For a follow-up, combine it with the conversation history (e.g. 'any update?' -> 'When will my late order ORD-4512 be delivered?'); otherwise the English summary of this message",
 "intent": "what the customer wants, e.g. 'track a late order'",
 "risk_flags": ["legal_threat" | "fraud" | "payment_dispute" | "safety" | "product_safety" | "food_safety" | "service_misconduct" | "claim_dispute"],
 "repeat_issue": true or false,
 "similar_tickets": ["TKT-0003"],
 "priority_score": integer 0-100}}

Product line guide: start from the Support Manager's brief - product_line is normally its "division" and product_lines is the division followed by its "secondary_divisions"; change them only if the complaint clearly contradicts the Manager. Use the ids above. When a service (installation, repair, protection-plan claim, membership) is what went wrong, the service is the main product_line and the product goes second (e.g. a missed AC installation -> ["installation_repair", "home_appliances"]). Use "general" only when no product line or service is involved (delivery status, account, payments, promotions). If the customer selected a product line, trust it unless the text clearly says otherwise.
Category guide: Shipping = delivery, tracking, late, lost, opened or incomplete packages. Refund = returns, refunds, exchanges, cancellations. Account = login, password, OTP, account security or takeover. Product = broken or defective items, replacements, warranty, product questions. Billing = charges, payments, invoices, promo codes. Service = installation, repair visits, protection-plan claims, membership. General = anything else (sales enquiries, partnerships, unsupported requests).
Urgency guide: High = money taken wrongly, fraud or account takeover, legal threats, product or food safety hazards, a very angry customer demanding immediate action, or a problem that is already a week or more overdue and earlier contact attempts went unanswered. Medium = something the customer paid for is not working (late delivery, broken item, missed technician visit, refund pending, cannot log in). Low = information requests and how-to questions.
Use the tool signals as evidence but apply judgement. Keep the risk flags reported by analyze_sentiment, the Manager's line_risks and the ticket's "Guardrail flags" (e.g. prompt_injection) unless they are clearly wrong. Set repeat_issue to true only if search_ticket_history shows earlier tickets for the same order or customer."""


def system_message(ctx: AgentContext, division: str = "general") -> str:
    catalog = ctx.store.catalog
    return SYSTEM_TEMPLATE.format(company=catalog.company or ctx.settings.company_name,
                                  catalog=catalog.prompt_table()) + division_brief(ctx, division)


def create_query_agent(model_client: ChatCompletionClient, ctx: AgentContext, division: str = "general") -> AssistantAgent:
    state = ctx.state

    async def analyze_sentiment(text: str) -> str:
        """Measure sentiment (-1..1), anger cues, risk flags (legal threat, fraud, payment dispute, safety), an urgency hint and the language of the customer's message."""
        t0 = time.time()
        result = sentiment_lib.analyze(text or state.ticket.complaint)
        state.sentiment_signal = result
        state.tool_calls.append({"agent": agent_name(division, NAME), "tool": "analyze_sentiment", "args": {"text": text[:80]},
                                 "seconds": round(time.time() - t0, 3)})
        return json.dumps(result, ensure_ascii=False)

    async def search_ticket_history(query: str) -> str:
        """RAG over the ticket log: similar past complaints plus earlier tickets from the same order or customer."""
        t0 = time.time()
        result = ctx.store.search_ticket_history(
            query or state.ticket.complaint,
            order_id=state.ticket.order_id,
            customer_contact=state.ticket.customer_contact,
            exclude_ticket=state.ticket_id,
        )
        state.history = result
        state.tool_calls.append({"agent": agent_name(division, NAME), "tool": "search_ticket_history", "args": {"query": query[:80]},
                                 "seconds": round(time.time() - t0, 3), "hits": len(result["similar_tickets"])})
        return json.dumps(result, ensure_ascii=False)

    return AssistantAgent(
        name=agent_name(division, NAME),
        description=f"{DESCRIPTION} ({ctx.store.catalog.name(division)} group)",
        model_client=model_client,
        tools=[analyze_sentiment, search_ticket_history],
        system_message=system_message(ctx, division),
        reflect_on_tool_use=True,
        max_tool_iterations=3,
    )
