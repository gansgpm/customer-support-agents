"""Hierarchical AutoGen team: a Support Manager on top of one agent group per division.

    SelectorGroupChat ("support_desk")
    ├── support_manager                      reads the ticket, picks the owning division, briefs the group
    ├── general_group          RoundRobin    general_query_agent → general_knowledge_agent → ... → general_log_agent
    ├── electronics_group      RoundRobin    electronics_query_agent → ... → electronics_log_agent
    ├── home_appliances_group  ...
    └── ... one group per product line / service in knowledge_base/catalog.json

The selector hands the floor to the Manager first and then to the group named in the Manager's
``ROUTE:`` line (the LLM selector is only a fallback if that line is missing). The run stops when the
group's Log Agent says LOGGED; a source-match condition and a message cap are the safety nets.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.conditions import (FunctionalTermination, MaxMessageTermination, SourceMatchTermination,
                                         TextMentionTermination)
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_core.models import ChatCompletionClient

from agents.context import AgentContext, agent_name
from agents.knowledge_agent import create_knowledge_agent
from agents.log_agent import create_log_agent
from agents.manager_agent import create_manager_agent
from agents.query_agent import create_query_agent
from agents.resolution_agent import create_resolution_agent
from engine.guardrail import manager_says_none
from engine.parsing import extract_json
from engine.state import GROUP_ROLES, MANAGER

SELECTOR_PROMPT = """You coordinate Remya Industries Ltd's AI customer support desk. The participants are:
{roles}

The Support Manager speaks first and names the division whose agent group must handle the ticket.
Then that division's group works the ticket (Customer Query -> Knowledge Base -> Resolution -> Log).

Conversation so far:
{history}

Read the conversation, then select the next participant from {participants}. Only return the name."""

INNER_SELECTOR_PROMPT = """You coordinate one division group of Remya Industries Ltd's AI support desk. The roles are:
{roles}

The normal flow is: query agent triages -> knowledge agent retrieves the facts -> resolution agent writes the reply ->
log agent records the ticket and decides escalation. Never pick a role whose inputs are not ready yet.

Conversation so far:
{history}

Read the conversation, then select the next role from {participants}. Only return the role name."""


def group_name(division: str) -> str:
    return f"{division}_group"


def parse_route(text: str, valid: Sequence[str]) -> Optional[str]:
    """Division named by the Manager: the ROUTE line first, then the JSON 'division' field."""
    m = re.search(r"ROUTE:\s*`?([a-z_]+)`?", text or "", flags=re.I)
    candidates = [m.group(1).lower()] if m else []
    obj = extract_json(text or "") or {}
    if obj.get("division"):
        candidates.append(str(obj["division"]).strip().lower())
    for c in candidates:
        c = c[:-6] if c.endswith("_group") else c
        if c in valid:
            return c
    return None


def build_division_group(model_client: ChatCompletionClient, ctx: AgentContext, division: str):
    agents = [
        create_query_agent(model_client, ctx, division),
        create_knowledge_agent(model_client, ctx, division),
        create_resolution_agent(model_client, ctx, division),
        create_log_agent(model_client, ctx, division),
    ]
    log_name = agent_name(division, "log_agent")
    termination = TextMentionTermination(ctx.settings.termination_word, sources=[log_name]) | MaxMessageTermination(
        ctx.settings.max_messages)
    line = ctx.store.catalog.get(division)
    description = f"{line.name} agent group: {line.description}"
    if ctx.settings.orchestration_mode == "selector":
        names = [a.name for a in agents]

        def remaining(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str]:
            spoken = {m.source for m in messages if isinstance(m, BaseChatMessage)}
            return [n for n in names if n not in spoken] or [names[-1]]

        team = SelectorGroupChat(agents, model_client=model_client, name=group_name(division), description=description,
                                 termination_condition=termination, selector_prompt=INNER_SELECTOR_PROMPT,
                                 allow_repeated_speaker=False, candidate_func=remaining)
    else:
        team = RoundRobinGroupChat(agents, name=group_name(division), description=description,
                                   termination_condition=termination)
    return team, agents


def build_team(model_client: ChatCompletionClient, ctx: AgentContext) -> Tuple[object, Dict[str, list]]:
    catalog = ctx.store.catalog
    divisions = catalog.ids()  # general + every product line and service
    manager = create_manager_agent(model_client, ctx)
    groups, members = [], {MANAGER: [manager]}
    for division in divisions:
        team, agents = build_division_group(model_client, ctx, division)
        groups.append(team)
        members[division] = agents

    log_agents = [agent_name(d, "log_agent") for d in divisions]
    group_agents = [agent_name(d, r) for d in divisions for r in GROUP_ROLES]
    def not_a_ticket(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> bool:
        """Manager guardrail: 'ROUTE: none' (not a support request) ends the run before any group works."""
        return any(isinstance(m, BaseChatMessage) and m.source == MANAGER
                   and manager_says_none(m.to_text(), ctx.state.ticket.complaint) for m in messages)

    termination = (TextMentionTermination(ctx.settings.termination_word, sources=log_agents)
                   | FunctionalTermination(not_a_ticket)
                   | SourceMatchTermination(group_agents)  # a group finished without logging -> stop, engine logs
                   | MaxMessageTermination(ctx.settings.max_messages + 4))

    def route(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> Optional[str]:
        chat = [m for m in messages if isinstance(m, BaseChatMessage)]
        manager_msgs = [m for m in chat if m.source == MANAGER]
        if not manager_msgs:
            return MANAGER
        division = parse_route(manager_msgs[-1].to_text(), divisions)
        if division is None:  # Manager forgot the ROUTE line: use its tool's detection, else let the model pick
            division = ctx.state.product_signal.get("primary") if ctx.state.product_signal else None
        ctx.state.division = division or ctx.state.division
        return group_name(division) if division else None

    desk = SelectorGroupChat(
        [manager] + groups,
        model_client=model_client,
        name="support_desk",
        termination_condition=termination,
        selector_prompt=SELECTOR_PROMPT,
        selector_func=route,
        allow_repeated_speaker=False,
    )
    return desk, members
