# Copyright 2026 Wes Jackson
# SPDX-License-Identifier: Apache-2.0

"""MemoryHub test agent — demonstrates governed cross-session memory in Kagenti ADK.

This agent uses two stores simultaneously:
- ContextStore (built-in): per-conversation A2A message replay
- MemoryStore (MemoryHub): persistent, governed, cross-session knowledge

On each message the agent searches MemoryHub for relevant memories, includes
them in the LLM prompt as context, and writes new memories when the user
explicitly asks it to remember something or the conversation produces
a noteworthy fact.
"""

import logging
import os
from typing import Annotated

import openai
from a2a.types import AgentSkill, Message
from a2a.utils.message import get_message_text
from kagenti_adk.a2a.extensions import (
    AgentDetail,
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
)
from kagenti_adk.a2a.types import AgentMessage
from kagenti_adk.server import Server
from kagenti_adk.server.context import RunContext
from kagenti_adk.server.dependencies import Depends
from kagenti_adk.server.store.memoryhub_memory_store import (
    MemoryHubMemoryStore,
    MemoryHubMemoryStoreInstance,
    create_memory_dependency,
)

logger = logging.getLogger(__name__)

# --- Memory store setup ---
memory_store = MemoryHubMemoryStore.from_env()
memory_dep = create_memory_dependency(memory_store)

# --- Server ---
server = Server()

SYSTEM_PROMPT = """\
You are a helpful assistant with persistent memory. You can remember things
across conversations because you have access to a governed memory store.

## Your memories
The following memories were retrieved from your memory store based on the
user's current message. Use them to personalize your response:

{memories}

## Instructions
- When the user says "remember" or asks you to keep something in mind,
  respond naturally AND include a line starting with "MEMORY_WRITE:" followed
  by a concise version of what to remember. Example:
    MEMORY_WRITE: User prefers Podman over Docker for container builds
- When recalling memories, reference them naturally in your response.
- If no memories are relevant, just respond normally.
- Be concise and helpful.
"""


def _format_memories(memories: list) -> str:
    if not memories:
        return "(No relevant memories found)"
    lines = []
    for m in memories:
        lines.append(f"- [{m.scope}, weight={m.weight}] {m.content}")
    return "\n".join(lines)


def _get_llm_config(llm: LLMServiceExtensionServer):
    """Extract LLM connection config from the ADK extension."""
    if llm and llm.data and llm.data.llm_fulfillments:
        config = llm.data.llm_fulfillments.get("default")
        if config:
            return config
    # Fallback to environment variables for local dev
    api_base = os.getenv("LLM_API_BASE", os.getenv("LLM_URL", "http://localhost:8000/v1"))
    api_key = os.getenv("LLM_API_KEY", "not-needed")
    model = os.getenv("LLM_MODEL", "gpt-oss-20b")
    return type("Config", (), {"api_base": api_base, "api_key": api_key, "api_model": model})()


async def _build_history(context: RunContext) -> list[dict]:
    """Build chat history from the context store."""
    messages = []
    try:
        async for item in context.load_history():
            msg = item if isinstance(item, Message) else getattr(item, "message", None)
            if msg is None:
                continue
            text = get_message_text(msg)
            if not text:
                continue
            role = "assistant" if hasattr(item, "artifact_id") else "user"
            messages.append({"role": role, "content": text})
    except Exception:
        pass
    return messages


@server.agent(
    name="MemoryHub Test Agent",
    version="0.1.0",
    default_input_modes=["text", "text/plain"],
    default_output_modes=["text", "text/plain"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting=(
            "Hi! I'm an agent with persistent memory powered by MemoryHub. "
            "I can remember things across conversations and even survive pod restarts. "
            "Try asking me to remember something!"
        ),
        framework="Python",
    ),
    skills=[
        AgentSkill(
            id="memory",
            name="Governed Memory",
            description=(
                "Remembers user preferences, project decisions, and context across "
                "conversations using MemoryHub's governed memory store."
            ),
            tags=["memory", "memoryhub", "persistence"],
            examples=[
                "Remember that I prefer Podman over Docker",
                "What do you know about my preferences?",
                "How should I containerize my Python app?",
            ],
        )
    ],
)
async def memoryhub_agent(
    input: Message,
    context: RunContext,
    llm: Annotated[LLMServiceExtensionServer, LLMServiceExtensionSpec.single_demand()],
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    memory: Annotated[MemoryHubMemoryStoreInstance, Depends(memory_dep)],
):
    """Agent with governed cross-session memory via MemoryHub."""
    # Store the incoming message in the context store (A2A replay)
    await context.store(input)
    user_input = get_message_text(input)

    # --- Search MemoryHub for relevant memories ---
    yield trajectory.trajectory_metadata(
        title="Searching memories", content=f"Query: {user_input[:100]}"
    )
    memories = await memory.search(user_input, max_results=5)
    memory_text = _format_memories(memories)
    logger.info("Found %d relevant memories", len(memories))

    yield trajectory.trajectory_metadata(
        title="Memories loaded",
        content=f"Found {len(memories)} relevant memory(ies)",
    )

    # --- Build prompt with memory context ---
    llm_config = _get_llm_config(llm)
    client = openai.AsyncOpenAI(
        api_key=llm_config.api_key, base_url=llm_config.api_base
    )
    model = llm_config.api_model

    system = SYSTEM_PROMPT.format(memories=memory_text)
    history = await _build_history(context)
    messages = [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": user_input},
    ]

    # --- Stream LLM response ---
    yield trajectory.trajectory_metadata(
        title="Generating response", content=f"Using model: {model}"
    )

    stream = await client.chat.completions.create(
        model=model, temperature=0.3, messages=messages, stream=True
    )

    buffer = ""
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            buffer += delta
            yield delta

    # --- Extract and write memories ---
    written_memories = []
    clean_lines = []
    for line in buffer.split("\n"):
        if line.strip().startswith("MEMORY_WRITE:"):
            content = line.strip().removeprefix("MEMORY_WRITE:").strip()
            if content:
                memory_id = await memory.write(content, scope="user", weight=0.8)
                if memory_id:
                    written_memories.append(content)
                    logger.info("Wrote memory: %s (id=%s)", content, memory_id)
        else:
            clean_lines.append(line)

    # Store the assistant response in context store
    clean_response = "\n".join(clean_lines).strip()
    await context.store(AgentMessage(text=clean_response))

    if written_memories:
        yield trajectory.trajectory_metadata(
            title="Memories saved",
            content=f"Wrote {len(written_memories)} memory(ies):\n"
            + "\n".join(f"- {m}" for m in written_memories),
        )

    yield trajectory.trajectory_metadata(
        title="Complete",
        content=f"Response delivered ({len(memories)} recalled, {len(written_memories)} written)",
    )


def run():
    try:
        server.run(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
