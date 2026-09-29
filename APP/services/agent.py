"""
services/agent.py — Vir's Agentic Reasoning Loop

Instead of hard-coded routing (LOOKUP / COMPUTE / HYBRID via regex),
Vir now:
  1. Receives the question and conversation history.
  2. Calls Groq with ALL tool definitions exposed.
  3. The LLM decides which tools to call, in what order.
  4. Tool results are fed back as messages.
  5. The LLM iterates (up to MAX_ROUNDS) until it produces a final text answer.
  6. Returns the final answer string.

This gives the model full agency over retrieval strategy, with no manual routing.

Public API:
  run_agent(question, history)    — blocking, returns dict (existing behaviour)
  stream_agent(question, history) — async generator, yields SSE lines for /chat/stream
"""

import json
import asyncio
import logging
from groq import Groq, RateLimitError, APIConnectionError, APITimeoutError, APIStatusError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
from config import GROQ_API_KEY, GROQ_MODEL
from services.agent_tools import AGENT_TOOL_DEFINITIONS, execute_agent_tool

logger = logging.getLogger(__name__)


def _groq_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (500, 502, 503, 504)
    return False


@retry(
    retry=retry_if_exception(_groq_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_groq(client: Groq, messages: list, tools: list, stream: bool = False, **kwargs):
    """Single retried Groq API call — shared by agent rounds and final synthesis."""
    return client.chat.completions.create( # i need to understand this code
        model=GROQ_MODEL,
        messages=messages,
        tools=tools if tools else None,
        tool_choice="auto" if tools else None,
        temperature=0.2,
        max_tokens=2048,
        stream=stream,
        **kwargs,
    )

_client = Groq(api_key=GROQ_API_KEY)

# Maximum reasoning rounds to prevent infinite loops
MAX_ROUNDS = 6

# ── System Prompt and we might need to change this  ──────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are Vir, the intelligent AI campus assistant for P.T. Lee Chengalvaraya Naicker College of Engineering and Technology (PT Lee CNCET).

You have access to the following tools:

1. **vector_search(query, top_k)** — Search uploaded documents (PDFs, regulations, policies, syllabus, transport schedules, prospectus, etc.) using semantic similarity.
   Use when: questions about academic rules, GPA/CGPA formulas, exam patterns, admission procedures, college facilities, regulations, course descriptions.

2. **sql_query(question)** — Query the campus SQLite database in natural language.
   Use when: questions about specific students (by name or reg no), marks, attendance, grades, arrears, GPA/CGPA values, faculty directory, batch statistics, department-wise data.

3. **find_path(source, destination)** — Get turn-by-turn navigation directions between two campus locations.
4. **list_rooms(query)** — Search for rooms, labs, offices, or facilities by name or category.
5. **get_room_info(room_id)** — Get details about a specific room.

## REASONING RULES

- **Think before calling tools.** Identify which tools are needed and in what order.
- **Call multiple tools when needed.** For example:
  - "Who has the highest marks and where is the exam office?" → sql_query + find_path
  - "Explain the GPA formula and tell me Aathi's current GPA" → vector_search + sql_query
- **If a tool returns no results,** try an alternate query before giving up.
- **Never fabricate data.** Only answer based on tool results. If tools return nothing relevant, say so honestly.
- **For aggregation questions** (average marks, top students, count by dept) → always use sql_query.
- **For concept/policy questions** (what is arrear, how is CGPA calculated) → always use vector_search.
- **For navigation** (how to get to room X, where is the library) → use find_path or list_rooms.

## FACULTY / STAFF LOOKUPS (CRITICAL RULE)
The SQLite database has a **faculty** table with columns: faculty_name, qualification, designation, department, phone_primary, phone_secondary, email, room_cabin_no, class_incharge_role.

**ALWAYS use sql_query** when asked about:
  - A professor or staff member by name (e.g. "who is Dr. Kumar", "find Prof. Priya")
  - A designation (e.g. "who is the HOD of CSE?", "who is the principal?", "find the director")
  - Faculty contact info (phone, email, cabin/room number of a professor)
  - All faculty in a department (e.g. "list all ECE faculty")

**NEVER use vector_search** for faculty/staff person lookups — they are NOT in documents.
Example sql_query calls for faculty:
  - "who is the HOD of IT?" -> sql_query(question="who is the HOD of IT department?")
  - "find Prof. Arularasu" -> sql_query(question="find faculty named Arularasu")
  - "principal contact number" -> sql_query(question="phone number of the principal")

## CITATIONS (IMPORTANT)
- When vector_search returns results, the tool output includes a SOURCES block listing the PDF filename and page numbers.
- Always end your answer with a **Sources:** line citing the relevant documents.
- Format: `**Sources:** Undergraduate Programme - Academic Regulations 2021.pdf (p. 15, 28)`
- For SQL data, no citation is needed (it comes from the live database).

## COLLEGE CONTEXT
- College: P.T. Lee Chengalvaraya Naicker College of Engineering and Technology
- Affiliated to: Anna University, Chennai
- Departments: CSE, IT, ECE, EEE, Mech, AI&DS
- Database contains: students from 2022-2026 batches, marks for IAT/model/university exams, attendance, faculty directory
"""


# ── Public API ─────────────────────────────────────────────────────────────────

def run_agent(question: str, history: list = None) -> dict:
    """
    Run Vir's full agentic reasoning loop.

    Args:
        question: The user's current question.
        history:  Prior conversation turns (list of {role, content} dicts).

    Returns:
        dict with keys:
            answer   — final answer string
            tools_used — list of tool names called
            rounds   — number of reasoning rounds used
    """
    if history is None:
        history = []

    # Build initial message list
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]

    # Include recent conversation history (last 6 turns) 
    for msg in history[-6:]:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Append the current question why do we ned these roles at all 
    messages.append({"role": "user", "content": question})

    tools_used = []
    rounds = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    print(f"\n{'='*60}")
    print(f"[Agent] Starting agentic loop for: {question[:100]}")
    print(f"[Agent] History turns: {len(history)}")

    for round_num in range(1, MAX_ROUNDS + 1):
        rounds = round_num
        print(f"\n[Agent] Round {round_num}/{MAX_ROUNDS} — {len(messages)} messages in context")

        try:
            response = _call_groq(_client, messages, AGENT_TOOL_DEFINITIONS)
        except Exception as e:
            print(f"[Agent] LLM call failed on round {round_num}: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {e}",
                "tools_used": tools_used,
                "rounds": rounds,
                "tokens": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

        # Track token usage for this round
        if hasattr(response, "usage") and response.usage:
            p_tok = getattr(response.usage, "prompt_tokens", 0)
            c_tok = getattr(response.usage, "completion_tokens", 0)
            t_tok = getattr(response.usage, "total_tokens", 0) or (p_tok + c_tok)
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_tokens += t_tok
            print(f"[Agent] Round {round_num} Tokens: Prompt={p_tok:,} | Completion={c_tok:,} | Total={t_tok:,}")

        choice = response.choices[0]
        message = choice.message

        # No tool calls → final text answer
        if not message.tool_calls:
            final_answer = message.content or ""
            print(f"[Agent] Final answer reached on round {round_num} ({len(final_answer)} chars)")
            print(f"[Agent] Tools used: {tools_used}")
            print(f"[Agent] Total Query Tokens: {total_tokens:,} (Prompt: {total_prompt_tokens:,} | Completion: {total_completion_tokens:,})")
            print(f"{'='*60}\n")
            return {
                "answer": final_answer,
                "tools_used": tools_used,
                "rounds": rounds,
                "tokens": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

        # Log tool calls
        print(f"[Agent] LLM requested {len(message.tool_calls)} tool call(s):")
        for tc in message.tool_calls:
            print(f"  → {tc.function.name}({tc.function.arguments[:120]})")

        # Append assistant's tool-call message to the conversation
        messages.append({
            "role": "assistant",
            "content": message.content,  # may be None
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })

        # Execute each tool and append results
        for tc in message.tool_calls:
            tool_name = tc.function.name
            if tool_name not in tools_used:
                tools_used.append(tool_name)

            tool_result = execute_agent_tool(tool_name, tc.function.arguments)

            print(f"[Agent] {tool_name} result preview: {str(tool_result)[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(tool_result),
            })

    # Exhausted all rounds — do one final synthesis call without tools
    print(f"[Agent] Max rounds ({MAX_ROUNDS}) reached. Forcing final synthesis.")
    messages.append({
        "role": "user",
        "content": "Please synthesize a final answer based on the tool results above.",
    })

    try:
        final_response = _call_groq(_client, messages, [], max_tokens=1024)
        if hasattr(final_response, "usage") and final_response.usage:
            p_tok = getattr(final_response.usage, "prompt_tokens", 0)
            c_tok = getattr(final_response.usage, "completion_tokens", 0)
            t_tok = getattr(final_response.usage, "total_tokens", 0) or (p_tok + c_tok)
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_tokens += t_tok
            print(f"[Agent] Final Synthesis Tokens: Prompt={p_tok:,} | Completion={c_tok:,} | Total={t_tok:,}")

        final_answer = final_response.choices[0].message.content or \
            "I was unable to produce a complete answer. Please try rephrasing your question."
    except Exception as e:
        final_answer = f"I encountered an error in the final synthesis step: {e}"

    print(f"[Agent] Synthesized answer ({len(final_answer)} chars)")
    print(f"[Agent] Total Query Tokens: {total_tokens:,} (Prompt: {total_prompt_tokens:,} | Completion: {total_completion_tokens:,})")
    print(f"{'='*60}\n")
    return {
        "answer": final_answer,
        "tools_used": tools_used,
        "rounds": rounds,
        "tokens": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        },
    }


# -- Streaming Agent (async generator for SSE) ---------------------------------

# Human-readable progress labels shown to the user while tools run
_TOOL_PROGRESS_LABELS = {
    "sql_query":     "Querying SQLite database...",
    "vector_search": "Searching documents...",
    "find_path":     "Calculating campus route...",
    "list_rooms":    "Looking up campus rooms...",
    "get_room_info": "Fetching room details...",
}


async def stream_agent(question: str, history: list = None):
    """
    Async generator -- yields raw SSE lines for the /chat/stream endpoint.

    Yielded line formats (each call produces one complete SSE message string):

      event: progress\\ndata: {"message": "..."}\\n\\n
      event: token\\ndata: {"text": "chunk"}\\n\\n
      event: done\\ndata: {"followups": [...], "tools_used": [...], ...}\\n\\n
      event: error\\ndata: {"message": "..."}\\n\\n

    Blocking Groq calls are offloaded with asyncio.to_thread so the event
    loop stays free to flush bytes to the client between tool rounds.
    """
    if history is None:
        history = []

    # Build initial message list (same logic as run_agent)
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    for msg in history[-6:]:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    tools_used = []
    rounds = 0

    try:
        # -- Agentic tool-call rounds (non-streaming; tool calls need full response) --
        for round_num in range(1, MAX_ROUNDS + 1):
            rounds = round_num

            # Blocking Groq call offloaded to thread pool
            response = await asyncio.to_thread(
                _call_groq, _client, messages, AGENT_TOOL_DEFINITIONS
            )

            choice = response.choices[0]
            message = choice.message

            # No tool calls -> ready for final streaming answer
            if not message.tool_calls:
                break

            # Emit one progress event per tool call the LLM requested
            for tc in message.tool_calls:
                label = _TOOL_PROGRESS_LABELS.get(
                    tc.function.name, f"Running {tc.function.name}..."
                )
                yield f'event: progress\ndata: {{"message": "{label}"}}\n\n'

            # Append assistant tool-call message
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            # Execute each tool in a thread and append results
            for tc in message.tool_calls:
                tool_name = tc.function.name
                if tool_name not in tools_used:
                    tools_used.append(tool_name)

                tool_result = await asyncio.to_thread(
                    execute_agent_tool, tool_name, tc.function.arguments
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(tool_result),
                })

        else:
            # MAX_ROUNDS exhausted -- nudge the model to synthesise
            messages.append({
                "role": "user",
                "content": "Please synthesize a final answer based on the tool results above.",
            })

        # -- Stream the final answer token-by-token ---------------------------------
        # The Groq SDK streaming API is synchronous, so we run it in a background
        # thread and pass tokens back via a queue so this async generator can yield
        # them without blocking the event loop.
        import queue as _queue

        chunk_queue: _queue.Queue = _queue.Queue()

        def _run_stream():
            """Run the blocking Groq stream in a thread; push chunks onto the queue."""
            try:
                stream = _client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    tools=None,
                    tool_choice=None,
                    temperature=0.2,
                    max_tokens=2048,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    text = getattr(delta, "content", None) if delta else None
                    if text:
                        chunk_queue.put(text)
            except Exception as exc:
                chunk_queue.put(exc)
            finally:
                chunk_queue.put(None)  # sentinel: stream is done

        loop = asyncio.get_event_loop()
        stream_future = loop.run_in_executor(None, _run_stream)

        full_answer_parts = []

        # Drain the queue, yielding each token to the HTTP response
        while True:
            try:
                item = chunk_queue.get_nowait()
            except _queue.Empty:
                await asyncio.sleep(0.01)  # release control briefly
                continue

            if item is None:  # sentinel -- stream finished
                break
            if isinstance(item, Exception):
                yield f'event: error\ndata: {{"message": "LLM stream error: {item}"}}\n\n'
                return

            full_answer_parts.append(item)
            # Escape characters that would break the inline JSON string
            safe = (
                item
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
            )
            yield f'event: token\ndata: {{"text": "{safe}"}}\n\n'

        await stream_future  # ensure the background thread is fully cleaned up

        # -- Generate followup questions (blocking, run in thread) ------------------
        from services.followups import generate_followup_questions
        full_answer = "".join(full_answer_parts)
        followups = await asyncio.to_thread(generate_followup_questions, question, full_answer)
        followups = followups or []

        # Final metadata event
        import json as _json
        done_payload = _json.dumps({
            "followups": followups,
            "tools_used": tools_used,
            "rounds": rounds,
            "source": "agent",
        })
        yield f"event: done\ndata: {done_payload}\n\n"

    except Exception as exc:
        logger.exception("[stream_agent] Unhandled error")
        yield f'event: error\ndata: {{"message": "Internal error: {exc}"}}\n\n'

