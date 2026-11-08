#!/usr/bin/env python
"""Week 9 - Task 2: Chatbot with memory, selectable persona and streaming output.

    python chatbot.py
    python chatbot.py --persona tutor
    python chatbot.py --paper paper.pdf          # chat about a paper (uses Task 1's reader)

Commands inside the chat:
    /persona <assistant|tutor|researcher|concise>   change personality (keeps the memory)
    /reset      forget the conversation
    /history    show the stored messages
    /exit       quit
"""
import argparse

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from llm_factory import get_llm

PERSONAS = {
    "assistant": "You are a friendly, helpful assistant.",
    "tutor": "You are a patient tutor. Explain step by step, use simple examples and finish by checking the student's understanding with one short question.",
    "researcher": "You are a careful research assistant. Be precise, say when you are unsure, and never invent references.",
    "concise": "You are a very concise assistant. Answer in at most three sentences.",
}

SYSTEM = """{persona}

{paper_context}
Answer in the language the user writes in. If you do not know something, say so instead of guessing."""

MAX_MESSAGES = 20          # keep only the last 20 messages (10 turns) so the prompt never grows without limit

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

_store = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    history = _store[session_id]
    if len(history.messages) > MAX_MESSAGES:      # simple sliding-window memory
        history.messages = history.messages[-MAX_MESSAGES:]
    return history


def build_chatbot(persona: str = "assistant", paper_text: str = ""):
    """Return a chain that remembers the conversation per session_id."""
    paper_context = ""
    if paper_text:
        paper_context = ("The user wants to discuss this paper. Use it as your main source.\n"
                         "--- PAPER (may be truncated) ---\n" + paper_text[:15000] + "\n--- END OF PAPER ---\n")
    chain = prompt.partial(persona=PERSONAS[persona], paper_context=paper_context) | get_llm(temperature=0.5, streaming=True) | StrOutputParser()
    return RunnableWithMessageHistory(chain, get_history, input_messages_key="input", history_messages_key="history")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", choices=PERSONAS, default="assistant")
    ap.add_argument("--paper", help="optional PDF/TXT to talk about")
    args = ap.parse_args()

    paper_text = ""
    if args.paper:
        from research_assistant import read_paper
        paper_text = read_paper(args.paper)
        print(f"Loaded paper: {args.paper} ({len(paper_text)} characters)")

    persona, session = args.persona, "cli"
    bot = build_chatbot(persona, paper_text)
    print(f"Chatbot ready (persona: {persona}). Type /exit to quit, /reset to forget, /persona <name> to switch.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            _store.pop(session, None)
            print("Memory cleared.\n")
            continue
        if user == "/history":
            for m in get_history(session).messages:
                print(f"  [{m.type}] {m.content[:100]}")
            print()
            continue
        if user.startswith("/persona"):
            name = user.split(maxsplit=1)[1].strip() if " " in user else ""
            if name in PERSONAS:
                persona = name
                bot = build_chatbot(persona, paper_text)          # same session_id -> memory is kept
                print(f"Persona is now '{persona}'.\n")
            else:
                print(f"Choose one of: {', '.join(PERSONAS)}\n")
            continue

        print("Bot: ", end="", flush=True)
        try:
            for token in bot.stream({"input": user}, config={"configurable": {"session_id": session}}):
                print(token, end="", flush=True)
        except Exception as e:                      # network / key / rate-limit problems
            print(f"\n[error] {e}")
        print("\n")


if __name__ == "__main__":
    main()
