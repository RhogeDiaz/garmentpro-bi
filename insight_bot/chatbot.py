import asyncio
import os
from collections.abc import AsyncIterator, Callable

from shiny import ui

from insight_bot.engines import create_chat_engine


def chat_ui():
    return ui.chat_ui(
        "insight_chat",
        greeting=(
            "## InsightBot\n\n"
            "Ask about the charts currently visible in GarmentPro BI. "
            "Responses use Gemini's free-tier API."
        ),
    )


def _next_chunk(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, ""


async def _response_chunks(engine, prompt: str) -> AsyncIterator[str]:
    response = await asyncio.to_thread(engine.stream_chat, prompt)
    chunks = iter(response.response_gen)
    while True:
        has_chunk, chunk = await asyncio.to_thread(_next_chunk, chunks)
        if not has_chunk:
            break
        yield str(chunk)


def register_chat(chart_context_provider: Callable[[], str]):
    chat = ui.Chat(id="insight_chat")
    chat_engine = None

    @chat.on_user_submit
    async def _respond(question: str):
        nonlocal chat_engine
        # Snapshot the active page's chart data for each turn before querying the local engines.
        context = chart_context_provider()
        if context.startswith("No charts are currently visible."):
            await chat.append_message(context)
            return

        stage = "initializing Gemini and the glossary index"
        try:
            if chat_engine is None:
                chat_engine = await asyncio.to_thread(create_chat_engine)
            prompt = (
                f"CURRENT DASHBOARD CONTEXT\n{context}\n\n"
                f"USER QUESTION\n{question}\n\n"
                "Answer in fewer than 200 words and distinguish association from causation."
            )
            stage = "querying Gemini"
            await chat.append_message_stream(_response_chunks(chat_engine, prompt))
        except Exception as error:
            error_text = str(error).lower()
            error_type = type(error).__name__
            detail = " ".join(str(error).split())[:240]
            if any(term in error_text for term in ["api key", "api_key", "credential", "api_key_invalid"]):
                message = (
                    "The Gemini API key is missing or invalid in the environment of "
                    "the process running Shiny. Set GOOGLE_API_KEY there, then restart the app."
                )
            elif any(term in error_text for term in ["429", "quota", "resource_exhausted", "rate limit"]):
                message = (
                    "The Gemini free-tier quota or rate limit was reached. "
                    "Wait for the quota window to reset and try again."
                )
            elif any(term in error_text for term in ["404", "not_found", "not found", "model"]):
                message = (
                    "Gemini could not find or use the configured model. Check that "
                    "the Free Tier can access the model and that its identifier is current."
                )
            elif any(term in error_text for term in ["connection", "connect", "timeout", "dns", "network"]):
                message = "The Shiny host could not connect to the Gemini API. Check outbound internet access."
            else:
                message = f"Gemini failed while {stage} ({error_type})."

            safe_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if safe_key:
                detail = detail.replace(safe_key, "[redacted API key]")
            if detail:
                message = f"{message} Details: {detail}"
            await chat.append_message(f"InsightBot could not answer: {message}")

    return chat