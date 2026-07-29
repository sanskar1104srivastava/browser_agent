from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import re
from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    cli,
)
from ai_clients import create_cerebras_llm, create_stt, create_tts, create_vad
from browser_tools import BrowserTools, create_driver


STOP_KEYWORDS = [
    "stop", "cancel", "abort", "forget it", "never mind", "quit",
    "don't bother", "leave it", "drop it", "stop searching", "stop looking"
]

STATUS_KEYWORDS = [
    "what's happening", "what are you doing", "still there", "you there",
    "any update", "what's going on", "how long", "still searching",
    "are you searching", "status", "progress", "done yet", "found anything"
]

KEEPALIVE_PHRASES = [
    "Still on it, give me just a moment...",
    "Almost there, bear with me...",
    "Loading the results, nearly done...",
    "Just a few more seconds, I promise...",
    "Still searching, this one's worth the wait...",
]

PRE_SEARCH_PHRASES = [
    "Sure, let me look that up for you!",
    "Give me a second, I'll check that now.",
    "On it, just a moment!",
    "Let me search that for you right away.",
]

# Kept as safety fallback in case model still emits <<RESULT>> blocks
RESULT_SEPARATOR = "<<RESULT>>"


def split_response(text: str) -> tuple[str, dict | None]:
    """
    Safety fallback: split LLM response into (spoken_text, result_dict).
    The LLM is no longer instructed to produce <<RESULT>> blocks,
    but this handles any edge cases where it does anyway.
    """
    if RESULT_SEPARATOR not in text:
        return text.strip(), None

    parts = text.split(RESULT_SEPARATOR, 1)
    spoken = parts[0].strip()
    raw_json = parts[1].strip()

    # strip optional ```json fences
    raw_json = re.sub(r'^```json\s*', '', raw_json, flags=re.IGNORECASE)
    raw_json = re.sub(r'\s*```$', '', raw_json)

    try:
        data = json.loads(raw_json.strip())
        return spoken, data
    except Exception as e:
        print(f"[split_response] JSON parse failed: {e}")
        return spoken, None


def build_result_from_tool_output(tool_name: str, result_text: str) -> dict | None:
    """
    Build a structured result dict from a tool result for emit_result.
    Called after browser tasks complete so the frontend gets structured data
    without asking the LLM to format JSON.
    """
    if not result_text or result_text.startswith("FAILED"):
        return None

    # Only emit structured results for page content tools
    if tool_name not in ("get_page_html", "navigate"):
        return None

    # Strip PAGE: prefix if present
    clean = re.sub(r'^PAGE:\s*[^\n]*\n', '', result_text).strip()

    if not clean:
        return None

    return {
        "title": "Browser Result",
        "intro": clean[:300],
        "sections": [],
        "note": f"Raw output from {tool_name}"
    }


class VoiceAssistant(Agent, BrowserTools):
    def __init__(self):
        self.driver = None
        self._driver_ready = asyncio.Event()
        self._driver_starting = False
        self._room = None
        self._session = None

        self._is_browsing = False
        self._stop_search = False
        self._keepalive_task = None
        self._phrase_index = 0
        self._pre_search_phrase_index = 0

        self._transcript_timer = None
        self._last_transcript_text = ""

        Agent.__init__(self, instructions="""You are a warm, helpful voice assistant with web browsing capability. Be concise — this is a voice conversation.

RULES:
- Never call tools on your own. Wait for the user to ask to search something or you have to answer something not known to you.
- Never mention tools, APIs, CSS selectors, or internal processes to the user
- Never output JSON, code blocks, or structured data in your responses
- Respond only in natural, conversational spoken language
- When asked to search, find, or look up anything, use browser tools immediately
- Start searches at https://www.google.com/search?q=... and complete in 4-5 tool calls maximum
- After completing a browser task, summarize findings in 1-3 natural spoken sentences only
- For general knowledge questions you already know the answer to, respond directly without browsing
- Keep all responses brief and natural — imagine you are speaking out loud

IMPORTANT: Never include <<RESULT>>, JSON objects, markdown, bullet points, or any structured formatting in your responses. Speak naturally at all times."""
        )

    # ── tts_node override — safety strip in case model still emits JSON ──
    async def tts_node(self, text, model_settings):
        """
        Called by LiveKit before text is sent to TTS.
        Collects full response, strips any accidental <<RESULT>> blocks,
        then passes only spoken text to TTS.
        """
        full_text = ""
        async for chunk in text:
            if isinstance(chunk, str):
                full_text += chunk
            elif hasattr(chunk, "text"):
                full_text += chunk.text

        # Safety fallback: strip JSON block if model still emits it
        spoken, result_data = split_response(full_text)

        print(f"[tts_node] spoken: {spoken[:120]}")
        if result_data:
            # Model still emitted a <<RESULT>> block despite instructions — handle it
            print(f"[tts_node] fallback result caught: {list(result_data.keys())}")
            asyncio.ensure_future(self.emit_result(result_data))
            asyncio.ensure_future(self.emit_browser_event("done", "Done"))

        async def _spoken_gen():
            yield spoken

        async for chunk in super().tts_node(_spoken_gen(), model_settings):
            yield chunk

    # ── driver lifecycle ──────────────────────────────────────────────

    async def wait_for_driver(self):
        if not self._driver_starting and self.driver is None:
            self._driver_starting = True
            asyncio.ensure_future(self._init_driver())
        await asyncio.wait_for(self._driver_ready.wait(), timeout=30)

    async def _init_driver(self):
        self.driver = await asyncio.to_thread(create_driver, True)
        self._driver_ready.set()
        print("[VoiceAssistant] Browser driver ready.")

    # ── emit helpers ──────────────────────────────────────────────────

    async def _publish(self, payload: dict):
        if self._room is None:
            return
        data = json.dumps(payload).encode("utf-8")
        try:
            await self._room.local_participant.publish_data(data, reliable=True)
        except Exception as e:
            print(f"[publish] {e}")

    async def emit_browser_event(self, event_type: str, message: str):
        await self._publish({
            "type": "browser_status",
            "event": event_type,
            "message": message
        })

    async def emit_transcript(self, role: str, text: str, is_final: bool = True):
        await self._publish({
            "type": "transcript",
            "role": role,
            "text": text.strip(),
            "is_final": is_final
        })

    async def emit_result(self, data: dict):
        await self._publish({
            "type": "result",
            "data": data
        })

    # ── browsing state ────────────────────────────────────────────────

    def start_browsing(self):
        self._is_browsing = True
        self._stop_search = False
        self._phrase_index = 0
        if self._keepalive_task:
            self._keepalive_task.cancel()
        self._keepalive_task = asyncio.ensure_future(self._browsing_keepalive())
        asyncio.ensure_future(self.emit_browser_event("start", "Searching..."))
        print("[BROWSING] Started")

    def stop_browsing(self):
        self._is_browsing = False
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        print("[BROWSING] Stopped")

    async def _browsing_keepalive(self):
        try:
            await asyncio.sleep(8)
            while self._is_browsing and not self._stop_search:
                phrase = KEEPALIVE_PHRASES[self._phrase_index % len(KEEPALIVE_PHRASES)]
                self._phrase_index += 1
                if self._session:
                    await self._session.say(phrase, allow_interruptions=True)
                await asyncio.sleep(9)
        except asyncio.CancelledError:
            pass

    def _classify_interruption(self, transcript: str) -> str:
        text = transcript.lower().strip()
        if any(kw in text for kw in STOP_KEYWORDS):
            return "stop"
        if any(kw in text for kw in STATUS_KEYWORDS):
            return "status"
        return "other"

    async def announce_search_start(self):
        if self._is_browsing:
            return
        phrase = PRE_SEARCH_PHRASES[
            self._pre_search_phrase_index % len(PRE_SEARCH_PHRASES)
        ]
        self._pre_search_phrase_index += 1
        print(f"[PRE-SEARCH] {phrase}")
        if self._session:
            await self._session.say(phrase, allow_interruptions=False)
        self.start_browsing()


server = AgentServer()


@server.rtc_session(agent_name="voice-bot")
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=create_stt(),
        llm=create_cerebras_llm(),   # fixed: was create_llm()
        tts=create_tts(),
        vad=create_vad(),
        max_tool_steps=8,
    )

    agent = VoiceAssistant()
    agent._room = ctx.room
    agent._session = session

    BROWSER_TOOLS = {
        "navigate", "get_page_html", "click_element",
        "type_into_field", "run_js", "get_current_url", "switch_tab"
    }

    @session.on("tool_calls_collected")
    def on_tool_calls(event):
        tool_names = [c.function.name for c in event.tool_calls]
        print(f"[TOOL CALL] {tool_names}")
        if any(n in BROWSER_TOOLS for n in tool_names):
            if not agent._is_browsing:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(agent.announce_search_start())

    @session.on("tool_calls_result")
    def on_tool_results(event):
        for r in event.tool_results:
            result_text = str(r.result)
            tool_name = getattr(r, "name", "") or getattr(r, "tool_name", "") or ""
            print(f"[TOOL RESULT] {tool_name}: {result_text[:200]}")

            # Emit structured result from tool output directly
            # This replaces the old <<RESULT>> JSON approach
            result_data = build_result_from_tool_output(tool_name, result_text)
            if result_data:
                asyncio.ensure_future(agent.emit_result(result_data))

    @session.on("user_input_transcribed")
    def on_user_input(event):
        transcript = event.transcript
        is_final = (
            getattr(event, "is_final", None)
            if getattr(event, "is_final", None) is not None
            else getattr(event, "final", None)
            if getattr(event, "final", None) is not None
            else True
        )
        print(f"\n[USER {'FINAL' if is_final else 'partial'}] {transcript}")
        agent._last_transcript_text = transcript

        if agent._transcript_timer:
            agent._transcript_timer.cancel()
            agent._transcript_timer = None

        if is_final:
            asyncio.ensure_future(agent.emit_transcript("user", transcript, is_final=True))
            _handle_browsing_interrupt(transcript)
        else:
            asyncio.ensure_future(agent.emit_transcript("user", transcript, is_final=False))
            loop = asyncio.get_event_loop()
            agent._transcript_timer = loop.call_later(
                1.5,
                lambda: asyncio.ensure_future(
                    agent.emit_transcript("user", agent._last_transcript_text, is_final=True)
                )
            )

    def _handle_browsing_interrupt(transcript: str):
        if not agent._is_browsing:
            return
        intent = agent._classify_interruption(transcript)
        print(f"[INTERRUPT INTENT] {intent}")
        if intent == "stop":
            agent._stop_search = True
            agent.stop_browsing()
            asyncio.ensure_future(
                session.say("Alright, I've stopped the search. What else can I help you with?")
            )
        elif intent == "status":
            asyncio.ensure_future(
                session.say("Still searching, just give me a moment!", allow_interruptions=True)
            )

    @session.on("conversation_item_added")
    def on_conversation_item(event):
        item = event.item
        if not hasattr(item, "role") or item.role != "assistant":
            return

        content = item.content
        raw = ""
        if isinstance(content, str):
            raw = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
                elif hasattr(block, "text"):
                    parts.append(str(block.text))
            raw = " ".join(p for p in parts if p).strip()
        elif hasattr(content, "text"):
            raw = str(content.text)
        else:
            raw = str(content)

        raw = raw.strip("[]'\"")
        if not raw:
            return

        # Safety strip in case model still emits <<RESULT>> block
        spoken, _ = split_response(raw)
        if spoken:
            print(f"[AGENT] {spoken[:200]}")
            asyncio.ensure_future(agent.emit_transcript("agent", spoken, is_final=True))

    @session.on("agent_state_changed")
    def on_state(event):
        print(f"[STATE] {event.old_state} → {event.new_state}")
        if event.new_state == "speaking" and agent._is_browsing:
            agent.stop_browsing()

    await session.start(agent=agent, room=ctx.room)
    await asyncio.sleep(1.5)
    await session.say("Hey there! I'm here and ready to chat. What's on your mind?")


if __name__ == "__main__":
    cli.run_app(server)