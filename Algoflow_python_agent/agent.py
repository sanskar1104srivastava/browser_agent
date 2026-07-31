from dotenv import load_dotenv
load_dotenv(override=True)

import asyncio
import json
import logging
import os
import re
from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    RoomOutputOptions,
    cli,
)
from ai_clients import create_cerebras_llm, create_stt, create_tts, create_vad, initialize_local_audio
from config import LocalAudioConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("voice_agent")


# Kept as safety fallback in case model still emits <<RESULT>> blocks
RESULT_SEPARATOR = "<<RESULT>>"
GREETING_TEXT = "नमस्ते! बताइए, मैं आपकी कैसे मदद कर सकता हूं?"


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


class VoiceAssistant(Agent):
    def __init__(self):
        self._room = None
        self._session = None

        self._transcript_timer = None
        self._last_transcript_text = ""

        Agent.__init__(self, instructions="""You are a warm, helpful Hindi voice assistant. Be concise because this is a voice conversation.

RULES:
- Always speak in natural Hindi unless the user explicitly asks for another language
- Use simple conversational Hindi that sounds good when spoken aloud
- Never mention tools, APIs, CSS selectors, or internal processes to the user
- Never output JSON, code blocks, or structured data in your responses
- Respond only in natural, conversational spoken Hindi
- Do not browse, search the web, open pages, click links, or call browser tools
- If the user asks for current or web-only information, say in Hindi that you cannot browse from this voice session
- For general knowledge questions you already know the answer to, respond directly in Hindi without browsing
- Keep responses very short, usually one sentence unless the user asks for detail

IMPORTANT: Never include <<RESULT>>, JSON objects, markdown, bullet points, or any structured formatting in your responses. Speak naturally in Hindi at all times."""
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
        if not spoken.strip():
            logger.warning(
                "stage=tts_node_empty_response_fallback raw_chars=%s",
                len(full_text),
            )
            spoken = "माफ कीजिए, मुझे जवाब बनाने में दिक्कत हुई। कृपया फिर से बोलिए।"

        print(f"[tts_node] spoken: {spoken[:120]}")
        if result_data:
            # Model still emitted a <<RESULT>> block despite instructions — handle it
            print(f"[tts_node] fallback result caught: {list(result_data.keys())}")
            asyncio.ensure_future(self.emit_result(result_data))

        async def _spoken_gen():
            yield spoken

        async for chunk in super().tts_node(_spoken_gen(), model_settings):
            yield chunk

    # ── emit helpers ──────────────────────────────────────────────────

    async def _publish(self, payload: dict):
        if self._room is None:
            return
        data = json.dumps(payload).encode("utf-8")
        try:
            await self._room.local_participant.publish_data(data, reliable=True)
        except Exception as e:
            print(f"[publish] {e}")

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

server = AgentServer()


async def send_startup_greeting(session: AgentSession, agent: VoiceAssistant) -> None:
    logger.info("stage=greeting_start text_chars=%s", len(GREETING_TEXT))
    await agent.emit_transcript("agent", GREETING_TEXT, is_final=True)
    handle = session.say(
        GREETING_TEXT,
        allow_interruptions=False,
        add_to_chat_ctx=False,
    )
    logger.info("stage=greeting_say_created speech_id=%s", getattr(handle, "id", "unknown"))

    def _on_done(done_handle):
        logger.info(
            "stage=greeting_handle_done speech_id=%s interrupted=%s done=%s",
            getattr(done_handle, "id", "unknown"),
            getattr(done_handle, "interrupted", None),
            done_handle.done(),
        )

    handle.add_done_callback(_on_done)

    async def _watch_playout() -> None:
        try:
            await handle.wait_for_playout()
            logger.info("stage=greeting_playout_done speech_id=%s", getattr(handle, "id", "unknown"))
        except Exception:
            logger.exception("stage=greeting_playout_failed speech_id=%s", getattr(handle, "id", "unknown"))

    asyncio.create_task(_watch_playout(), name="startup_greeting_playout_watch")


@server.rtc_session(agent_name="voice-bot")
async def entrypoint(ctx: JobContext):
    logger.info("stage=entrypoint_start room=%s", getattr(ctx.room, "name", "unknown"))
    logger.info("stage=local_audio_initialize_start")
    initialize_local_audio()
    audio_config = LocalAudioConfig.from_env()
    logger.info("stage=local_audio_initialize_done")

    logger.info("stage=livekit_connect_start")
    await ctx.connect()
    logger.info("stage=livekit_connect_done room=%s", getattr(ctx.room, "name", "unknown"))

    vad_enabled = os.getenv("LOCAL_ENABLE_SILERO_VAD", "1").strip().lower() in {"1", "true", "yes"}
    if audio_config.stt_provider == "whisper" and not vad_enabled:
        logger.warning("stage=vad_forced_for_whisper reason=stream_adapter_requires_endpointing")

    stt_vad = create_vad() if audio_config.stt_provider == "whisper" else None
    session_vad_enabled = (
        os.getenv("LOCAL_ENABLE_LIVEKIT_SESSION_VAD", "0" if audio_config.stt_provider == "whisper" else "1")
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )
    session_vad = create_vad() if session_vad_enabled and vad_enabled else None
    turn_detection_mode = "stt" if audio_config.stt_provider == "whisper" or session_vad is None else "vad"
    session_kwargs = {
        "stt": create_stt(vad=stt_vad),
        "llm": create_cerebras_llm(),
        "tts": create_tts(),
        "turn_handling": {
            "turn_detection": turn_detection_mode,
            "endpointing": {
                "mode": "fixed",
                "min_delay": 0.25,
                "max_delay": 0.8,
            },
            "interruption": {
                "enabled": False,
                "discard_audio_if_uninterruptible": True,
            },
        },
        "max_tool_steps": 8,
    }
    if session_vad is not None:
        session_kwargs["vad"] = session_vad

    session = AgentSession(**session_kwargs)
    logger.info(
        "stage=agent_session_created stt_provider=%s tts_provider=%s stt_adapter=%s stt_vad=%s session_vad=%s turn_detection=%s interruption_enabled=false endpointing_mode=fixed endpointing_min_delay=0.25 endpointing_max_delay=0.8",
        audio_config.stt_provider,
        audio_config.tts_provider,
        "livekit_stream_adapter" if audio_config.stt_provider == "whisper" else "native_stream",
        "silero" if stt_vad is not None else "none",
        "silero" if session_vad is not None else "none",
        turn_detection_mode,
    )

    agent = VoiceAssistant()
    agent._room = ctx.room
    agent._session = session

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
        transcript_text = transcript.strip()
        logger.info(
            "stage=user_transcript_received final=%s chars=%s preview=%r",
            is_final,
            len(transcript_text),
            transcript_text[:120],
        )
        print(f"\n[USER {'FINAL' if is_final else 'partial'}] {transcript}")
        agent._last_transcript_text = transcript

        if agent._transcript_timer:
            agent._transcript_timer.cancel()
            agent._transcript_timer = None

        if is_final:
            logger.info("stage=user_transcript_emit final=true chars=%s", len(transcript_text))
            asyncio.ensure_future(agent.emit_transcript("user", transcript, is_final=True))
        else:
            logger.info("stage=user_transcript_emit final=false chars=%s", len(transcript_text))
            asyncio.ensure_future(agent.emit_transcript("user", transcript, is_final=False))
            loop = asyncio.get_event_loop()
            agent._transcript_timer = loop.call_later(
                1.5,
                lambda: asyncio.ensure_future(
                    agent.emit_transcript("user", agent._last_transcript_text, is_final=True)
                )
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

    logger.info("stage=session_start_begin")
    await session.start(
        agent=agent,
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            audio_sample_rate=audio_config.tts_sample_rate,
            audio_num_channels=audio_config.num_channels,
        ),
    )
    logger.info("stage=session_start_done")
    await asyncio.sleep(0.2)
    await send_startup_greeting(session, agent)


if __name__ == "__main__":
    cli.run_app(server)
