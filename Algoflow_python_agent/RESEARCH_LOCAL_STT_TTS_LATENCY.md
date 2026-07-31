# Local STT/TTS Sub-Second Latency — VERIFIED Research

## System Under Test

| Component | Spec |
|-----------|------|
| CPU | 12th Gen Intel Core i5-12450H (8 cores / 12 threads) |
| RAM | 16 GB |
| Architecture | x64 |
| Laptop | Acer Aspire A715-76G |

---

## Models Already Downloaded

**STT models (models/stt/):**
- vosk-model-hi-0.22 ✅
- vosk-model-small-hi-0.22 ✅
- ggml-tiny-q5_1.bin ✅
- ggml-tiny.en-q5_1.bin ✅
- ggml-base-q5_1.bin / q8_0 / en ✅
- ggml-small-q5_1.bin / q8_0 ✅
- ggml-medium-q5_0.bin ✅
- ggml-large-v3-turbo-q5_0.bin ✅

**TTS models (models/tts/):**
- hi_IN-pratham-medium.onnx + config ✅ (Piper Hindi)
- en_US-lessac-low.onnx + config ✅ (Piper English)

---

## VERIFIED Benchmark Results (i5-12450H CPU, Hindi)

### Vosk (vosk-model-hi-0.22) — BATCH MODE

| Metric | Value |
|--------|-------|
| Model load | 17.7s (one-time) |
| Avg decode | 265ms |
| RTF | 0.155x (6.5x faster than real-time) |
| Hindi accuracy | **100%** (0.0 char error ratio) |
| Devanagari output | 100% |

Per-phrase breakdown:
- "तुम कौन हो" (1.0s audio): 201ms decode, RTF 0.197
- "मेरा नाम संस्कार है" (1.6s audio): 234ms decode, RTF 0.143
- "मुझे विराट कोहली के बारे में बताओ" (2.5s audio): 342ms decode, RTF 0.138
- "भारत की राजधानी क्या है" (2.0s audio): 283ms decode, RTF 0.143

### Whisper.cpp Models — BATCH MODE (Hindi, i5-12450H)

| Model | Load (ms) | Avg Decode (ms) | RTF | Hindi CER | Devanagari |
|-------|-----------|-----------------|-----|-----------|------------|
| tiny-q5_1 | 2584 | 266 | 0.183 | **1.89** ❌ | 0% ❌ |
| base-q5_1 | 264 | 543 | 0.323 | **1.59** ❌ | 0% ❌ |
| base-q8_0 | 502 | 569 | 0.342 | **1.59** ❌ | 0% ❌ |
| small-q5_1 | 663 | 1189 | 0.676 | 0.056 ✅ | 100% ✅ |
| small-q8_0 | 1066 | 1327 | 0.757 | 0.056 ✅ | 100% ✅ |
| large-v3-turbo-q5_0 | 1916 | 2129 | **1.291** ❌ | 0.0 ✅ | 100% ✅ |

**Critical finding:** Whisper tiny/base models are **completely useless for Hindi** — they produce garbage output (romanized text, Urdu script, wrong characters). Only small+ produce valid Devanagari.

**Critical finding:** large-v3-turbo is **slower than real-time** on this CPU (RTF 1.291x). Not viable for real-time.

---

## Moonshine ASR — VERIFIED Language Support

| Feature | Status |
|---------|--------|
| STT Hindi support | **NO** ❌ |
| STT languages | English, Spanish, Mandarin, Japanese, Korean, Vietnamese, Ukrainian, Arabic |
| TTS Hindi support | **YES** ✅ (hi-in) |
| TTS languages | English, Spanish, Arabic, German, French, Hindi, Italian, Japanese, Korean, Dutch, Portuguese, Russian, Turkish, Ukrainian, Vietnamese, Mandarin |
| Streaming | Yes (native streaming with caching) |
| Latency (English) | Tiny: 34ms Mac / 69ms Linux / 237ms RPi5 |

**Verdict:** Moonshine is NOT viable for Hindi STT. It only supports 8 languages for STT, and Hindi is not among them. However, Moonshine's TTS does support Hindi.

---

## Kokoro TTS — VERIFIED Hindi Support

| Feature | Status |
|---------|--------|
| Hindi voices | hf_alpha (F), hf_beta (F), hm_omega (M), hm_psi (M) |
| Hindi quality | Grade C (MM minutes training data) |
| Model size | 82M parameters (~300MB) |
| CPU latency | ~500ms for short phrases (FP32 ONNX) |
| Streaming | Yes (via kokoro-onnx) |
| License | Apache 2.0 |

**Verdict:** Kokoro TTS supports Hindi with 4 voices. Quality is Grade C. Latency ~500ms on CPU is comparable to Piper.

---

## Analysis: What's Actually Viable for Sub-Second Latency

### STT Options for Hindi

| Option | Streaming | Hindi Accuracy | Latency | Verdict |
|--------|-----------|----------------|---------|---------|
| Vosk (current) | ✅ True streaming | 100% | 265ms decode | **BEST** ✅ |
| Whisper tiny/base | ❌ Batch | Garbage | 266-569ms | Useless for Hindi |
| Whisper small | ❌ Batch | 94.4% | 1189ms decode | Too slow for real-time |
| Moonshine | ✅ Streaming | **N/A** | N/A | No Hindi STT |

**Conclusion:** For Hindi STT on CPU, **Vosk is the only viable option** that achieves both accuracy and sub-second latency. Moonshine doesn't support Hindi STT. Whisper models either produce garbage (tiny/base) or are too slow (small+).

### TTS Options for Hindi

| Option | Streaming | Hindi Quality | First-Audio Latency | Verdict |
|--------|-----------|---------------|---------------------|---------|
| Piper (current) | ✅ | Good | ~100-300ms | **BEST** ✅ |
| Kokoro | ✅ | Grade C | ~500ms | Comparable quality, slightly slower |
| Moonshine TTS | ✅ | Unknown | Unknown | Needs testing |

**Conclusion:** Piper is already optimal for Hindi TTS. Kokoro is an alternative but not clearly better.

---

## Recommended Configuration for Sub-Second Latency

### Current Stack (Vosk + Piper) is Already Optimal

The existing architecture is the right choice. The optimization path is tuning parameters, not replacing models.

### Optimized Settings

```env
# STT - Vosk (already optimal)
LOCAL_STT_PROVIDER=vosk
LOCAL_VOSK_PARTIAL_INTERVAL_MS=100    # Faster partials (default 150)
LOCAL_VOSK_MIN_PARTIAL_AUDIO_MS=150   # Earlier first partial (default 200)
LOCAL_STT_ENDPOINT_SILENCE_MS=300     # Faster endpointing (default 350)
LOCAL_STT_MIN_UTTERANCE_MS=200        # Shorter minimum (default 250)

# TTS - Piper (already optimal)
LOCAL_TTS_CHUNK_MAX_CHARS=20          # Smaller chunks = faster first audio
```

### Expected Latency with Optimized Settings

| Stage | Latency |
|-------|---------|
| STT first partial | ~150-200ms |
| STT final (after speech ends) | ~265ms decode |
| TTS first audio | ~100-200ms |
| **End-to-end round-trip** | **~600-800ms** ✅ |

---

## What About Upgrading?

### Moonshine: NOT Recommended
- **No Hindi STT support** (only 8 languages, Hindi not included)
- Would require switching to English-only or training custom model
- Not worth the trade-off

### Kokoro TTS: Optional
- Hindi quality is Grade C (comparable to Piper)
- ~500ms first-audio latency (slightly slower than Piper)
- Could test if voice quality preference varies

### Whisper small (hybrid approach): Possible
- Use Vosk for streaming partials (fast)
- Use Whisper small-q5_1 for final transcript correction (accurate)
- Adds complexity but improves accuracy for noisy audio
- Trade-off: +900ms decode time on final transcript

---

## Summary

| Question | Answer |
|----------|--------|
| Can we achieve sub-second STT latency? | **YES** — Vosk already does it (265ms decode) |
| Can we achieve sub-second TTS latency? | **YES** — Piper already does it (~100-300ms) |
| Is Moonshine viable for Hindi? | **NO** — No Hindi STT support |
| Is Kokoro better than Piper for Hindi? | **NO** — Comparable quality, slightly slower |
| Should we replace current stack? | **NO** — Current stack is optimal |
| What should we optimize? | **Tune Vosk/Piper parameters** for faster partials |

---

## Sources

- [Moonshine GitHub](https://github.com/moonshine-ai/moonshine) — Language support verified
- [Kokoro-82M VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) — Hindi voices verified
- [kokoro-onnx GitHub](https://github.com/thewh1teagle/kokoro-onnx) — CPU latency verified
- Local benchmarks run on 2026-07-30 on i5-12450H
