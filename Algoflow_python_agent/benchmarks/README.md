# Local STT/TTS R&D Benchmarks

Track the R&D acceptance metrics here as the native runtimes become available:

- STT latency
- LLM first-token latency
- End-to-end latency
- TTS first-audio latency
- CPU percent
- RAM
- Model load time
- Realtime factor
- Dropped frames

## Hindi STT Trials

Logged trial artifacts:

- `benchmarks/stt_trials/20260730T050653Z_hindi_stt_matrix_hi.json`
- `benchmarks/stt_trials/20260730T050844Z_hindi_stt_matrix_hi.json`
- `benchmarks/stt_trials/20260730T051349Z_hindi_stt_vosk.json`
- `benchmarks/stt_trials/20260730T052801Z_vosk_stt_service_latency.json`
- `benchmarks/stt_trials/20260730T052818Z_vosk_stt_service_latency.json`

Current controlled Hindi fixture result, using the same local Piper-generated Hindi phrases:

| Runtime | Model | Avg decode ms | Avg realtime factor | Avg char error ratio | Devanagari rate | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| whisper.cpp | `ggml-tiny-q5_1.bin` | 610.2 | 0.393 | 1.9227 | 0.0 | Fast, unusable Hindi transcription. |
| whisper.cpp | `ggml-base-q5_1.bin` | 1154.2 | 0.757 | 1.8500 | 0.0 | Often Roman/Urdu/English output; not acceptable for Hindi mode. |
| whisper.cpp | `ggml-base-q8_0.bin` | 978.9 | 0.641 | 1.8667 | 0.0 | Faster than base-q5 in this run, but still wrong script/content. |
| whisper.cpp | `ggml-small-q5_1.bin` | 4096.6 | 2.624 | 0.1121 | 1.0 | Readable Hindi, too slow for sub-1s CPU target. |
| whisper.cpp | `ggml-small-q8_0.bin` | 3382.3 | 2.208 | 0.1121 | 1.0 | Best whisper.cpp tradeoff so far, still too slow for smooth agent turns. |
| Vosk | `vosk-model-small-hi-0.22` | 592.0 | 0.375 | 0.0000 | 1.0 | Best controlled Hindi result so far; must be validated on real mic/noise/code-switch audio before replacing whisper.cpp. |

Decision so far:

- Keep whisper.cpp isolated as the reference native STT path; `libwhisper` can only load Whisper-family ggml models.
- Do not use `auto` language detection for Hindi calls; force `hi` to avoid short Hindi utterances being decoded as English.
- `ggml-small-q8_0.bin` is the best whisper.cpp candidate tested, but CPU decode latency is still above the conversational target.
- The whisper.cpp default is therefore `ggml-small-q8_0.bin` for Hindi accuracy while the lower-latency runtime decision is being validated.
- Vosk small Hindi is the best latency baseline in the controlled trial and is worth a real-mic LiveKit trial if the project accepts a Vosk/Kaldi runtime.
- Sherpa-ONNX remains the strongest architecture candidate for true streaming, but a suitable Hindi/Hindi-English streaming model must be verified before integration.

Standalone Vosk service latency test:

```powershell
$env:LOCAL_STT_PROVIDER='vosk'
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe benchmarks\vosk_stt_service_latency.py
```

Latest raw service result:

- Final text: `मुझे विराट कोहली के बारे में बताओ`
- First partial: `177.0 ms`
- Finalization after all chunks: `11.7 ms`
- STT processing time: `633.2 ms` for `2101.4 ms` audio
- Processing realtime factor: `0.301`

Latest simulated live-feed result:

- Final text: `मुझे विराट कोहली के बारे में बताओ`
- First partial: `939.6 ms`
- Finalization after all chunks: `43.6 ms`
- STT processing time: `834.9 ms` for `2194.3 ms` audio
- Processing realtime factor: `0.380`

For real microphone recordings, pass a 16-bit PCM WAV file:

```powershell
$env:LOCAL_STT_PROVIDER='vosk'
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe benchmarks\vosk_stt_service_latency.py --wav path\to\sample.wav --realtime
```

Realtime microphone test from terminal:

```powershell
$env:LOCAL_STT_PROVIDER='vosk'
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe benchmarks\vosk_stt_mic_realtime.py --device 1
```

Detailed realtime microphone debug with captured Vosk input audio:

```powershell
.\.venv\Scripts\python.exe benchmarks\vosk_stt_mic_realtime.py --device 1 --json --debug --show-rms --show-events --save-wav benchmarks\stt_trials\mic_live.wav
```

Replay the captured audio through the independent WAV latency test:

```powershell
.\.venv\Scripts\python.exe benchmarks\vosk_stt_service_latency.py --wav benchmarks\stt_trials\mic_live.wav --realtime
```

This terminal test uses Silero VAD by default and streams accepted speech audio into Vosk. RMS VAD is only a fallback mode:

```powershell
.\.venv\Scripts\python.exe benchmarks\vosk_stt_mic_realtime.py --device 1 --vad rms --show-rms
```

List microphone devices:

```powershell
.\.venv\Scripts\python.exe benchmarks\vosk_stt_mic_realtime.py --list-devices
```

Useful tuning while testing:

```powershell
.\.venv\Scripts\python.exe benchmarks\vosk_stt_mic_realtime.py --device 1 --show-rms
.\.venv\Scripts\python.exe benchmarks\vosk_stt_mic_realtime.py --device 1 --speech-rms-threshold 500 --endpoint-silence-ms 300 --partial-interval-ms 120
```

Use `--show-rms` for calibration only when running `--vad rms`: keep quiet for two seconds, then speak normally. Set `--speech-rms-threshold` above the quiet RMS and below the speaking RMS.

Worst-case scenarios from the detailed design:

- 60-minute continuous conversation
- Very long utterances
- Rapid speaker changes
- Background noise
- High CPU utilization
- Multiple concurrent calls
- Container restart and model reload
- Memory pressure
