#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <whisper.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

struct WhisperContextDeleter {
  void operator()(whisper_context *ctx) const {
    if (ctx != nullptr) {
      whisper_free(ctx);
    }
  }
};

void disable_whisper_log(enum ggml_log_level, const char *, void *) {}

bool stt_use_gpu_enabled() {
  const char *value = std::getenv("LOCAL_STT_USE_GPU");
  if (value == nullptr) {
    return true;
  }

  const std::string normalized(value);
  return normalized != "0" && normalized != "false" && normalized != "FALSE" && normalized != "no" && normalized != "NO";
}

int env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  try {
    return std::stoi(value);
  } catch (...) {
    return fallback;
  }
}

float env_float(const char *name, float fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  try {
    return std::stof(value);
  } catch (...) {
    return fallback;
  }
}

std::vector<float> pcm_i16_to_float_mono(const std::string &pcm_bytes, int channels) {
  if (channels <= 0) {
    throw std::invalid_argument("num_channels must be positive");
  }
  if ((pcm_bytes.size() % sizeof(int16_t)) != 0) {
    throw std::invalid_argument("PCM payload must be signed 16-bit little-endian audio");
  }

  const auto *samples = reinterpret_cast<const int16_t *>(pcm_bytes.data());
  const size_t total_samples = pcm_bytes.size() / sizeof(int16_t);
  const size_t mono_samples = total_samples / static_cast<size_t>(channels);
  std::vector<float> out;
  out.reserve(mono_samples);

  for (size_t i = 0; i < mono_samples; ++i) {
    int accum = 0;
    for (int ch = 0; ch < channels; ++ch) {
      accum += samples[(i * static_cast<size_t>(channels)) + static_cast<size_t>(ch)];
    }
    const float mono = static_cast<float>(accum) / static_cast<float>(channels);
    out.push_back(mono / 32768.0f);
  }

  return out;
}

float mean_abs_amplitude(const std::vector<float> &audio) {
  if (audio.empty()) {
    return 0.0f;
  }

  double sum = 0.0;
  for (const float sample : audio) {
    sum += std::abs(sample);
  }
  return static_cast<float>(sum / static_cast<double>(audio.size()));
}

std::string valid_utf8_prefix(const char *input) {
  if (input == nullptr) {
    return "";
  }

  const std::string text(input);
  std::string out;
  out.reserve(text.size());

  for (size_t i = 0; i < text.size();) {
    const auto c = static_cast<unsigned char>(text[i]);
    size_t len = 0;
    if (c <= 0x7F) {
      len = 1;
    } else if ((c & 0xE0) == 0xC0) {
      len = 2;
    } else if ((c & 0xF0) == 0xE0) {
      len = 3;
    } else if ((c & 0xF8) == 0xF0) {
      len = 4;
    } else {
      ++i;
      continue;
    }

    if (i + len > text.size()) {
      break;
    }

    bool valid = true;
    for (size_t j = 1; j < len; ++j) {
      const auto cc = static_cast<unsigned char>(text[i + j]);
      if ((cc & 0xC0) != 0x80) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      ++i;
      continue;
    }

    out.append(text, i, len);
    i += len;
  }

  return out;
}

}  // namespace

class WhisperStream {
 public:
  explicit WhisperStream(const std::string &model_path, int sample_rate, const std::string &language)
      : sample_rate_(sample_rate), language_(language), use_gpu_(stt_use_gpu_enabled()) {
    if (sample_rate_ != 16000) {
      throw std::invalid_argument("whisper.cpp expects 16 kHz PCM; resample before STT");
    }

    whisper_log_set(disable_whisper_log, nullptr);
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = use_gpu_;
    ctx_.reset(whisper_init_from_file_with_params(model_path.c_str(), cparams));
    if (!ctx_) {
      throw std::runtime_error("failed to load whisper.cpp model: " + model_path);
    }
  }

  py::dict accept_pcm_i16(const py::bytes &pcm, int sample_rate, int channels) {
    if (sample_rate != sample_rate_) {
      throw std::invalid_argument("input sample rate does not match recognizer sample rate");
    }

    std::string pcm_bytes = pcm;
    std::vector<float> audio = pcm_i16_to_float_mono(pcm_bytes, channels);
    audio_.insert(audio_.end(), audio.begin(), audio.end());

    py::dict result;
    result["partial"] = "";
    result["speech_started"] = !audio.empty();
    return result;
  }

  py::dict partial(int max_window_ms) {
    py::dict result;
    result["speech_ended"] = false;

    if (audio_.empty()) {
      result["partial"] = "";
      result["segments"] = std::vector<std::string>{};
      result["confidence"] = 0.0;
      return result;
    }

    const size_t max_samples =
        max_window_ms > 0 ? static_cast<size_t>((static_cast<int64_t>(sample_rate_) * max_window_ms) / 1000) : audio_.size();
    const size_t start = max_samples < audio_.size() ? audio_.size() - max_samples : 0;
    std::vector<float> window(audio_.begin() + static_cast<std::ptrdiff_t>(start), audio_.end());

    return decode(window, false, true);
  }

  py::dict flush() {
    py::dict result;
    result["speech_ended"] = true;

    if (audio_.empty()) {
      result["final"] = "";
      result["segments"] = std::vector<std::string>{};
      result["confidence"] = 0.0;
      return result;
    }

    result = decode(audio_, true, false);
    audio_.clear();
    return result;
  }

  void reset() {
    audio_.clear();
  }

 private:
  py::dict decode(const std::vector<float> &audio, bool speech_ended, bool interim) {
    py::dict result;
    result["speech_ended"] = speech_ended;
    const float energy = mean_abs_amplitude(audio);
    result["audio_energy"] = energy;
    if (energy < env_float("LOCAL_STT_MIN_AUDIO_ENERGY", 0.0060f)) {
      if (interim) {
        result["partial"] = "";
      } else {
        result["final"] = "";
      }
      result["segments"] = std::vector<std::string>{};
      result["confidence"] = 0.0;
      result["no_speech_prob"] = 1.0;
      return result;
    }

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.no_timestamps = true;
    params.translate = false;
    params.language = language_.c_str();
    const int default_threads =
        use_gpu_ ? 2 : static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
    params.n_threads = std::max(1, env_int("LOCAL_STT_THREADS", default_threads));
    params.no_context = true;
    params.single_segment = true;
    params.suppress_blank = true;
    params.suppress_nst = true;
    params.temperature = 0.0f;
    params.temperature_inc = 0.0f;
    params.no_speech_thold = env_float("LOCAL_STT_NO_SPEECH_THRESHOLD", 0.35f);
    params.logprob_thold = env_float("LOCAL_STT_LOGPROB_THRESHOLD", -0.50f);
    params.entropy_thold = env_float("LOCAL_STT_ENTROPY_THRESHOLD", 2.4f);
    params.audio_ctx = env_int("LOCAL_STT_AUDIO_CTX", 512);
    params.max_tokens = env_int("LOCAL_STT_MAX_TOKENS", interim ? 24 : 48);

    int status = 0;
    {
      py::gil_scoped_release release;
      status = whisper_full(ctx_.get(), params, audio.data(), static_cast<int>(audio.size()));
    }
    if (status != 0) {
      throw std::runtime_error("whisper.cpp transcription failed");
    }

    const int n_segments = whisper_full_n_segments(ctx_.get());
    std::vector<std::string> segments;
    std::string final_text;
    float max_no_speech_prob = 0.0f;
    segments.reserve(static_cast<size_t>(std::max(0, n_segments)));
    for (int i = 0; i < n_segments; ++i) {
      const float no_speech_prob = whisper_full_get_segment_no_speech_prob(ctx_.get(), i);
      max_no_speech_prob = std::max(max_no_speech_prob, no_speech_prob);
      const char *segment_text = whisper_full_get_segment_text(ctx_.get(), i);
      if (segment_text == nullptr) {
        continue;
      }
      std::string text = valid_utf8_prefix(segment_text);
      if (!text.empty()) {
        segments.push_back(text);
        if (!final_text.empty()) {
          final_text += " ";
        }
        final_text += text;
      }
    }
    if (max_no_speech_prob >= env_float("LOCAL_STT_MAX_NO_SPEECH_PROB", 0.65f)) {
      final_text.clear();
      segments.clear();
    }

    if (interim) {
      result["partial"] = final_text;
    } else {
      result["final"] = final_text;
    }
    result["segments"] = segments;
    result["confidence"] = final_text.empty() ? 0.0 : std::max(0.0f, 1.0f - max_no_speech_prob);
    result["no_speech_prob"] = max_no_speech_prob;
    return result;
  }

  int sample_rate_;
  std::string language_;
  bool use_gpu_;
  std::unique_ptr<whisper_context, WhisperContextDeleter> ctx_;
  std::vector<float> audio_;
};

PYBIND11_MODULE(_local_stt, m) {
  m.doc() = "Local whisper.cpp STT runtime for LiveKit PCM frames";

  py::class_<WhisperStream>(m, "WhisperStream")
      .def(py::init<const std::string &, int, const std::string &>(), py::arg("model_path"), py::arg("sample_rate"),
           py::arg("language"))
      .def("accept_pcm_i16", &WhisperStream::accept_pcm_i16, py::arg("pcm"), py::arg("sample_rate"),
           py::arg("channels"))
      .def("partial", &WhisperStream::partial, py::arg("max_window_ms"))
      .def("flush", &WhisperStream::flush)
      .def("reset", &WhisperStream::reset);
}
