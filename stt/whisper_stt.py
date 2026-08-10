"""
Speech-to-text wrapper using faster-whisper.
Supports an optional forced_language to skip auto-detection and decode
directly in a specified language/script — needed to fix Hindi/Bengali
auto-detect confusion when the user manually overrides via the UI.
"""

from faster_whisper import WhisperModel
from config import config
from logger import get_logger

logger = get_logger(__name__)


class WhisperSTT:
    def __init__(self):
        try:
            self.model = WhisperModel(
                config.stt.model_size,
                device=config.stt.device,
                compute_type=config.stt.compute_type,
            )
            logger.info(f"Loaded Whisper model '{config.stt.model_size}' on {config.stt.device}")
        except Exception as e:
            logger.exception("Failed to load Whisper model")
            raise RuntimeError(f"Could not load STT model: {e}") from e

    def transcribe(self, audio_path: str, forced_language: str = None) -> dict:
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=forced_language,
                vad_filter=True,
                beam_size=5,
            )
            segments = list(segments)
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_path}")
            return {
                "text": "", "language": None, "language_confidence": 0.0,
                "segments": [], "error": "Audio file not found.",
            }
        except Exception as e:
            logger.exception(f"Transcription failed for {audio_path}")
            return {
                "text": "", "language": None, "language_confidence": 0.0,
                "segments": [], "error": f"Transcription failed: {e}",
            }

        full_text = " ".join(seg.text.strip() for seg in segments).strip()
        detected_lang = info.language
        lang_probability = info.language_probability

        if forced_language:
            logger.info(f"Transcribed with forced language='{forced_language}': {full_text[:80]}")
        else:
            if lang_probability < 0.6:
                logger.warning(
                    f"Low-confidence language detection: '{detected_lang}' at "
                    f"{lang_probability:.2f} confidence — may be misidentified "
                    f"(especially Hindi/Bengali confusion on short utterances)"
                )
            logger.info(f"Transcribed ({detected_lang}, confidence={lang_probability:.2f}): {full_text[:80]}")

        return {
            "text": full_text,
            "language": detected_lang,
            "language_confidence": lang_probability,
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
        }


if __name__ == "__main__":
    import sys
    stt = WhisperSTT()
    result = stt.transcribe(sys.argv[1])
    print(f"Language: {result['language']} (confidence: {result.get('language_confidence', 0):.2f})")
    print(f"Text: {result['text']}")