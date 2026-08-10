"""
Unified TTS interface: MMS-TTS for Hindi/Bengali (and other Indic
languages), Coqui for English/voice cloning. No MeloTTS dependency —
that integration was rolled back due to a dependency conflict.
"""

from logger import get_logger

logger = get_logger(__name__)


class TTSRouter:
    def __init__(self, preload_languages: list = None):
        self._coqui = None
        self._mms = None
        if preload_languages:
            self._warmup(preload_languages)

    def _warmup(self, languages: list):
        logger.info(f"Warming up TTS engines for: {languages}")
        for lang in languages:
            try:
                if lang in ("hi", "bn", "pa", "mr", "ml"):
                    self.mms._load_language(lang)
                else:
                    _ = self.coqui
            except Exception as e:
                logger.warning(f"Warmup failed for '{lang}': {e}")
        logger.info("TTS warmup complete")

    @property
    def coqui(self):
        if self._coqui is None:
            from tts.coqui_tts import CoquiTTS
            self._coqui = CoquiTTS()
            logger.info("Coqui TTS engine loaded")
        return self._coqui

    @property
    def mms(self):
        if self._mms is None:
            from tts.mms_tts import MMSTTS
            self._mms = MMSTTS()
            logger.info("MMS-TTS engine loaded")
        return self._mms

    def synthesize(self, text: str, language: str = "en", out_path: str = "output.wav") -> str:
        if language in ("hi", "bn", "pa", "mr", "ml"):
            try:
                return self.mms.synthesize(text, language=language, out_path=out_path)
            except Exception as e:
                logger.error(f"MMS-TTS failed ({language}), falling back to Coqui/English: {e}")
                return self.coqui.synthesize(text, language="en", out_path=out_path)

        try:
            return self.coqui.synthesize(text, language=language, out_path=out_path)
        except Exception as e:
            logger.exception(f"Coqui TTS failed: {e}")
            raise RuntimeError("Both TTS engines failed to synthesize a reply.") from e