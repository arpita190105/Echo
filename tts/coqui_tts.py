"""
Text-to-speech wrapper using Coqui TTS.
Supports multilingual synthesis and voice cloning via a reference speaker_wav
(useful later for your fine-tuned Indian-accent voice).
"""

import os
from TTS.api import TTS
from config import config

# Coqui's language codes for YourTTS / XTTS-style multilingual models
LANG_CODE_MAP = {
    "en": "en",
    "hi": "hi",
    "bn": "bn",
}


class CoquiTTS:
    def __init__(self):
        self.tts = TTS(model_name=config.tts.model_name, progress_bar=False)
        self.speaker_wav = config.tts.speaker_wav

    def synthesize(self, text: str, language: str = None, out_path: str = "output.wav") -> str:
        """
        Convert text to speech and save as a .wav file.

        Args:
            text: text to speak
            language: "en" / "hi" / "bn" — falls back to config default
            out_path: output file path

        Returns:
            path to generated audio file
        """
        lang = LANG_CODE_MAP.get(language or config.tts.default_language, "en")

        kwargs = {"text": text, "file_path": out_path}

        # Multilingual models need a language code
        if self.tts.is_multi_lingual:
            kwargs["language"] = lang

        # Voice cloning: pass a reference sample if we have one
        if self.speaker_wav and os.path.exists(self.speaker_wav):
            kwargs["speaker_wav"] = self.speaker_wav
        elif self.tts.is_multi_speaker:
            # fall back to the model's first built-in speaker
            kwargs["speaker"] = self.tts.speakers[0]

        self.tts.tts_to_file(**kwargs)
        return out_path


if __name__ == "__main__":
    # quick manual test: python tts/coqui_tts.py
    tts = CoquiTTS()
    path = tts.synthesize("Hello, this is a test of the voice agent.", language="en")
    print(f"Saved audio to {path}")