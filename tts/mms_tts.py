"""
Meta MMS-TTS wrapper — ungated, per-language VITS checkpoints via transformers.
Replaces the gated AI4Bharat Indic Parler-TTS as the Hindi/Bengali engine.
"""

import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer
from logger import get_logger

logger = get_logger(__name__)

# One checkpoint per language — MMS-TTS trains a separate model per language
# rather than one shared multilingual model.
MODEL_IDS = {
    "hi": "facebook/mms-tts-hin",
    "bn": "facebook/mms-tts-ben",
}


class MMSTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models = {}      # lazy-loaded per language, so we don't load both upfront
        self._tokenizers = {}

    def _load_language(self, language: str):
        if language not in self._models:
            model_id = MODEL_IDS.get(language)
            if model_id is None:
                raise ValueError(f"No MMS-TTS checkpoint configured for language '{language}'")
            logger.info(f"Loading MMS-TTS model for '{language}': {model_id}")
            self._models[language] = VitsModel.from_pretrained(model_id).to(self.device)
            self._tokenizers[language] = AutoTokenizer.from_pretrained(model_id)
        return self._models[language], self._tokenizers[language]

    def synthesize(self, text: str, language: str = "hi", out_path: str = "output.wav") -> str:
        model, tokenizer = self._load_language(language)

        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = model(**inputs).waveform

        audio = output.squeeze().cpu().numpy()
        sf.write(out_path, audio, model.config.sampling_rate)
        logger.info(f"MMS-TTS synthesized {language} audio -> {out_path}")
        return out_path


if __name__ == "__main__":
    tts = MMSTTS()
    path = tts.synthesize("नमस्ते, आप कैसे हैं?", language="hi")
    print(f"Saved audio to {path}")