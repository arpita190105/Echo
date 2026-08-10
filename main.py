"""
End-to-end pipeline glue (file-based test loop) for the study assistant:

  1. Record/accept a .wav file
  2. Transcribe it (STT)
  3. Classify intent + retrieve notes + generate response (LLM via Groq)
  4. Speak the reply back (TTS)
"""

import sys
from stt.whisper_stt import WhisperSTT
from tts.tts_router import TTSRouter
from llm.dialogue_manager import DialogueManager


def run_turn(audio_in_path: str, audio_out_path: str = "reply.wav"):
    stt = WhisperSTT()
    tts_router = TTSRouter()
    dialogue = DialogueManager()

    # 1. Speech -> text
    stt_result = stt.transcribe(audio_in_path)
    user_text = stt_result["text"]
    language = stt_result["language"]
    print(f"[Student ({language})]: {user_text}")

    if not user_text:
        print("No speech detected.")
        return

    # 2. Text -> intent + RAG + LLM reply
    result = dialogue.get_response(user_text, language=language)
    print(f"[Mode: {result['intent']} | Used notes: {result['used_context']}]")
    print(f"[Assistant]: {result['reply']}")

    # 3. Text -> speech
    out_path = tts_router.synthesize(result["reply"], language=language, out_path=audio_out_path)
    print(f"[Saved reply audio to]: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_input_audio.wav>")
        sys.exit(1)

    run_turn(sys.argv[1])