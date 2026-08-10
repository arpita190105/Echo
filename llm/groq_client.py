"""
Groq API client wrapper — OpenAI-compatible, free tier.
Reply language is explicitly enforced based on STT's detected language,
rather than inferred from (possibly noisy) transcribed text.
"""

import os
from groq import Groq
from dotenv import load_dotenv
from config import config
from logger import get_logger

load_dotenv()
logger = get_logger(__name__)

FALLBACK_REPLY = "Sorry, I'm having trouble responding right now. Please try again in a moment."

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali"}


class GroqClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY missing from environment/.env")
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
                "and add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", config.llm.model_name)

    def generate(self, mode: str, user_query: str, context: str = "",
                 history: list = None, language: str = "en") -> str:
        system_prompt = (
            config.llm.quiz_system_prompt if mode == "quiz" else config.llm.explain_system_prompt
        )

        lang_name = LANGUAGE_NAMES.get(language, "English")
        # Explicit, hard instruction — not left to inference from the text.
        language_directive = (
            f"\n\nIMPORTANT: You MUST reply entirely in {lang_name}, regardless of "
            f"how clear or garbled the transcribed input text looks. If the input "
            f"seems unclear or mistranscribed, politely say so IN {lang_name.upper()} "
            f"and ask the student to repeat — do not switch to a different language."
        )

        context_block = (
            f"CONTEXT (retrieved notes):\n{context}\n\n"
            if context
            else "CONTEXT: No matching notes were found — answer from general "
                 "knowledge, but mention that this isn't grounded in the student's notes.\n\n"
        )

        messages = [{"role": "system", "content": system_prompt + language_directive}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": context_block + f"STUDENT QUERY: {user_query}"})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,
            )
            reply = response.choices[0].message.content
            logger.info(f"Groq call succeeded (mode={mode}, language={language}, model={self.model})")
            return reply

        except Exception as e:
            logger.error(f"Groq API call failed (mode={mode}, model={self.model}): {e}")
            return FALLBACK_REPLY


if __name__ == "__main__":
    client = GroqClient()
    reply = client.generate(
        mode="explain",
        user_query="Explain Newton's second law with an example",
        context="Newton's second law states force equals mass times acceleration (F=ma).",
        language="hi",
    )
    print(reply)