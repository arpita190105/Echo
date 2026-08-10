"""
Dialogue manager for the study assistant: ties together spaCy intent
classification, RAG retrieval, and Groq generation.

Flow: user text -> classify intent (explain/quiz) -> retrieve relevant
notes -> generate grounded response via Groq -> return to caller for TTS.
"""

from nlu.intent_classifier import classify_intent
from rag.retriever import Retriever
from llm.groq_client import GroqClient
from logger import get_logger

logger = get_logger(__name__)


class DialogueManager:
    def __init__(self):
        self.retriever = Retriever()
        self.groq = GroqClient()
        self.history = []  # list of {"role": ..., "content": ...} for follow-ups

    def get_response(self, user_text: str, language: str = "en") -> dict:
        """
        Args:
            user_text: transcribed student query
            language: detected language from STT — explicitly forwarded to
                      the LLM so the reply language is enforced, not inferred

        Returns:
            {"reply": str, "intent": str, "used_context": bool}
        """
        if not user_text or not user_text.strip():
            logger.warning("Empty user_text passed to get_response")
            return {"reply": "I didn't catch that — could you repeat it?", "intent": "explain", "used_context": False}

        intent = classify_intent(user_text)
        context = self.retriever.retrieve(user_text)

        reply = self.groq.generate(
            mode=intent,
            user_query=user_text,
            context=context,
            history=self.history[-6:],
            language=language,
        )

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})

        logger.info(f"Turn complete — intent={intent}, language={language}, used_context={bool(context)}")

        return {
            "reply": reply,
            "intent": intent,
            "used_context": bool(context),
        }

    def reset(self):
        self.history = []
        logger.debug("Conversation history reset")


if __name__ == "__main__":
    # quick manual test: python -m llm.dialogue_manager
    dm = DialogueManager()

    result = dm.get_response("Explain Newton's second law with an example", language="en")
    print(f"[{result['intent']}] {result['reply']}")

    result = dm.get_response("Give me 2 MCQs on the same topic", language="en")
    print(f"[{result['intent']}] {result['reply']}")