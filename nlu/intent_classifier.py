"""
Local spaCy-based intent classifier: routes each query to EXPLAIN or QUIZ mode.

Deliberately lightweight (pattern/keyword matching, not deep NLP) since there
are only two intents — a full classifier would be overkill and slower than
needed for a real-time voice pipeline. English lemmatization is used where
available; Hindi/Bengali rely on substring trigger matching since spaCy's
small English model doesn't lemmatize those languages.
"""

import spacy
from logger import get_logger

logger = get_logger(__name__)

nlp = spacy.load("en_core_web_sm")

QUIZ_TRIGGERS = {
    "quiz", "test", "mcq", "mcqs", "questions", "question", "practice",
    "प्रश्न", "क्विज़", "টেস্ট", "প্রশ্ন", "কুইজ",
}

EXPLAIN_TRIGGERS = {
    "explain", "what is", "how does", "how do", "understand", "teach", "describe",
    "समझाओ", "समझाइए", "क्या है", "बताओ", "বোঝাও", "কী", "ব্যাখ্যা",
}


def classify_intent(text: str) -> str:
    """
    Returns "quiz" or "explain". Defaults to "explain" for ambiguous input,
    since an unrequested explanation is a safer fallback than an unrequested quiz.
    """
    if not text or not text.strip():
        logger.warning("Empty text passed to intent classifier, defaulting to 'explain'")
        return "explain"

    text_lower = text.lower()

    try:
        doc = nlp(text_lower)
        lemmas = {token.lemma_ for token in doc}
    except Exception as e:
        logger.error(f"spaCy processing failed, falling back to substring match: {e}")
        lemmas = set()

    quiz_match = bool(lemmas & QUIZ_TRIGGERS) or any(t in text_lower for t in QUIZ_TRIGGERS)
    explain_match = bool(lemmas & EXPLAIN_TRIGGERS) or any(t in text_lower for t in EXPLAIN_TRIGGERS)

    if quiz_match and not explain_match:
        intent = "quiz"
    elif explain_match and not quiz_match:
        intent = "explain"
    elif quiz_match and explain_match:
        # both matched (e.g. "explain and then quiz me") — quiz is more specific/actionable
        intent = "quiz"
    else:
        intent = "explain"

    logger.info(f"Intent classified as '{intent}' for query: {text[:60]}")
    return intent


if __name__ == "__main__":
    # quick manual test: python -m nlu.intent_classifier
    tests = [
        "Explain photosynthesis to me",
        "Give me 3 MCQs on Newton's laws",
        "न्यूटन का नियम समझाओ",
        "आज का क्विज़ दो",
        "What is gravity",
    ]
    for t in tests:
        print(f"{t!r} -> {classify_intent(t)}")
        