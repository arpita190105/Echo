"""
Central configuration for the study voice assistant.
"""

from dataclasses import dataclass, field


@dataclass
class STTConfig:
    model_size: str = "medium"
    device: str = "cpu"
    compute_type: str = "int8"
    supported_languages: list = field(default_factory=lambda: ["en", "hi", "bn"])


@dataclass
class TTSConfig:
    model_name: str = "tts_models/multilingual/multi-dataset/your_tts"
    default_language: str = "en"
    speaker_wav: str | None = None


@dataclass
class LLMConfig:
    provider: str = "groq"
    model_name: str = "openai/gpt-oss-120b"

    explain_system_prompt: str = (
        "You are a patient, friendly study tutor for Indian students, fluent in "
        "English, Hindi, and Bengali. Respond in the SAME language the student used. "
        "If they mixed languages, mirror that mix naturally. Use the provided CONTEXT "
        "(retrieved notes) as your primary source of truth — if it doesn't cover "
        "something, say so before adding general knowledge. Structure every "
        "explanation as: a one-line summary, a step-by-step breakdown, one concrete "
        "example, then a short recap. Keep sentences short and speakable, since this "
        "will be converted to speech — avoid markdown, bullet symbols, or tables; use "
        "natural spoken phrasing like 'first... next... finally...' instead."
    )

    quiz_system_prompt: str = (
        "You are a quiz generator for Indian students, fluent in English, Hindi, and "
        "Bengali. Respond in the SAME language the student used. Use the provided "
        "CONTEXT (retrieved notes) to generate accurate questions — do not invent "
        "facts not supported by it. Generate exactly the number and type of questions "
        "requested (default: 3 MCQs if unspecified). For each question: state it "
        "clearly, then say 'Option A... Option B...' spoken-style (not symbols like "
        "'A)'), then state the correct answer with a one-line reason. Keep phrasing "
        "natural for speech."
    )


@dataclass
class RAGConfig:
    collection_name: str = "study_notes"
    persist_dir: str = "data/chroma_db"
    embedding_model: str = "BAAI/bge-m3"

    top_k: int = 8
    distance_threshold: float = 0.45     # tightened now that this is genuine cosine distance
    min_results: int = 1
    max_results: int = 4
    min_keyword_overlap: float = 0.15    # hard filter floor for Latin-script queries

@dataclass
class AgentConfig:
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    sample_rate: int = 16000


config = AgentConfig()