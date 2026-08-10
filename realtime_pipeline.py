"""
Real-time voice agent pipeline using Pipecat: mic -> VAD -> STT -> LLM
(with tool calling) -> TTS -> speaker, with barge-in support.

This replaces the file-based main.py loop with live streaming audio.

NOTE: Pipecat's exact class names/signatures shift between versions.
This is written against the pipeline/frame-processor pattern in
pipecat-ai >= 0.0.50. If an import fails, check `pipecat.frames.frames`
and `pipecat.pipeline` in your installed version and adjust imports —
the pipeline *shape* below (VAD -> STT -> LLM -> TTS) stays the same.
"""

import asyncio

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TextFrame, TranscriptionFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from stt.whisper_stt import WhisperSTT
from llm.dialogue_manager import DialogueManager
from tts.tts_router import TTSRouter


class WhisperSTTProcessor(FrameProcessor):
    """Wraps our faster-whisper STT as a Pipecat frame processor."""

    def __init__(self):
        super().__init__()
        self.stt = WhisperSTT()

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Expect raw audio chunks buffered upstream into a temp wav by the
        # transport/VAD layer; here we assume `frame.audio_path` is set
        # once an utterance is complete (see transport VAD config below).
        if hasattr(frame, "audio_path"):
            result = self.stt.transcribe(frame.audio_path)
            if result["text"]:
                await self.push_frame(
                    TranscriptionFrame(result["text"], "", result["language"]),
                    direction,
                )
        else:
            await self.push_frame(frame, direction)


class DialogueLLMProcessor(FrameProcessor):
    """Wraps our Ollama-based dialogue manager (with tool calling) as a processor."""

    def __init__(self):
        super().__init__()
        self.dialogue = DialogueManager()

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            language = getattr(frame, "language", "en") or "en"
            reply = self.dialogue.get_response(frame.text, language=language)
            out_frame = TextFrame(reply)
            out_frame.language = language  # carry language forward for TTS routing
            await self.push_frame(out_frame, direction)
        else:
            await self.push_frame(frame, direction)


class RoutedTTSProcessor(FrameProcessor):
    """Wraps the Coqui/Indic-Parler TTS router as a processor."""

    def __init__(self):
        super().__init__()
        self.router = TTSRouter()

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            language = getattr(frame, "language", "en") or "en"
            audio_path = self.router.synthesize(frame.text, language=language, out_path="_reply.wav")
            await self.push_frame(TTSAudioRawFrame(audio_path=audio_path), direction)
        else:
            await self.push_frame(frame, direction)


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),  # handles turn-taking + barge-in
        )
    )

    pipeline = Pipeline([
        transport.input(),
        WhisperSTTProcessor(),
        DialogueLLMProcessor(),
        RoutedTTSProcessor(),
        transport.output(),
    ])

    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    print("Starting real-time voice agent. Speak into your mic (Ctrl+C to stop)...")
    asyncio.run(main())