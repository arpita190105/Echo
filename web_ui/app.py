"""
Gradio web UI for the study voice assistant.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import order matters here on Windows: load torch-dependent modules
# BEFORE gradio, to avoid a native-library (DLL) conflict where gradio's
# own dependencies load incompatible copies of shared runtime libraries
# first, causing torch's c10.dll to fail on init afterward.
from stt.whisper_stt import WhisperSTT
from tts.tts_router import TTSRouter
from llm.dialogue_manager import DialogueManager

import gradio as gr

stt = WhisperSTT()
tts_router = TTSRouter()
dialogue = DialogueManager()


def handle_turn(audio_path, selected_language):
    if audio_path is None:
        return "No audio received.", None, "", ""

    forced_lang = None if selected_language == "auto" else selected_language
    stt_result = stt.transcribe(audio_path, forced_language=forced_lang)

    user_text = stt_result["text"]
    detected_language = stt_result["language"]
    confidence = stt_result.get("language_confidence", 0.0)

    language = selected_language if selected_language != "auto" else detected_language

    if not user_text:
        return "Couldn't hear anything — try again.", None, "", ""

    result = dialogue.get_response(user_text, language=language)
    reply_audio_path = tts_router.synthesize(
        result["reply"], language=language, out_path="_ui_reply.wav"
    )

    override_note = "" if selected_language == "auto" else " (manually forced)"
    transcript_display = f"You ({language}{override_note}): {user_text}\n\nAssistant: {result['reply']}"
    mode_display = (
        f"Mode: {result['intent'].upper()}  |  Grounded in your notes: "
        f"{'Yes' if result['used_context'] else 'No'}"
    )
    if selected_language == "auto":
        mode_display += f"  |  Auto-detected: {detected_language} (confidence: {confidence:.2f})"

    return transcript_display, reply_audio_path, mode_display, language

def reset_conversation():
    dialogue.reset()
    return "Conversation reset.", None, "", ""


with gr.Blocks(title="Echo") as demo:
    gr.HTML(
        """
        <h1 style="font-size: 2.5em; font-weight: 800; margin-bottom: 0;">
             Echo
        </h1>
        <p style="font-size: 1.1em; color: #a0a0a0; margin-top: 4px;">
            Multilingual Study Voice Assistant &nbsp;•&nbsp; English / Hindi / Bengali
        </p>
        """
    )

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Ask a question")

    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=[
                ("Auto-detect", "auto"),
                ("English", "en"),
                ("Hindi", "hi"),
                ("Bengali", "bn"),
            ],
            value="auto",
            label="Choose Language",
        )

    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        reset_btn = gr.Button("Reset conversation")

    transcript_box = gr.Textbox(label="Conversation", lines=6)
    mode_box = gr.Textbox(label="Detection info", lines=1)
    audio_output = gr.Audio(label="Assistant's reply", autoplay=True)
    lang_box = gr.Textbox(label="Detected language", visible=True)

    submit_btn.click(
        fn=handle_turn,
        inputs=[audio_input, language_dropdown],
        outputs=[transcript_box, audio_output, mode_box, lang_box],
    )
    reset_btn.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[transcript_box, audio_output, mode_box, lang_box],
    )

if __name__ == "__main__":
    demo.launch()