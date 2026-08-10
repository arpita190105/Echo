"""
Prepare a small custom-voice dataset for fine-tuning Coqui VITS/YourTTS.

Expected input: a folder of raw recordings (any length/format) that you
record yourself reading sentences aloud (30-60 min total is a good target
for a decent fine-tune; even 5-10 min works for voice cloning with YourTTS).

This script:
  1. Converts everything to 22050Hz mono WAV (Coqui's expected format)
  2. Splits long recordings into utterance-level clips using silence detection
  3. Builds the metadata.csv file Coqui's formatter expects (LJSpeech format)

You still need a transcript for each clip. The easiest free workflow:
  - Record yourself reading from a fixed script (recommended), so you
    already know the text for each line, OR
  - Use WhisperSTT (from stt/whisper_stt.py) to auto-transcribe clips,
    then manually correct the transcripts (auto-transcripts are NOT
    reliable enough to train on without a human pass).
"""

import os
import csv
import glob
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence

RAW_DIR = "data/raw_recordings"
CLIPS_DIR = "data/clips"
METADATA_PATH = "data/metadata.csv"
TARGET_SR = 22050


def convert_and_split(min_silence_len=400, silence_thresh=-40, keep_silence=200):
    os.makedirs(CLIPS_DIR, exist_ok=True)
    clip_index = 0
    rows = []

    audio_files = glob.glob(os.path.join(RAW_DIR, "*.*"))
    if not audio_files:
        print(f"No files found in {RAW_DIR}. Add recordings there first.")
        return

    for filepath in audio_files:
        print(f"Processing {filepath}...")
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_frame_rate(TARGET_SR).set_channels(1)

        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
        )

        for chunk in chunks:
            # skip clips that are too short or too long to be useful
            duration_s = len(chunk) / 1000.0
            if duration_s < 1.0 or duration_s > 15.0:
                continue

            clip_name = f"clip_{clip_index:04d}.wav"
            clip_path = os.path.join(CLIPS_DIR, clip_name)
            chunk.export(clip_path, format="wav")

            rows.append({"clip": clip_name, "text": ""})  # fill in text manually or via STT
            clip_index += 1

    write_metadata(rows)
    print(f"Done. {clip_index} clips saved to {CLIPS_DIR}")
    print(f"IMPORTANT: fill in the 'text' column in {METADATA_PATH} before training.")


def write_metadata(rows):
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        for row in rows:
            # LJSpeech format: clip_id|text|normalized_text
            clip_id = row["clip"].replace(".wav", "")
            writer.writerow([clip_id, row["text"], row["text"]])


def auto_transcribe_clips():
    """
    Optional helper: auto-fill metadata.csv text using WhisperSTT.
    ALWAYS manually review/correct the output before training —
    ASR errors baked into training data will degrade the fine-tuned voice.
    """
    from stt.whisper_stt import WhisperSTT

    stt = WhisperSTT()
    rows = []
    clip_files = sorted(glob.glob(os.path.join(CLIPS_DIR, "*.wav")))

    for clip_path in clip_files:
        result = stt.transcribe(clip_path)
        clip_id = os.path.basename(clip_path).replace(".wav", "")
        rows.append({"clip": clip_id + ".wav", "text": result["text"]})
        print(f"{clip_id}: {result['text']}")

    write_metadata(rows)
    print(f"Auto-transcription done. REVIEW {METADATA_PATH} before training.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--transcribe":
        auto_transcribe_clips()
    else:
        convert_and_split()