"""
Fine-tune a Coqui VITS/YourTTS checkpoint on your custom voice dataset.

Prereqs:
  - Run prepare_dataset.py first and fill in data/metadata.csv with
    correct transcripts
  - pip install TTS trainer (already in requirements.txt via `TTS`)

This uses Coqui's standard recipe pattern. Adjust `batch_size` and
`epochs` down if you're training on CPU or a small GPU — this is
tuned for a modest fine-tune, not from-scratch training.
"""

import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.config.shared_configs import BaseDatasetConfig

OUTPUT_PATH = "training/output"
DATA_PATH = "data"

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",  # our metadata.csv follows LJSpeech format
    meta_file_train="metadata.csv",
    path=DATA_PATH,
)

config = VitsConfig(
    audio={"sample_rate": 22050},
    batch_size=8,          # lower to 2-4 on CPU/small GPU
    eval_batch_size=4,
    num_loader_workers=2,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=100,             # fine-tuning needs far fewer epochs than training from scratch
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    print_step=25,
    save_step=500,
    output_path=OUTPUT_PATH,
    datasets=[dataset_config],
    # start from a pretrained checkpoint instead of random init:
    # download one via: tts --model_name tts_models/multilingual/multi-dataset/your_tts
)


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    ap = AudioProcessor.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_size=0.1,
    )

    model = Vits(config, ap)

    trainer = Trainer(
        TrainerArgs(
            # point this at a pretrained checkpoint to fine-tune rather
            # than train from scratch, e.g.:
            # restore_path="~/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/model_file.pth"
        ),
        config,
        OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()