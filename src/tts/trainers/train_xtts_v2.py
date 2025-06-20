import json
from pathlib import Path

import click
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
)
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

from src.constants import DATA_COQUI, MODEL_CONFIG_PATH, MODEL_OUTPUT_PATH
from src.logger_definition import get_logger

logger = get_logger(__file__)


def prepare_xtts_v2_checkpoints(output_path: Path) -> tuple[str, str, str, str]:
    """Prepare required XTTS v2.0 checkpoint files for training or inference.

    This function ensures the DVAE model, mel statistics, XTTS model checkpoint,
    and tokenizer vocab file are downloaded into a local output directory.
    If any file is missing, it will be automatically downloaded.

    Args:
        output_path (Path): Directory where model checkpoint files will be stored.

    Returns:
        Tuple[str, str, str, str]: Paths to:
            - dvae_checkpoint
            - mel_norm_file
            - xtts_checkpoint
            - tokenizer_file
    """
    output_path.mkdir(parents=True, exist_ok=True)

    files = {
        "dvae": "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
        "mel_stats": "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth",
        "tokenizer": "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
        "xtts": "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
    }

    local_paths = {key: output_path / Path(url).name for key, url in files.items()}

    missing_urls = [url for key, url in files.items() if not local_paths[key].is_file()]

    if missing_urls:
        print(" > Downloading missing XTTS v2.0 files...")
        ModelManager._download_model_files(missing_urls, output_path, progress_bar=True)

    return (
        str(local_paths["dvae"]),
        str(local_paths["mel_stats"]),
        str(local_paths["xtts"]),
        str(local_paths["tokenizer"]),
    )


@click.command()
@click.option(
    "--voice",
    type=str,
    required=True,
    help="Dataset voice name to use for training.",
)
def main(voice: str):
    # speaker reference to be used in training test sentences
    SPEAKER_REFERENCE = [
        str(MODEL_OUTPUT_PATH / voice / "speaker_references" / "ref_en.wav")
    ]
    output_path = str(MODEL_OUTPUT_PATH / voice)

    # Load local config
    config_path = MODEL_CONFIG_PATH / "xttsv2-config.json"
    with open(config_path, encoding="utf-8") as f:
        xtts_config = json.load(f)

    # Load checkpoints
    CHECKPOINTS_OUT_PATH = MODEL_OUTPUT_PATH / "XTTS_v2.0_original_model_files/"
    dvae_checkpoint, mel_norm_file, xtts_checkpoint, tokenizer_file = (
        prepare_xtts_v2_checkpoints(CHECKPOINTS_OUT_PATH)
    )

    # init args and config
    xtts_config["base_args"]["path"] = str(DATA_COQUI / voice)
    config_dataset = BaseDatasetConfig(**xtts_config["base_args"])

    model_args = GPTArgs(
        dvae_checkpoint=dvae_checkpoint,
        xtts_checkpoint=xtts_checkpoint,
        mel_norm_file=mel_norm_file,
        tokenizer_file=tokenizer_file,
        **xtts_config["gpt_args"],
    )
    # audio config
    audio_config = XttsAudioConfig(**xtts_config["audio_args"])

    # training parameters config
    config = GPTTrainerConfig(
        logger_uri=None,
        output_path=output_path,
        model_args=model_args,
        audio=audio_config,
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it"
                " I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": config_dataset.language,
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": config_dataset.language,
            },
        ],
        **xtts_config["gpt_trainer_args"],
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        [config_dataset],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and run
    trainer = Trainer(
        TrainerArgs(
            **xtts_config["trainer_args"],
            restore_path=None,
            skip_train_epoch=False,
        ),
        config,
        output_path=str(output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
