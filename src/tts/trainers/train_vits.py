"""Train a VITS model using a custom dataset and the Coqui Trainer with speaker adaptation"""

import json
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsArgs, VitsAudioConfig, VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def load_config_from_json(
    local_config_path: Path, base_config_path: Path
) -> VitsConfig:
    """
    Load a VitsConfig from the base pretrained config and override it with user-defined local config

    Args:
        local_config_path (Path): Path to the local JSON file with training overrides.
        base_config_path (Path): Path to the base pretrained model config.

    Returns:
        VitsConfig: Final config with overrides applied.
    """
    with open(base_config_path, encoding="utf-8") as f:
        base_dict = json.load(f)

    with open(local_config_path, encoding="utf-8") as f:
        local_dict = json.load(f)

    # Merge dataset and audio overrides if provided
    if "datasets" in local_dict:
        base_dict["datasets"] = local_dict.pop("datasets")
    if "audio" in local_dict:
        base_dict["audio"] = local_dict.pop("audio")

    # Update remaining fields
    base_dict.update(local_dict)

    # Parse nested fields into Coqpit-compatible classes
    base_dict["datasets"] = [BaseDatasetConfig(**ds) for ds in base_dict["datasets"]]
    base_dict["audio"] = VitsAudioConfig(**base_dict["audio"])
    if "characters" in base_dict:
        base_dict["characters"] = CharactersConfig(**base_dict["characters"])
    if "model_args" in base_dict:
        base_dict["model_args"] = VitsArgs(**base_dict["model_args"])

    return VitsConfig(**base_dict)


def main() -> None:
    """
    Load configuration by merging local overrides with a base pretrained config,
    initialize model and tokenizer using a pretrained VITS base,
    prepare the dataset, and fine-tune using the Coqui Trainer.
    """
    local_config_path = Path("src/tts/configs/vits-config.json").resolve()
    pretrained_dir = Path(
        "/Users/guzman.vitar/Library/Application Support/tts/tts_models--en--vctk--vits"
    ).expanduser()
    base_config_path = pretrained_dir / "config.json"

    # Merge base config with local overrides
    config = load_config_from_json(local_config_path, base_config_path)

    # Initialize audio processor and tokenizer
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Initialize model and load base checkpoint
    model = Vits(config, ap, tokenizer)
    model.load_checkpoint(
        config,
        checkpoint_path=pretrained_dir / "model_file.pth",
        eval=False,
        strict=False,
    )

    # Load speaker list from pretrained_dir
    speakers_file = pretrained_dir / "speaker_ids.json"
    with open(speakers_file, encoding="utf-8") as f:
        speaker_list = json.load(f)

    # Ignore all speakers except 'p230'
    ignored_speakers = [spk for spk in speaker_list if spk != "p230"]
    config.datasets[0]["ignored_speakers"] = ignored_speakers

    # Convert dataset dict to BaseDatasetConfig object
    dataset_cfg = BaseDatasetConfig(**config.datasets[0])

    # Load training samples
    train_samples, eval_samples = load_tts_samples(
        dataset_cfg, eval_split=True, eval_split_size=0.1
    )

    # Make sure all your custom samples are labeled with the speaker you want to fine-tune
    for sample in train_samples:
        sample["speaker_name"] = "p230"

    if eval_samples:
        for sample in eval_samples:
            sample["speaker_name"] = "p230"

    # Configure trainer
    trainer_args = TrainerArgs()
    trainer_args.logging_level = "INFO"

    trainer = Trainer(
        args=trainer_args,
        config=config,
        output_path=config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
