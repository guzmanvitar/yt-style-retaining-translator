"""Audio preprocessing script for resampling and downmixing WAV files."""

from pydub import AudioSegment

from src.constants import DATA_PROCESSED, DATA_RECORDINGS
from src.logger_definition import get_logger

logger = get_logger(__file__)


def convert_audio():
    """
    Convert a stereo 48kHz WAV file to mono 22.05kHz WAV.

    This function reads the original audio file from the
    `DATA_RECORDINGS` directory, downmixes it to mono, resamples it to 22050 Hz,
    and exports the result to a new file in the `DATA_PROCESSED` directory.

    The output format matches the expectations of the VITS base model (`en/vctk/vits`),
    which uses 16-bit PCM, mono, 22.05kHz WAV input.
    """
    # Paths
    input_path = DATA_RECORDINGS / "tolkien-speech-clean.wav"
    output_path = DATA_PROCESSED / "tolkien-speech-clean_22k.wav"

    # Load original WAV
    audio = AudioSegment.from_wav(input_path)

    # Convert to mono + 22050 Hz
    audio = audio.set_channels(1).set_frame_rate(22050)

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(output_path, format="wav")

    logger.info("Saved 22.05kHz mono WAV to %s", output_path)


if __name__ == "__main__":
    convert_audio()
