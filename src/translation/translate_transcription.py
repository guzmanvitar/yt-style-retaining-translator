"""
Translate segmented audio transcripts from English to Spanish using an LLM service.

This script reads CSVs containing segmented transcripts from a directory,
batches the text for efficient LLM translation with context preservation,
and writes a single CSV with the translated segments.

Usage:
    python translate_segments.py
"""

from pathlib import Path

import click
import pandas as pd

from src.constants import DATA_INFERENCE, DATA_PROCESSED
from src.logger_definition import get_logger
from src.translation.llm_service.services import LLMServiceFactory

logger = get_logger(__file__)


def load_segment_csvs(segments_dir: Path) -> pd.DataFrame:
    """Load all segment CSVs from a directory into one DataFrame.

    Args:
        segments_dir (Path): Path to the directory containing CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame with 'filename', 'start', 'end', and 'text'.
    """
    dfs = []
    for file in sorted(segments_dir.glob("*.csv")):
        df = pd.read_csv(file)
        df["filename"] = file.stem
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def format_text_for_prompt(texts: list[str]) -> str:
    """Wrap texts in <s> and </s> tags for LLM prompt.

    Args:
        texts (List[str]): List of input phrases.

    Returns:
        str: Prompt-ready text block.
    """
    return "\n".join(f"<s>{t}</s>" for t in texts)


def extract_translations(response: str) -> list[str]:
    """Extract <s>translated text</s> lines from LLM response.

    Args:
        response (str): Raw response from the LLM.

    Returns:
        List[str]: List of translated strings.
    """
    return [
        line.strip()[3:-4]
        for line in response.strip().splitlines()
        if line.startswith("<s>") and line.endswith("</s>")
    ]


def translate_segments(df: pd.DataFrame, batch_size: int = 80) -> pd.DataFrame:
    """Translate a DataFrame of segments using the LLM in batches.

    Args:
        df (pd.DataFrame): DataFrame with segments to translate.
        batch_size (int): Number of rows per LLM call.

    Returns:
        pd.DataFrame: Translated DataFrame.
    """
    service = LLMServiceFactory("gpt-4", "translator").get_service()
    translated_batches = []
    len_df = len(df)

    for i in range(0, len_df, batch_size):
        batch = df.iloc[i : i + batch_size]
        prompt_text = format_text_for_prompt(batch["text"].tolist())

        if not service.initial_prompt:
            raise ValueError("Missing system prompt for translation service")
        prompt = service.initial_prompt.format(sliced_text=prompt_text)

        response = service.generate_formatted_response(prompt)
        translations = extract_translations(response)

        batch_copy = batch.copy()
        batch_copy["text"] = translations
        translated_batches.append(batch_copy)
        logger.info(f"Processed batch {min(i + batch_size, len_df)}/{len_df}")

    return pd.concat(translated_batches, ignore_index=True)


@click.command()
def main():
    """Translate all segment CSVs and output a combined translated CSV."""
    input_dir = DATA_PROCESSED / "segments"
    output_path = DATA_INFERENCE / "translated_segments.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_segment_csvs(input_dir)
    translated_df = translate_segments(df)
    translated_df.to_csv(output_path, index=False)

    logger.info(f"Translated segments written to: {output_path}")


if __name__ == "__main__":
    main()
