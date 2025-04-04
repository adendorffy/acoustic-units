import argparse
from pathlib import Path

import pandas as pd
import textgrids
from typing import List


def extract_alignments(align_dir: Path) -> pd.DataFrame:
    print("Getting alignments .. ")
    textgrid_paths = list(align_dir.rglob("**/*.TextGrid"))
    total_files = len(textgrid_paths)
    prev_progress = -1

    if not textgrid_paths:
        print(f"⚠️ No TextGrid files found in {align_dir}.")
        return pd.DataFrame()

    alignment_records: List[dict] = []
    num_words = 0
    for i, textgrid_path in enumerate(textgrid_paths, start=1):
        try:
            grid = textgrids.TextGrid(textgrid_path)
            filename = textgrid_path.stem

            words_tier = grid.get("words")
            phones_tier = grid.get("phones")

            if not words_tier or not phones_tier:
                print(f"⚠️ Missing 'words' or 'phones' tier in {filename}. Skipping.")
                continue

            for idx, word_interval in enumerate(words_tier, 1):
                word_text = word_interval.text.strip()
                word_start, word_end = word_interval.xmin, word_interval.xmax

                word_phones = [
                    phone_interval.text.strip()
                    for phone_interval in phones_tier
                    if phone_interval.xmax > word_start
                    and phone_interval.xmin < word_end
                ]

                alignment_records.append(
                    {
                        "word_id": idx,
                        "filename": filename,
                        "word_start": word_start,
                        "word_end": word_end,
                        "text": word_text,
                        "phones": ",".join(word_phones),
                    }
                )

            num_words += idx

        except Exception as e:
            print(f"⚠️ Error processing {textgrid_path}: {e}")

        progress = int((i / total_files) * 100)
        if progress % 10 == 0 and progress > prev_progress:
            print(f"\r🟢 Progress: {progress}% ({i}/{total_files} files)")
            prev_progress = progress

    if not alignment_records:
        print(f"⚠️ No valid alignments extracted from {align_dir}.")
        return pd.DataFrame()

    print(f"Total words processed: {num_words}")
    alignments_df = pd.DataFrame(alignment_records)
    alignments_df.to_csv(align_dir / "alignments.csv")
    print(f"Stored alignments to {align_dir / 'alignments.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract alignments from TextGrid files."
    )

    parser.add_argument(
        "align_dir", type=Path, help="Path to the directory containing TextGrid files."
    )

    args = parser.parse_args()

    print(f"extract_alignment.py [{args.align_dir}]", flush=True)
    extract_alignments(args.align_dir)
