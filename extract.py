import argparse
import textgrids
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def extract_alignments(align_dir: Path) -> pd.DataFrame:
    """
    Extract all the alignment data into a CSV with the ground truth boundaries & text for each word.
    """
    textgrid_paths = list(align_dir.rglob("**/*.TextGrid"))
    csv_path = align_dir / "alignments.csv"

    alignments_df = pd.DataFrame(
        columns=["word_id", "filename", "word_start", "word_end", "text", "phones"]
    )

    for textgrid_path in tqdm(textgrid_paths, desc="Extracting Alignments"):
        grid = textgrids.TextGrid(textgrid_path)
        filename = textgrid_path.stem

        words_tier = grid["words"]
        phones_tier = grid["phones"]

        for idx, word_interval in enumerate(words_tier, start=1):
            word_text = word_interval.text.strip()

            word_start, word_end = word_interval.xmin, word_interval.xmax
            word_phones = [
                phone_interval.text.strip()
                for phone_interval in phones_tier
                if phone_interval.xmax > word_start and phone_interval.xmin < word_end
            ]

            new_row = pd.DataFrame(
                [
                    [
                        idx - 1,
                        filename,
                        word_start,
                        word_end,
                        word_text,
                        ",".join(word_phones),
                    ]
                ],
                columns=alignments_df.columns,
            )

            alignments_df = pd.concat([alignments_df, new_row], ignore_index=True)

    alignments_df.to_csv(csv_path, index=False)
    print(f"Wrote alignments to {csv_path}")
    return alignments_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract alignments from TextGrid files."
    )
    parser.add_argument(
        "align_dir", type=Path, help="Path to the directory containing TextGrid files."
    )

    args = parser.parse_args()
    extract_alignments(args.align_dir)
