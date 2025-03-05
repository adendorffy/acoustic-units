import textgrids
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def extract_alignments(align_dir: Path) -> pd.DataFrame:
    """
    Extract all the alignment data into a csv with the ground truth boundaries & text for each word.
    """
    textgrid_paths = list(align_dir.rglob("**/*.TextGrid"))
    csv_path = align_dir / "alignments.csv"

    alignments_df = pd.DataFrame(
        columns=["word_id", "filename", "word_start", "word_end", "text"]
    )

    for textgrid_path in tqdm(textgrid_paths, desc="Extracting Alignments"):
        grid = textgrids.TextGrid(textgrid_path)
        filename = textgrid_path.stem

        words_tier = grid["words"]
        for idx, interval in enumerate(words_tier, start=1):
            new_row = pd.DataFrame(
                [[idx - 1, filename, interval.xmin, interval.xmax, interval.text]],
                columns=alignments_df.columns,
            )
            alignments_df = pd.concat([alignments_df, new_row], ignore_index=True)

    alignments_df.to_csv(csv_path, index=False)
    print(f"Wrote alignments to {csv_path}")
    return alignments_df
