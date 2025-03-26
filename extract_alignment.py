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

    for i, textgrid_path in enumerate(textgrid_paths, start=1):
        try:
            grid = textgrids.TextGrid(textgrid_path)
            filename = textgrid_path.stem

            words_tier = grid.get("words")
            phones_tier = grid.get("phones")

            if not words_tier or not phones_tier:
                print(f"⚠️ Missing 'words' or 'phones' tier in {filename}. Skipping.")
                continue

            for idx, word_interval in enumerate(words_tier):
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

        except Exception as e:
            print(f"⚠️ Error processing {textgrid_path}: {e}")

        progress = int((i / total_files) * 100)
        if progress % 10 == 0 and progress > prev_progress:
            print(f"\r🟢 Progress: {progress}% ({i}/{total_files} files)")
            prev_progress = progress

    if not alignment_records:
        print(f"⚠️ No valid alignments extracted from {align_dir}.")
        return pd.DataFrame()

    alignments_df = pd.DataFrame(alignment_records)

    return alignments_df


def match_to_feat_paths(
    gamma: float,
    layer: int,
    alignments_df: pd.DataFrame,
    feat_dir: Path,
    out_path: Path,
) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / "alignments_aligned_to_features.csv"

    if file_path.exists():
        print("Alignments are already aligned to features.")
        return

    paths = sorted(
        Path(feat_dir / str(gamma) / str(layer)).rglob("**/*.npy"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )

    texts = []
    phones = []
    starts, ends = [], []
    filenames = []
    prev_progress = -1

    print("Getting Alignments in order...", flush=True)
    for i, path in enumerate(paths):
        filename_parts = path.stem.split("_")
        wav_df = alignments_df[alignments_df["filename"] == filename_parts[0]]
        word_df = wav_df[wav_df["word_id"] == int(filename_parts[1])]

        texts.append(str(word_df["text"].iloc[0]))

        word_phones = str(word_df["phones"].iloc[0]).split(",")
        word_phones = " ".join(word_phones)
        phones.append(word_phones)

        starts.append(word_df["word_start"].iloc[0])
        ends.append(word_df["word_end"].iloc[0])

        filenames.append(filename_parts[0])

        progress = int((i / len(paths)) * 100)
        if progress % 10 == 0 and progress > prev_progress:
            print(f"🟢 Progress: {progress}% ({i}/{len(paths)} files)")
            prev_progress = progress

    df = pd.DataFrame(
        {
            "filename": filenames,
            "text": texts,
            "phones": phones,
            "start": starts,
            "end": ends,
        }
    )

    df.to_csv(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract alignments from TextGrid files."
    )
    parser.add_argument(
        "gamma", type=float, help="Gamma value used for feature extraction."
    )
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument(
        "align_dir", type=Path, help="Path to the directory containing TextGrid files."
    )
    parser.add_argument(
        "feat_dir", type=Path, help="Path to the directory with encodings."
    )
    parser.add_argument("out_path", type=Path, help="Path to store alignments at.")

    args = parser.parse_args()

    alignments_df = extract_alignments(args.align_dir)

    match_to_feat_paths(
        args.gamma, args.layer, alignments_df, args.feat_dir, args.out_path
    )
