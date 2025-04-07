from pathlib import Path
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import random
import argparse
from sklearn.metrics import roc_auc_score, roc_curve
import editdistance
import numpy as np
import joblib
import matplotlib.pyplot as plt


def label_pair(row):
    same_word = row["text1"] == row["text2"]
    same_speaker = row["speaker1"] == row["speaker2"]
    if same_word and same_speaker:
        return "C1"
    elif same_word and not same_speaker:
        return "C2"
    elif not same_word and same_speaker:
        return "C3"
    else:
        return "C4"


def create_samediff_dataset(align_df: pd.DataFrame, out_dir: Path):
    segments = []
    seen_speakers = set()
    for filename in tqdm(align_df["filename"].unique(), desc="Processing files"):
        speaker = filename.split("-")[0]
        if speaker not in seen_speakers:
            seen_speakers.add(speaker)
        file_df = align_df[align_df["filename"] == filename]

        for word_id in file_df["word_id"].unique():
            word_df = file_df[file_df["word_id"] == word_id]
            if len(word_df) == 0:
                continue
            base_filename = filename.split("_")[0]
            text = str(word_df["text"].values[0])
            if not text or text == "<unk>":
                continue
            segment = {
                "filename": base_filename,
                "speaker": speaker,
                "text": text,
                "word_start": word_df["word_start"].values[0],
                "word_end": word_df["word_end"].values[0],
                "phones": word_df["phones"].values[0],
            }
            segments.append(segment)
    segment_df = pd.DataFrame(segments)

    min_instances = 5
    word_counts = segment_df["text"].value_counts()
    valid_words = word_counts[word_counts >= min_instances].index
    subset_df = segment_df[segment_df["text"].isin(valid_words)].reset_index(drop=True)

    max_pairs_per_word = 200
    max_word_instances = 100
    same_pairs = []
    for _, group in tqdm(
        subset_df.groupby("text"), desc="Creating same pairs", total=len(valid_words)
    ):
        group = group[:max_word_instances]  # Keep first 100 occurrences only

        items = group.to_dict("records")
        all_combos = list(combinations(items, 2))
        sampled_combos = random.sample(
            all_combos, min(len(all_combos), max_pairs_per_word)
        )

        for a, b in sampled_combos:
            label = 1
            pair_type = label_pair(
                pd.Series(
                    {
                        "text1": a["text"],
                        "text2": b["text"],
                        "speaker1": a["speaker"],
                        "speaker2": b["speaker"],
                    }
                )
            )
            same_pairs.append((a, b, label, pair_type))

    all_items = subset_df.to_dict("records")
    diff_pairs = []
    attempts = 0
    max_attempts = len(same_pairs) * 10

    while len(diff_pairs) < len(same_pairs) and attempts < max_attempts:
        a, b = random.sample(all_items, 2)
        if a["text"] != b["text"]:
            label = 0
            pair_type = label_pair(
                pd.Series(
                    {
                        "text1": a["text"],
                        "text2": b["text"],
                        "speaker1": a["speaker"],
                        "speaker2": b["speaker"],
                    }
                )
            )
            diff_pairs.append((a, b, label, pair_type))
        attempts += 1

    pair_records = []
    for a, b, label, pair_type in same_pairs + diff_pairs:
        pair_records.append(
            {
                "file1": a["filename"],
                "speaker1": a["speaker"],
                "text1": a["text"],
                "start1": a["word_start"],
                "end1": a["word_end"],
                "phones1": a["phones"],
                "file2": b["filename"],
                "speaker2": b["speaker"],
                "text2": b["text"],
                "start2": b["word_start"],
                "end2": b["word_end"],
                "phones2": b["phones"],
                "label": label,
                "pair_type": pair_type,
            }
        )

    pairs_df = pd.DataFrame(pair_records)

    out_file = out_dir / "same_diff_pairs_labeled.csv"
    pairs_df.to_csv(out_file, index=False)
    print(f"✅ Saved {len(pairs_df)} labeled pairs to {out_file}")

    return pairs_df


def load_encodings(feat_dir: Path, align_df: pd.DataFrame, filenames_in_pairs_df: set):
    if (feat_dir / "word_encodings.pkl").exists():
        print("✅ Word encodings already exist. Loading...")
        return joblib.load(feat_dir / "word_encodings.pkl")
    encodings = {}

    for file in tqdm(
        feat_dir.glob("**/*.npy"),
        desc="Loading encodings",
        total=len(list(feat_dir.glob("**/*.npy"))),
    ):
        filename = file.stem.split("_")[0]
        word_id = int(file.stem.split("_")[1])
        if (
            filename not in align_df["filename"].values
            or filename not in filenames_in_pairs_df
        ):
            continue
        file_df = align_df[align_df["filename"] == filename]

        if word_id not in file_df["word_id"].values:
            print(f"⚠️ Word ID {word_id} not found in {filename}. Skipping.")
            continue

        word_df = file_df[file_df["word_id"] == word_id]
        start_time = word_df["word_start"].iloc[0]
        end_time = word_df["word_end"].iloc[0]

        key = (filename, float(start_time), float(end_time))
        encodings[key] = np.load(file)

    joblib.dump(encodings, feat_dir / "word_encodings.pkl")
    return encodings


def same_different_evaluation(
    pairs_df: pd.DataFrame, feat_dir: Path, align_df: pd.DataFrame, out_dir: Path
):
    file_info = str(feat_dir).split("/")
    out_file = (
        out_dir / f"roc_{file_info[0]}_{file_info[1]}_{file_info[2]}_{file_info[3]}.png"
    )

    filenames_in_pairs_df = set(pairs_df["file1"]).union(set(pairs_df["file2"]))
    word_encodings = load_encodings(feat_dir, align_df, filenames_in_pairs_df)

    skipped = 0
    labels = []
    scores = []

    for _, row in pairs_df.iterrows():
        key1 = (row["file1"], row["start1"], row["end1"])
        key2 = (row["file2"], row["start2"], row["end2"])
        seq1 = word_encodings.get(key1)
        seq2 = word_encodings.get(key2)

        if seq1 is None or len(seq1) == 0:
            skipped += 1
            continue
        if seq2 is None or len(seq2) == 0:
            skipped += 1
            continue

        d = editdistance.eval(seq1, seq2) / max(len(seq1), len(seq2))
        score = -d

        labels.append(row["label"])
        scores.append(score)

    print(f"Skipped {skipped} pairs due to missing encodings.")

    if len(set(labels)) < 2:
        print("⚠️ Not enough positive/negative samples to compute ROC.")
        return

    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    # Plot once
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Same-Different ROC Curve")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Save first, then show (optional)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"✅ ROC curve saved to {out_file}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create same/different pairs dataset")

    parser.add_argument(
        "feat_dir",
        type=Path,
        default="features/",
        help="Input file path for the features",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        default="librispeech/dev-clean/",
        help="Output file path for the labeled dataset",
    )
    parser.add_argument(
        "align_dir",
        type=Path,
        default="librispeech/dev-clean/",
        help="Input file path for the alignments",
    )
    args = parser.parse_args()

    align_df = pd.read_csv(
        args.align_dir / "alignments.csv",
    )
    align_df["text"] = align_df["text"].str.lower()

    if Path(args.out_dir / "same_diff_pairs_labeled.csv").exists():
        print(
            f"Labeled pairs file {args.out_dir / 'same_diff_pairs_labeled.csv'} already exists"
        )
        pairs_df = pd.read_csv(args.out_dir / "same_diff_pairs_labeled.csv")
    else:
        pairs_df = create_samediff_dataset(align_df, args.out_dir)

    same_different_evaluation(pairs_df, args.feat_dir, align_df, args.out_dir)
