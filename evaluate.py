import argparse
from pathlib import Path
from convert_partition import get_partition_path
from intervaltree import Interval
import pandas as pd
import re
from collections import defaultdict, Counter
from intervaltree import IntervalTree
import editdistance
from typing import Iterable, Tuple
import statistics
import itertools


def clean_phones(dirty_phones: str):
    return tuple(
        re.sub(r"[012]", "", phn).lower()
        for phn in str(dirty_phones).split(",")
        if phn.strip() not in {"sil", "sp", "", "spn"}
    )


def convert_to_intervals(align_df: pd.DataFrame):
    fragments = []
    non_silence_fragments = 0
    dirty_fragments = []
    for _, row in align_df.iterrows():
        speaker = row["filename"]
        start_time = row["word_start"]
        end_time = row["word_end"]
        phones = clean_phones(row["phones"])
        word = row["text"]
        if len(phones) > 0:
            non_silence_fragments += 1
        else:
            dirty_fragments.append(
                (speaker, Interval(float(start_time), float(end_time)), row["phones"])
            )
        fragments.append(
            (speaker, Interval(float(start_time), float(end_time)), phones, word)
        )

    return fragments, non_silence_fragments, dirty_fragments


def build_speaker_trees(gold_fragments):
    trees = defaultdict(IntervalTree)
    for speaker, interval, phones, word in gold_fragments:
        trees[speaker].addi(interval.begin, interval.end, (phones, word))
    return trees


def check_boundary(gold: Interval, disc: Interval) -> bool:
    if gold.contains_interval(disc):
        return True

    gold_duration = round(gold.end - gold.begin, 2)
    overlap_duration = round(gold.overlap_size(disc), 2)
    overlap_percentage = overlap_duration / gold_duration if gold_duration > 0 else 0

    duration_condition = gold_duration >= 0.06 and overlap_duration >= 0.03
    percentage_condition = gold_duration < 0.06 and overlap_percentage > 0.5
    return duration_condition or percentage_condition


def transcribe(
    discovered_fragments: list[tuple[str, Interval, int]],
    trees: dict[str, IntervalTree],
) -> list[tuple[str, Interval, tuple[str, ...], str]]:
    enriched_fragments = []

    for speaker, disc_interval, cluster in discovered_fragments:
        match_found = False
        if speaker in trees:
            overlaps = trees[speaker].overlap(disc_interval.begin, disc_interval.end)
            for match in overlaps:
                gold_interval = Interval(match.begin, match.end)
                phones, word = match.data
                match_found = True

                if check_boundary(gold_interval, disc_interval):
                    # match_found = True
                    if len(phones) > 0:
                        enriched_fragments.append(
                            (cluster, speaker, disc_interval, phones, word)
                        )

                    break
        if not match_found:
            print(f"No match found for {speaker} at {disc_interval}")

    return enriched_fragments


def distance(p: Tuple[str, ...], q: Tuple[str, ...]) -> float:
    length = max(len(p), len(q))
    return editdistance.eval(p, q) / length if length > 0 else 1


def ned(
    discovered_transcriptions: Iterable[
        Tuple[int, str, Interval, Tuple[str, ...], str]
    ],
    out_path: Path,
):
    sorted_transcriptions = sorted(discovered_transcriptions, key=lambda x: x[0])
    overall_distances = []
    with open(out_path, "w") as f:
        for cluster_id, group in itertools.groupby(
            sorted_transcriptions, key=lambda x: x[0]
        ):
            group_list = list(group)
            phones_list = [x[3] for x in group_list]

            # Write cluster summary
            f.write(f"\n{'-' * 60}\n")
            f.write(f"🧩 Cluster {cluster_id} | Size: {len(phones_list)}\n")
            f.write(f"{'-' * 60}\n")

            token_counter = Counter(phones_list)
            for tokens, count in token_counter.most_common():
                token_str = " ".join(tokens)
                f.write(f"{token_str:<30} → {count} times\n")

            if len(group_list) < 2:
                continue

            cluster_distances = []
            for p, q in itertools.combinations(group_list, 2):
                d = distance(p[3], q[3])

                cluster_distances.append(d)

            if cluster_distances:
                avg_cluster_ned = statistics.mean(cluster_distances)
                overall_distances.extend(cluster_distances)
                f.write(f"→ Avg NED for Cluster {cluster_id}: {avg_cluster_ned:.4f}\n")
                f.write(f"{'=' * 60}\n")

        if overall_distances:
            overall_avg = statistics.mean(overall_distances)
            f.write(f"\n🔍 Overall NED across all clusters: {overall_avg:.4f}\n")
        else:
            overall_avg = 0.0
            f.write("\n⚠️ No valid clusters with multiple elements found.\n")
    print(f"✅ NED report for [NED = {overall_avg * 100:.3f}%] written to: {out_path}")
    return overall_avg


def convert_dirty_fragments_to_csv(dirty_fragments, save_path):
    filenames = [filename for filename, _, _ in dirty_fragments]
    starts = [interval.begin for _, interval, _ in dirty_fragments]
    ends = [interval.end for _, interval, _ in dirty_fragments]
    phones = [phone for _, _, phone in dirty_fragments]

    df = pd.DataFrame(
        {"filename": filenames, "start": starts, "end": ends, "phones": phones},
        columns=["filename", "start", "end", "phones"],
    )
    df.to_csv(save_path, index=False)
    print(f"Saved dirty fragments to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "gamma", type=float, help="Gamma value used for feature extraction."
    )
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument("threshold", type=float, help="Threshold at whic to extract.")
    parser.add_argument(
        "output_dir",
        help="path to the discovered fragments.",
        type=Path,
    )

    parser.add_argument(
        "align_dir",
        help="path to the directory of alignments.",
        type=Path,
    )
    parser.add_argument(
        "--override",
        help="To override the normal path",
        default=None,
        type=Path,
    )
    args = parser.parse_args()
    if args.override:
        print(f"Use override path: {args.override}")
        list_dir = args.override
    else:
        _, resolution = get_partition_path(
            args.gamma, args.layer, args.threshold, args.output_dir
        )
        print(f"Resolution found: {resolution}")
        list_dir = (
            args.output_dir / str(args.gamma) / str(args.layer) / f"{resolution:.6f}"
        )
    files = list_dir.rglob("*.list")
    discovered_fragments = []
    for file in files:
        with open(file, "r") as f:
            start_time = 0.0
            for line in f:
                if len(line.split(" ")) == 2:
                    end_time, cluster = line.split(" ")
                    speaker = file.stem
                    discovered_fragments.append(
                        (
                            speaker,
                            Interval(float(start_time), float(end_time)),
                            int(cluster),
                        )
                    )

                    start_time = float(end_time)

    discovered_clusters = [cluster for _, _, cluster in discovered_fragments]

    print(
        f"Number of discovered clusters: {len(set(discovered_clusters))}, Number of discovered 'words': {len(discovered_fragments)}"
    )

    alignment_df = pd.read_csv(args.align_dir / "alignments.csv")
    gold_fragments, total_non_silence, dirty_fragments = convert_to_intervals(
        alignment_df
    )
    print(f"Dirty fragments: {len(dirty_fragments)}, ex: {dirty_fragments[0]}")

    convert_dirty_fragments_to_csv(dirty_fragments, "output/dirty_fragments.csv")
    trees = build_speaker_trees(gold_fragments)

    discovered_transcriptions = transcribe(discovered_fragments, trees)
    print(f"Example transcription: {discovered_transcriptions[0]}")
    print(
        f"Correct number of tokens (non-silence fragments): {'YES' if len(discovered_transcriptions) == total_non_silence else 'NO'} [{total_non_silence}|{len(discovered_transcriptions)}]"
    )

    # ned_value = ned(discovered_transcriptions, list_dir / "00_ned.txt")
