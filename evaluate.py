import argparse
from pathlib import Path
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
        re.sub(r"[012]", "", phn)
        for phn in str(dirty_phones).split(",")
        if phn.strip() not in {"sil", "sp", "", "spn", "nan"}
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

    return fragments, non_silence_fragments


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
    silences: bool = False,
) -> list[tuple[str, Interval, tuple[str, ...], str]]:
    enriched_fragments = []

    for speaker, disc_interval, cluster in discovered_fragments:
        match_found = False
        if speaker not in trees:
            print(f"⚠️ Speaker {speaker} not in trees.")

        if speaker in trees:
            overlaps = trees[speaker].overlap(disc_interval.begin, disc_interval.end)
            for match in overlaps:
                gold_interval = Interval(match.begin, match.end)
                phones, word = match.data
                match_found = True

                if check_boundary(gold_interval, disc_interval):
                    if not phones and not silences:
                        continue

                    enriched_fragments.append(
                        (cluster, speaker, disc_interval, phones, word)
                    )

                    break
        if not match_found:
            phones = tuple()
            word = ""
            enriched_fragments.append((cluster, speaker, disc_interval, phones, word))

    return enriched_fragments


def distance(p: Tuple[str, ...], q: Tuple[str, ...]) -> float:
    if p and q:
        length = max(len(p), len(q))
        return editdistance.eval(p, q) / length
    return 1


def ned(
    discovered_transcriptions: Iterable[
        Tuple[int, str, Interval, Tuple[str, ...], str]
    ],
    log_path: Path,
) -> float:
    discovered_transcriptions = sorted(discovered_transcriptions, key=lambda x: x[0])
    overall_distances = []
    lines = []

    for cluster_id, group in itertools.groupby(
        discovered_transcriptions, key=lambda x: x[0]
    ):
        group_list = list(group)
        phones_list = [x[3] for x in group_list]

        lines.append(f"\n{'-' * 60}")
        lines.append(f"🧩 Cluster {cluster_id} | Size: {len(phones_list)}")
        lines.append(f"{'-' * 60}")

        # Collect speakers per token sequence
        token_speakers = {}
        for x in group_list:
            phones = x[3]
            speaker = x[1]
            token_speakers.setdefault(phones, []).append(speaker)

        # Display with speakers and count
        for tokens, speakers in sorted(
            token_speakers.items(), key=lambda item: len(item[1]), reverse=True
        ):
            token_str = " ".join(tokens)
            count = len(speakers)
            lines.append(f"{token_str:<20}  → {count} times")

        if len(group_list) < 2:
            continue

        # Pairwise NED calculations
        cluster_distances = []
        for p, q in itertools.combinations(group_list, 2):
            d = distance(p[3], q[3])
            cluster_distances.append(d)
            overall_distances.append(d)

        avg_cluster_ned = statistics.mean(cluster_distances)

        lines.append(f"→ Avg NED for Cluster {cluster_id}: {avg_cluster_ned:.4f}")
        lines.append(f"{'=' * 60}")

    # Final summary
    if overall_distances:
        overall_avg = statistics.mean(overall_distances)
        lines.append(f"\n🔍 Overall NED across all clusters: {overall_avg:.4f}")
    else:
        overall_avg = 0.0
        lines.append("\n⚠️ No valid clusters with multiple elements found.")

    # Write to file
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("\n".join(lines))
        print(
            f"✅ NED report for [NED = {overall_avg * 100:.3f}%] written to: {log_path}"
        )
    else:
        print("\n".join(lines))

    return overall_avg


def word_purity(
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
            word_list = [x[4] for x in group_list]
            f.write(f"\n{'-' * 60}\n")
            f.write(f"🧩 Cluster {cluster_id} | Size: {len(word_list)}\n")
            f.write(f"{'-' * 60}\n")

            token_counter = Counter(word_list)
            for token, count in token_counter.most_common():
                f.write(f"{token if token != 'nan' else '':<30}  → {count} times\n")

            if len(group_list) < 2:
                continue
            cluster_distances = []
            for p, q in itertools.combinations(group_list, 2):
                p4 = str(p[4])
                q4 = str(q[4])

                if p4 == "nan":
                    p4 = ""
                if q4 == "nan":
                    q4 = ""

                d = distance(p4, q4)
                cluster_distances.append(d)

            if cluster_distances:
                avg_cluster_ned = statistics.mean(cluster_distances)

                f.write(
                    f"→ Avg word purity for Cluster {cluster_id}: {avg_cluster_ned:.4f}\n"
                )
                f.write(f"{'=' * 60}\n")

                overall_distances.extend(cluster_distances)

        overall_avg = statistics.mean(overall_distances)
        f.write(f"\n🔍 Overall WP across all clusters: {overall_avg:.4f}\n")
    print(f"✅ WP report for [WP = {overall_avg * 100:.3f}%] written to: {out_path}")
    return overall_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "model",
        type=str,
        default="hubert_base",
        help="Model name from torchaudio.pipelines",
    )
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument(
        "gamma", type=float, help="Gamma value used for feature extraction."
    )

    parser.add_argument("threshold", type=float, help="Threshold at whic to extract.")

    parser.add_argument(
        "align_dir",
        help="path to the directory of alignments.",
        type=Path,
    )

    args = parser.parse_args()

    file = (
        Path("partitions")
        / args.model
        / f"layer{args.layer}"
        / f"gamma{args.gamma}"
        / f"k{args.n_clusters}"
        / f"{args.model}_l{args.layer}_g{args.gamma}_t{args.threshold}.txt"
    )

    discovered_fragments = []
    with open(file, "r") as f:
        start_time = 0.0
        cluster_id = None
        end_time = 0.0
        for line in f:
            if "Class" in line:
                cluster_id = int(line.split(" ")[1])
            if len(line.split(" ")) == 3:
                speaker, start_time, end_time = line.split(" ")
                discovered_fragments.append(
                    (
                        speaker,
                        Interval(float(start_time), float(end_time)),
                        int(cluster_id),
                    )
                )

                start_time = float(end_time)

    discovered_clusters = [cluster for _, _, cluster in discovered_fragments]

    print(
        f"Number of discovered clusters: {len(set(discovered_clusters))}, Number of discovered 'words': {len(discovered_fragments)}"
    )

    alignment_df = pd.read_csv(args.align_dir / "alignments.csv")
    gold_fragments, total_non_silence = convert_to_intervals(alignment_df)

    trees = build_speaker_trees(gold_fragments)

    discovered_transcriptions = transcribe(discovered_fragments, trees, args.silences)
    print(f"Example transcription: {discovered_transcriptions[0]}")
    print(
        f"Correct number of tokens (non-silence fragments): {'YES' if len(discovered_transcriptions) == total_non_silence else 'NO'} [{total_non_silence}|{len(discovered_transcriptions)}]"
    )

    out_dir = (
        Path("clusters")
        / args.model
        / f"layer{args.layer}"
        / f"gamma{args.gamma}"
        / f"k{args.n_clusters}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ned_value = ned(
        discovered_transcriptions,
        out_dir / f"{args.model}_l{args.layer}_g{args.gamma}_t{args.threshold}_ned.txt",
    )
    word_purity_value = word_purity(
        discovered_transcriptions,
        out_dir / f"{args.model}_l{args.layer}_g{args.gamma}_t{args.threshold}_wp.txt",
    )
