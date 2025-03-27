import argparse
from pathlib import Path

import dataclasses
import re
import itertools
from typing import Iterable, List, Tuple
import statistics

import editdistance
from intervaltree import IntervalTree, Interval
from textgrid import TextGrid, IntervalTier
from collections import Counter


@dataclasses.dataclass(frozen=True)
class Fragment:
    speaker: str
    interval: Interval


@dataclasses.dataclass(frozen=True)
class Transcription:
    intervals: List[Interval]

    @property
    def tokens(self) -> Tuple[str, ...]:
        return tuple(
            interval.data
            for interval in self.intervals
            if interval.data != "sil"
            and interval.data != "sp"
            and interval.data != "spn"
        )

    @property
    def bounds(self) -> Interval:
        return Interval(self.intervals[0].begin, self.intervals[-1].end)


def distance(p: Tuple[str, ...], q: Tuple[str, ...]) -> float:
    length = max(len(p), len(q))
    return editdistance.eval(p, q) / length if length > 0 else 1


def ned(
    discovered: Iterable[Tuple[Fragment, int, Transcription]],
    out_path: Path = Path("ned_output.txt"),
) -> float:
    discovered = sorted(discovered, key=lambda x: x[1])
    overall_distances = []
    with open(out_path, "w") as f:
        all_words = 0
        silences = 0
        for cluster_id, group in itertools.groupby(discovered, key=lambda x: x[1]):
            group_list = list(group)
            all_words += len(group_list)
            tokens_list = [x[2].tokens for x in group_list]
            for x_tokens in tokens_list:
                if len(x_tokens) < 1:
                    silences += 1

            token_counter = Counter(tokens_list)

            # Write cluster summary
            f.write(f"\n{'-' * 60}\n")
            f.write(f"🧩 Cluster {cluster_id} | Size: {len(tokens_list)}\n")
            f.write(f"{'-' * 60}\n")

            for tokens, count in token_counter.most_common():
                tokens = [token for token in tokens if token != ""]
                if len(tokens) < 1:
                    f.write("❗❗ SILENCE\n")
                    continue
                token_str = " ".join(tokens)

                f.write(f"{token_str:<30} → {count} times\n")
            if len(group_list) < 2:
                continue

            cluster_distances = []
            for p, q in itertools.combinations(group_list, 2):
                d = distance(p[2].tokens, q[2].tokens)

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
    f.close()
    print(
        f"All words: {all_words}, Silences: {silences} === Actual words: {all_words - silences}"
    )
    print(f"✅ NED report written to: {out_path}")
    return overall_avg


def tokens(
    gold: Iterable[Fragment],
    disc: Iterable[Fragment],
) -> Tuple[float, float, float]:
    gold_fragments = set(gold)
    disc_fragments = set(disc)
    intersection = gold_fragments & disc_fragments
    precision = len(intersection) / len(disc_fragments)
    recall = len(intersection) / len(gold_fragments)
    fscore = 2 * (precision * recall) / (precision + recall)
    return precision, recall, fscore


def check_boundary(gold: Interval, disc: Interval) -> bool:
    if gold.contains_interval(disc):
        return True

    gold_duration = round(gold.end - gold.begin, 2)
    overlap_duration = round(gold.overlap_size(disc), 2)
    overlap_percentage = overlap_duration / gold_duration
    duration_condition = gold_duration >= 0.06 and overlap_duration >= 0.03
    percentage_condition = gold_duration < 0.06 and overlap_percentage > 0.5
    return duration_condition or percentage_condition


def treeify(grid: TextGrid) -> IntervalTree:
    intervals = [
        (interval.minTime, interval.maxTime, re.sub("\d", "", interval.mark))
        for interval in grid.tiers[1]
    ]
    return IntervalTree.from_tuples(intervals)


def words(grid: TextGrid, tree: IntervalTree) -> List[Transcription]:
    overlaps = [
        tree.overlap(interval.minTime, interval.maxTime)
        for interval in grid.tiers[0]
        if interval.mark != "<eps>"
    ]
    overlaps = [
        sorted(intervals, key=lambda x: x.begin)
        for intervals in overlaps
        if all(interval.data not in ["sp", "spn", "sil"] for interval in intervals)
    ]
    overlaps = [Transcription(intervals) for intervals in overlaps]
    return overlaps


def transcribe(fragment: Fragment, tree: IntervalTree) -> Transcription:
    transcription = sorted(tree.overlap(fragment.interval), key=lambda x: x.begin)
    transcription = [
        interval
        for interval in transcription
        if check_boundary(interval, fragment.interval)
    ]
    return Transcription(transcription)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "disc_path",
        metavar="disc-path",
        help="path to the discovered fragments.",
        type=Path,
    )
    parser.add_argument(
        "gold_dir",
        metavar="gold-dir",
        help="path to the directory of alignments.",
        type=Path,
    )
    parser.add_argument(
        "--alignment_format",
        metavar="--alignment-format",
        help="extension of the alignment files.",
        default=".TextGrid",
        type=str,
    )
    args = parser.parse_args()

    files = args.disc_path.rglob("**/*" + ".list")
    fragments = []
    for file in files:
        with open(file, "r") as f:
            start_time = 0.0
            for line in f:
                if len(line.split(" ")) == 2:  # end_time class
                    end_time, cluster = line.split(" ")
                    speaker = file.stem
                    fragments.append(
                        (
                            speaker,
                            Interval(float(start_time), float(end_time)),
                            int(cluster),
                        )
                    )
                    start_time = float(end_time)

    disc_fragments = [Fragment(speaker, interval) for speaker, interval, _ in fragments]
    disc_clusters = [cluster for _, _, cluster in fragments]

    print("Number of clusters:", len(set(disc_clusters)))

    grids = {}
    files = args.gold_dir.rglob("**/*" + args.alignment_format)
    for file in files:  # alignment files
        if args.alignment_format == ".TextGrid":
            grids[file.stem] = TextGrid.fromFile(file)
        elif args.alignment_format == ".txt":
            with open(file, "r") as f:
                grids[file.stem] = TextGrid()
                interval_tier = IntervalTier(name="phones")
                for line in f:
                    line = line.split()
                    interval_tier.add(float(line[0]), float(line[1]), line[2])
                grids[file.stem].append(interval_tier)

    trees = {speaker: treeify(grid) for speaker, grid in grids.items()}

    disc_transcriptions = [
        transcribe(fragment, trees[fragment.speaker]) for fragment in disc_fragments
    ]

    disc_tokens = [
        " ".join(transcription.tokens) for transcription in disc_transcriptions
    ]

    gold_words = {
        speaker: words(grids[speaker], trees[speaker]) for speaker in grids.keys()
    }
    gold_fragments = [
        Fragment(speaker, word.bounds)
        for speaker, words in gold_words.items()
        for word in words
    ]
    gold_transcriptions = [word for words in gold_words.values() for word in words]

    print(
        "NED",
        ned(
            zip(disc_fragments, disc_clusters, disc_transcriptions),
            args.disc_path / "ned_output.txt",
        ),
    )
