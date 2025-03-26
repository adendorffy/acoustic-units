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
            and interval.data != "spn"
            and interval.data != "sp"
        )

    @property
    def bounds(self) -> Interval:
        return Interval(self.intervals[0].begin, self.intervals[-1].end)


def distance(p: Tuple[str, ...], q: Tuple[str, ...]) -> float:
    length = max(len(p), len(q))
    return editdistance.eval(p, q) / length if length > 0 else 1


def ned(discovered: Iterable[Tuple[Fragment, int, Transcription]]) -> float:
    discovered = sorted(discovered, key=lambda x: x[1])
    distances = [
        distance(p[2].tokens, q[2].tokens)
        for _, group in itertools.groupby(discovered, key=lambda x: x[1])
        for p, q in itertools.combinations(group, 2)
    ]
    return statistics.mean(distances)


def coverage(
    disc: Iterable[Tuple[Fragment, Transcription]],
    gold: Iterable[Transcription],
):
    covered = {
        (fragment.speaker, interval.begin, interval.end, interval.data)
        for fragment, transcription in disc
        for interval in transcription.intervals
        if interval.data != "sil" and interval.data != "spn" and interval.data != "sp"
    }
    total = [
        interval.data
        for transcription in gold
        for interval in transcription.intervals
        if interval.data != "sil" and interval.data != "spn" and interval.data != "sp"
    ]
    return len(covered) / len(total)


def types(
    gold: Iterable[Transcription],
    disc: Iterable[Transcription],
) -> Tuple[float, float, float]:
    gold_types = {transcription.tokens for transcription in gold}
    disc_types = {transcription.tokens for transcription in disc}
    intersection = gold_types & disc_types
    precision = len(intersection) / len(disc_types)
    recall = len(intersection) / len(gold_types)
    fscore = 2 * (precision * recall) / (precision + recall)
    return precision, recall, fscore


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
        if all(interval.data not in ["sp", "spn"] for interval in intervals)
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

    print("NED", ned(zip(disc_fragments, disc_clusters, disc_transcriptions)))
    print(
        "Coverage",
        coverage(zip(disc_fragments, disc_transcriptions), gold_transcriptions),
    )
    print("Types", types(gold_transcriptions, disc_transcriptions))
