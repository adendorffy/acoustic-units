from pathlib import Path
from utils.features import DataSet
import textgrids
import pandas as pd
import argparse
from tqdm import tqdm


def extract_alignments(dataset):
    in_paths = list(dataset.align_dir.rglob("**/*.TextGrid"))
    out_path = dataset.align_dir / "alignments.csv"

    alignments_df = pd.DataFrame(
        columns=["word_id", "filename", "word_start", "word_end", "text"]
    )

    for in_path in tqdm(in_paths, desc="Extracting Alignments"):
        grid = textgrids.TextGrid(in_path)
        filename = in_path.stem

        words_tier = grid["words"]
        for idx, interval in enumerate(words_tier, start=1):
            new_row = pd.DataFrame(
                [[idx - 1, filename, interval.xmin, interval.xmax, interval.text]],
                columns=alignments_df.columns,
            )
            alignments_df = pd.concat([alignments_df, new_row], ignore_index=True)

    alignments_df.to_csv(out_path, index=False)
    print(f"Wrote alignments to {out_path}")


def pair_generator(num_paths):
    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            yield i, j


def get_batch_of_paths(num_paths, chunk_limit=100):
    """Generate sequential batches of (i, j) path pairs."""
    pairs = pair_generator(num_paths)  # Generate all possible pairs
    chunk = []  # Single list for sequential processing

    for idx, (i, j) in enumerate(pairs, 1):
        chunk.append((i, j))

        if idx % chunk_limit == 0:
            yield chunk  # Yield the current batch
            chunk = []  # Reset chunk

    if chunk:  # Yield any remaining pairs
        yield chunk


def fill_chunck(dist_mat, chunk):
    chunk_size = len(chunk)
    for i in range(chunk_size):
        dist_mat[chunk[i][0], chunk[i][1]] = 1

    return dist_mat


def main():
    parser = argparse.ArgumentParser(
        description="Process dataset configurations for alignment extraction."
    )

    parser.add_argument(
        "name", type=str, default="librispeech-dev-clean", help="Name of the dataset."
    )
    parser.add_argument(
        "in_dir",
        type=Path,
        default=Path("data/dev-clean"),
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "align_dir",
        type=Path,
        default=Path("data/alignments/dev-clean"),
        help="Path to the alignments directory.",
    )
    parser.add_argument(
        "feat_dir",
        type=Path,
        default=Path("features"),
        help="Path to store extracted features.",
    )
    parser.add_argument(
        "--file_extension",
        type=str,
        default=".flac",
        help="File extension for audio files.",
    )

    args = parser.parse_args()

    dataset = DataSet(
        name=args.name,
        in_dir=args.in_dir,
        align_dir=args.align_dir,
        feat_dir=args.feat_dir,
        audio_ext=args.file_extension,
    )

    extract_alignments(dataset)


if __name__ == "__main__":
    main()
