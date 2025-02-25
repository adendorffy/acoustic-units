from pathlib import Path
from features import DataSet
import textgrids
import  pandas as pd
import argparse
from tqdm import tqdm

def extract_alignments(dataset):
    in_paths = list(dataset.align_dir.rglob("**/*.TextGrid"))
    out_path = dataset.align_dir/ "alignments.csv"
    
    alignments_df = pd.DataFrame(columns=['word_id', 'filename', 'word_start', 'word_end', 'text'])

    for in_path in tqdm(in_paths, desc="Extracting Alignments"):
        grid = textgrids.TextGrid(in_path)
        filename = in_path.stem

        words_tier = grid['words']
        for idx, interval in enumerate(words_tier, start=1):
            new_row = pd.DataFrame([[idx-1, filename, interval.xmin, interval.xmax, interval.text]], columns=alignments_df.columns)
            alignments_df = pd.concat([alignments_df, new_row], ignore_index=True)
        
    alignments_df.to_csv(out_path, index=False)
    print(f"Wrote alignments to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Process dataset configurations for alignment extraction.")
    
    parser.add_argument("name", type=str, default="librispeech-dev-clean", help="Name of the dataset.")
    parser.add_argument("in_dir", type=Path, default=Path("data/dev-clean"), help="Path to the dataset directory.")
    parser.add_argument("align_dir", type=Path, default=Path("data/alignments/dev-clean"), help="Path to the alignments directory.")
    parser.add_argument("feat_dir", type=Path, default=Path("features"), help="Path to store extracted features.")
    parser.add_argument("--file_extension", type=str, default=".flac", help="File extension for audio files.")

    args = parser.parse_args()

    dataset = DataSet(
        name=args.name,
        in_dir=args.in_dir,
        align_dir=args.align_dir,
        feat_dir=args.feat_dir,
        audio_ext=args.file_extension
    )

    extract_alignments(dataset)

if __name__ == "__main__":
    main()