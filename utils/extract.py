from pathlib import Path
from features import DataSet, WordUnit
import textgrids
import  pandas as pd

def extract_alignments(dataset):
    in_paths = list(dataset.align_dir.rglob("**/*.TextGrid"))
    out_path = dataset.align_dir/ "alignments.csv"
    
    alignments_df = pd.DataFrame(columns=['word_id', 'filename', 'word_start', 'word_end', 'text'])

    for in_path in in_paths:
        grid = textgrids.TextGrid(in_path)
        filename = in_path.stem

        words_tier = grid['words']
        for idx, interval in enumerate(words_tier, start=1):
            new_row = pd.DataFrame([[idx-1, filename, interval.xmin, interval.xmax, interval.text]], columns=alignments_df.columns)
            alignments_df = pd.concat([alignments_df, new_row], ignore_index=True)
        
    alignments_df.to_csv(out_path, index=False)
    print(f"Wrote alignments to {out_path}")




current_dir = Path.cwd()

dataset = DataSet(
    "librispeech-dev-clean",
    Path("data/dev-clean"),
    Path("data/alignments/dev-clean"),
    Path("features"), 
    "wavlm_base",
    7,
    ".flac" 
)

extract_alignments(dataset)