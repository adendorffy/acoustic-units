# Acoustic Units To Build a Lexicon

## 1. Extract `alignments.csv` from .TextGrid files.

Run ``python utils/extract.py $dataset_name $data_dir $alignments_dir $features_dir`` .


Optionally ``--file_extension=.wav`` can be added. Default is `.flac`. 
This extracts all the alignment files for the given dataset into a csv file with `word_id, filename, word_start, word_end, text` for each word.

## 2. `DEMO.ipynb` explains the rest quite well.