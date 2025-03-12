# Edit-Distance Clusters For Building a Lexicon from Acoustic Units

## 1. Download Data
- Librispeech dev-clean and their alignments (`data/dev-clean` and `data/alignments/dev-clean`)

## 2. Extract alignment info into csv
- Future: Use prom-seg & ES-KMeans for word boundaries

## 3. Encode the features (acoustic units) for each word
- Benji's HuBERT model + DPDP algorithm 
- Need to vary gamma. Currently 0.1 amd 0.2.

## 4. Calculate distances
- Sample `sample_size` amount of words.
- Generate batch of pairs of `chunk_limit` size (default = 5000000).
- For each pair in the batch `(i, j)` calculate the edit distance between `features[i]`, `features[j]`.
- For each batch save the rows, columns and values for that batch as `ouptput/{gamma}/temp/temp_rows_{chunk_idx}.npy` (or cols/vals).


## 5. Clustering
- Get the text for each `word_id` in the dataset by sorting the filepaths and getting the text for the associated index in the `alignments_df`.
- Initialise the graph with a vertex for each sample point in the dataset.
- For each chunk of distances:
    - Read in its `rows`, `cols` and `vals`.
    - Filter the data points for `vals < 0.4` and edges between the corresponding nodes (`i`, `j`) with the corresponding `weight` (val).
- Cluster using `leidenalg` package's `find_partition` with `partition_type=la.CPMVertexPartition` and `resolution_parameter=0.0277` (found to correspond closely with `num_clusters=13967` as needed).

## 6. Evaluation and Visualisation
- Cluster transcriptions can be printed using `transcribe_clusters(partition, texts)` - where `texts` refers to an array with the corresponding text of each `word_id` at that index - and `print_clusters(cluster_transcriptions)`. 
- NED is calculated using edit distance on the whole dataset with `ned(cluster_transcriptions)`.

## Gamma, Resolution, and NED Comparisons

The table below shows the comparison of different `gamma` values, their corresponding best `res` values, and the computed `NED`.

| Gamma | Best Resolution (`res`) | NED Value | Cluster Diff |
|-------|-------------------------|-----------|--------------|
| 0.1   | 0.020                   | 0.084     | 50           |
| 0.2   | 0.022                   | 0.086     | 13           |
| 0.3   | 0.----                  | -----     | --           |
| 0.4   | 0.----                  | 0.---     | --           |
| 0.5   | 0.----                  | 0.---     | --           |
