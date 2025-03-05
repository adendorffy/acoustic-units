# Edit-Distance Clusters For Building a Lexicon from Acoustic Units

## 1. Download Data
- Librispeech dev-clean and their alignments (`data/dev-clean` and `data/alignments/dev-clean`)

## 2. Extract alignment info into csv
- Future: Use prom-seg & ES-KMeans for word boundaries

## 3. Encode the features (acoustic units) for each word
- Benji's HuBERT model + DPDP algorithm 
- Need to vary gamma. At the moment only 0.2 

## 4. Calculate distance matrix
- Sample `sample_size` amount of words.
- Generate batch of pairs of `chunk_limit` size.
- For each batch of pairs:
    - Load the acoustic units for the pairs into a dict of `(i,j) : (feature_i, feature_j)`.
    - Calculate the distances between each of the pairs in the batch and store them in the distance matrix of shape `(sample_size, sample_size)`.
    - Save the filenames and word indices to a csv so that the words associated with the distances can be loaded in for evaluation. 
    - Save a compressed distance matrix.

## 5. Clustering
- Load in the distance matrix and associated words from `info.csv`.
- Cluster usiing graph-clustering or kmeans-clustering.
- Covert int clusters to true word clusters. 

## 6. Evaluation and Visualisation
- Plot the pairwise edit distance matrix to visualise clusters quickly.
- Calculate ned across clusters and for each cluster.
- Print the pure/inpure clusters.