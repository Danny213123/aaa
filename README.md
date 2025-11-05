# CPS842: Vector Space Model Search with Cosine Similarity

## Overview

This assignment implements a vector space model (VSM) search engine using TF-IDF weighting and cosine similarity for document ranking. The system consists of four main components:

1. **invert.py** - Creates the inverted index from document collection
2. **ui.py** - Interactive GUI-based search interface with detailed result output
3. **eval.py** - Evaluation program for query performance testing
4. **search.py** - Command-line search interface (legacy/debugging)

The actual tokenization, normalization, postings lists, and inverted index has been ported over from the previous assignment, since the inverted index and postings list was done using a BST sorted based on document ID, it was easily ported.

## Running the Programs

The main functionality of the program runs through `search.py`, in this file, all of the calculations are happening, the query taken by the user is tokenized and normalized similarly to how it was done on `test.py`.

### Step 1: Build the Inverted Index

```shell
python invert.py --i cacm/cacm.all --o output/output.txt --stopwords --stemming --stopwords-file cacm/common_words
```

**Arguments:**
```shell
options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to cacm.all file
  --output OUTPUT, -o OUTPUT
                        Path to output file
  --stopwords           Remove stopwords
  --stopwords-file STOPWORDS_FILE
                        Path to stopwords file
  --stemming            Enable Porter stemming
```

### Step 2: Running UI Interface

This is the main UI interface for search.py, it is best to run the main program through here because you will get a CLI.

```shell
python ui.py -i output/index.pkl.gz -p output/postings.pkl.gz -d output/documents.pkl.gz --stopwords --stemming
```

**Arguments:**
```shell
options:
  -h, --help            show this help message and exit
  --index INDEX, -i INDEX
                        Path to index pickle file (index.pkl.gz)
  --postings POSTINGS, -p POSTINGS
                        Path to postings pickle file (postings.pkl.gz)
  --documents DOCUMENTS, -d DOCUMENTS
                        Path to documents pickle file (documents.pkl.gz)
  --stopwords           Remove stopwords from query
  --stopwords-file STOPWORDS_FILE
                        Path to stopwords file
  --stemming            Enable Porter stemming for query
  --top-k TOP_K, -k TOP_K
                        Number of top results to return (default: 10)
```

### Step 3: Evaluation

The `eval.py` program will go through each query within `query.text` and compare the results to `qrels.text` with results from `search.py`, it will then compare the retrieved documents with the relevant documents, to calculate a precision and recall score. The r-precision is calculated by looking at the total number of relevant documents / the number of top-k retrieved documents.

Recall is the total number of relevant documents found in the retrieved divided by the total number of relevant documents
Precision@k is the total number of relevant documents found within the top-k of retrieved documents
R-precision is the total precision at a certain k
MAP is the average precision across all queries
Average Precision (AP) is the average precision at each recall step

```shell
python eval.py
```

**Arguments:**
```shell
options:
  -h, --help            show this help message and exit
  --queries QUERIES     Path to query.text file
  --qrels QRELS         Path to qrels.text file
  --postings POSTINGS   Path to postings pickle file
  --documents DOCUMENTS
                        Path to documents pickle file
  --stopwords           Remove stopwords from queries
  --stemming            Enable Porter stemming
  --top-k TOP_K         Number of results to retrieve per query
  --query-id QUERY_ID   Run only a specific query ID
  --use-static-quality  Use static quality score combined with cosine similarity
  --quality-weight QUALITY_WEIGHT
                        Weight for static quality score (0.0 to 1.0), default 0.3
```

## Algorithms

The main algorithm used within this program is the standard calculation of the Cosine similarity between the query and document vectors with length normalization for both query and document vectors. First the `idf` dictionary is created, this is buy looping through the index, which can be found within `index.txt`, and using the formula `log(N/df)`, where `N` is the total number of documents, the `df` is the document frequency of a term. Once that is found, the tf-vectors are created by looping through each document and creating a tf-vector for each term-vector using `1+log(f)`, where `f` is the frequency of a term within a document. After that the weight-vectors are created by taking the tf-vectors for each document, and multiplying by the idf-vector, same goes for the query-vector. The cosine similarity is calculated by using the formula `sim(dj, q) = sum(wiq*wij) / sqrt(sum(wij^2)) * sqrt(sum(wiq^2))`.

A detailed table of all calculations, exactly how they are displayed in lec2 can be found in the `query_results.txt` file within the output/ directory.

## Additional Stuff

### query / document term highlighting

The return query will have terms that are in both the query and document highlighted

```shell
Context: A mechanical procedure is derived for determining whether a given context-free phrase structure **grammar** is a simple precedence grammar. This procedure consists of elementary operations on suitably defined Boolean matrices. Application of the procedure to operator grammars is...
```

### Top-K

The Top-K method used in this assignment is just a threshold filter, the search.py will return the top results that have a cosine similarity score above the one defined by the user

## Examples

### Step 1
```shell
python invert.py --i cacm/cacm.all --o output/output.txt --stopwords --stemming --stopwords-file cacm/common_words
```

### Step 2
```shell
python ui.py -i output/index.pkl.gz -p output/postings.pkl.gz -d output/documents.pkl.gz --stopwords --stemming
```

### Step 4
```shell
python eval.py --stopwords --stemming
```

### directly prompting the search.py file
```shell
python search.py -i output/index.pkl.gz -p output/postings.pkl.gz -d output/documents.pkl.gz -q "your query" -t 0.15 --stopwords --stemming
```
