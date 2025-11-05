# PageRank Implementation and Evaluation Results

## Overview
This document summarizes the implementation of the PageRank algorithm for the CACM collection and its integration with the vector space model search engine.

## Implementation Details

### 1. PageRank Algorithm
- **File:** `pagerank.py`
- **Method:** Power Iteration
- **Damping Factor:** 0.85
- **Citation Graph:** Built from the .X field in cacm.all
- **Documents:** 3,204 documents
- **Citations:** 28,830 citations
- **Convergence:** 20 iterations
- **Normalization:** Min-max normalization to [0, 1] range

### 2. Top Documents by PageRank
The top 10 documents with highest PageRank scores:
1. Doc 1781: 1.0000 - "Translator Writing systems"
2. Doc 1491: 0.6556 - "EULER: A Generalization ALGOL, and its Formal Definition: Part I*"
3. Doc 3184: 0.6068 - "Revised Report on the Algorithmic Language ALGOL 60"
4. Doc 1787: 0.5524 - "Use of Transition Matrices in Compiling"
5. Doc 1945: 0.5259 - "The Role of Programming in a Ph.D. Computer Science Program"
6. Doc 1265: 0.4926 - "On the Relative Efficiencies of Context-Free Grammar Recognizers"
7. Doc  196: 0.4902 - "Report on the Algorithmic Language ALGOL 60"
8. Doc  680: 0.4785 - "An Error-Correcting Parse Algorithm"
9. Doc 1860: 0.4780 - "An Algol-Based Associative Language"
10. Doc 1496: 0.4704 - "A Formal Semantics for Computer Languages and its Application In a Com"

### 3. Search Integration
Modified `search.py` and `eval.py` to combine cosine similarity with PageRank:
- **Formula:** score(d, q) = w1 * cos-score(d, q) + w2 * pagerank(d)
- **Constraint:** w1 + w2 = 1
- **Implementation:** Added parameters for PageRank scores and weights

## Evaluation Results

### Test Configuration
- **Dataset:** CACM collection (3,204 documents)
- **Queries:** 64 queries
- **Stopwords:** Enabled (default)
- **Stemming:** Enabled (default)
- **Top-K:** 10 documents per query
- **Metrics:** Precision@10, Recall@10, R-Precision, MAP (Mean Average Precision)

### Results Comparison

#### Baseline (No PageRank - Pure Cosine Similarity)
```
Average Precision@10:  0.2344
Average Recall@10:     0.2288
Average R-Precision:   0.2022
Mean Average Precision (MAP): 0.1600
```

#### With PageRank (w1=0.7, w2=0.3)
```
Average Precision@10:  0.1359
Average Recall@10:     0.1599
Average R-Precision:   0.1161
Mean Average Precision (MAP): 0.0952
```

#### With PageRank (w1=0.5, w2=0.5)
```
Average Precision@10:  0.0594
Average Recall@10:     0.0872
Average R-Precision:   0.0375
Mean Average Precision (MAP): 0.0347
```

### Performance Summary Table

| Configuration | P@10  | R@10  | R-Precision | MAP   |
|---------------|-------|-------|-------------|-------|
| Baseline (1.0, 0.0) | 0.2344 | 0.2288 | 0.2022 | 0.1600 |
| PageRank (0.7, 0.3) | 0.1359 | 0.1599 | 0.1161 | 0.0952 |
| PageRank (0.5, 0.5) | 0.0594 | 0.0872 | 0.0375 | 0.0347 |

### Relative Performance (compared to baseline)

| Configuration | P@10 | R@10 | R-Precision | MAP |
|---------------|------|------|-------------|-----|
| PageRank (0.7, 0.3) | -42.0% | -30.1% | -42.6% | -40.5% |
| PageRank (0.5, 0.5) | -74.7% | -61.9% | -81.5% | -78.3% |

## Analysis and Conclusions

### Key Findings

1. **PageRank Hurts Performance**: Adding PageRank to the ranking function consistently decreases all evaluation metrics compared to pure cosine similarity.

2. **Higher Cosine Weight is Better**: When PageRank is used, giving more weight to cosine similarity (0.7 vs 0.5) significantly improves performance:
   - MAP increases from 0.0347 to 0.0952 (174% improvement)
   - Precision@10 increases from 0.0594 to 0.1359 (129% improvement)

3. **Query-Dependent vs Query-Independent Ranking**:
   - Cosine similarity is **query-dependent**: it measures how well a document matches the specific query
   - PageRank is **query-independent**: it measures general document importance based on citations
   - For specific information retrieval tasks, query-dependent ranking is more effective

### Why PageRank Decreases Performance

1. **Citation Bias**: Documents that are highly cited may not be relevant to specific queries. For example, Doc 1781 (Translator Writing systems) has the highest PageRank but may not be relevant to queries about specific algorithms or data structures.

2. **Collection Characteristics**: The CACM collection is relatively small and focused on computer science. Citation patterns may reflect historical importance rather than topical relevance.

3. **Query Specificity**: The test queries are very specific (e.g., "articles by Prieve or Udo Pooch"). PageRank doesn't help find documents matching specific criteria.

4. **Score Mismatch**: Even with normalization, combining PageRank scores with cosine similarity can dilute the relevance signal, especially when query-document similarity is the primary factor for relevance.

### Recommendations

1. **For CACM Collection**: Use pure cosine similarity without PageRank for best performance.

2. **When PageRank Might Help**:
   - Web search where link structure indicates authority
   - Breaking ties between documents with similar cosine scores
   - Broad, navigational queries where popular documents are preferred
   - Collections where citation patterns correlate with relevance

3. **Optimal Weight Selection**: If PageRank must be used, keep w2 (PageRank weight) low (≤ 0.3) to maintain most of the query-dependent signal.

## Running the Code

### Compute PageRank Scores
```bash
python3 pagerank.py --normalize
```

### Run Evaluation with PageRank
```bash
# With weights (0.5, 0.5)
python3 eval.py --use-pagerank --pagerank-weight 0.5

# With weights (0.7, 0.3)
python3 eval.py --use-pagerank --pagerank-weight 0.3

# Baseline (no PageRank)
python3 eval.py
```

### Files Generated
- `output/pagerank_scores.txt` - PageRank scores in text format
- `output/pagerank_scores.pkl` - PageRank scores in pickle format for fast loading
- `output/eval_pagerank_w05_w05.txt` - Evaluation results with (0.5, 0.5) weights
- `output/eval_pagerank_w07_w03.txt` - Evaluation results with (0.7, 0.3) weights

## Technical Notes

### PageRank Formula
```
PR(d) = (1-α)/N + α * Σ(PR(c) / out_count(c))
```
where:
- α = damping factor (0.85)
- N = total number of documents
- c = documents that cite document d
- out_count(c) = number of outgoing citations from document c

### Combined Scoring Formula
```
score(d, q) = w1 * cos_similarity(d, q) + w2 * pagerank(d)
```
where w1 + w2 = 1

### Normalization
PageRank scores are normalized using min-max normalization:
```
normalized_PR(d) = (PR(d) - min(PR)) / (max(PR) - min(PR))
```

This ensures PageRank scores are in [0, 1] range, similar to cosine similarity scores.
