#!/usr/bin/env python3
"""
Evaluate search results against qrels.text relevance judgments
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set

from search import (
    calculate_idf_dict,
    calculate_static_quality_scores,
    calculate_tf_vectors,
    calculate_weight_vectors,
    load_documents,
    load_postings,
    load_stopwords,
    search_with_weight_vectors,
)


def read_queries(query_file: Path) -> Dict[int, str]:
    """
    Read queries from query.text file
    :param query_file: Path to query.text file
    :return: Dictionary mapping query ID to query text
    """
    queries = {}
    current_id = None
    query_lines = []
    in_query = False

    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith(".I"):
                # Save previous query if exists
                if current_id is not None and query_lines:
                    queries[current_id] = " ".join(query_lines).strip()
                    query_lines = []

                # Start new query
                parts = line.split()
                current_id = int(parts[1])
                in_query = False

            elif line.startswith(".W"):
                in_query = True

            elif line.startswith(".A") or line.startswith(".N"):
                in_query = False

            elif in_query and line.strip():
                query_lines.append(line.strip())

        # Save last query
        if current_id is not None and query_lines:
            queries[current_id] = " ".join(query_lines).strip()

    return queries


def read_qrels(qrels_file: Path) -> Dict[int, Set[int]]:
    """
    Read relevance judgments from qrels.text file
    :param qrels_file: Path to qrels.text file
    :return: Dictionary mapping query ID to set of relevant document IDs
    """
    qrels = {}

    with open(qrels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                query_id = int(parts[0])
                doc_id = int(parts[1])

                if query_id not in qrels:
                    qrels[query_id] = set()
                qrels[query_id].add(doc_id)

    return qrels


def calculate_precision_recall(retrieved: List[int], relevant: Set[int]) -> tuple:
    """
    Calculate precision and recall
    :param retrieved: List of retrieved document IDs
    :param relevant: Set of relevant document IDs
    :return: Tuple of (precision, recall, retrieved_relevant_count)
    """
    if not retrieved:
        return 0.0, 0.0, 0

    retrieved_set = set(retrieved)
    retrieved_relevant = retrieved_set & relevant

    precision = len(retrieved_relevant) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(retrieved_relevant) / len(relevant) if relevant else 0.0

    return precision, recall, len(retrieved_relevant)


def calculate_average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    """
    Calculate Average Precision (AP) for a single query
    :param retrieved: List of retrieved document IDs (in ranked order)
    :param relevant: Set of relevant document IDs
    :return: Average Precision score
    """
    if not relevant or not retrieved:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            relevant_count += 1
            precision_at_i = relevant_count / i
            precision_sum += precision_at_i

    if relevant_count == 0:
        return 0.0

    return precision_sum / len(relevant)


def calculate_r_precision(retrieved: List[int], relevant: Set[int]) -> float:
    """
    Calculate R-Precision for a single query

    Suppose R is the set of known relevant documents, in top |R| results,
    r of them are relevant, r/|R| is called R-precision

    :param retrieved: List of retrieved document IDs (in ranked order)
    :param relevant: Set of relevant document IDs (R)
    :return: R-Precision score = r / |R|
    """
    if not relevant or not retrieved:
        return 0.0

    # |R| is the number of relevant documents
    R = len(relevant)

    # Take the top |R| retrieved documents
    top_R_retrieved = retrieved[:R]

    # Count how many of the top |R| are relevant (r)
    r = len(set(top_R_retrieved) & relevant)

    # R-Precision = r / |R|
    return r / R


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate search results against qrels.text"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("cacm/query.text"),
        help="Path to query.text file",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("cacm/qrels.text"),
        help="Path to qrels.text file",
    )
    parser.add_argument(
        "--postings",
        type=Path,
        default=Path("output/postings.pkl.gz"),
        help="Path to postings pickle file",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path("output/documents.pkl.gz"),
        help="Path to documents pickle file",
    )
    parser.add_argument(
        "--stopwords",
        action="store_true",
        default=True,
        help="Remove stopwords from queries",
    )
    parser.add_argument(
        "--stemming", action="store_true", default=True, help="Enable Porter stemming"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results to retrieve per query"
    )
    parser.add_argument("--query-id", type=int, help="Run only a specific query ID")
    parser.add_argument(
        "--use-static-quality",
        action="store_true",
        default=False,
        help="Use static quality score combined with cosine similarity",
    )
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=0.3,
        help="Weight for static quality score (0.0 to 1.0), default 0.3",
    )

    args = parser.parse_args()

    # Load data
    print("Loading index, postings, and documents...")
    terms_dict = load_postings(args.postings)
    document_dict = load_documents(args.documents)
    print(f"Loaded {len(terms_dict)} terms and {len(document_dict)} documents.\n")

    # Calculate IDF and weight vectors
    print("Calculating IDF values and weight vectors...")
    idf_dict = calculate_idf_dict(terms_dict, len(document_dict))
    tf_vectors = calculate_tf_vectors(document_dict, terms_dict)
    weight_vectors = calculate_weight_vectors(document_dict, terms_dict, idf_dict)
    print("Weight vectors calculated.\n")

    # Calculate static quality scores if enabled
    if args.use_static_quality:
        print("Calculating static quality scores...")
        quality_scores = calculate_static_quality_scores(document_dict)
        print(f"Static quality scores calculated (weight: {args.quality_weight}).\n")

    # Load stopwords
    stopwords_file = Path("cacm/common_words")
    stopwords = (
        load_stopwords(stopwords_file)
        if args.stopwords and stopwords_file.exists()
        else None
    )

    # Read queries and qrels
    print("Reading queries and relevance judgments...")
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    print(
        f"Loaded {len(queries)} queries and relevance judgments for {len(qrels)} queries.\n"
    )

    # Run evaluation
    results_summary = []

    query_ids = [args.query_id] if args.query_id else sorted(queries.keys())

    for query_id in query_ids:
        if query_id not in queries:
            print(f"Warning: Query {query_id} not found in query file")
            continue

        query_text = queries[query_id]
        relevant_docs = qrels.get(query_id, set())

        print("=" * 80)
        print(f"Query {query_id:02d}: {query_text}")
        print(f"Relevant documents: {sorted(relevant_docs)}")
        print("-" * 80)

        # Search
        results, query_stats, query_terms = search_with_weight_vectors(
            query_text,
            terms_dict,
            document_dict,
            stopwords_enabled=args.stopwords,
            stopwords=stopwords,
            stemming_enabled=args.stemming,
            top_k=args.top_k,
            detailed=False,
            use_static_quality=args.use_static_quality,
            quality_weight=args.quality_weight,
        )

        # Extract retrieved document IDs
        retrieved_docs = [doc_id for doc_id, score, common_terms in results]

        # Calculate metrics
        precision, recall, relevant_retrieved = calculate_precision_recall(
            retrieved_docs, relevant_docs
        )
        avg_precision = calculate_average_precision(retrieved_docs, relevant_docs)
        r_precision = calculate_r_precision(retrieved_docs, relevant_docs)

        print(f"Retrieved: {retrieved_docs}")
        print(f"Relevant retrieved: {relevant_retrieved}/{len(relevant_docs)}")
        print(f"Precision@{args.top_k}: {precision:.4f}")
        print(f"Recall@{args.top_k}: {recall:.4f}")
        print(f"R-Precision: {r_precision:.4f}")
        print(f"Average Precision (AP): {avg_precision:.4f}")
        print()

        # Show top results with scores
        if results:
            print(f"Top {min(10, len(results))} results:")
            for rank, (doc_id, score, common_terms) in enumerate(results[:10], 1):
                doc = document_dict[doc_id]
                is_relevant = "[RELEVANT]" if doc_id in relevant_docs else "[NOT RELEVANT]"
                print(f"  {rank}. Doc {doc_id:4d} | Score: {score:.4f} | {is_relevant}")
                print(f"      {doc.title[:70]}")
        print()

        results_summary.append(
            {
                "query_id": query_id,
                "precision": precision,
                "recall": recall,
                "avg_precision": avg_precision,
                "r_precision": r_precision,
                "relevant_retrieved": relevant_retrieved,
                "total_relevant": len(relevant_docs),
                "total_retrieved": len(retrieved_docs),
            }
        )

    # Print summary statistics
    if len(results_summary) > 1:
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        avg_precision = sum(r["precision"] for r in results_summary) / len(
            results_summary
        )
        avg_recall = sum(r["recall"] for r in results_summary) / len(results_summary)
        map_score = sum(r["avg_precision"] for r in results_summary) / len(
            results_summary
        )
        avg_r_precision = sum(r["r_precision"] for r in results_summary) / len(
            results_summary
        )

        print(f"Queries evaluated: {len(results_summary)}")
        print(f"Average Precision@{args.top_k}: {avg_precision:.4f}")
        print(f"Average Recall@{args.top_k}: {avg_recall:.4f}")
        print(f"Average R-Precision: {avg_r_precision:.4f}")
        print(f"Mean Average Precision (MAP): {map_score:.4f}")
        print()

if __name__ == "__main__":
    main()
