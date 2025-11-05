#!/usr/bin/env python3
"""
PageRank implementation for CACM collection
Uses citation information from .X field to compute PageRank scores
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Set


def parse_cacm_citations(cacm_file: Path) -> tuple[Dict[int, Set[int]], Set[int]]:
    """
    Parse citation information from CACM collection
    The .X field contains citations in the format:
    cited_doc_id field_type citing_doc_id

    :param cacm_file: Path to cacm.all file
    :return: Tuple of (citation graph, set of all document IDs)
    """
    citations = {}  # doc_id -> set of docs it cites
    all_docs = set()
    current_doc_id = None

    with open(cacm_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # Document ID line
            if line.startswith('.I '):
                doc_id = int(line.split()[1])
                current_doc_id = doc_id
                all_docs.add(doc_id)
                if doc_id not in citations:
                    citations[doc_id] = set()

            # Citation line (under .X field)
            elif line and not line.startswith('.') and current_doc_id is not None:
                # Try to parse citation data
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        cited_doc_id = int(parts[0])
                        citing_doc_id = int(parts[2])

                        # Verify this is the current document
                        if citing_doc_id == current_doc_id:
                            citations[current_doc_id].add(cited_doc_id)
                            all_docs.add(cited_doc_id)
                    except ValueError:
                        # Not a citation line, skip
                        pass

    return citations, all_docs


def compute_pagerank(
    citations: Dict[int, Set[int]],
    all_docs: Set[int],
    damping_factor: float = 0.85,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-6
) -> Dict[int, float]:
    """
    Compute PageRank scores using Power Iteration method

    :param citations: Dictionary mapping doc_id to set of docs it cites
    :param all_docs: Set of all document IDs
    :param damping_factor: Probability of following a link (default 0.85)
    :param max_iterations: Maximum number of iterations (default 100)
    :param convergence_threshold: Convergence threshold (default 1e-6)
    :return: Dictionary mapping document IDs to PageRank scores
    """
    N = len(all_docs)

    # Initialize PageRank scores uniformly
    pagerank = {doc_id: 1.0 / N for doc_id in all_docs}

    # Build reverse citation graph (who cites this document)
    incoming_citations = {doc_id: set() for doc_id in all_docs}
    for citing_doc, cited_docs in citations.items():
        for cited_doc in cited_docs:
            if cited_doc in all_docs:
                incoming_citations[cited_doc].add(citing_doc)

    # Count outgoing links for each document
    outgoing_counts = {doc_id: len(citations.get(doc_id, set())) for doc_id in all_docs}

    # Power Iteration
    for iteration in range(max_iterations):
        new_pagerank = {}
        max_change = 0.0

        for doc_id in all_docs:
            # Random surfer component
            rank = (1 - damping_factor) / N

            # Citation component
            for citing_doc in incoming_citations[doc_id]:
                out_count = outgoing_counts[citing_doc]
                if out_count > 0:
                    rank += damping_factor * (pagerank[citing_doc] / out_count)

            new_pagerank[doc_id] = rank
            max_change = max(max_change, abs(new_pagerank[doc_id] - pagerank[doc_id]))

        pagerank = new_pagerank

        # Check convergence
        if max_change < convergence_threshold:
            print(f"Converged after {iteration + 1} iterations")
            break
    else:
        print(f"Reached maximum iterations ({max_iterations})")

    return pagerank


def normalize_pagerank_scores(pagerank: Dict[int, float]) -> Dict[int, float]:
    """
    Normalize PageRank scores to [0, 1] range using min-max normalization

    :param pagerank: Dictionary of PageRank scores
    :return: Dictionary of normalized PageRank scores
    """
    if not pagerank:
        return {}

    min_score = min(pagerank.values())
    max_score = max(pagerank.values())

    # If all scores are the same, return uniform distribution
    if max_score == min_score:
        return {doc_id: 0.5 for doc_id in pagerank}

    normalized = {
        doc_id: (score - min_score) / (max_score - min_score)
        for doc_id, score in pagerank.items()
    }

    return normalized


def save_pagerank_scores(pagerank: Dict[int, float], output_file: Path):
    """
    Save PageRank scores to a file

    :param pagerank: Dictionary of PageRank scores
    :param output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id in sorted(pagerank.keys()):
            f.write(f"{doc_id}\t{pagerank[doc_id]:.10f}\n")


def save_pagerank_pickle(pagerank: Dict[int, float], output_file: Path):
    """
    Save PageRank scores to a pickle file for easy loading

    :param pagerank: Dictionary of PageRank scores
    :param output_file: Path to output pickle file
    """
    with open(output_file, 'wb') as f:
        pickle.dump(pagerank, f)


def main():
    parser = argparse.ArgumentParser(
        description='Compute PageRank scores for CACM collection'
    )
    parser.add_argument(
        '--cacm',
        type=Path,
        default=Path('cacm/cacm.all'),
        help='Path to cacm.all file (default: cacm/cacm.all)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('output/pagerank_scores.txt'),
        help='Path to output file (default: output/pagerank_scores.txt)'
    )
    parser.add_argument(
        '--output-pickle',
        type=Path,
        default=Path('output/pagerank_scores.pkl'),
        help='Path to output pickle file (default: output/pagerank_scores.pkl)'
    )
    parser.add_argument(
        '--damping',
        type=float,
        default=0.85,
        help='Damping factor (default: 0.85)'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=100,
        help='Maximum number of iterations (default: 100)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize PageRank scores to [0, 1] range'
    )

    args = parser.parse_args()

    # Parse citations
    print(f"Parsing citations from {args.cacm}...")
    citations, all_docs = parse_cacm_citations(args.cacm)

    print(f"Found {len(all_docs)} documents")
    print(f"Found {sum(len(cited) for cited in citations.values())} citations")

    # Compute PageRank
    print(f"\nComputing PageRank with damping factor {args.damping}...")
    pagerank = compute_pagerank(
        citations,
        all_docs,
        damping_factor=args.damping,
        max_iterations=args.max_iterations
    )

    # Normalize if requested
    if args.normalize:
        print("\nNormalizing PageRank scores to [0, 1] range...")
        pagerank = normalize_pagerank_scores(pagerank)

    # Print statistics
    print(f"\nPageRank Statistics:")
    print(f"  Min score: {min(pagerank.values()):.10f}")
    print(f"  Max score: {max(pagerank.values()):.10f}")
    print(f"  Mean score: {sum(pagerank.values()) / len(pagerank):.10f}")

    # Print top 10 documents by PageRank
    print(f"\nTop 10 documents by PageRank:")
    sorted_docs = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    for rank, (doc_id, score) in enumerate(sorted_docs[:10], 1):
        print(f"  {rank}. Doc {doc_id}: {score:.10f}")

    # Save results
    print(f"\nSaving PageRank scores to {args.output}...")
    save_pagerank_scores(pagerank, args.output)

    print(f"Saving PageRank scores to {args.output_pickle}...")
    save_pagerank_pickle(pagerank, args.output_pickle)

    print("\nDone!")


if __name__ == '__main__':
    main()
