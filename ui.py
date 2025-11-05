import argparse
import math
import os
import sys
from pathlib import Path

from search import *


def write_results_to_file(
    query,
    query_vector,
    document_dict,
    doc_vectors,
    doc_lengths,
    terms_dict,
    query_terms,
    top_k=10,
    output_file="output/query_results.txt",
):
    """
    Write top-k results to file
    :param query: The original query string
    :param query_vector: weight vector for query
    :param document_dict: Dictionary of Document objects
    :param doc_vectors: Pre-computed document TF-IDF vectors
    :param doc_lengths: vector length normalization
    :param terms_dict: term dictionary
    :param query_terms: List of processed query terms
    :param top_k: Number of top results to write to file
    :param output_file: Path to the output file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Calculate query vector length (magnitude) using query terms
    query_length = math.sqrt(sum(w**2 for w in query_vector.values()))

    all_scores = []

    for doc_id in document_dict.keys():
        if doc_id not in doc_vectors:
            continue

        doc_vector = doc_vectors[doc_id]

        # Calculate document vector length
        # sqrt(w1q^2 + w2q^2 + ... + wiq^2)
        doc_length_query_terms = math.sqrt(
            sum(doc_vector.get(term, 0) ** 2 for term in query_vector.keys())
        )

        # Calculate cosine similarity with per-term normalization
        # sim(d,q) = E[(wiq/|q|) * (wij/|d|)] for each term
        if query_length == 0 or doc_length_query_terms == 0:
            score = 0.0
        else:
            # Calculate normalized dot product
            score = 0.0
            for term, q_weight in query_vector.items():
                if term in doc_vector:
                    d_weight = doc_vector[term]
                    # Normalized product for this term: (wiq/|q|) * (wij/|d|)
                    score += (q_weight / query_length) * (
                        d_weight / doc_length_query_terms
                    )

        all_scores.append((doc_id, score, doc_length_query_terms))

    # Sort by score in descending order
    all_scores.sort(key=lambda x: x[1], reverse=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"QUERY: {query}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Query Weight Vector (TF-IDF):\n")
        f.write("-" * 80 + "\n")
        for term, weight in sorted(query_vector.items()):
            f.write(f"  '{term}': {weight:.6f}\n")
        f.write(f"\nQuery Vector Length |q|: {query_length:.6f}\n")
        f.write(
            f"Calculation: |q| = E({' + '.join([f'{w:.6f}²' for w in [query_vector[term] for term in sorted(query_vector.keys())]])})\n"
        )
        f.write("-" * 80 + "\n\n")

        if not all_scores:
            f.write("No documents found.\n\n")
            return

        # Limit to top-k results
        top_scores = all_scores[:top_k]

        f.write(
            f"Cosine Similarity for TOP {len(top_scores)} documents (out of {len(all_scores)} total):\n"
        )
        f.write(f"Using formula: sim(d,q) = E[(wij * wiq) / (|d| * |q|)]\n")
        f.write(f"where |d| and |q| are calculated using ONLY query terms\n\n")

        for rank, (doc_id, score, doc_length_query_terms) in enumerate(top_scores, 1):
            doc = document_dict[doc_id]
            authors = ", ".join(doc.authors) if doc.authors else "Unknown"

            f.write(f"{rank}. Document ID: {doc_id}\n")
            f.write(f"   Title: {doc.title}\n")
            f.write(f"   Authors: {authors}\n")

            # Write query and document weight vectors side by side
            f.write(f"   Weight Vectors (TF-IDF for query terms):\n")
            if doc_id in doc_vectors:
                doc_vector = doc_vectors[doc_id]

                # Header
                f.write(
                    f"   {'Term':<12} {'Q-f':<6} {'Q-tf':<10} {'D-f':<6} {'D-tf':<10} {'IDF':<10} {'Q-wiq':<10} {'nw-Q':<10} {'D-wij':<10} {'nw-D':<10} {'Product':<10}\n"
                )
                f.write(
                    f"   {'-'*12} {'-'*6} {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}\n"
                )

                # Calculate dot product with detailed breakdown
                dot_product = 0.0
                dot_product_terms = []
                for term in sorted(query_vector.keys()):
                    q_weight = query_vector.get(term, 0.0)
                    d_weight = doc_vector.get(term, 0.0)

                    # Get raw term frequencies (f)
                    q_freq = query_terms.count(term)
                    d_freq = 0
                    if term in terms_dict and doc_id in terms_dict[term].postings:
                        d_freq = terms_dict[term].postings[doc_id].tf

                    # Calculate TF weights: 1 + log10(f)
                    q_tf = calculate_tf(q_freq)
                    d_tf = calculate_tf(d_freq)

                    # Get IDF value
                    idf = 0.0
                    if term in terms_dict:
                        doc_freq = terms_dict[term].postings.size
                        idf = calculate_idf(doc_freq, len(document_dict))

                    # Calculate normalized weights: (wiq / |q|) and (wij / |d|)
                    normalized_q = q_weight / query_length if query_length > 0 else 0.0
                    normalized_d = (
                        d_weight / doc_length_query_terms
                        if doc_length_query_terms > 0
                        else 0.0
                    )
                    product = normalized_q * normalized_d
                    dot_product += product
                    dot_product_terms.append(
                        f"({q_weight:.6f}/{query_length:.6f})*({d_weight:.6f}/{doc_length_query_terms:.6f})"
                    )
                    f.write(
                        f"   {term:<12} {q_freq:<6} {q_tf:<10.6f} {d_freq:<6} {d_tf:<10.6f} {idf:<10.6f} {q_weight:<10.6f} {normalized_q:<10.6f} {d_weight:<10.6f} {normalized_d:<10.6f} {product:<10.6f}\n"
                    )

                f.write(
                    f"\n   Document Vector Length |d| (query terms only): {doc_length_query_terms:.6f}\n"
                )
                f.write(
                    f"   Calculation: |d| = sqrt({' + '.join([f'{doc_vector.get(term, 0.0):.6f}^2' for term in sorted(query_vector.keys())])})\n"
                )

                f.write(f"\n   Normalized Dot Product: {dot_product:.6f}\n")
                f.write(
                    f"   Calculation: SUM[(wiq/|q|) * (wij/|d|)] = {' + '.join(dot_product_terms)} = {dot_product:.6f}\n"
                )

                f.write(f"\n   Cosine Similarity: {dot_product:.6f}\n")
                f.write(
                    f"   Note: The similarity is already normalized since we computed (wiq/|q|)*(wij/|d|) for each term\n"
                )
            else:
                f.write(f"     No vector available\n")
            f.write("\n")

        f.write("=" * 80 + "\n\n")


def display_results(
    results,
    document_dict,
    query_terms,
    terms_dict,
    term_stats,
    stemming_enabled,
    query_vector,
    doc_vectors,
):
    """
    Display search results with ranking, title, authors, and highlighted text
    :param results: results from search
    :param document_dict: documents
    :param query_terms: highlight terms that appear in the query
    :param terms_dict: terms
    :param term_stats: document frequency + idf
    :param stemming_enabled: stemming bool
    :param query_vector: query weight vector
    :param doc_vectors: documnent weight vector
    """
    if not results:
        print("No relevant documents found.\n")
        return

    print(f"\nFound {len(results)} relevant document(s):\n")

    # Print query weight vector
    print("Query Weight Vector (TF-IDF):")
    print("-" * 80)
    for term, weight in sorted(query_vector.items()):
        print(f"  '{term}': {weight:.6f}")
    print("-" * 80)
    print()

    print("=" * 80)

    for rank, (doc_id, score) in enumerate(results, 1):
        doc = document_dict[doc_id]
        authors = ", ".join(doc.authors) if doc.authors else "Unknown"

        print(f"{rank}. Document ID: {doc_id}")
        print(f"   Score: {score:.6f}")
        print(f"   Title: {doc.title}")
        print(f"   Authors: {authors}")

        # Print document weight vector (only for query terms)
        print(f"   Document Weight Vector (TF-IDF for query terms):")
        if doc_id in doc_vectors:
            doc_vector = doc_vectors[doc_id]
            for term in sorted(query_vector.keys()):
                weight = doc_vector.get(term, 0.0)
                print(f"     '{term}': {weight:.6f}")
        else:
            print(f"     No vector available")

        # Print TF for each query term in this document
        print(f"   Term Frequencies:")
        for term in term_stats.keys():
            if term in terms_dict and doc_id in terms_dict[term].postings:
                tf = terms_dict[term].postings[doc_id].tf
                print(f"     '{term}': {tf}")

        snippet = extract_context_snippet(
            doc.text,
            query_terms,
            terms_dict,
            doc_id,
            context_words=25,
            stemming_enabled=stemming_enabled,
        )
        print(f"   Context: {snippet}")
        print()

    print("=" * 80)


def read_cli():
    """Read command line arguments"""
    parser = argparse.ArgumentParser(
        prog="search_ui",
        description="Interactive Search User Interface - Search documents and view results",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=Path,
        required=True,
        help="Path to index pickle file (index.pkl.gz)",
    )
    parser.add_argument(
        "--postings",
        "-p",
        type=Path,
        required=True,
        help="Path to postings pickle file (postings.pkl.gz)",
    )
    parser.add_argument(
        "--documents",
        "-d",
        type=Path,
        required=True,
        help="Path to documents pickle file (documents.pkl.gz)",
    )
    parser.add_argument(
        "--stopwords",
        action="store_true",
        default=False,
        help="Remove stopwords from query",
    )
    parser.add_argument(
        "--stopwords-file",
        type=Path,
        default=Path("cacm/common_words"),
        help="Path to stopwords file",
    )
    parser.add_argument(
        "--stemming",
        action="store_true",
        default=False,
        help="Enable Porter stemming for query",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Number of top results to return (default: 10)",
    )
    return parser.parse_args()


def main():
    """Main interactive search interface"""
    args = read_cli()

    # Validate paths
    if not args.index.exists():
        print(f"Error: Index file {args.index} does not exist.", file=sys.stderr)
        sys.exit(1)
    if not args.postings.exists():
        print(f"Error: Postings file {args.postings} does not exist.", file=sys.stderr)
        sys.exit(1)
    if not args.documents.exists():
        print(
            f"Error: Documents file {args.documents} does not exist.", file=sys.stderr
        )
        sys.exit(1)

    # Load data
    print("Loading index, postings, and documents...")
    terms_dict = load_postings(args.postings)
    document_dict = load_documents(args.documents)
    print(f"Loaded {len(terms_dict)} terms and {len(document_dict)} documents.")


    idf_dict = calculate_idf_dict(terms_dict, len(document_dict))
    tf_vectors = calculate_tf_vectors(document_dict, terms_dict)
    weight_vectors = calculate_weight_vectors(document_dict, terms_dict, idf_dict)
    load_index(args.index)
    

    # load stopwords
    stopwords = set()
    if args.stopwords:
        stopwords = load_stopwords(args.stopwords_file)
        print(f"Loaded {len(stopwords)} stopwords from {args.stopwords_file}")

    # Pre-compute document vectors and lengths
    print("Pre-computing document weight vectors...")
    doc_vectors = build_document_term_vectors(terms_dict, len(document_dict))
    doc_lengths = compute_document_lengths(doc_vectors)
    print("Pre-computation complete.\n")

    # Display configuration
    print("=" * 80)
    print("SEARCH.PY UI")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Stopwords: {'Enabled' if args.stopwords else 'Disabled'}")
    print(f"  - Stemming: {'Enabled' if args.stemming else 'Disabled'}")
    print(f"\nType 'exit' or 'quit' to exit the program")
    print("=" * 80)

    # Interactive loop
    while True:
        try:
            query = input("\nEnter your query: ").strip()

            # Check for exit commands
            if query.lower() in ["exit", "quit"]:
                print("\n Quitting.")
                break

            # Skip empty queries
            if not query:
                print("Please enter a valid query.")
                continue

            # Ask for top-k
            top_k_input = input(f"Enter number of top results to display (default: {args.top_k}): ").strip()
            if top_k_input:
                try:
                    top_k = int(top_k_input)
                    if top_k <= 0:
                        print("Invalid top-k value. Using default.")
                        top_k = args.top_k
                except ValueError:
                    print("Invalid input. Using default top-k value.")
                    top_k = args.top_k
            else:
                top_k = args.top_k

            # Ask for threshold
            threshold_input = input("Enter minimum similarity threshold (default: 0.0): ").strip()
            if threshold_input:
                try:
                    threshold = float(threshold_input)
                    if threshold < 0 or threshold > 1:
                        print("Invalid threshold. Using default (0.0).")
                        threshold = 0.0
                except ValueError:
                    print("Invalid input. Using default threshold (0.0).")
                    threshold = 0.0
            else:
                threshold = 0.0

            # Perform search
            results, term_stats, query_terms = search(
                query,
                terms_dict,
                document_dict,
                doc_vectors,
                doc_lengths,
                args.stopwords,
                stopwords,
                args.stemming,
                threshold,
                detailed=True,
            )

            # Apply top-k filtering to results
            results = results[:top_k]

            # Compute query vector for display
            processed_query_terms = process_query(
                query, args.stopwords, stopwords, args.stemming
            )
            query_vector = compute_query_vector(
                processed_query_terms, terms_dict, len(document_dict)
            )

            # Write top-k results to file
            write_results_to_file(
                query,
                query_vector,
                document_dict,
                doc_vectors,
                doc_lengths,
                terms_dict,
                processed_query_terms,
                top_k,
            )
            print(f"\nTop {top_k} results (with threshold >= {threshold}) saved to output/query_results.txt")

            # Display results
            display_results(
                results,
                document_dict,
                query_terms,
                terms_dict,
                term_stats,
                args.stemming,
                query_vector,
                doc_vectors,
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            print("Please try again.\n")


if __name__ == "__main__":
    main()
