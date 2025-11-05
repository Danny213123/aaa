import argparse
import gzip
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from nltk.tokenize import word_tokenize

from document import Document
from stemming import PorterStemmer
from term import Term

global index
index: dict[str, int]

global terms_dict
terms_dict: dict[str, Term] = {}

global document_dict
document_dict: dict[int, Document] = {}

global idf_dict
idf_dict: dict[str, float] = {}


def load_documents(path: Path) -> dict:
    """
    Load documents from the pickle gzip file
    :param path: Path to the gzip file
    :return: Dictionary of document IDs and their metadata
    """
    global documents

    with gzip.open(path, "rb") as f:
        documents = pickle.load(f)
    return documents


def load_index(path: Path) -> dict:
    """
    Load index from the pickle gzip file
    :param path: Path to the gzip file
    :return: Dictionary of terms and their document frequencies
    """
    global index

    with gzip.open(path, "rb") as f:
        index = pickle.load(f)
    return index


def load_postings(path: Path) -> dict:
    """
    Load postings from the pickle gzip file
    :param path: Path to the gzip file
    :return: Dictionary of Term objects
    """
    with gzip.open(path, "rb") as f:
        snapshot: dict[str, dict] = pickle.load(f)

    terms_dict = {}
    for term, payload in snapshot.items():
        term_obj = Term(term, frequency=0)
        for doc_id, tf, positions in payload["postings"]:
            if positions:
                for pos in positions:
                    term_obj.add_occurrence(doc_id, pos)
            else:
                for _ in range(tf):
                    term_obj.add_occurrence(doc_id, None)
        terms_dict[term_obj.term] = term_obj
    return terms_dict


def normalize(text: str) -> str:
    """
    Normalize text by removing punctuation and digits
    :param text: Input text
    :return: Normalized text
    """
    text = "".join(char for char in text if char.isalnum() or char.isspace())
    if not text.strip():
        return ""
    return text.strip().lower()


def tokenize(text: str) -> List[str]:
    """
    Tokenize text using nltk's word_tokenize
    :param text: Input text
    :return: List of tokens
    """
    try:
        return word_tokenize(text)
    except Exception:
        punctuation = ".,!?;:'\"()-[]{}/"
        result = []
        current_word = ""
        for char in text:
            if char in punctuation:
                if current_word:
                    result.append(current_word)
                    current_word = ""
                result.append(char)
            elif char.isspace():
                if current_word:
                    result.append(current_word)
                    current_word = ""
            else:
                current_word += char
        if current_word:
            result.append(current_word)
        return result


def process_query(
    query: str,
    stopwords_enabled: bool = False,
    stopwords: set = None,
    stemming_enabled: bool = False,
) -> List[str]:
    """
    Process a query by tokenizing, normalizing, removing stopwords, and stemming
    :param query: query
    :param stopwords_enabled: remove stopwords
    :param stopwords: set of stopwords
    :param stemming_enabled: stemming bool
    :return: List of processed terms
    """
    tokens = tokenize(query)
    processed_terms = []

    for token in tokens:
        normalized = normalize(token)
        if not normalized:
            continue

        if stopwords_enabled and stopwords and normalized in stopwords:
            continue

        if stemming_enabled:
            normalized = PorterStemmer().stem(normalized, 0, len(normalized) - 1)

        processed_terms.append(normalized)

    return processed_terms


def extract_context_snippet(
    doc_text: str,
    query_terms: List[str],
    terms_dict: dict,
    doc_id: int,
    context_words: int = 25,
    stemming_enabled: bool = False,
) -> str:
    """
    Generate a summary of the document highlighting the first occurrence of the term with n words in its context
    """
    if not doc_text or not query_terms:
        return doc_text[:200] + "..." if len(doc_text) > 200 else doc_text

    words = doc_text.split()

    term_position = -1

    # Find the first occurrence of any query term by iterating through words
    for index, word in enumerate(words):
        normalized_word = "".join(char for char in word.lower() if char.isalnum())

        # Check if this word matches any query term directly
        if normalized_word in query_terms:
            term_position = index
            break

        # Try stemming if enabled
        if stemming_enabled:
            stemmed_word = PorterStemmer().stem(
                normalized_word, 0, len(normalized_word) - 1
            )
            if stemmed_word in query_terms:
                term_position = index
                break

    # If still no match, return beginning of text
    if term_position == -1:
        snippet_words = words[: context_words * 2]
        return " ".join(snippet_words) + (
            "..." if len(words) > context_words * 2 else ""
        )

    # Extract context around the term position
    start_pos = max(0, term_position - context_words)
    end_pos = min(len(words), term_position + context_words)

    context_words_list = words[start_pos:end_pos]

    # Highlight query terms in the snippet
    highlighted_words = []
    for index, word in enumerate(context_words_list):
        if start_pos + index == term_position:
            highlighted_words.append(f"**{word}**")
        else:
            highlighted_words.append(word)

    snippet = " ".join(highlighted_words)

    # add the three dots at the beginning and end if needed
    if start_pos > 0:
        snippet = "..." + snippet
    if end_pos < len(words):
        snippet = snippet + "..."

    return snippet


def calculate_tf(term_freq: int) -> float:
    """
    tf = 1 + log10(tf) if tf > 0 else 0
    :param term_freq: frequency of a term in a specific document
    :return: term frequency value
    """
    if term_freq == 0:
        return 0.0
    else:
        return 1 + math.log10(term_freq)


def calculate_idf(doc_freq: int, total_docs: int) -> float:
    """
    df = log10(N / df), where N is total number of documents
    :param doc_freq: total # of documnents that has a specific term
    :param total_docs: total # of documents
    :return: IDF value
    """
    if doc_freq == 0:
        return 0.0
    return math.log10(total_docs / doc_freq)


def compute_query_vector(
    query_terms: List[str], terms_dict: dict, total_docs: int
) -> Dict[str, float]:
    """
    Compute TF-IDF weighted query vector
    :param query_terms: List of processed query terms
    :param terms_dict: Dictionary of Term objects from the index
    :param total_docs: Total number of documents
    :return: Dictionary mapping terms to their TF-IDF weights
    """
    query_vector = {}
    term_counts = {}

    # Count term frequencies in query
    for term in query_terms:
        term_counts[term] = term_counts.get(term, 0) + 1

    # Calculate TF-IDF for each unique term
    for term, count in term_counts.items():
        if term not in terms_dict:
            continue

        tf = calculate_tf(count)
        doc_freq = terms_dict[term].postings.size
        idf = calculate_idf(doc_freq, total_docs)
        query_vector[term] = tf * idf

    return query_vector


def compute_query_tf_idf_vectors(
    query_terms: List[str], terms_dict: dict, total_docs: int
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Compute TF vector, IDF values, and TF-IDF weighted query vector separately
    :param query_terms: List of processed query terms
    :param terms_dict: Dictionary of Term objects from the index
    :param total_docs: Total number of documents
    :return: Tuple of (tf_vector, idf_vector, weight_vector)
    """
    tf_vector = {}
    idf_vector = {}
    weight_vector = {}
    term_counts = {}

    # Count term frequencies in query
    for term in query_terms:
        term_counts[term] = term_counts.get(term, 0) + 1

    # Calculate TF, IDF, and weight for each unique term
    for term, count in term_counts.items():
        if term not in terms_dict:
            continue

        # Calculate TF weight
        tf = calculate_tf(count)
        tf_vector[term] = tf

        # Calculate IDF
        doc_freq = terms_dict[term].postings.size
        idf = calculate_idf(doc_freq, total_docs)
        idf_vector[term] = idf

        # Calculate weight vector tf * idf
        weight_vector[term] = tf * idf

    return tf_vector, idf_vector, weight_vector


def build_document_term_vectors(
    terms_dict: dict, total_docs: int
) -> Dict[int, Dict[str, float]]:
    """
    Pre-compute full TF-IDF vectors for all documents
    :param terms_dict: Dictionary of Term objects
    :param total_docs: Total number of documents
    :return: Dictionary mapping document IDs to their full term vectors
    """
    doc_vectors = {}

    for term, term_obj in terms_dict.items():
        doc_freq = term_obj.postings.size
        idf = calculate_idf(doc_freq, total_docs)

        # Get all documents containing this term
        for doc_id, tf in term_obj.postings.inorder():
            if doc_id not in doc_vectors:
                doc_vectors[doc_id] = {}

            tf_weight = calculate_tf(tf)
            doc_vectors[doc_id][term] = tf_weight * idf

    return doc_vectors


def compute_document_lengths(
    doc_vectors: Dict[int, Dict[str, float]],
) -> Dict[int, float]:
    """
    vector length norm
    :param doc_vectors: Dictionary of document vectors
    :return: Dictionary mapping document IDs to their vector lengths
    """
    doc_lengths = {}
    for doc_id, vector in doc_vectors.items():
        doc_lengths[doc_id] = math.sqrt(sum(w**2 for w in vector.values()))
    return doc_lengths


def cosine_similarity(
    query_vector: Dict[str, float],
    doc_vector: Dict[str, float],
    query_length: float,
    doc_length: float,
) -> float:
    """
    Calculate cosine similarity between query and document vectors
    :param query_vector: Query TF-IDF weight vector
    :param doc_vector: Document TF-IDF weight vector
    :param query_length: Pre-computed length of the query vector
    :param doc_length: Pre-computed length of the document vector
    """
    # Calculate document vector length using ONLY query terms
    doc_length_query_terms = math.sqrt(
        sum(doc_vector.get(term, 0) ** 2 for term in query_vector.keys())
    )

    if query_length == 0 or doc_length_query_terms == 0:
        return 0.0

    score = 0.0
    for term, q_weight in query_vector.items():
        if term in doc_vector:
            d_weight = doc_vector[term]
            score += (q_weight / query_length) * (d_weight / doc_length_query_terms)

    return score


def cosine_similarity_common_terms(
    query_vector: Dict[str, float], doc_vector: Dict[str, float]
) -> tuple[float, set[str]]:
    """
    Calculate cosine similarity between query and document vectors
    Uses full vector magnitudes for proper normalization
    :param query_vector: Query TF-IDF weight vector
    :param doc_vector: Document TF-IDF weight vector
    :return: Tuple of (cosine similarity score, set of common terms)
    """
    # Step 1: Find common terms between query and document
    common_terms = set(query_vector.keys()) & set(doc_vector.keys())

    if not common_terms:
        return 0.0, set()

    # Step 2: Calculate dot product using common terms
    dot_product = 0.0
    for term in common_terms:
        dot_product += query_vector[term] * doc_vector[term]

    # Step 3: Calculate full vector magnitudes (using all terms, not just common ones)
    query_magnitude = math.sqrt(sum(w**2 for w in query_vector.values()))
    doc_magnitude = math.sqrt(sum(w**2 for w in doc_vector.values()))

    if query_magnitude == 0 or doc_magnitude == 0:
        return 0.0, common_terms

    similarity = dot_product / (query_magnitude * doc_magnitude)

    return similarity, common_terms


def search_with_weight_vectors(
    query: str,
    terms_dict: dict,
    document_dict: dict,
    stopwords_enabled: bool = False,
    stopwords: set = None,
    stemming_enabled: bool = False,
    top_k: int = 10,
    detailed: bool = True,
    use_static_quality: bool = False,
    quality_weight: float = 0.3,
) -> Tuple[
    List[Tuple[int, float, set[str]]], Dict[str, Tuple[float, float, float]], List[str]
]:
    """
    Search using weight vectors stored in documents with common terms approach
    :param query: Free text query
    :param terms_dict: Dictionary of Term objects
    :param document_dict: Dictionary of Document objects
    :param stopwords_enabled: Whether to remove stopwords
    :param stopwords: Set of stopwords
    :param stemming_enabled: Whether to apply stemming
    :param top_k: Number of top results to return
    :param detailed: Whether to print query term statistics
    :param use_static_quality: Whether to combine static quality scores
    :param quality_weight: Weight of static quality score in final ranking
    :return: Tuple of (list of (document_id, score, common_terms) tuples, dict of term statistics, list of query terms)
    """
    # Process query
    query_terms = process_query(query, stopwords_enabled, stopwords, stemming_enabled)

    if not query_terms:
        return [], {}, []

    total_docs = len(document_dict)

    # Compute query TF, IDF, and TF-IDF vectors
    query_tf_vector, query_idf_vector, query_tfidf_vector = (
        compute_query_tf_idf_vectors(query_terms, terms_dict, total_docs)
    )

    if not query_tfidf_vector:
        return [], {}, query_terms

    # Collect query term statistics
    query_stats = {}
    if detailed:
        print("\nQuery Term Statistics:")
        print("-" * 80)
        for term in query_tfidf_vector.keys():
            tf = query_tf_vector[term]
            idf = query_idf_vector[term]
            tfidf = query_tfidf_vector[term]
            doc_freq = terms_dict[term].postings.size if term in terms_dict else 0
            query_stats[term] = (tf, idf, tfidf)
            print(f"Term: '{term}'")
            print(f"  Document Frequency (df): {doc_freq}")
            print(f"  Query TF: {tf:.6f}")
            print(f"  IDF: {idf:.6f}")
            print(f"  Query TF-IDF Weight: {tfidf:.6f}\n")
        print("-" * 80 + "\n")

    # Calculate cosine similarity for each document using common terms approach
    scores = []
    for doc_id, doc_obj in document_dict.items():
        doc_weight_vector = doc_obj.get_weight_vector()

        if not doc_weight_vector:
            continue

        # Calculate cosine similarity using only common terms
        similarity, common_terms = cosine_similarity_common_terms(
            query_tfidf_vector, doc_weight_vector
        )

        if similarity > 0:
            # Combine with static quality score if enabled
            if use_static_quality:
                quality_score = doc_obj.get_static_quality_score()
                # score(d, q) = alpha * g(d) + (1 - alpha) * cos(d, q)
                final_score = (
                    quality_weight * quality_score + (1 - quality_weight) * similarity
                )
            else:
                final_score = similarity

            scores.append((doc_id, final_score, common_terms))

    # Sort by score in descending order and return top K
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k], query_stats, query_terms


def search(
    query: str,
    terms_dict: dict,
    document_dict: dict,
    doc_vectors: Dict[int, Dict[str, float]],
    doc_lengths: Dict[int, float],
    stopwords_enabled: bool = False,
    stopwords: set = None,
    stemming_enabled: bool = False,
    threshold: float = 0.0,
    detailed: bool = True,
) -> Tuple[List[Tuple[int, float]], Dict[str, Tuple[int, float]], List[str]]:
    """
    Search for relevant documents using vector space model with cosine similarity
    :param query: Free text query
    :param terms_dict: Dictionary of Term objects
    :param document_dict: Dictionary of Document objects
    :param doc_vectors: Pre-computed document TF-IDF vectors
    :param doc_lengths: Pre-computed document vector lengths
    :param stopwords_enabled: Whether to remove stopwords
    :param stopwords: Set of stopwords
    :param stemming_enabled: Whether to apply stemming
    :param threshold: Minimum cosine similarity score threshold (return all documents with score >= threshold)
    :param detailed: Whether to print query term statistics
    :return: Tuple of (list of (document_id, score) tuples, dict of term statistics, list of query terms)
    """
    # Process query
    query_terms = process_query(query, stopwords_enabled, stopwords, stemming_enabled)

    if not query_terms:
        return [], {}, []

    total_docs = len(document_dict)

    # Compute query vector
    query_vector = compute_query_vector(query_terms, terms_dict, total_docs)

    if not query_vector:
        return [], {}, query_terms

    # Collect term statistics (df and idf)
    term_stats = {}
    if detailed:
        print("\nQuery Term Statistics:")
        print("-" * 60)
        for term in query_vector.keys():
            if term in terms_dict:
                df = terms_dict[term].postings.size
                idf = calculate_idf(df, total_docs)
                term_stats[term] = (df, idf)
                print(f"Term: '{term}'")
                print(f"  Document Frequency (df): {df}")
                print(f"  Inverse Document Frequency (idf): {idf:.6f}")
                print(f"  Query Term Frequency (tf): {query_vector[term]:.6f}")
                print(f"  Query Weight: {query_vector[term]:.6f}\n")
        print("-" * 60 + "\n")

    # Compute query vector length
    query_length = math.sqrt(sum(w**2 for w in query_vector.values()))

    # Find candidate documents (documents containing at least one query term)
    candidate_docs = set()
    for term in query_vector.keys():
        if term in terms_dict:
            for doc_id, _ in terms_dict[term].postings.inorder():
                candidate_docs.add(doc_id)

    # Calculate cosine similarity for each candidate document
    scores = []
    for doc_id in candidate_docs:
        if doc_id not in doc_vectors:
            continue

        score = cosine_similarity(
            query_vector, doc_vectors[doc_id], query_length, doc_lengths[doc_id]
        )

        if score >= threshold:
            scores.append((doc_id, score))

    # Sort by score in descending order and return all documents above threshold
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores, term_stats, query_terms


def load_stopwords(stopwords_file: Path) -> set:
    """
    Load stopwords from file
    :param stopwords_file: Path to stopwords file
    :return: Set of stopwords
    """
    stopwords = set()
    if stopwords_file.exists():
        with stopwords_file.open("r", encoding="utf-8") as f:
            for line in f:
                stopwords.add(normalize(line.strip()))
    return stopwords


def calculate_idf_dict(terms_dict: dict, total_docs: int) -> dict:
    global idf_dict

    with open("output/idf_values.txt", "w", encoding="utf-8") as f:

        f.write("Calculating IDF values for terms...\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Documents: {total_docs}\n\n")

        for term, term_obj in terms_dict.items():
            doc_freq = term_obj.postings.size
            idf = calculate_idf(doc_freq, total_docs)
            idf_dict[term] = idf
            f.write(f"Term: '{term}' | Document Frequency: {doc_freq}\n")
            f.write(f"Term: '{term}' | Inverse Document Frequency: {idf}\n")

        f.write("-" * 50 + "\n")

        return idf_dict


def calculate_tf_vectors(document_dict: dict, terms_dict: dict) -> dict:
    """
    Generate weight vectors for each document using calculate_tf
    :param document_dict: Dictionary of Document objects
    :param terms_dict: Dictionary of Term objects
    :return: Dictionary mapping document IDs to their TF weight vectors
    """
    with open("output/tf_vectors.txt", "w", encoding="utf-8") as f:
        f.write("Calculating TF weight vectors for documents...\n")
        f.write("-" * 50 + "\n")
        tf_vectors = {}

        for doc_id, doc_obj in document_dict.items():
            tf_vector = {}
            f.write(f"\nDocument ID: {doc_id}\n")
            f.write(f"Title: {doc_obj.title}\n")
            f.write(f"Terms in document: {len(doc_obj.terms)}\n")

            # For each term in the document, calculate TF weight
            for term in doc_obj.terms:
                if term in terms_dict and doc_id in terms_dict[term].postings:
                    # Get raw term frequency
                    raw_tf = terms_dict[term].postings[doc_id].tf
                    # Calculate weighted term frequency using calculate_tf
                    tf_weight = calculate_tf(raw_tf)
                    tf_vector[term] = tf_weight
                    f.write(
                        f"  Term: '{term}' | Raw TF: {raw_tf} | Weighted TF: {tf_weight:.6f}\n"
                    )

            f.write("-" * 50 + "\n")

            # Store TF vector in the document object
            doc_obj.set_tf_vector(tf_vector)
            tf_vectors[doc_id] = tf_vector

        f.write(f"\nTotal documents processed: {len(tf_vectors)}\n")
        return tf_vectors


def calculate_static_quality_scores(document_dict: dict) -> dict:
    """
    Calculate static quality scores g(d) for all documents based on number of unique terms
    This is used for top-k ranking
    :param document_dict: Dictionary of Document objects
    :return: Dictionary mapping document IDs to their static quality scores (0.0 to 1.0)
    """
    quality_scores = {}

    unique_terms_counts = []

    for doc_id, doc in document_dict.items():
        unique_terms = len(doc.terms)
        unique_terms_counts.append(unique_terms)

    min_unique_terms = min(unique_terms_counts) if unique_terms_counts else 1
    max_unique_terms = max(unique_terms_counts) if unique_terms_counts else 1

    for doc_id, doc in document_dict.items():
        unique_terms = len(doc.terms)

        if max_unique_terms > min_unique_terms:
            quality_score = (unique_terms - min_unique_terms) / (
                max_unique_terms - min_unique_terms
            )
        else:
            quality_score = 0.5

        doc.set_static_quality_score(quality_score)
        quality_scores[doc_id] = quality_score

    return quality_scores


def calculate_weight_vectors(
    document_dict: dict, terms_dict: dict, idf_dict: dict
) -> dict:
    """
    The weight vector for a document is calculated as:
    weight(t, d) = tf(t, d) * idf(t)
    :param document_dict: Dictionary of Document objects
    :param terms_dict: Dictionary of Term objects
    :param idf_dict: Dictionary of IDF values for terms
    :return: Dictionary mapping document IDs to their TF-IDF weight vectors
    """
    weight_vectors = {}

    for doc_id, doc_obj in document_dict.items():
        weight_vector = {}

        # For each term in the document, calculate tf*idf weight
        for term in doc_obj.terms:
            if (
                term in terms_dict
                and doc_id in terms_dict[term].postings
                and term in idf_dict
            ):
                # Get raw term frequency
                raw_tf = terms_dict[term].postings[doc_id].tf
                tf_weight = calculate_tf(raw_tf)
                idf = idf_dict[term]
                tfidf_weight = tf_weight * idf
                weight_vector[term] = tfidf_weight
        doc_obj.set_weight_vector(weight_vector)
        weight_vectors[doc_id] = weight_vector

    return weight_vectors


def read_cli() -> argparse.Namespace:
    """
    Read command line arguments
    """
    parser = argparse.ArgumentParser(
        prog="search",
        description="CPS842: Information Retrieval - Vector Space Model Search with Cosine Similarity",
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
        "--query",
        "-q",
        type=str,
        help="Free text query (if not provided, interactive mode)",
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
        "--threshold",
        "-t",
        type=float,
        default=0.0,
        help="Minimum cosine similarity score threshold (default: 0.0, returns all results with score >= threshold)",
    )
    return parser.parse_args()


def main():
    """
    Main function for vector space model search with cosine similarity
    """
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

    # Calculate IDF values for all terms
    print("Calculating IDF values for all terms...")
    idf_dict = calculate_idf_dict(terms_dict, len(document_dict))
    print(f"IDF values saved to output/idf_values.txt")

    # Calculate TF weight vectors for all documents
    print("Calculating TF weight vectors for all documents...")
    tf_vectors = calculate_tf_vectors(document_dict, terms_dict)
    print(f"TF weight vectors saved to output/tf_vectors.txt")

    # Calculate TF-IDF weight vectors for all documents
    print("Calculating TF-IDF weight vectors for all documents...")
    weight_vectors = calculate_weight_vectors(document_dict, terms_dict, idf_dict)
    print(f"TF-IDF weight vectors saved to output/tfidf_weight_vectors.txt")

    # Pre-compute document vectors and lengths
    print("Pre-computing document TF-IDF vectors...")
    doc_vectors = build_document_term_vectors(terms_dict, len(document_dict))
    doc_lengths = compute_document_lengths(doc_vectors)
    print("Pre-computation complete.")

    # Load stopwords if enabled
    stopwords = None
    if args.stopwords:
        stopwords = load_stopwords(args.stopwords_file)
        print(f"Loaded {len(stopwords)} stopwords.")

    # Single query mode or interactive mode
    if args.query:
        # Single query mode
        print(f"\nQuery: {args.query}")
        print(f"Stopwords: {'enabled' if args.stopwords else 'disabled'}")
        print(f"Stemming: {'enabled' if args.stemming else 'disabled'}")
        print(f"Threshold: {args.threshold}\n")

        results, term_stats, query_terms = search(
            args.query,
            terms_dict,
            document_dict,
            doc_vectors,
            doc_lengths,
            args.stopwords,
            stopwords,
            args.stemming,
            args.threshold,
        )

        if not results:
            print("No relevant documents found.")
        else:
            print(f"Found {len(results)} documents with score >= {args.threshold}:\n")
            for rank, (doc_id, score) in enumerate(results, 1):
                doc = document_dict[doc_id]
                print(f"{rank}. Document ID: {doc_id}")
                print(f"   Score: {score:.6f}")
                print(f"   Title: {doc.title}")

                # Print TF for each query term in this document
                print(f"   Term Frequencies:")
                for term in term_stats.keys():
                    if term in terms_dict and doc_id in terms_dict[term].postings:
                        tf = terms_dict[term].postings[doc_id].tf
                        print(f"     '{term}': {tf}")

                # Extract and print context snippet
                snippet = extract_context_snippet(
                    doc.text,
                    query_terms,
                    terms_dict,
                    doc_id,
                    context_words=25,
                    stemming_enabled=args.stemming,
                )
                print(f"   Context: {snippet}")
                print()
    else:
        # Interactive mode
        print("\nInteractive search mode (type 'exit' to quit)")
        print(f"Stopwords: {'enabled' if args.stopwords else 'disabled'}")
        print(f"Stemming: {'enabled' if args.stemming else 'disabled'}")
        print(f"Threshold: {args.threshold}\n")

        while True:
            try:
                query = input("Enter query: ").strip()
                if query.lower() == "exit":
                    print("Exiting...")
                    break

                if not query:
                    print("Please enter a valid query.\n")
                    continue

                results, term_stats, query_terms = search(
                    query,
                    terms_dict,
                    document_dict,
                    doc_vectors,
                    doc_lengths,
                    args.stopwords,
                    stopwords,
                    args.stemming,
                    args.threshold,
                )

                if not results:
                    print("No relevant documents found.\n")
                else:
                    print(f"\nFound {len(results)} documents with score >= {args.threshold}:\n")
                    for rank, (doc_id, score) in enumerate(results, 1):
                        doc = document_dict[doc_id]
                        print(f"{rank}. Document ID: {doc_id}")
                        print(f"   Score: {score:.6f}")
                        print(f"   Title: {doc.title}")

                        # Print TF for each query term in this document
                        print(f"   Term Frequencies:")
                        for term in term_stats.keys():
                            if (
                                term in terms_dict
                                and doc_id in terms_dict[term].postings
                            ):
                                tf = terms_dict[term].postings[doc_id].tf
                                print(f"     '{term}': {tf}")

                        # Extract and print context snippet
                        snippet = extract_context_snippet(
                            doc.text,
                            query_terms,
                            terms_dict,
                            doc_id,
                            context_words=25,
                            stemming_enabled=args.stemming,
                        )
                        print(f"   Context: {snippet}")
                        print()
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
