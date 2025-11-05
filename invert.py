import argparse
import gzip
import pickle
import ssl
import sys
import time
from pathlib import Path
from typing import List, Optional

import nltk

nltk.download("punkt")
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


def read_documents(file_path: Path) -> List[Document]:
    """
    Read documents from a file in the specified format
    :param file_path: Path to the input file
    :return: List of Document objects
    """

    if not file_path.exists():
        print(f"Error: {file_path} does not exist.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Reading document(s) from {file_path}")

    # declare current document state
    document_id: Optional[int] = None
    pointer: Optional[str] = None  # one of {"T","W","B","A","N","X"} or None

    title_parts: List[str] = []
    text_parts: List[str] = []
    date_parts: List[str] = []
    authors: List[str] = []
    n_parts: List[str] = []
    x_parts: List[str] = []

    documents: List[Document] = []

    def flush_current():
        global document_dict

        nonlocal document_id, title_parts, text_parts, date_parts, authors, n_parts, x_parts
        if document_id is None:
            return
        doc = Document(
            document_id,
            " ".join(title_parts).strip(),
            " ".join(text_parts).strip(),
            " ".join(date_parts).strip(),
            [a.strip() for a in authors if a.strip()],
            n_parts[:],
            x_parts[:],
        )
        documents.append(doc)

        document_dict[document_id] = doc

        # reset
        document_id = None
        title_parts.clear()
        text_parts.clear()
        date_parts.clear()
        authors.clear()
        n_parts.clear()
        x_parts.clear()

    with open(file_path, "r", encoding="utf-8") as file:
        for raw in file:
            line = raw.rstrip("\n").replace("\t", " ")

            if not line:
                if pointer == "T":
                    title_parts.append("")
                elif pointer == "W":
                    text_parts.append("")
                elif pointer == "B":
                    date_parts.append("")
                elif pointer == "A":
                    authors.append("")
                elif pointer == "N":
                    n_parts.append("")
                elif pointer == "X":
                    x_parts.append("")
                continue

            if line.startswith(".I"):
                flush_current()
                parts = line.split()
                if len(parts) < 2 or not parts[1].isdigit():
                    print(f"[ERROR]: Invalid Document ID line: {line}", file=sys.stderr)
                    sys.exit(1)
                document_id = int(parts[1])
                pointer = None
                continue

            if line in (".T", ".W", ".B", ".A", ".N", ".X"):
                pointer = line[1]
                continue

            if pointer == "T":
                title_parts.append(line)
            elif pointer == "W":
                text_parts.append(line)
            elif pointer == "B":
                date_parts.append(line)
            elif pointer == "A":
                authors.append(line)
            elif pointer == "N":
                n_parts.append(line)
            elif pointer == "X":
                x_parts.append(line)
            else:
                pass

    if document_id is not None:
        flush_current()
    else:
        if not documents:
            print(f"[ERROR]: No documents found in {file_path}", file=sys.stderr)
            sys.exit(1)

    return documents


def write_documents(file_path: Path, documents: List[Document]) -> None:
    """
    Write documents to a file
    :param file_path: Path to the output file
    :param documents: List of Document objects
    :return: None
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.__repr__() + "\n")


def write_terms(file_path: Path, terms: List[str]) -> None:
    """
    Write terms to a file
    :param file_path: Path to the output file
    :param terms: List of terms to write
    :return: None
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for term in terms:
            f.write(term + "\n")


def write_text(file_path: Path, text: str) -> None:
    """
    Write text to a file
    :param file_path: Path to the output file
    :param text: Text to write
    :return: None
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def write_index(file_path: Path, index: dict[str, int]) -> None:
    """
    Write the index dictionary to a file
    :param file_path: Path to the output file
    :param index: Index dictionary
    :return: None
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for term, count in index.items():
            f.write(f"{term}: {count}\n")


def pickle_index(path: Path) -> None:
    """
    Pickle the index dictionary
    :param path: Path to the gzip file
    :return: None
    """
    global index
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_postings_list(file_path: Path) -> None:
    """
    Write the postings list for all terms to a file
    :param file_path: Path to the output file
    :return: None
    """
    global terms_dict
    # make sure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # write postings for each term in the same file
    with file_path.open("w", encoding="utf-8") as f:
        for term, term_obj in terms_dict.items():
            postings_list = grab_postings_list(term_obj)
            f.write(f"{term_obj}\n")


def pickle_postings_list(path) -> None:
    """
    Pickle the postings list for all terms
    :param path: Path to the gzip file
    :return: None
    """
    global terms_dict

    snapshot = {}
    for term, term_obj in terms_dict.items():
        # requires the inorder_with_positions() helper we added earlier
        plist = term_obj.postings.inorder_with_positions()
        snapshot[term] = {"freq": term_obj.frequency, "postings": plist}

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_documents(path: Path) -> None:
    """
    Pickle the documents dictionary
    :param path: Path to the gzip file
    :return: None
    """
    global document_dict

    with open("output/document_dict_debug.txt", "w", encoding="utf-8") as f:
        for doc_id, doc in document_dict.items():
            f.write(f"{doc_id}: {doc}\n")
            f.write(f"Document Terms: {doc.get_terms()}\n\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(document_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def tokenize(text: str) -> List[str]:
    """Tokenize text using nltk's word_tokenize
    :param text: Input text
    :return: List of tokens
    """
    try:
        return word_tokenize(text)
    except Exception as e:
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

        # Add last word if exists
        if current_word:
            result.append(current_word)

        return result


def normalize(text: str) -> str:
    """
    Normalize text by removing punctuation and digits.
    :param text: Input text
    :return: Normalized text
    """
    # Remove punctuation
    text = "".join(char for char in text if char.isalnum() or char.isspace())

    # only keep non-numeric characters
    # text = "".join(char for char in text if not char.isdigit())

    # check for empty string
    if not text.strip():
        return ""

    # Lowercase and strip whitespace
    return text.strip().lower()


def grab_terms(doc: Document) -> dict[str, List[str]]:
    """
    Grab terms from a specific document (includes title, authors, and text)
    :param doc: Document object
    :return: List of terms
    """
    # grab terms from a specific document
    terms = []
    # Add terms from title first
    for term in tokenize(doc.title):
        terms.append(normalize(term))
    # Add terms from authors
    if doc.authors:
        for author in doc.authors:
            for term in tokenize(author):
                terms.append(normalize(term))
    # Then add terms from text
    for term in tokenize(doc.text):
        terms.append(normalize(term))
    return terms


def grab_terms_from_all_documents(
    Documents: List[Document], stopwords: bool, stopwords_file: Path, stemming: bool
) -> None | List[str]:
    """
    Grab terms from all documents
    :param Documents: List of Document objects
    :param stopwords: Whether to remove stopwords
    :param stopwords_file: Path to stopwords file
    :param stemming: Whether to apply Porter stemming
    :return: List of unique terms
    """
    global terms_dict
    terms_dict = {}

    global document_dict

    # check if stopwords removal is enabled and if file exists
    stopword_set = set()
    if stopwords and stopwords_file.exists():
        with stopwords_file.open("r", encoding="utf-8") as f:
            for line in f:
                stopword_set.add(normalize(line.strip()))

    debug_path = Path("debug/docs_terms_debug.txt")
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    # Debug file to track term addition
    debug_add_path = Path("debug/term_addition_debug.txt")
    debug_add_path.parent.mkdir(parents=True, exist_ok=True)

    with debug_path.open("w", encoding="utf-8") as f, debug_add_path.open(
        "w", encoding="utf-8"
    ) as f_add:
        for doc in Documents:
            doc_terms = grab_terms(doc)

            f.write(
                f"Document ID: {doc.document_id}\n"
                f"Title: {doc.title}\n"
                f"Text: {doc.text}\n"
                f"Terms: {sorted(doc_terms)}\n\n"
            )

            position_pointer = 0

            # Get the document from document_dict to ensure we're modifying the right object
            current_doc = document_dict[doc.document_id]

            f_add.write(f"\n=== Processing Document ID: {doc.document_id} ===\n")
            f_add.write(f"Before processing - doc.terms: {doc.terms}\n")
            f_add.write(f"Before processing - current_doc.terms: {current_doc.terms}\n")
            f_add.write(f"Same object? {doc is current_doc}\n")

            processed_terms_added = []

            # count terms in global index
            for term in doc_terms:
                # Skip empty terms
                if not term:
                    position_pointer += 1
                    continue

                # Skip stopwords but still increment position
                if stopwords and term in stopword_set:
                    position_pointer += 1
                    continue

                # Apply stemming if enabled
                processed_term = term
                if stemming:
                    processed_term = PorterStemmer().stem(term, 0, len(term) - 1)

                # Get or create term object
                term_obj = terms_dict.get(processed_term)
                if term_obj is None:
                    term_obj = Term(processed_term)
                    terms_dict[processed_term] = term_obj

                # Add occurrence with current position
                term_obj.add_occurrence(doc.document_id, position_pointer)
                current_doc.add_term(processed_term)
                processed_terms_added.append(processed_term)

                position_pointer += 1

            f_add.write(f"Processed terms added: {len(processed_terms_added)}\n")
            f_add.write(
                f"After processing - current_doc.terms: {current_doc.terms[:10]}...\n"
            )
            f_add.write(f"Total unique terms in document: {len(current_doc.terms)}\n")

    return list(terms_dict.keys())


def grab_postings_list(term: Term) -> List[int]:
    """
    Grab postings list for a specific term
    :param term: Term object
    :return: List of document IDs where the term appears
    """
    postings = term.grab_postings()
    return [doc_id for doc_id, tf in postings.inorder()]


def indexer() -> dict[str, int]:
    """
    return index
    """
    global index
    index = {}
    for term in sorted(terms_dict.keys()):
        index[term] = terms_dict[term].postings.size
    return index


def read_cli() -> argparse.Namespace:
    """
    Read command line arguments
    """
    parser = argparse.ArgumentParser(
        prog="cli",
        description="CPS842: Information Retrieval and Web Search - Assignment 1",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to cacm.all file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to output file",
    )
    parser.add_argument(
        "--stopwords",
        action="store_true",
        default=False,
        help="Remove stopwords",
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
        help="Enable Porter stemming",
    )
    return parser.parse_args()


def main():
    """
    You need to write a program invert to do the index construction. The input to the program is the document collection.
    The output includes two files - a dictionary file and a postings lists file. Each entry in the dictionary should include
    a term and its document frequency. You should use a proper data structure to build the dictionary (e.g. hashmap or search
    tree or others). The structure should be easy for random lookup and insertion of new terms. All the terms should be sorted
    in alphabetical order. Postings list for each term should include postings for all documents the term occurs in (in the
    order of document ID), and the information saved in a posting includes document ID, term frequency in the document, and
    positions of all occurrences of the term in the document. There is a one-to-one correspondence between the term in the
    dictionary file and its postings list in the postings lists file.
    """
    global index, terms_dict

    start = time.time()

    args = read_cli()

    # make sure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    docs = read_documents(args.input)

    #   write_documents(args.output, docs)
    #   print(f"\nParsed {len(docs)} documents -> {args.output}")

    terms = grab_terms_from_all_documents(
        docs, args.stopwords, args.stopwords_file, args.stemming
    )
    print(f"Extracted {len(terms)} unique terms.")
    print(f"Extracted {len(docs)} documents.")

    index = indexer()
    #   print(f"Created index with {len(index)} unique terms.")
    index_output_path = args.output.parent / "index.txt"
    write_index(index_output_path, index)

    #   terms_output_path = args.output.parent / "terms.txt"
    #   write_terms(terms_output_path, sorted(terms))

    #   text_output_path = args.output.parent / "all_text.txt"
    #   all_text = "\n".join(doc.text for doc in docs)
    #   write_text(text_output_path, all_text)

    # write postings for each term in the same file
    postings_dir = args.output.parent / "postings.txt"
    write_postings_list(postings_dir)

    pickle_postings_list(postings_dir.parent / "postings.pkl.gz")

    pickle_index(index_output_path.parent / "index.pkl.gz")

    pickle_documents(index_output_path.parent / "documents.pkl.gz")

    duration = time.time() - start
    print(f"Indexing completed in {duration:.6f} seconds.")


if __name__ == "__main__":
    main()
