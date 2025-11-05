from postings import PostingsList


class Term:
    """
    Class representing a term in the inverted index
    """

    def __init__(self, term, frequency=0, postings=None):
        """
        Initialize a Term object
        """
        self.term = term
        self.frequency = frequency
        self.postings = postings if postings is not None else PostingsList()

    def add_occurrence(self, document_id, position=None) -> None:
        """
        Add an occurrence of the term in a document at a specific position & increment term frequency
        :param document_id: Document ID where the term occurs
        """
        self.frequency += 1
        self.postings.insert(document_id, position)

    def get_occurrence(self, document_id) -> dict | None:
        """
        Grab the occurence of the term with a specific document ID
        :param document_id: Document ID to look up
        :return: Dictionary with document ID, term frequency, positions, and term if found, None otherwise
        """
        node = self.postings.__getitem__(document_id)
        if node:
            return {
                "document_id": node.document_id,
                "tf": node.tf,
                "positions": node.positions,
                "term": self.term,
            }
        return None

    def grab_frequency(self) -> int:
        """
        Grab term frequency
        :return: Frequency of the term
        """
        return self.frequency

    def grab_postings(self) -> PostingsList:
        """
        Get the postings list for the term
        :return: PostingsList object
        """
        return self.postings

    def __str__(self) -> str:
        return f"Term({self.term}, Frequency: {self.frequency}, Postings: {self.postings.inorder_with_positions()})"

    def __repr__(self) -> str:
        return f"Term({self.term}, Frequency: {self.frequency}, Postings: {self.postings.inorder_with_positions()})"
