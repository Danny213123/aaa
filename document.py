class Document:
    """
    Class representing a document with metadata
    """

    def __init__(self, document_id, title, text, publication_date, authors, n, x):
        """
        Initialize a Document object
        """
        self.document_id = document_id
        self.title = title
        self.text = text
        self.publication_date = publication_date
        self.authors = authors
        self.n = n
        self.x = x
        self.terms = []
        self.tf_vector = {}
        self.weight_vector = {}
        self.static_quality_score = 0.0

    def word_count(self):
        """
        Get the word count of the document text
        :return: Word count
        """
        return len(self.text.split())

    def get_terms(self):
        """
        Get the list of terms in the document
        :return: List of terms
        """
        return self.terms

    def add_term(self, term):
        """
        Add a term to the document's term list
        :param term: Term to add
        """
        if term not in self.terms:
            self.terms.append(term)
            self.terms.sort()

    def set_tf_vector(self, tf_vector):
        """
        Set the TF weight vector for this document
        :param tf_vector: Dictionary mapping term -> TF weight
        """
        self.tf_vector = tf_vector

    def get_tf_vector(self):
        """
        Get the TF weight vector for this document
        :return: Dictionary mapping term -> TF weight
        """
        return self.tf_vector

    def set_weight_vector(self, weight_vector):
        """
        Set the TF-IDF weight vector for this document
        :param weight_vector: Dictionary mapping term -> TF-IDF weight
        """
        self.weight_vector = weight_vector

    def get_weight_vector(self):
        """
        Get the TF-IDF weight vector for this document
        :return: Dictionary mapping term -> TF-IDF weight
        """
        return self.weight_vector

    def set_static_quality_score(self, score):
        """
        Set the static quality score for this document
        :param score: Static quality score (0.0 to 1.0)
        """
        self.static_quality_score = score

    def get_static_quality_score(self):
        """
        Get the static quality score for this document
        :return: Static quality score
        """
        return self.static_quality_score

    def check_text_empty(self):
        """
        Check if the document text is empty
        :return: True if text is empty, False otherwise
        """
        return len(self.text.strip()) == 0

    def __str__(self):
        if not self.check_text_empty():
            return f"Document ID: {self.document_id}, Title: {self.title}, Text: {self.text}, Publication Date: {self.publication_date}, Authors: {self.authors}, N: {self.n}, X: {self.x}"
        return self.text

    def __repr__(self):
        if not self.check_text_empty():
            return f"Document({self.document_id} | Title: {self.title} | Text: {self.text} | Publication Date: {self.publication_date} | Authors: {self.authors} | N: {self.n} | X: {self.x})"
        return f"Document({self.document_id} | Title: {self.title} | Publication Date: {self.publication_date} | Authors: {self.authors} | N: {self.n} | X: {self.x})"
