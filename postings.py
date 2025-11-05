class Node:
    def __init__(self, document_id, tf=0):
        self.left = None
        self.right = None
        self.document_id = document_id
        self.tf = tf
        self.positions = []  # position pointer within the text


class PostingsList:
    """Binary Search Tree for Postings List"""

    def __init__(self):
        """Initialize the root and size."""
        self.root = None
        self.size = 0

    def insert(self, document_id, position=None):
        """Insert an occurrence for `document_id`; record `position` if given."""
        if self.root is None:
            self.root = Node(document_id, tf=1)
            if position is not None:
                self.root.positions.append(position)
            self.size += 1
            return

        cur = self.root
        while True:

            # left if smaller, right if larger
            if document_id == cur.document_id:
                cur.tf += 1
                if position is not None:
                    cur.positions.append(position)
                return
            elif document_id < cur.document_id:
                if cur.left is None:
                    cur.left = Node(document_id, tf=1)
                    if position is not None:
                        cur.left.positions.append(position)
                    self.size += 1
                    return
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = Node(document_id, tf=1)
                    if position is not None:
                        cur.right.positions.append(position)
                    self.size += 1
                    return
                cur = cur.right

    def inorder(self):
        """Inorder traversal of the BST, left-root-right."""
        result = []
        stack = []
        cur = self.root

        # while there are nodes to process
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            result.append((cur.document_id, cur.tf))
            cur = cur.right
        return result

    def inorder_with_positions(self):
        """inorder but with positions, vertix model to inverted index"""
        result = []
        stack = []
        cur = self.root
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            result.append((cur.document_id, cur.tf, list(cur.positions)))
            cur = cur.right
        return result

    def sorted_by_tf(self):
        """Return postings sorted by term frequency (tf) in descending order."""
        nodes = []
        stack = []
        cur = self.root

        # Inorder traversal to collect all nodes
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            nodes.append(cur)
            cur = cur.right

        # Sort nodes by tf in descending order
        nodes.sort(key=lambda node: node.tf, reverse=True)

        # Return list of (document_id, tf)
        return [(node.document_id, node.tf) for node in nodes]

    def items(self):
        """returns all nodes in the postings list"""
        nodes = []
        stack = []
        cur = self.root

        # Inorder traversal to collect all nodes
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            nodes.append((cur.document_id, cur))
            cur = cur.right

        return nodes

    def __contains__(self, document_id):
        """Check if a term shows up in a certain document"""
        cur = self.root
        while cur:
            if document_id == cur.document_id:
                return True
            cur = cur.left if document_id < cur.document_id else cur.right
        return False

    def __getitem__(self, document_id):
        """Get the postings for a specific document"""
        cur = self.root
        while cur:
            if document_id == cur.document_id:
                return cur
            cur = cur.left if document_id < cur.document_id else cur.right
        raise KeyError(f"Document ID {document_id} not found in postings list.")

    def grab_positions(self, document_id):
        """grab positions for a specific document"""
        cur = self.root
        while cur:
            if document_id == cur.document_id:
                return list(cur.positions)  # return a copy
            cur = cur.left if document_id < cur.document_id else cur.right
        raise KeyError(f"Document ID {document_id} not found in postings list.")

    def __len__(self):
        return self.size
