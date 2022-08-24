import numpy as np
from typing import List, Tuple, Mapping
from collections import Counter, deque


class CoefficientTreeNode:
    def __init__(self, val: int, level: int, loc: tuple, quad: int, children: list = []):
        """A class is used to construct a tree based on the location correspondence between
        points in lower sub-band and points in higher sub-bands in iterative process.

        Args:
            val (int): An integer represents value of current element in coefficient matrices.
            level (int): An integer represents sub-band level of current element in coefficient matrix.
            loc (tuple): A two-dimensional tuple location of current element in coefficient matrix.
            quad (int): An integer represents quadrant of current element in coefficient matrix.
            children (list): A list contains children of current node.
        """
        self.val = val
        self.lev = level
        self.loc = loc
        self.quad = quad
        self.children = children
        self.code = None

    def zero_code(self):
        """Marker of tree node."""
        # Travel all children of current node
        for child in self.children:
            child.zero_code()
        # If the value of current node is more than 0
        if abs(self.val) > 0:
            self.code = "N"
        else:  # If not
            # Mark current node as "Z" if all marks of its children are not "T", otherwise mark "T" for it.
            self.code = "Z" if any([child.code != "T" for child in self.children]) else "T"

    @staticmethod
    def build_trees(coeffs):
        """Construct forest.

        Args:
            coeffs (list): Coefficient matrices.

        Returns:
            trees (list): A list of `CoefficientTree`s.
        """
        def build_child_tree(lev: int, parent_loc: tuple, quad: int):
            """Build children trees.

            Args:
                lev (int): An integer represents sub-band level of current element in coefficient matrix.
                parent_loc (tuple): A two-dimensional tuple location of parent of current element in coefficient matrix.
                quad (int): An integer represents quadrant of current element in coefficient matrix.

            Returns:
                children (list): A list of `CoefficientTree`s.

            """
            # The end of iteration
            if lev + 1 > len(coeffs):
                return []

            (i, j) = parent_loc  # Parent location
            (H, W) = coeffs[lev][quad].shape  # The shape of current quadrant

            # Coordinates of children of current element
            loclis = [(2 * i, 2 * j),
                      (2 * i, 2 * j + 1),
                      (2 * i + 1, 2 * j),
                      (2 * i + 1, 2 * j + 1)]
            children = []

            # Travel all coordinates
            for loc in loclis:
                # If search is out of region.
                if loc[0] >= H or loc[1] >= W:
                    continue
                # Create `CoefficientTreeNode` object for available children
                node = CoefficientTreeNode(coeffs[lev][quad][loc[0]][loc[1]], lev, loc, quad)
                # Find children of current available children
                node.children = build_child_tree(lev + 1, loc, quad)
                # Add current available children into the list `children`
                children.append(node)
            return children

        LL = coeffs[0]  # coefficients of low frequency sub-band
        trees = []   # forest
        (H, W) = LL.shape  # the shape of LL

        # travel around all points in LL
        for i in range(H):
            for j in range(W):
                children = [CoefficientTreeNode(subband[i][j], 1, (i, j), quad, children=build_child_tree(2, (i, j), quad)) for quad, subband in enumerate(coeffs[1])]
                trees.append(CoefficientTreeNode(LL[i, j], 0, (i, j), None, children=children))
        return trees


class ZeroTreeEncoder:
    def __init__(self, coeffs) -> None:
        """A class is used to travel all nodes in Coefficient Tree Nodes in a certain way.

        Args:
            coeffs: A list containing coefficient matrices corresponding to sub-band areas.

        """
        self.trees = CoefficientTreeNode.build_trees(coeffs)  # A list of ` CoefficientTree`s.

    def travel(self) -> Tuple[List[str], List[Mapping[str, int]], List[int]]:
        """Travel all nodes.

        Returns:
            A tuple (sym_seq, sym_freq, nonzero_lis), where sym_seq is a list containing all marks, sym_freq is a list containing all mappings from mark to the frequency of mark, and nonzero_lis is a list containing nonzero integers.

        """
        nonzero_lis = []    # Non-zero list
        sym_seq = []    # Symbol sequence
        q = deque()  # Queue

        for parent in self.trees:
            parent.zero_code()
            q.append(parent)

        # BFS travel
        while len(q) != 0:
            node = q.popleft()

            if node.code != "T":
                for child in node.children:
                    q.append(child)

            if node.code == "N":
                if node.quad != None:
                    # add value of node whose symbol is 'N'
                    nonzero_lis.append(node.val)
                node.val = 0

            # add node.code
            sym_seq.append(node.code)

        counter = Counter(sym_seq)
        syms = sorted(counter.keys(), key=lambda x: x)
        freq = []
        for sym in syms:
            freq.append(counter.get(sym))
        sym_freq = list(zip(syms, freq))
        nonzero_lis = np.array(nonzero_lis)

        return sym_seq, sym_freq, nonzero_lis


class ZeroTreeDecoder:
    """A class used to reconstruct zero tree.

    Attributes:
        coeffs: A list containing coefficient matrices corresponding to sub-band areas.
        trees: A list of objects of `CoefficientTreeNode`.

    """
    def __init__(self, coeffs: List[List[np.ndarray]]):
        """A class is used to refill values of all nodes by using non-zero list in a certain way.

        Args:
            coeffs: A list containing coefficient matrices corresponding to sub-band areas.

        """
        self.coeffs = coeffs
        self.trees = CoefficientTreeNode.build_trees(self.coeffs)

    def refill(self,
               code_list: List[str],
               non_zeros_ls: List[int]):
        """Refill values of all nodes.

        Args:
            code_list: A list of marks.
            non_zeros_ls: A list of nonzero numbers.

        """
        q = deque()
        for parent in self.trees:
            q.append(parent)

        for code in code_list:
            if len(q) == 0:
                break
            node = q.popleft()
            # If mark is not "T"
            if code != "T":
                for child in node.children:
                    q.append(child)
            # If mark is "N" and quadrant of current node is not None
            if code == "N" and node.quad != None:
                node.val = non_zeros_ls.popleft()
                self.coeffs[node.lev][node.quad][node.loc] = node.val
