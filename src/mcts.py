from typing import Optional

from src.node import Node
from src.llm import get_GPT2_probabilities

class MCTS:
    def __init__(self, root: Optional[Node] = None, num_inferences: int = 1000, max_timeout: int = 100):
        """
        Args:
            root (Node): the root node of the tree
            num_inferences (int): the number of inferences to make
            max_timeout (int): the maximum number of timeouts to allow
        """
        self.root = root
        self.num_inferences = num_inferences
        self.max_timeout = max_timeout

    def search_leaf(self, node):
        """
        Searches the tree for the best leaf node to expand
        Args:
            node (Node): the node to start the search from
        Returns:
            Node: the best leaf node
        """
        if len(node.children) == 0:
            return node
        else:
            best_ucb_child = node.ucb_children_ranking[0]  # this operation leaves the list unchanged
            return self.search_leaf(best_ucb_child)

    # expands a node by doing LLM inference on it
    def expand(self, node, return_type: str, ):
        """
        Expands the node by adding all possible children
        Args:
            node (Node): the node to expand
        """
        if not node.is_terminal:
            print("A request was made to expand a non terminal node. Returning without doing anything.")
            return
        probabilities = get_GPT2_probabilities(node.full_sentence, return_type, )
        for token, probability in probabilities.items():
            node.add_child(token, probability)

    # instead of rollouts, we will be using inferences
    def rollout(self, node):
        """
        Does an LLM inference for the whole sentence  from the current node and returns the resulting value
        Args:
            node (Node): the node to simulate from
        Returns:
            float: the value of the node
        """
        while not node.is_terminal:
            node.calculate_ucb_children_ranking()
            best_ucb_child = node.ucb_children_ranking[0]
            node = best_ucb_child
        return node.value




