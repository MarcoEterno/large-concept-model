# To better draw the parallelism with Alphazero, it is useful to see every set of tokens as a node,
# analogously to a chess position

import numpy as np


class Node:
    def __init__(self, latest_token: int, latest_token_probability: float, parent: ['Node'] = None):
        """
        Args:
            latest_token (int): the token that was just added to the sentence
            latest_token_probability (float): the probability of the latest token
            parent (Node): the parent node
        """
        # latest token is an int for now but other formats might be more convenient
        self.parent = parent
        self.full_sentence = (parent.full_sentence if parent is not None else []) + [latest_token]
        self.latest_token = latest_token
        self.latest_token_probability = latest_token_probability
        self.visits = 0
        self.value = self.calculate_value()
        self.children = {}  # children have tokens as keys and nodes as values
        self.ucb_children_ranking = []  # children sorted by ucb value in a list of nodes

    def calculate_value(self):
        """
        Calculate the value of the node with the perplexity
        :return: value of the node
        """
        return ((0 if self.parent is None
                 else self.parent.value)
                - self.latest_token_probability * np.log(self.latest_token_probability))

    @property
    def is_terminal(self):
        return len(self.children) == 0

    @property
    def ucb_value(self):
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return self.value
        else:
            return self.value + np.sqrt(2 * np.log(self.parent.visits) / self.visits)


    def add_child(self, token, probability):
        self.children[token] = Node(
            latest_token=token,
            latest_token_probability=probability,
            parent=self
        )
        # this child will have infinite ucb value, so we need to make it first in the ucb_value ranking
        self.ucb_children_ranking = [self.children[token]] + self.ucb_children_ranking

    def calculate_ucb_children_ranking(self):
        """
        In case there is doubt on ucb ranking this function resorts the children,
        at the expense of listing them all and sort them in time O(n log n) for n = number of children.
        """
        self.ucb_children_ranking = sorted(self.children.values(), key=lambda x: x.ucb_value, reverse=True)

    def get_best_ucb_child(self):
        """
        Get the child with the highest ucb value
        :return: the child with the highest value
        """
        return self.ucb_children_ranking[0]



if __name__ == '__main__':
    root = Node(0, 1)
    child = Node(1, 1, root)
    print(child.full_sentence)
    print(child.value)
