import logging
from typing import Optional

import numpy as np

from tictactoe import TicTacToe


class Node:
    def __init__(self,
                 game_state: Optional['TicTacToe'] = None,
                 parent: Optional['Node'] = None):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.wins = 0
        self.game_state = game_state if game_state is not None else self.construct_game_state()

    def construct_game_state(self):
        if self.parent is None:
            return TicTacToe()
        # find the key corresponding to the children value that is equal to the current node
        for move, child in self.parent.children.items():
            if child == self:
                new_game_state = self.parent.game_state.get_updated_game_state(move)
                return new_game_state

    @property
    def last_move(self):
        return self.game_state.get_last_move()

    def add_child_given_move(self, move: int):
        child = Node(game_state=self.game_state.get_updated_game_state(move), parent=self)
        self.children[move] = child

    def update(self, result):
        self.visits += 1
        self.wins += result

    @property
    def ucb(self):
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return self.wins / self.visits
        else:
            return self.wins / self.visits + 2 * np.sqrt(np.log(self.parent.visits) / self.visits)

    @property
    def average_wins(self):
        return self.wins / self.visits if self.visits > 0 else 0

    def get_best_move_from_possible_children(self):
        """
        Returns the move that leads to the child with the highest UCB value
        :return: the best move or None if there are no possible moves
        """
        # handle the case where there are no possible moves
        self.add_all_children()
        if len(self.children) == 0:
            logging.info("No possible moves were found")
            return None
        # find the key corresponding to the highest ucb value in children
        best_move = max(self.children, key=lambda x: self.children[x].ucb)
        if best_move not in self.game_state.get_possible_moves():
            logging.info("Best move is not in possible moves")
            return None
        return best_move

    @property
    def best_child(self):
        """
        Returns the child with the highest UCB value from all the possible children
        if there are no children, return None
        :return: the child with the highest UCB value
        """
        self.add_all_children()
        if len(self.children) > 0:
            return max(self.children.values(), key=lambda x: x.ucb)
        else:
            logging.info("No children found")
            return None

    def add_all_children(self):
        for move in self.game_state.get_possible_moves():
            if move not in self.children.keys():
                self.add_child_given_move(move)

    def __repr__(self):
        return (f"State: {self.game_state.game_history}, "
                f"Visits: {self.visits},"
                f" Wins: {self.wins}, "
                f"UCB: {round(self.ucb, 3)}")

    def __getstate__(self):
        # Capture what is normally pickled
        state = self.__dict__.copy()
        # You can add custom handling here if needed
        return state

    def __setstate__(self, state):
        # Restore state
        self.__dict__.update(state)
        # Ensure all attributes are initialized properly


if __name__ == "__main__":
    game = TicTacToe()
    node = Node(game)
    node.update(1)
    node.add_all_children()
    print(node.children[4])
    node.children[4].update(-1)
    print(node.children[4])
    print(node.get_best_move_from_possible_children())

    for move, child in node.children.items():
        child.update(1)
    print(node.get_best_move_from_possible_children())

    node.children[2].add_child_given_move(3)
    node.children[2].children[3].game_state.print_board()
    print(node.children[2].children[3].game_state.game_history)
