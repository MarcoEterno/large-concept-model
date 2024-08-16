import copy
import logging
from typing import Optional

import numpy as np

from tictactoe import TicTacToe
from treenode import Node
from utils import timer


class MCTS:
    def __init__(self, game: TicTacToe, num_simulations=1000, root: Optional[Node] = None):
        """
        Initializes the MCTS algorithm
        :param game: the game to play
        :param num_simulations: the number of simulations an agent will do before considering a move
        """
        self.game = game
        self.num_simulations = num_simulations
        self.root = Node(game) if root is None else root

    def search_leaf(self, node):
        """Searches the tree for the best leaf node to expand
        :param node: the node to start the search from
        :return: the best leaf node
        """
        if len(node.children) == 0:
            return node
        else:
            best_move = node.get_best_move_from_possible_children()  # function already checks if the move is possible

            if best_move is not None:
                return self.search_leaf(node.children[best_move])
            return node

    def rollout(self, node):
        """Simulates a random game from the current node to the end and returns the winner
        :param node: the node to simulate from
        """
        if node is None:
            logging.warning("Node is None in rollout")
            return 0
        game = copy.deepcopy(node.game_state)
        while not game.is_over:
            moves = game.get_possible_moves()
            move = np.random.choice(moves)
            game.make_move(move)
        return game.return_winner()

    def backpropagate(self, node, result):
        """Backpropagates the result of a simulation to the root nodes
        :param node: the node to start the backpropagation from
        :param result: the result of the simulation
        """
        while node != None:
            node.visits += 1
            node.wins += result
            node = node.parent
            result *= -1  # a win for a player in a given node is a loss for the player in the parent node

    def do_one_step(self, node: Optional[Node] = None):
        """
        Does one step of the MCTS algorithm
        """
        node = self.root if node is None else node
        leaf = self.search_leaf(node)
        if leaf.visits > 0:
            leaf.add_all_children()
            if leaf.best_child:
                result = self.rollout(leaf.best_child)
                self.backpropagate(leaf.best_child, result)
        if leaf.visits == 0:
            result = self.rollout(leaf)
            self.backpropagate(leaf, result)

    @timer
    def build_mcts_tree(self, node: Optional[Node] = None, print_progress: bool = False):
        """
        Builds the MCTS tree until the number of simulations is reached
        :param node: the node to start the tree building from
        """
        node = self.root if node is None else node
        while node.visits < self.num_simulations:
            self.do_one_step(node)
            if print_progress:
                # print a progress bar of the tree building
                print(f"\r{node.visits}/{self.num_simulations}", end="")


    def print_tree(self, node: Optional[Node] = None):
        """Prints the tree of moves played by the MCTS algorithm"""

        def print_node(node, indent):
            if node is None:
                return
            if node.visits != 0:
                print(f"{' ' * indent} State: {node.game_state.game_history} Visits: {node.visits}, Wins: {node.wins}")
            for child in node.children.values():
                print_node(child, indent + 4)

        node = self.root if node is None else node
        print_node(node, 0)

    def find_best_move_with_mcts(self, node: Node, print_tree: bool = False):
        """
        Finds the move with highest win rate using the MCTS algorithm
        :param node: the node to start the search from
        :return: the best move
        """
        self.build_mcts_tree(node)
        if print_tree:
            self.print_tree(node)
        best_move = max(node.children, key=lambda x: node.children[x].average_wins)
        return best_move

    def __getstate__(self):
        # Return a dictionary representing the object's state
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore the object's state from the dictionary
        self.__dict__.update(state)


if __name__ == "__main__":
    from saver import save_mcts, load_mcts
    mcts = MCTS(TicTacToe(), num_simulations=1000000)
    mcts = load_mcts(mcts)
    print(type(mcts))
    print(mcts.find_best_move_with_mcts(mcts.root, print_tree=False))
    print(mcts.root.children)
    save_mcts(mcts)
    mcts2 = MCTS(TicTacToe(), num_simulations=1000000)
    mcts2 = load_mcts(mcts2)
    print(mcts2.root.children)

    """
    print(mcts.root.children)
    mcts2 = MCTS(TicTacToe())
    mcts2.load_mcts(int(mcts.num_simulations))
    print(type(mcts2))
    print(mcts2.root.children)
    print(sum([child.visits for child in mcts2.root.children.values()]))
    game = TicTacToe()"""
