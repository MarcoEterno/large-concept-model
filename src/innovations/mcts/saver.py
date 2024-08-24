import math
import pickle

from src.innovations.mcts.mcts import MCTS
from utils import timer

"""
This module contains functions to save and load the MCTS tree
Functions were not placed in the MCTS class due to picle not being able 
to serialize the MCTS class since it contains a reference to the game object
"""


def save_mcts(mcts):
    """Saves the mcts to a file"""
    try:
        with open(f'mcts_10^{round(math.log10(mcts.num_simulations), 2)}', 'wb') as f:
            pickle.dump(mcts, f)
    except Exception as e:
        print(f"Error saving MCTS tree: {e}")
        exit(1)


@timer
def load_mcts(mcts, num_simulations=None):
    """Loads the mcts from a file
    :param num_simulations: the number of simulations of the mcts we want to load"""
    num_simulations = int(num_simulations) if num_simulations is not None else mcts.num_simulations
    filename = f'mcts_10^{round(math.log10(num_simulations), 2)}'
    try:
        with open(filename, 'rb') as f:
            mcts = pickle.load(f)
            print(mcts.__dict__)
            return mcts

    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading MCTS tree: {e}")
        print("Do you want to build a new tree? (y/n): ")
        answer = input()
        if answer == "y":
            mcts = MCTS(TicTacToe(), num_simulations)
            mcts.build_mcts_tree(print_progress=True)
            return mcts

        else:
            # exit the program if the user does not want to build a new tree
            exit(0)
