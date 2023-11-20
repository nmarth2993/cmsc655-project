import numpy as np

# package python-chess
from chess.pgn import read_game

def gen_eval_list():
    eval_list = []
    with open("databases/pgntest3.pgn") as dbfile:
        first_game = read_game(dbfile)

        # print(first_game)

        # board = second_game.board()
        for node in first_game.mainline():
            # printthe current move and eval (evals are stored in comments)
            print(node.san())
            print(node.comment)

            # add the evaluation to the eval list as a float to be plotted
            eval_list.append(float(node.comment))
    return eval_list

# fitting a line of best fit to a single game's eval scores
from least_mean_squares import plot_fit, lms, FIT_DEGREE

evals = gen_eval_list()

# create a "time-series" of moves (one move per eval annotation)
# there must be len(evals) number of points to match x and y dimensions
move_counts = np.linspace(1, len(evals), len(evals))

# this is currently only a single dataset, but will need to support multiple games in the future
dataset = list(zip(move_counts, evals))

# compute the LMS weights and plot the fit against the dataset
weight_vector = lms(dataset, FIT_DEGREE)
plot_fit(0, len(evals), dataset, weight_vector)