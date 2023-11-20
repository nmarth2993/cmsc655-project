import numpy as np

# package python-chess
from chess.pgn import read_game


def validate_games():
    # simple method to validate assumptions of eval comments
    # each node is assumed to have a comment and comments should be eval scores only
    all_valid = True
    with open("annotated/db2kclean.pgn") as dbfile:
        game = read_game(dbfile)

        game_count = 1
        while game is not None:
            move_count = 1
            for node in game.mainline():
                try:
                    x = float(node.comment)
                except ValueError as e:
                    all_valid = False
                    print(f"error in game {game_count}, move {move_count}: {e}")

                # keep track of half-moves
                move_count += 0.5
            game = read_game(dbfile)
            game_count += 1

    if all_valid:
        print("all games valid")


# extract all evaluation scores from comments and return them in a list
def gen_eval_list():
    eval_list = []
    with open("annotated/db200.pgn") as dbfile:
        first_game = read_game(dbfile)

        for node in first_game.mainline():
            # print the current move and eval
            # the node's standard algebraic notation holds the current half-move
            # the node's comment holds the eval score
            print(node.san())
            print(node.comment)

            # add the evaluation to the eval list as a float to be plotted
            eval_list.append(float(node.comment))
    return eval_list


if __name__ == "__main__":
    print("validating games")
    validate_games()

    """
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
    """
