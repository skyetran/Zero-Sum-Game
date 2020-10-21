import numpy as np
from warnings import filterwarnings


# Code for the pivot method for solving games, as described in
# https://www.math.ucla.edu/~tom/Game_Theory/mat.pdf
# Page 41, Section 4.5 - "4.5 Description of the Pivot Method for Solving Games"

# In the author's words:
# "The following algorithm for solving finite games is essentially the simplex method for solving [them]"
# python version of https://www.math.ucla.edu/~tom/gamesolve.html


# avoid divide by zero warning printing to console
filterwarnings("ignore", category=RuntimeWarning)


# create the simplex tableau to perform the pivoting on
# returns the left column and top row of labels separately and the augmented simplex tableau
def create_pivot_tableau(matrix):

    # create left column and top row side, since can't have strings and numbers in same array
    row_labels = []
    col_labels = []

    # fill top and left sides with labels
    for i in range(matrix.shape[0]):
        row_labels.append(f"x{i}")
    for i in range(matrix.shape[1]):
        col_labels.append(f"y{i}")
    # rowtest = np.transpose(col_strats)

    # STEP 1
    # add a constant so the value of the game so the value of the game stays positive while pivoting
    # will subtract the same amount from the value later to obtain the real value of the game
    matrix = np.add(matrix, np.full(matrix.shape, 20))

    # STEP 2
    # augment the matrix with -1's on the bottom and 1's on the right to create the main tableau
    # set the bottom right corner empty
    augment_1 = np.hstack((matrix, np.ones((matrix.shape[0], 1))))
    simplex_tableau = np.vstack((augment_1, np.full((1, augment_1.shape[1]), -1)))
    simplex_tableau[-1][-1] = 0

    return [row_labels, col_labels, simplex_tableau]


# STEP 3
# selects a pivot based on the criteria from STEP 3
# returns tuple (the value of the pivot, the pivot-border ratio, and the coordinates of the pivot cell)
def select_pivot(tableau):
    # modify indices for simplicity (...?)
    m = tableau.shape[0] - 1
    n = tableau.shape[1] - 1

    # initialize pivot for comparison, should always be overwritten
    pivot = (tableau[0][0], 999, (0, 0))

    # for r in range(m):
    #     for c in range(n):
    #         current_cell = (tableau[r][c], tableau[r][n] / tableau[r][c], (r, c))
    #         if tableau[m][c] < 0 < tableau[r][c] and current_cell[1] < pivot[1]:
    #             pivot = current_cell

    for c in range(n):
        # Step 3 criteria a)
        if tableau[m][c] < 0:
            # checking for Step 3 criteria c)
            for r in range(m):
                current_cell = (tableau[r][c], tableau[r][n] / tableau[r][c], (r, c))
                # Step 3 criteria b) and c)
                if tableau[r, c] >= 0 and 0 <= current_cell[1] < pivot[1]:
                    pivot = current_cell
            break
    return pivot


# STEP 4 (and 5)
# performs one pivot on the tableau
# returns the new tableau after the pivot was performed; check return of create_simplex_tableau() for details
def execute_pivot(row_labels, col_labels, tableau):
    m = tableau.shape[0]-1
    n = tableau.shape[1]-1

    # Step 3
    # find and get pivot for current form of the tableau
    dump = select_pivot(tableau)
    pivot_val = dump[0]
    p, q = dump[2]

    # create new tableau to write to, to avoid overwriting values in the original tableau needed to perform the pivot
    new_tableau = np.ones(tableau.shape)

    # Step 4a
    for i in range(m+1):
        for j in range(n+1):
            if i != p and j != q:
                new_tableau[i, j] = tableau[i, j] - tableau[p, j] * tableau[i, q] / tableau[p, q]

    # Step 4b
    for col in range(n+1):
        if col != q:
            new_tableau[p, col] = tableau[p][col] / pivot_val

    # Step 4c
    for row in range(m+1):
        if row != p:
            new_tableau[row, q] = -tableau[row, q] / pivot_val

    # Step 4d
    new_tableau[p, q] = 1 / pivot_val

    # STEP 5
    temp = row_labels[p]
    row_labels[p] = col_labels[q]
    col_labels[q] = temp

    return row_labels, col_labels, new_tableau


# Combine all steps to fully solve a game matrix
# returns the strategies for the row player, column player, and the value of the game
def simplex(matrix):
    # STEP 1 & 2
    tableau = create_pivot_tableau(matrix)

    # STEP 3, 4, and 5
    tableau = execute_pivot(tableau[0], tableau[1], tableau[2])

    # setup for Step 6
    bottom_row = list(tableau[2][tableau[2].shape[0] - 1, :])
    V = bottom_row.pop(-1)

    # STEP 6
    while any(elem < 0 for elem in bottom_row):
        tableau = execute_pivot(tableau[0], tableau[1], tableau[2])
        bottom_row = list(tableau[2][tableau[2].shape[0] - 1, :])
        V = bottom_row.pop(-1)

    # setup for STEP 7
    left_side = tableau[0]
    top_side = tableau[1]
    x_strats = {}
    y_strats = {}


    # STEP 7

    for label in range(len(top_side)):
        # Step 7b
        if "x" in top_side[label]:
            index = int(top_side[label].split("x")[1]) + 1
            x_strats[f"row {index}"] = round(tableau[2][tableau[2].shape[0]-1, label] / V, 5)   # rounding here
        # Step 7c
        else:
            index = int(top_side[label].split("y")[1]) + 1
            y_strats[f"column {index}"] = 0

    for label in range(len(left_side)):
        # Step 7c
        if "y" in left_side[label]:
            index = int(left_side[label].split("y")[1]) + 1
            y_strats[f"column {index}"] = round(tableau[2][label, tableau[2].shape[1]-1] / V, 5)    # rounding here
        # Step 7b
        else:
            index = int(left_side[label].split("x")[1]) + 1
            x_strats[f"row {index}"] = 0

    # Sort and format results according to personal preference
    # x_strategy = np.reshape(np.array([value for (key, value) in sorted(x_strats.items())]), (len(x_strats), 1))
    # x_strategy = np.array([value for (key, value) in sorted(x_strats.items())])[:, np.newaxis]
    # y_strategy = np.atleast_2d(np.array([value for (key, value) in sorted(y_strats.items())])).T
    x_strategy = np.array([value for (key, value) in sorted(x_strats.items())]).tolist()
    y_strategy = np.array([value for (key, value) in sorted(y_strats.items())]).tolist()

    # Step 7a / undoing Step 1
    original_game_value = 1/V - 20

    return x_strategy, y_strategy, original_game_value


# testing stuff
# test = np.random.randint(-10, 10, size=(randint(2, 5), randint(2, 5)))
# test = np.array([[2, -1, 6], [0, 1, -1], [-2, 2, 1]])
# test = np.array([[-4, -2, -7], [-3, -10, -5], [-3, 4, -5]])
# test = np.array([[5, -1, -5, -5], [-6, 2, -9, -9], [5, 0, 2, 6]])
# s = simplex(test)
# print(test)
# print(f"\nX strategy:\n{s[0]}")
# print(f"\nY strategy:\n{s[1]}")
# print(f"\nV: {s[2]}")


# paramenter payoff_matrix is np.array object
def battle_solver(matrix):
    if matrix.shape[0] <= 2 or matrix.shape[1] <= 2:
        return matrix
    else:
        copy = matrix
        new_matrix = np.empty(matrix.shape)
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                temp = np.delete(copy, r, axis=0)
                temp = np.delete(temp, c, axis=1)
                new_matrix[r, c] = matrix[r, c] + simplex(battle_solver(temp))[2]
                copy = matrix
        return new_matrix

# return action of player 1 and 2
# the range is between the 1 and the column or row size of the current payoff matrix
# be careful of off-by-1 error
def get_action(max_range):
    p1 = int(input("Player 1 action: "))
    p2 = int(input("Player 2 action: "))
    # validate input
    while p1 < 1 or p1 > max_range or p2 < 1 or p2 > max_range:
        print("Input is out of range, please reenter your selection")
        p1 = int(input("Player 1 action: "))
        p2 = int(input("Player 2 action: "))
    return p1, p2

# 5x5 with each cell is a 4x4
def initialize_game():
    
    GAME = np.array([[[[-2, -5, -5, 4], [4, 5, -1, -1], [3, -2, 1, -4], [8, 4, -1, 0]], [[-5, 2, -7, 4], [-9, 5, -3, 7], [-4, -8, 9, -9], [0, -2, -7, -7]], [[-4, -7, -4, 5], [9, -1, -8, 5], [-2, 8, -7, -9], [3, 1, -7, 3]], [[0, -2, 8, -4], [4, 3, -7, -8], [-7, 1, -2, 9], [-5, 4, -6, 7]], [[0, -4, 2, -2], [3, -1, 5, 7], [-2, -4, 2, -4], [7, 1, -9, 1]]], [[[0, 3, -6, 1], [-4, 5, 6, 3], [-2, -3, 7, -5], [-5, 2, 5, -6]], [[-4, -2, 2, 0], [-9, 4, -1, 3], [-8, 7, 8, 8], [1, 2, 3, 5]], [[0, -6, 1, 3], [-4, 7, 5, -2], [6, 1, -5, 6], [-4, 4, -1, 8]], [[8, 6, 5, 6], [3, 4, 4, 2], [-7, 6, -9, -9], [-6, 2, -2, -2]], [[5, -7, 3, -8], [6, 9, -1, -2], [3, 7, -8, 5], [0, -5, 4, 0]]], [[[5, 3, 2, 4], [-9, 3, 1, 2], [1, -5, -7, -7], [-1, 4, 9, 3]], [[8, -3, 4, -7], [3, -2, 3, 2], [-8, 8, -5, 0], [-6, 4, 2, 1]], [[7, 8, 4, 2], [2, -8, -9, 9], [-7, -5, 8, 1], [2, -1, -7, -9]], [[6, 1, -6, -1], [3, -9, 1, -9], [-9, -4, 8, -5], [5, 7, 2, 9]], [[0, 8, -6, 9], [2, 8, 9, -3], [-2, 9, 8, 3], [1, 6, 3, 8]]], [[[-1, 2, -7, 5], [2, 3, 7, 9], [-5, 9, 6, -9], [3, 8, 1, 2]], [[1, 9, 5, -3], [1, -9, -7, -5], [0, -5, 4, -6], [6, 2, 6, 6]], [[6, -3, 4, 1], [1, 4, -7, -3], [-9, 9, -5, 3], [-6, -6, -1, -3]], [[4, -1, -5, 9], [-1, -3, 7, -2], [-4, -6, -1, 7], [-6, 5, -6, -5]], [[8, 7, 3, 3], [-2, 3, -1, 1], [1, 2, -6, -4], [-8, -5, -6, -1]]], [[[-7, -7, -5, -6], [9, 8, 5, 1], [-6, -2, -2, -1], [3, -8, -1, -5]], [[-2, -5, -1, -2], [-3, -4, -2, 6], [1, 5, -7, 5], [9, 2, -6, -3]], [[-7, -4, -6, 3], [-8, -8, 4, -8], [-9, 9, 7, 8], [-1, -5, 9, -8]], [[-1, 5, 4, 2], [3, 6, -2, 9], [-9, 7, 3, 4], [9, 4, -3, 8]], [[-9, -7, 0, -7], [5, -3, 3, -6], [5, -2, -6, -7], [-5, -5, 3, -8]]]])
    return GAME

# the game
def main():
    
    game = initialize_game()
    game_payoff = []
    
    # loop through gam
    for i in range(5):
        row_payoff = []
        for j in range(5):
            battle = game[i, j]
            row_payoff.append(round(simplex(battle_solver(battle))[2], 3))
        game_payoff.append(row_payoff)
    print(game_payoff)
    print(simplex(np.array(game_payoff)))

main()