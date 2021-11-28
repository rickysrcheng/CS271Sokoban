from constants import *
import numpy as np
#I used https://levelup.gitconnected.com/solve-a-maze-with-python-e9f0580979a1
#Returns [] if there is no path.
def shortest_path_actions(board, end_position, start_position):
    board = np.array(board)
    board[board == DEADLOCK] = 0
    board[board == FLOOR] = 0
    board[board == WALL] = 1
    print(board)
    start = start_position[0], start_position[1]
    end = end_position[0], end_position[1]
    m = []
    for i in range(len(board)):
        m.append([])
        for j in range(len(board[i])):
            m[-1].append(0)
    i, j = start
    m[i][j] = 1

    k = 0
    actions = []  # takes actions 0, 1, 2, 3
    step = True
    while m[end[0]][end[1]] == 0:
        k += 1
        step = make_step(k, m, board)
        if not step:
            return actions
    i, j = end
    k = m[i][j]
    the_path = [(i, j)]

    while k > 1:
        if i > 0 and m[i - 1][j] == k - 1:
            i, j = i - 1, j
            the_path.append((i, j))
            actions.append(0)
            k -= 1
        elif j > 0 and m[i][j - 1] == k - 1:
            i, j = i, j - 1
            the_path.append((i, j))
            actions.append(2)
            k -= 1
        elif i < len(m) - 1 and m[i + 1][j] == k - 1:
            i, j = i + 1, j
            the_path.append((i, j))
            actions.append(1)
            k -= 1
        elif j < len(m[i]) - 1 and m[i][j + 1] == k - 1:
            i, j = i, j + 1
            the_path.append((i, j))
            actions.append(3)
            k -= 1
    return actions

def make_step(k, m, board):
    count = 0
    for i in range(len(m)):
        for j in range(len(m[i])):
            if m[i][j] == k:
                count += 1
                if i > 0 and m[i - 1][j] == 0 and board[i - 1][j] == 0:
                    m[i - 1][j] = k + 1
                if j > 0 and m[i][j - 1] == 0 and board[i][j - 1] == 0:
                    m[i][j - 1] = k + 1
                if i < len(m) - 1 and m[i + 1][j] == 0 and board[i + 1][j] == 0:
                    m[i + 1][j] = k + 1
                if j < len(m[i]) - 1 and m[i][j + 1] == 0 and board[i][j + 1] == 0:
                    m[i][j + 1] = k + 1
    return count != 0

#A few test cases
#bad_board = [[-1001, -1001, -1001, -1001], [-1001, -1, -1001, -1001], [-1001, -1001, -1, -1001], [-1001, -1001, -1001, -1001]]
#good_board = [[-1001, -1001, -1001, -1001], [-1001, -1, -1001, -1001], [-1001, -1, -1, -1001], [-1001, -1001, -1001, -1001]]
#actions = shortest_path_actions(bad_board, [1, 1], [2, 2])
#print(actions)
#shortest_path_actions(bad_board, [1, 1], [2, 2])