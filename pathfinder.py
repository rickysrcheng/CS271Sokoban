from constants import *
import numpy as np
#I used https://levelup.gitconnected.com/solve-a-maze-with-python-e9f0580979a1
#Returns [-1] if there is no path. Returns [] if already at correct location.
def shortest_path_actions(board, start_position, end_position, box_positions):
    board_copy = np.array(board).copy()
    board_copy[board_copy == DEADLOCK] = 0
    board_copy[board_copy == FLOOR] = 0
    board_copy[board_copy == WALL] = 1
    #Box positions need to be considered as locations that are blocked off, because one box can block another box.
    for box in box_positions:
        board_copy[box[0], box[1]] = 1
    start = start_position[0], start_position[1]
    end = end_position[0], end_position[1]
    m = []
    for i in range(len(board_copy)):
        m.append([])
        for j in range(len(board_copy[i])):
            m[-1].append(0)
    i, j = start
    m[i][j] = 1

    k = 0
    actions = []  # takes actions 0, 1, 2, 3
    step = True
    while m[end[0]][end[1]] == 0:
        k += 1
        step = make_step(k, m, board_copy)
        if not step:
            return [-1]
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

def make_step(k, m, board_copy):
    count = 0
    for i in range(len(m)):
        for j in range(len(m[i])):
            if m[i][j] == k:
                count += 1
                if i > 0 and m[i - 1][j] == 0 and board_copy[i - 1][j] == 0:
                    m[i - 1][j] = k + 1
                if j > 0 and m[i][j - 1] == 0 and board_copy[i][j - 1] == 0:
                    m[i][j - 1] = k + 1
                if i < len(m) - 1 and m[i + 1][j] == 0 and board_copy[i + 1][j] == 0:
                    m[i + 1][j] = k + 1
                if j < len(m[i]) - 1 and m[i][j + 1] == 0 and board_copy[i][j + 1] == 0:
                    m[i][j + 1] = k + 1
    return count != 0

#A few test cases
#bad_board = [[-1001, -1001, -1001, -1001], [-1001, -1, -1001, -1001], [-1001, -1001, -1, -1001], [-1001, -1001, -1001, -1001]]
#good_board = [[-1001, -1001, -1001, -1001], [-1001, -1, -1001, -1001], [-1001, -1, -1, -1001], [-1001, -1001, -1001, -1001]]
#actions = shortest_path_actions(bad_board, [1, 1], [1, 2], [])
#print(actions)
#actions = shortest_path_actions(bad_board, [1, 1], [2, 2])
#print(actions)
#
# test_board = [[-1001, -1001, -1001, -1001, -1001, -1001, -1001, -1001],
#  [-1001,    -1, -1000, -1001, -1000,    -1, -1000, -1001],
#  [-1001,    -1,    -1,    -1,    -1,    -1, -1000, -1001],
#  [-1001, -1000,    -1, -1000, -1001,    -1, -1001, -1001],
#  [-1001, -1001,    -1, -1001, -1000,    -1,    -1, -1001],
#  [-1001, -1000,    -1,    -1,    -1,    -1,    -1, -1001],
#  [-1001, -1000,    -1,    -1, -1001, -1000, -1000, -1001],
#  [-1001, -1001, -1001, -1001, -1001, -1001, -1001, -1001]]
#                                         #end position, start position
# print(shortest_path_actions(test_board, [5, 5], [6, 6], [[2, 3], [4, 5], [5, 4]]))