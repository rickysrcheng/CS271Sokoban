import numpy as np
from constants import *
def preprocess(rows, columns, walls, boxes, goal):
    board = np.full((rows, columns), FLOOR)
    for r in range(1, rows + 1):
        for c in range(1, columns + 1):
            if([r,c] in walls):
                board[r - 1, c - 1] = BLOCKED
            elif([r,c] in goal):
                board[r - 1, c - 1] = GOAL
            #check if corners
            elif [r + 1, c] in walls and [r, c + 1] in walls:
                board[r-1, c-1] = BLOCKED
            elif [r - 1, c] in walls and [r, c + 1] in walls:
                board[r-1, c-1] = BLOCKED
            elif [r - 1, c] in walls and [r, c - 1] in walls:
                board[r-1, c-1] = BLOCKED
            elif [r, c - 1] in walls and [r + 1, c] in walls:
                board[r-1, c-1] = BLOCKED
    return board

