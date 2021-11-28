import numpy as np
from constants import *
def preprocess(rows, columns, walls, boxes, goal):
    board = np.full((rows, columns), FLOOR)
    for r in range(1, rows + 1):
        for c in range(1, columns + 1):
            if([r,c] in walls):
                board[r - 1, c - 1] = WALL
            # elif([r,c] in goal):              # Removed goal reward on board, so agent doesn't get reward if it lands on goal position
            #     board[r - 1, c - 1] = GOAL
            #check if non-goal corners
            elif [r,c] not in goal and [r + 1, c] in walls and [r, c + 1] in walls:
                board[r-1, c-1] = DEADLOCK
            elif [r,c] not in goal and [r - 1, c] in walls and [r, c + 1] in walls:
                board[r-1, c-1] = DEADLOCK
            elif [r,c] not in goal and [r - 1, c] in walls and [r, c - 1] in walls:
                board[r-1, c-1] = DEADLOCK
            elif [r,c] not in goal and [r, c - 1] in walls and [r + 1, c] in walls:
                board[r-1, c-1] = DEADLOCK

    # Correct box and storage indices
    for i in range(len(boxes)):
        for j in range(len(boxes[0])):
            boxes[i][j] -= 1
            goal[i][j] -= 1

    return board

