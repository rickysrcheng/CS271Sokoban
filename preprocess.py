import numpy as np
from constants import *
def preprocess(rows, columns, walls, boxes, goal):
    board = np.full((rows, columns), FLOOR)
    corners = []

    for r in range(1, rows + 1):
        for c in range(1, columns + 1):
            if([r,c] in walls):
                board[r - 1, c - 1] = WALL
            # elif([r,c] in goal):              # Removed goal reward on board, so agent doesn't get reward if it lands on goal position
            #     board[r - 1, c - 1] = GOAL
            #check if non-goal corners
            elif [r,c] not in goal and [r + 1, c] in walls and [r, c + 1] in walls:
                board[r-1, c-1] = DEADLOCK
                corners.append([r-1,c-1])
            elif [r,c] not in goal and [r - 1, c] in walls and [r, c + 1] in walls:
                board[r-1, c-1] = DEADLOCK
                corners.append([r-1,c-1])
            elif [r,c] not in goal and [r - 1, c] in walls and [r, c - 1] in walls:
                board[r-1, c-1] = DEADLOCK
                corners.append([r-1,c-1])
            elif [r,c] not in goal and [r, c - 1] in walls and [r + 1, c] in walls:
                board[r-1, c-1] = DEADLOCK
                corners.append([r-1,c-1])

    # Correct box and storage indices
    for i in range(len(boxes)):
        for j in range(len(boxes[0])):
            boxes[i][j] -= 1
            goal[i][j] -= 1

    for i in range(len(corners)):
        corner1 = corners[i]
        if corner1 not in goal:
            for j in range(i+1, len(corners)):
                corner2 = corners[j]
                if corner1[0] == corner2[0]:
                    r = corner1[0]
                    minC = min(corner1[1], corner2[1])
                    maxC = max(corner1[1], corner2[1])

                    rowSet = list(set(board[r, minC+1:maxC]))
                    aboveSet = list(set(board[r-1, minC+1:maxC]))
                    belowSet = list(set(board[r+1, minC+1:maxC]))

                    for c in range(minC+1,maxC):
                        if [r,c] in goal:
                            break
                    else:
                        if len(rowSet) == 1 and rowSet[0] == FLOOR:
                            if (len(aboveSet) == 1 and aboveSet[0] == WALL) or (len(belowSet) == 1 and belowSet[0] == WALL):
                                board[r, minC+1:maxC] = DEADLOCK

                if corner1[1] == corner2[1]:
                    c = corner1[1]
                    minR = min(corner1[0], corner2[0])
                    maxR = max(corner1[0], corner2[0])

                    colSet = list(set(board[minR+1:maxR, c]))
                    leftSet = list(set(board[minR+1:maxR, c-1]))
                    rightSet = list(set(board[minR+1:maxR, c+1]))

                    for r in range(minR+1, maxR):
                        if [r,c] in goal:
                            break
                    else:
                        if len(colSet) == 1 and colSet[0] == FLOOR:
                            if (len(leftSet) == 1 and leftSet[0] == WALL) or (len(rightSet) == 1 and rightSet[0] == WALL):
                                board[minR+1:maxR, c] = DEADLOCK


    return board

