import queue
from constants import *
import sys
from collections import deque

#taken from https://www.techiedelight.com/lee-algorithm-shortest-path-in-a-maze/
# Below lists detail all four possible movements from a cell
row = [-1, 0, 0, 1]
col = [0, -1, 1, 0]


# Function to check if it is possible to go to position (row, col)
# from the current position. The function returns false if row, col
# is not a valid position or has a value 0 or already visited.
def isValid(mat, visited, row, col):
    return (row >= 0) and (row < len(mat)) and (col >= 0) and (col < len(mat[0])) and (mat[row][col] == -1 or mat[row][col] == -1000) and not visited[row][col]

def findShortestPathLength(mat, src, dest):
    i, j = src
    x, y = dest
    if not mat or len(mat) == 0 or mat[i][j] == 0 or mat[x][y] == 0:
        return -1
    (M, N) = (len(mat), len(mat[0]))
    visited = [[False for x in range(N)] for y in range(M)]
    q = deque()
    visited[i][j] = True
    q.append((i, j, 0))
    min_dist = sys.maxsize

    while q:
        (i, j, dist) = q.popleft()
        if i == x and j == y:
            min_dist = dist
            break

        for k in range(4):
            if isValid(mat, visited, i + row[k], j + col[k]):
                visited[i + row[k]][j + col[k]] = True
                q.append((i + row[k], j + col[k], dist + 1))

    if min_dist != sys.maxsize:
        return min_dist
    else:
        return -1


if __name__ == '__main__':

    mat = [[-1001, -1001, -1001, -1001, -1001, -1001, -1001, -1001],
             [-1001, -1000, -1000, -1001, -1000, -1, -1000, -1001],
             [-1001, -1, -1, -1, -1, -1, -1000, -1001],
             [-1001, -1000, -1, -1000, -1001, -1, -1001, -1001],
             [-1001, -1001, -1, -1001, -1000, -1, -1000, -1001],
             [-1001, -1000, -1, -1, -1, -1, -1, -1001],
             [-1001, -1000, -1, -1000, -1001, -1000, -1000, -1001],
             [-1001, -1001, -1001, -1001, -1001, -1001, -1001, -1001]]

    src = (6, 6)
    dest = (1, 1)

    min_dist = findShortestPathLength(mat, src, dest)

    if min_dist != -1:
        print("The shortest path from source to destination has length", min_dist)
    else:
        print("Destination cannot be reached from source")
