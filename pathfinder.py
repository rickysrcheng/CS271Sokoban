import queue
from constants import *
import sys
from collections import deque
import numpy as np

a = [[-1001, -1001, -1001, -1001, -1001, -1001, -1001, -1001],
             [-1001, -1000, -1000, -1001, -1000, -1, -1000, -1001],
             [-1001, -1, -1, -1, -1, -1, -1000, -1001],
             [-1001, -1000, -1, -1000, -1001, -1, -1001, -1001],
             [-1001, -1001, -1, -1001, -1000, -1, -1000, -1001],
             [-1001, -1000, -1, -1, -1, -1, -1, -1001],
             [-1001, -1000, -1, -1000, -1001, -1000, -1000, -1001],
             [-1001, -1001, -1001, -1001, -1001, -1001, -1001, -1001]]
a = np.array(a)
a[a == -1000] = 0
a[a == -1] = 0
a[a == -1001] = 1
start = 1,1
end = 6,6

def make_step(k):
  for i in range(len(m)):
    for j in range(len(m[i])):
      if m[i][j] == k:
        if i>0 and m[i-1][j] == 0 and a[i-1][j] == 0:
          m[i-1][j] = k + 1
        if j>0 and m[i][j-1] == 0 and a[i][j-1] == 0:
          m[i][j-1] = k + 1
        if i<len(m)-1 and m[i+1][j] == 0 and a[i+1][j] == 0:
          m[i+1][j] = k + 1
        if j<len(m[i])-1 and m[i][j+1] == 0 and a[i][j+1] == 0:
           m[i][j+1] = k + 1

def print_m(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            print( str(m[i][j]).ljust(2),end=' ')
        print()

m = []
for i in range(len(a)):
    m.append([])
    for j in range(len(a[i])):
        m[-1].append(0)
i,j = start
m[i][j] = 1

k = 0
while m[end[0]][end[1]] == 0:
    k += 1
    make_step(k)

i, j = end
k = m[i][j]
the_path = [(i,j)]
actions = [] #takes actions 0, 1, 2, 3
while k > 1:
  if i > 0 and m[i - 1][j] == k-1:
    i, j = i-1, j
    the_path.append((i, j))
    actions.append(0)
    k-=1
  elif j > 0 and m[i][j - 1] == k-1:
    i, j = i, j-1
    the_path.append((i, j))
    actions.append(2)
    k-=1
  elif i < len(m) - 1 and m[i + 1][j] == k-1:
    i, j = i+1, j
    the_path.append((i, j))
    actions.append(1)
    k-=1
  elif j < len(m[i]) - 1 and m[i][j + 1] == k-1:
    i, j = i, j+1
    the_path.append((i, j))
    actions.append(3)
    k -= 1

print(actions)
