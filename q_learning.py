import numpy as np

def q(rows, columns, board):
    DISCOUNT = 0.5
    LEARN_RATE = 0.6

    #initialize q-table. key = state (a coordinate pair), value = list of 4 actions and their rewards, in order U, D, L, R.
    q = {}
    for r in range(rows):
        for c in range(columns):
            left = check_left_reward([r,c], board)
            right = check_right_reward([r,c], board, columns)
            up = check_up_reward([r, c], board)
            down = check_down_reward([r, c], board, rows)
            q[tuple([r,c])] = [left, right, up, down] #can't use list as key, hence the tuple
    print(q)


    #new_q[cur_pos, action] = (1 - LEARN_RATE) * current_q + LEARN_RATE * (reward + DISCOUNT * max_future_q)

#TODO: make all of this one function that accepts a direction (instead of 4 functions for 4 directions)
def check_left_reward(cur_position, board):
    row = cur_position[0]
    column = cur_position[1]
    if(column != 0): #if column value is not already at the very left
        return board[row, column - 1]
    return None

def check_right_reward(cur_position, board, num_columns):
    row = cur_position[0]
    column = cur_position[1]
    if (column != num_columns - 1):  # if column value is not already at the very right
        return board[row, column + 1]
    return None

def check_up_reward(cur_position, board):
    row = cur_position[0]
    column = cur_position[1]
    if (row != 0):  # if row value is not already at the very top
        return board[row - 1, column]
    return None

def check_down_reward(cur_position, board, num_rows):
    row = cur_position[0]
    column = cur_position[1]
    if (row != num_rows - 1):  # if row value is not already at the very bottom
        return board[row + 1, column]
    return None
