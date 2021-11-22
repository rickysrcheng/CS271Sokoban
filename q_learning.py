import numpy as np
from constants import *
# def reward(rows, columns, board):
#     #initialize q-table. key = state (a coordinate pair), value = list of 4 actions and their rewards, in order L, R, U, D.
#     reward = {}
#     for r in range(rows):
#         for c in range(columns):
#             left = check_left_reward([r,c], board)
#             right = check_right_reward([r,c], board, columns)
#             up = check_up_reward([r, c], board)
#             down = check_down_reward([r, c], board, rows)
#             reward[tuple([r,c])] = [left, right, up, down] #can't use list as key, hence the tuple
#     return reward

def q_learn(board, start_position, goal_position, rows, columns):
    cur_position = start_position
    q_vals = np.zeros((rows, columns, 4))
    #actions = [UP, DOWN, LEFT, RIGHT]

    for k in range(100):
        while not check_if_goal(cur_position, goal_position):
            action = epsilon_greedy_get_action(cur_position, EPSILON, q_vals)
            old_position = cur_position
            next_location(cur_position, action)
            reward = board[cur_position[0], cur_position[1]]
            print("Current position :", cur_position)
            print("reward", reward)
            old_q_val = q_vals[old_position[0], old_position[1], action]
            td = reward + (DISCOUNT * np.max(q_vals[cur_position[0], cur_position[1]])) - old_q_val
            print("TD :", td)
            new_q_val = old_q_val + (LEARN_RATE * td)
            print("New Q value :", new_q_val)
            print("")
            q_vals[old_position[0], old_position[1], action] = new_q_val
    print(q_vals)


def optimal_route(rows, columns, reward):
    return []
#new_q[cur_pos, action] = (1 - LEARN_RATE) * current_q + LEARN_RATE * (reward + DISCOUNT * max_future_q)

def check_if_goal(cur_position, goal_position):
    if cur_position == goal_position:
        return True
    return False

def epsilon_greedy_get_action(curr_position, epsilon, q_values):
    if np.random.random() < epsilon:
        #return argmax of q values for current position, will return an action
        return np.argmax(q_values[curr_position[0], curr_position[1]])
    #return np.random.choice([UP, DOWN, LEFT, RIGHT]) #returns a random number from 0 to 4, which will indicate which of the 4 directions
    return np.random.choice([0, 1, 2, 3])

def next_location(cur_position, action):
    if action == 0: #UP
        cur_position[0] += -1
    elif action == 1: #DOWN
        cur_position[0] += 1
    elif action == 2: #LEFT
        cur_position[1] += -1
    else: #current action = RIGHT
        cur_position[1] += 1

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
