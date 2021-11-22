import numpy as np
from constants import *

def q_learn(board, start_position, goal_position, rows, columns):
    cur_position = start_position
    q_vals = np.zeros((rows, columns, 4))
    print(cur_position)
    for k in range(100):
        while not goal_found(cur_position, goal_position):
            action = epsilon_greedy_get_action(cur_position, EPSILON, q_vals, rows, columns)
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

#TODO
def optimal_route(rows, columns, reward):
    return []

def goal_found(cur_position, goal_position):
    if cur_position == goal_position:
        return True
    return False

#TODO: implement checks so it can't hit walls. Maybe change WALL constant to None and check for "None" or something similar.
def epsilon_greedy_get_action(curr_position, epsilon, q_values, rows, columns):
    if np.random.random() < epsilon: #return random from 0 to 1
        #return argmax of q values for current position, will return an action
        print(q_values[curr_position[0], curr_position[1]])
        print(np.argmax(q_values[curr_position[0], curr_position[1]]))
        return np.argmax(q_values[curr_position[0], curr_position[1]])
    actions = []     #need to check if valid first
    if check_left_valid(curr_position):
        actions.append(2)
    if check_right_valid(curr_position, columns):
        actions.append(3)
    if check_down_valid(curr_position, rows):
        actions.append(1)
    if check_up_valid(curr_position):
        actions.append(0)
    return np.random.choice(actions)


def next_location(cur_position, action):
    if action == 0: #UP
        cur_position[0] += -1
    elif action == 1: #DOWN
        cur_position[0] += 1
    elif action == 2: #LEFT
        cur_position[1] += -1
    else: #current action = RIGHT
        cur_position[1] += 1

def check_left_valid(cur_position):
    column = cur_position[1]
    if column != 0: #if column value is not already at the very left
        return True
    return False

def check_right_valid(cur_position,num_columns):
    column = cur_position[1]
    if (column != num_columns - 1):  # if column value is not already at the very right
        return True
    return False

def check_up_valid(cur_position):
    row = cur_position[0]
    if (row != 0):  # if row value is not already at the very top
        return True
    return False

def check_down_valid(cur_position, num_rows):
    row = cur_position[0]
    if (row != num_rows - 1):  # if row value is not already at the very bottom
        return True
    return False
