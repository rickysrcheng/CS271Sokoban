import numpy as np
from constants import *

def q_learn(board, start_position, goal_position, rows, columns):
    cur_position = start_position
    q_vals = np.zeros((rows, columns, 4))
    print(cur_position)
    for k in range(100):
        while not goal_found(cur_position, goal_position):
            action = epsilon_greedy_get_action(cur_position, EPSILON, q_vals, rows, columns, board)
            old_position = cur_position
            next_location(cur_position, action)
            reward = board[cur_position[0], cur_position[1]]
            print("Current position :", cur_position)
            #print("reward", reward)
            old_q_val = q_vals[old_position[0], old_position[1], action]
            td = reward + (DISCOUNT * np.max(q_vals[cur_position[0], cur_position[1]])) - old_q_val
            #print("TD :", td)
            new_q_val = old_q_val + (LEARN_RATE * td)
            #print("New Q value :", new_q_val)
            #print("")
            q_vals[old_position[0], old_position[1], action] = new_q_val
    print(q_vals)

#TODO
def optimal_route(rows, columns, reward):
    return []

def goal_found(cur_position, goal_position):
    if cur_position == goal_position:
        return True
    return False

def epsilon_greedy_get_action(curr_position, epsilon, q_values, rows, columns, board):
    actions = []  #making list of valid moves
    if check_up_valid(curr_position, board):
        actions.append(0)
    if check_down_valid(curr_position, board, rows):
        actions.append(1)
    if check_left_valid(curr_position, board):
        actions.append(2)
    if check_right_valid(curr_position, board, columns):
        actions.append(3)

    total_actions = [0, 1, 2, 3]
    exclude = [i for i in actions + total_actions if i not in actions or i not in total_actions]
    m = np.zeros(4, dtype=bool)
    m[exclude] = True

    if np.random.random() < epsilon: #return random from 0 to 1
        #return argmax of q values for current position, will return an action
        #only allow actions that were deemed allowed by the above validity checks
        q_values_masked = np.ma.array(q_values[curr_position[0], curr_position[1]], mask=m)
        return np.argmax(q_values_masked)

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

#Checks that you won't go off the left side of the map, and that you're not running into a wall.
def check_left_valid(cur_position, board):
    column = cur_position[1]
    check_inaccessible = board[cur_position[0], cur_position[1] - 1]
    if column != 0 and check_inaccessible != WALL:
        return True
    return False

#Checks that you won't go off right side of map, and that you're not running into a wall.
def check_right_valid(cur_position,board, num_columns):
    column = cur_position[1]
    check_inaccessible = board[cur_position[0], cur_position[1] + 1]
    if (column != num_columns - 1) and check_inaccessible != WALL:  # if column value is not already at the very right
        return True
    return False

#Checks that you won't go off the top side of the map, and that you're not running into a wall.
def check_up_valid(cur_position, board):
    row = cur_position[0]
    check_inaccessible = board[cur_position[0] - 1, cur_position[1]]
    if (row != 0) and check_inaccessible != WALL:  # if row value is not already at the very top
        return True
    return False

#Checks that you won't go off the bottom side of the map, and that you're not running into a wall.
def check_down_valid(cur_position, board, num_rows):
    row = cur_position[0]
    check_inaccessible = board[cur_position[0] + 1, cur_position[1]]
    if (row != num_rows - 1) and check_inaccessible != WALL:  # if row value is not already at the very bottom
        return True
    return False
