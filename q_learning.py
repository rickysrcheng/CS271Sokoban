import numpy as np
from constants import *

def q_learn(board, start_boxes, start_position, goal_positions, rows, columns):
    q_vals = np.zeros((rows, columns, 4))
    print(board)
    for k in range(100):
        # Restart episode
        print("\nStart episode")
        cur_position = [start_pos for start_pos in start_position] #deep copy
        cur_boxes = [row[:] for row in start_boxes]
        step = 0

        print("Current position :", cur_position)
        print("Boxes positions: ", start_boxes)

        # Episode ends if all boxes are on goals or # of steps is reached
        while not goal_found(cur_position, goal_positions, cur_boxes) and not deadlocks(cur_boxes, board, goal_positions) and step != 300:
            action = epsilon_greedy_get_action(cur_position, EPSILON, q_vals, rows, columns, board, cur_boxes, k)
            old_position = [cur_pos for cur_pos in cur_position] # deep copy, otherwise old_position is overwritten after next_location()

            # Agent recieves reward if it pushes a box onto a goal
            box_reward = next_location(cur_position, action, board, cur_boxes, goal_positions)
            reward = -1 + box_reward

            print("\nCurrent position :", cur_position)
            print("Boxes positions: ", cur_boxes)

            #print("reward", reward)
            old_q_val = q_vals[old_position[0], old_position[1], action]
            td = reward + (DISCOUNT * np.max(q_vals[cur_position[0], cur_position[1]])) - old_q_val
            #print("TD :", td)
            new_q_val = old_q_val + (LEARN_RATE * td)
            #print("New Q value :", new_q_val)
            #print("")
            q_vals[old_position[0], old_position[1], action] = new_q_val
            step += 1
    # Final q_values after all episodes ran
    print(q_vals)

def deadlocks(boxes, board, goal_positions):
    for box in boxes:
        if board[box[0], box[1]] == DEADLOCK and box not in goal_positions:
            return True
    return False

#TODO
def optimal_route(rows, columns, reward):
    return []

def goal_found(cur_position, goal_positions, boxes):
    remaining_boxes = len(boxes)
    
    for box in boxes:
        if box in goal_positions:
            remaining_boxes -= 1

    if not remaining_boxes:
        return True
    return False

def epsilon_greedy_get_action(curr_position, epsilon, q_values, rows, columns, board, boxes, k):
    actions = []  #making list of valid moves
    if check_up_valid(curr_position, board, boxes):
        actions.append(0)
    if check_down_valid(curr_position, board, rows, boxes):
        actions.append(1)
    if check_left_valid(curr_position, board, boxes):
        actions.append(2)
    if check_right_valid(curr_position, board, columns, boxes):
        actions.append(3)

    total_actions = [0, 1, 2, 3]
    exclude = [i for i in actions + total_actions if i not in actions or i not in total_actions]
    m = np.zeros(4, dtype=bool)
    m[exclude] = True

    ep = epsilon

    if k > 50:
        ep = .2

    if np.random.random() > ep: #return random from 0 to 1
        #return argmax of q values for current position, will return an action
        #only allow actions that were deemed allowed by the above validity checks
        q_values_masked = np.ma.array(q_values[curr_position[0], curr_position[1]], mask=m)
        return np.argmax(q_values_masked)

    return np.random.choice(actions)


def next_location(cur_position, action, board, boxes, goals):
    if action == 0: #UP
        reward = move_box(cur_position, board, boxes, -1, 0, goals)
        cur_position[0] += -1
    elif action == 1: #DOWN
        reward = move_box(cur_position, board, boxes, 1, 0, goals)
        cur_position[0] += 1
    elif action == 2: #LEFT
        reward = move_box(cur_position, board, boxes, 0, -1, goals)
        cur_position[1] += -1    
    else: #current action = RIGHT
        reward = move_box(cur_position, board, boxes, 0, 1, goals)
        cur_position[1] += 1
    return reward
        
#Checks that you won't go off the left side of the map, and that you're not running into a wall.
def check_left_valid(cur_position, board, boxes):
    column = cur_position[1]
    check_inaccessible = board[cur_position[0], cur_position[1] - 1]
    valid = False
    is_box, can_push = can_move_box(cur_position, board, boxes, 0, -1)
    if column != 0 and check_inaccessible != WALL:
        valid = True

        if is_box and not can_push: # Invalid move if can't push a box
            valid = False 

    return valid 

#Checks that you won't go off right side of map, and that you're not running into a wall.
def check_right_valid(cur_position, board, num_columns, boxes):
    column = cur_position[1]
    check_inaccessible = board[cur_position[0], cur_position[1] + 1]
    valid = False
    is_box, can_push = can_move_box(cur_position, board, boxes, 0, 1)
    if (column != num_columns - 1) and check_inaccessible != WALL:  # if column value is not already at the very right
        valid = True

        if is_box and not can_push: # Invalid move if can't push a box
            valid = False 

    return valid 

#Checks that you won't go off the top side of the map, and that you're not running into a wall.
def check_up_valid(cur_position, board, boxes):
    row = cur_position[0]
    check_inaccessible = board[cur_position[0] - 1, cur_position[1]]
    valid = False
    is_box, can_push = can_move_box(cur_position, board, boxes, -1, 0)
    if (row != 0) and check_inaccessible != WALL:  # if row value is not already at the very top
        valid = True

        if is_box and not can_push: # Invalid move if can't push a box
            valid = False 

    return valid 

#Checks that you won't go off the bottom side of the map, and that you're not running into a wall.
def check_down_valid(cur_position, board, num_rows, boxes):
    row = cur_position[0]
    check_inaccessible = board[cur_position[0] + 1, cur_position[1]]
    valid = False
    is_box, can_push = can_move_box(cur_position, board, boxes, 1, 0)
    if (row != num_rows - 1) and check_inaccessible != WALL:  # if row value is not already at the very bottom
        valid = True

        if is_box and not can_push: # Invalid move if can't push a box
            valid = False 

    return valid 

# If there is a box, checks if the box can be pushed
# Cannot push box into wall or into another box 
def can_move_box(cur_position, board, boxes, move_row, move_col): 
    is_box = True if [cur_position[0] + move_row, cur_position[1] + move_col] in boxes else False
    can_push = True if is_box and board[cur_position[0] + move_row * 2, cur_position[1] + move_col * 2] != WALL and [cur_position[0] + move_row * 2, cur_position[1] + move_col * 2] not in boxes else False

    return is_box, can_push

# If there is a pushable box, changes the position of that box
# If box is moved onto a goal, return GOAL reward. If not, return no reward
def move_box(cur_position, board, boxes, move_row, move_col, goals):
    is_box, can_push = can_move_box(cur_position, board, boxes, move_row, move_col)

    if is_box and can_push:
        idx = boxes.index([cur_position[0] + move_row, cur_position[1] + move_col])
        boxes[idx] = [cur_position[0] + move_row * 2, cur_position[1] + move_col * 2]

        if board[boxes[idx][0], boxes[idx][1]] == DEADLOCK and boxes[idx] not in goals:
            return DEADLOCK

    return GOAL if is_box and can_push and boxes[idx] in goals else 0 


