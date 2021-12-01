import numpy as np
from constants import *
from pathfinder import *
import queue
import random

def q_learn(board, start_boxes, start_position, goal_positions, rows, columns):
    q_vals = np.zeros((rows, columns, 4))
    for k in range(1000):
        # Restart episode
        print("\nStart episode")
        cur_position = [start_pos for start_pos in start_position] #deep copy
        cur_boxes = [row[:] for row in start_boxes]
        step = 0

        print("Current position :", cur_position)
        print("Boxes positions: ", start_boxes)

        #Episode ends if all boxes are on goals or # of steps is reached
        while not goal_found(cur_position, goal_positions, cur_boxes) and step != 300:
            box_and_moves = possible_box_moves(cur_boxes, board, q_vals, cur_position) #returns overall best box, overall best action
            if box_and_moves[0] == [-1]:
                break
            #action = epsilon_greedy_get_action(cur_position, .5, q_vals, rows, columns, board, cur_boxes)
            action = []
            if (epsilon_greedy(EPSILON)): #if less than epsilon, than choose optimal value.
                action = box_and_moves[0]
            else: #action is randomly chosen from list of paths.
                paths = []
                for b in box_and_moves[1]:
                    p = box_and_moves[1].get(b)
                    for k in p:
                        paths.append(p[k])
                action = random.choice(paths)
            old_position = [cur_pos for cur_pos in cur_position] # deep copy, otherwise old_position is overwritten after next_location()

            # Agent receives reward if it pushes a box onto a goal
            box_reward = next_location(cur_position, action, board, cur_boxes, goal_positions)
            reward = -1 + box_reward

            print("\nCurrent position :", cur_position)
            print("Boxes positions: ", cur_boxes)

            #print("reward", reward)
            old_q_val = q_vals[old_position[0], old_position[1], action]
            td = reward + (DISCOUNT * np.max(q_vals[cur_position[0], cur_position[1]])) - old_q_val
            new_q_val = old_q_val + (LEARN_RATE * td)
            q_vals[old_position[0], old_position[1], action] = new_q_val
            step += 1

        if goal_found(cur_position, goal_positions, cur_boxes):
            break
    # Final q_values after all episodes ran
    print(q_vals)

#Checks where boxes can be potentially moved. Once we have this, we can use the Q table to check which move would result
#in the highest q value, and move our agent to perform the agent.
def possible_box_moves(cur_boxes, board, q_vals, cur_position):
    #in order for a box to be moved into a certain position,
    #1) the position it is moving into must not be obstructed.
    #2) The agent must be able to push from the opposite side of the box.
    box_dict = {}
    boxes_and_pathways = {}
    for box in cur_boxes:
        #Check whether box has space to move up, down, left, right. Only considers if space is wall.
        #Also checks that the box location in question is actually reachable by using pathfinder.
        actions_to_q_vals = {}
        pathways = {}
        up = is_up_potential_move(box, board, cur_position, cur_boxes)
        down = is_down_potential_move(box, board, cur_position, cur_boxes)
        left = is_left_potential_move(box, board, cur_position, cur_boxes)
        right = is_right_potential_move(box, board, cur_position, cur_boxes)
        if up != [INVALID]: #First check if box can be moved up and agent can reach location.
            #calculate q value of action 0's at box position:
            q_value = q_vals[box[0], box[1], UP]
            actions_to_q_vals[UP] = q_value
            pathways[UP] = up
        if down != [INVALID]:
            #calculate q value of action 1 at box's position:
            q_value = q_vals[box[0], box[1], DOWN]
            actions_to_q_vals[DOWN] = q_value
            pathways[DOWN] = down
        if left != [INVALID]:
            #calculate q value of action 2 at box's position:
            q_value = q_vals[box[0], box[1], LEFT]
            actions_to_q_vals[LEFT] = q_value
            pathways[LEFT] = left
        if right != [INVALID]:
            #calculate q value of action 3 at box's position:
            q_value = q_vals[box[0], box[1], RIGHT]
            actions_to_q_vals[RIGHT] = q_value
            pathways[RIGHT] = right
        if(pathways):
            boxes_and_pathways[tuple(box)] = pathways
        box_dict[tuple(box)] = actions_to_q_vals
    #print(box_dict)
    print("boxes and pathways", boxes_and_pathways)
    return [choose_box_and_action(box_dict, boxes_and_pathways), boxes_and_pathways]

#Accepts the dictionary of dictionary that is {box_locations: {actions: q values}}
#From the dictionary, finds the box and its corresponding action that results in the highest q value.
#TODO: if ties then randomize choice?
def choose_box_and_action(box_dict, boxes_and_pathways):
    max_choices = {} #This will be a dict of tuples. {Box_Coordinate : (max action: q_value)}
    overall_best_box = [INVALID, INVALID]
    overall_best_action = INVALID
    overall_best_q = -10000
    for box in box_dict:
        if(box_dict.get(box)):
            best_action = max(box_dict.get(box), key=box_dict.get(box).get)
            best_q = box_dict[box][best_action]
            if best_q > overall_best_q:
                overall_best_q = best_q
                overall_best_action = best_action
                overall_best_box = list(box)
            max_choices[box] = [best_action, best_q]
    if(overall_best_box == [INVALID, INVALID]):
        return [INVALID]
    print("Overall best box: ", overall_best_box)
    print("Overall best action: ", overall_best_action)
    best_path = boxes_and_pathways[tuple(overall_best_box)][overall_best_action]
    print("Overall best path: ", best_path)
    return best_path

#Parameters: accepts a box's coordinates, current agent position, and the board.
#Returns the path needed to push the box into the position.
def is_left_potential_move(box, board, cur_position, box_positions):
    #check if to the left is a wall
    #check that agent can reach the location to the right of the box using pathfinder.
    path = shortest_path_actions(board, [box[0], box[1] + 1], cur_position, box_positions)
    #print("Left path: ", path)
    if(board[box[0], box[1] - 1] != WALL and path != [INVALID]):
        path.append(LEFT)
        return path
    return [INVALID]

# Parameters: accepts a box's coordinates, current agent position, and the board.
def is_right_potential_move(box, board, cur_position, box_positions):
    # check if to the right is a wall
    # check if to the left is an empty space (so the agent can occupy it)
    path = shortest_path_actions(board, [box[0], box[1] - 1], cur_position, box_positions)
    if (board[box[0], box[1] + 1] != WALL and path != [INVALID]):
        path.append(RIGHT)
        return path
    return [INVALID]

# Parameters: accepts a box's coordinates, current agent position, and the board.
def is_up_potential_move(box, board, cur_position, box_positions):
    # check if up is a wall
    # check if down is an empty space (so the agent can occupy it)
    path = shortest_path_actions(board, [box[0] + 1, box[1]], cur_position, box_positions)
    if (board[box[0] - 1, box[1]] != WALL and path != [INVALID]):
        path.append(UP)
        return path
    return [INVALID]

# Parameters: accepts a box's coordinates, current agent position, and the board.
def is_down_potential_move(box, board, cur_position, box_positions):
    # check if down is a wall
    # check if up is an empty space (so the agent can occupy it)
    path = shortest_path_actions(board, [box[0] - 1, box[1]], cur_position, box_positions)
    if (board[box[0] + 1, box[1]] != WALL and path != [INVALID]):
        path.append(DOWN)
        return path
    return [INVALID]

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

def epsilon_greedy(epsilon):
    return np.random.random() < epsilon  #Returns true if less than epsilon, false otherwise

def epsilon_greedy_get_action(curr_position, epsilon, q_values, rows, columns, board, boxes):
    actions = []  #making list of valid moves
    if check_up_valid(curr_position, board, boxes):
        actions.append(UP)
    if check_down_valid(curr_position, board, rows, boxes):
        actions.append(DOWN)
    if check_left_valid(curr_position, board, boxes):
        actions.append(LEFT)
    if check_right_valid(curr_position, board, columns, boxes):
        actions.append(RIGHT)

    total_actions = [UP, DOWN, LEFT, RIGHT]
    exclude = [i for i in actions + total_actions if i not in actions or i not in total_actions]
    m = np.zeros(4, dtype=bool)
    m[exclude] = True

    if np.random.random() < epsilon: #return random from 0 to 1
        #return argmax of q values for current position, will return an action
        #only allow actions that were deemed allowed by the above validity checks
        q_values_masked = np.ma.array(q_values[curr_position[0], curr_position[1]], mask=m)
        return np.argmax(q_values_masked)

    return np.random.choice(actions)


def next_location(cur_position, actions, board, boxes, goals):
    reward = 0
    for action in actions:
        if action == UP:                                 #move_row, move_column
            reward = move_box(cur_position, board, boxes, -1, 0, goals)
            cur_position[0] += -1
        elif action == DOWN:
            reward = move_box(cur_position, board, boxes, 1, 0, goals)
            cur_position[0] += 1
        elif action == LEFT:
            reward = move_box(cur_position, board, boxes, 0, -1, goals)
            cur_position[1] += -1
        else:
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

# Check if we push the box into a deadlock
def detect_deadlock(cur_position, board, boxes, move_row, move_col, goals):
    idx = boxes.index([cur_position[0] + move_row * 2, cur_position[1] + move_col * 2])
    curBoxRow = boxes[idx][0]
    curBoxCol = boxes[idx][1]
    isDeadlock = False
    if boxes[idx] in goals:
        return isDeadlock

    # if in a preprocessed deadlock state (non-goal corners), it's a deadlock
    if board[boxes[idx]] == DEADLOCK:
        return True


    # TODO:
    # non-goal corners are preprocessed into reward board
    # check deadlocks caused by walls and boxes in 3x3 grid
    # ex   #$_
    #      _$#
    #      ___ is a deadlock for box just pushed
    # idea: create a hashtable for deadlocks to save on processing speed every iteration

    # TODO:
    # if box by a wall, if there are no clear directions for the box to reach a storage along the wall, it's a deadlock

    return isDeadlock

# If there is a pushable box, changes the position of that box
# If box is moved onto a goal, return GOAL reward. If not, return no reward
def move_box(cur_position, board, boxes, move_row, move_col, goals):
    is_box, can_push = can_move_box(cur_position, board, boxes, move_row, move_col)

    if is_box and can_push:
        idx = boxes.index([cur_position[0] + move_row, cur_position[1] + move_col])
        boxes[idx] = [cur_position[0] + move_row * 2, cur_position[1] + move_col * 2]

    return GOAL if is_box and can_push and boxes[idx] in goals else 0 


