import numpy as np
from constants import *
from pathfinder import *
import queue
import random
from sys import getsizeof
from datetime import datetime
from scipy.special import comb

verbose = False
printBoard = True
experimental = True

def q_learn(board, start_boxes, start_position, goal_positions, rows, columns):
    q_vals = np.zeros((rows, columns, 4))
    start_tuples = make_tuple_key(start_boxes)
    print(start_tuples)
    q_vals_exp = {start_tuples:np.zeros((len(start_tuples), 4))}
    
    unique, counts = np.unique(board, return_counts=True)
    elements = dict(zip(unique, counts))
    print(f'Number of possible floor space: {elements[FLOOR]}')
    print(f'Number of boxes: {len(start_boxes)}')
    totalStates = comb(elements[FLOOR], len(start_boxes))
    print(f'Number of possible states: {totalStates}')
    start_time = datetime.now()
    moves = []
    for episode in range(10000):
        # Restart episode
        #print(f"\nStart episode {episode}")
        cur_position = [start_pos for start_pos in start_position] #deep copy
        cur_boxes = [row[:] for row in start_boxes]
        step = 0
        boardSolution = ""
        if printBoard:
            boardSolution += print_board(board, cur_position, cur_boxes, goal_positions)
        if verbose:
            print("Current position :", cur_position)
            print("Boxes positions: ", start_boxes)
        deadLock = False
        box_and_moves = possible_box_moves(cur_boxes, board, q_vals_exp, cur_position)
        
        paths = []
        action = []
        for b in box_and_moves[1]:
            p = box_and_moves[1].get(b)
            for k in p:
                paths.append([p[k], b])
        
        #Episode ends if all boxes are on goals or # of steps is reached
        while not goal_found(cur_position, goal_positions, cur_boxes) and step != 300 and not deadLock:
            old_boxes = make_tuple_key(cur_boxes)
            #box_and_moves = possible_box_moves(cur_boxes, board, q_vals_exp, cur_position) #returns overall best box, overall best action

            if box_and_moves[0] == [-1]:
                break
            #action = epsilon_greedy_get_action(cur_position, .5, q_vals, rows, columns, board, cur_boxes)
            action = []
            if (epsilon_greedy(EPSILON)): #if less than epsilon, than choose optimal value.
                action = box_and_moves[0][0]
                best_box = box_and_moves[0][1]
            else: #action is randomly chosen from list of paths.
                randomChoice = random.choice(paths)
                action = randomChoice[0]
                best_box = randomChoice[1]
            old_position = [cur_pos for cur_pos in cur_position] # deep copy, otherwise old_position is overwritten after next_location()

            moves += action 

            # Agent receives reward if it pushes a box onto a goal
            box_reward, deadLock = next_location(cur_position, action, board, cur_boxes, goal_positions)
            reward = -(step**2) + box_reward

            if verbose:
                print("\nCurrent position :", cur_position)
                print("Boxes positions: ", cur_boxes)
                print(f'Actions chosen: {action}')
            if printBoard:
                boardSolution += print_board(board, cur_position, cur_boxes, goal_positions)
            
            #print(print_board(board, cur_position, cur_boxes, goal_positions))
            #print("reward", reward)
            currKey = make_tuple_key(cur_boxes)
            if currKey not in q_vals_exp:
                q_vals_exp[currKey] = np.zeros((len(start_tuples), 4))

            #old_q_val = q_vals[old_position[0], old_position[1], action]
            #td = reward + (DISCOUNT * np.max(q_vals[cur_position[0], cur_position[1]])) - old_q_val
            #new_q_val = old_q_val + (LEARN_RATE * td)
            #q_vals[old_position[0], old_position[1], action] = new_q_val

            # for next state
            box_and_moves = possible_box_moves(cur_boxes, board, q_vals_exp, cur_position)

            # get max q-value of possible actions at s_{t+1}

            # if no moves, end current episode AND update q-values
            if box_and_moves[0] == [-1]:
                if not goal_found(cur_position, goal_positions, cur_boxes):
                    old_q_val = q_vals_exp[old_boxes][old_boxes.index(tuple(best_box)), action[-1]]
                    td = reward + NOMOVES - old_q_val
                    new_q_val = old_q_val + (LEARN_RATE * td)
                    q_vals_exp[old_boxes][old_boxes.index(tuple(best_box)), action[-1]] = new_q_val
                break
            paths = []
            qValActions = []
            for b in box_and_moves[1]:
                p = box_and_moves[1].get(b)
                for k in p:
                    paths.append([p[k], b])
                    qValActions.append(q_vals_exp[currKey][currKey.index(tuple(b)), p[k][-1]])
            
            if verbose:
                print(f'Old q val: {old_q_val}')
            # new experimental q-table
            if experimental:
                #print(f'Old Key = {old_boxes}')
                #print(f'Cur Key = {currKey}')
                #print(f'Best box: {best_box}')
                #print(q_vals_exp[old_boxes])
                #print(f'max curr state: {np.max(qValActions)}')
                old_q_val = q_vals_exp[old_boxes][old_boxes.index(tuple(best_box)), action[-1]]
                td = reward + (DISCOUNT * np.max(qValActions)) - old_q_val
                new_q_val = old_q_val + (LEARN_RATE * td)
                q_vals_exp[old_boxes][old_boxes.index(tuple(best_box)), action[-1]] = new_q_val
            
            step += 1
        if episode % 10000 == 0:
            print(f'Episode {episode} steps: {step}')
            print(f'Size of q-table in bytes: {getsizeof(q_vals_exp)}')
            print(f'Number of states in q-table: {len(q_vals_exp)}/{totalStates} {len(q_vals_exp)/totalStates:.2f}% of total state space traversed')
            print(f'Current Time: {datetime.now()}')

        if step > 200 and printBoard and deadLock:
            print(f'Episode {episode} steps: {step}')
            print(f'Size of q-table in bytes: {getsizeof(q_vals_exp)}')
            print(f'Number of states in q-table: {len(q_vals_exp)}')
            #print(print_board(board, cur_position, cur_boxes, goal_positions))
        
        if goal_found(cur_position, goal_positions, cur_boxes):
            print(f'Episode {episode} steps: {step} # of moves: {len(moves)}')
            print(f'Size of q-table in bytes: {getsizeof(q_vals_exp)}')
            print(f'Number of states in q-table: {len(q_vals_exp)}')
            print(f'Current Time: {datetime.now()}')
            #print_moves(moves)
            #print(boardSolution)
            with open('solve.txt', 'w') as f:
                f.write(f'Episode {episode} steps: {step} # of moves: {len(moves)}\n')
                f.write(f'Size of q-table in bytes: {getsizeof(q_vals_exp)}\n')
                f.write(f'Number of states in q-table: {len(q_vals_exp)}/{totalStates} {len(q_vals_exp)/totalStates:.2f}% of total state space traversed\n')
                f.write(f'Started: {start_time}\nFound: {datetime.now()}\n')
                f.write(print_moves(moves) + '\n')
                f.write(boardSolution)
            break
            
            
        moves = []
    # Final q_values after all episodes ran
    # for k,v in q_vals_exp.items():
    #     print(k)
    #     print(v)

def print_board(board, playerPosition, boxesPositions, goalPositions):
    m,n = board.shape
    retString = ""
    for r in range(m):
        rowString = ""
        for c in range(n):
            if [r,c] == playerPosition:
                rowString += "@"
            elif [r,c] in boxesPositions:
                if [r,c] in goalPositions:
                    rowString += "*"
                else:
                    rowString += "$"
            elif [r,c] in goalPositions:
                rowString += "."
            elif board[r,c] == WALL:
                rowString += "#"
            elif board[r,c] == FLOOR or board[r,c] == DEADLOCK:
                rowString += " "
        retString += rowString + "\n"
        #print(f'{rowString}')

    return retString

# Prints the moves of the agent's final solution 
def print_moves(moves):
    moves = [str(m) for m in moves]
    str_moves = " ".join(moves)
    str_moves = str_moves.replace("0", "U")
    str_moves = str_moves.replace("1", "D")
    str_moves = str_moves.replace("2", "L")
    str_moves = str_moves.replace("3", "R")

    print(len(moves), str_moves)
    return str_moves

def make_tuple_key(boxes):
    return tuple(sorted([tuple(box) for box in boxes]))

#Checks where boxes can be potentially moved. Once we have this, we can use the Q table to check which move would result
#in the highest q value, and move our agent to perform the agent.
def possible_box_moves(cur_boxes, board, q_vals, cur_position):
    #in order for a box to be moved into a certain position,
    #1) the position it is moving into must not be obstructed.
    #2) The agent must be able to push from the opposite side of the box.
    box_dict = {}
    boxes_and_pathways = {}
    boxKey = make_tuple_key(cur_boxes)
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
            #q_value = q_vals[box[0], box[1], UP]
            q_value = q_vals[boxKey][boxKey.index(tuple(box)), UP]
            actions_to_q_vals[UP] = q_value
            pathways[UP] = up
        if down != [INVALID]:
            #calculate q value of action 1 at box's position:
            #q_value = q_vals[box[0], box[1], DOWN]
            q_value = q_vals[boxKey][boxKey.index(tuple(box)), DOWN]
            actions_to_q_vals[DOWN] = q_value
            pathways[DOWN] = down
        if left != [INVALID]:
            #calculate q value of action 2 at box's position:
            #q_value = q_vals[box[0], box[1], LEFT]
            q_value = q_vals[boxKey][boxKey.index(tuple(box)), LEFT]
            actions_to_q_vals[LEFT] = q_value
            pathways[LEFT] = left
        if right != [INVALID]:
            #calculate q value of action 3 at box's position:
            q_value = q_vals[boxKey][boxKey.index(tuple(box)), RIGHT]
            actions_to_q_vals[RIGHT] = q_value
            pathways[RIGHT] = right
        if(pathways):
            boxes_and_pathways[tuple(box)] = pathways
        box_dict[tuple(box)] = actions_to_q_vals
    if verbose:
        print(box_dict)
        print("boxes and pathways", boxes_and_pathways)
    return [choose_box_and_action(box_dict, boxes_and_pathways), boxes_and_pathways]

#Accepts the dictionary of dictionary that is {box_locations: {actions: q values}}
#From the dictionary, finds the box and its corresponding action that results in the highest q value.
#TODO: if ties then randomize choice?
def choose_box_and_action(box_dict, boxes_and_pathways):
    max_choices = {} #This will be a dict of tuples. {Box_Coordinate : (max action: q_value)}
    overall_best_box = [INVALID, INVALID]
    overall_best_action = INVALID
    overall_best_q = -np.inf
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
    best_path = boxes_and_pathways[tuple(overall_best_box)][overall_best_action]
    if verbose:
        print("Overall best box: ", overall_best_box)
        print("Overall best action: ", overall_best_action)
        print("Overall best path: ", best_path)
    return best_path, overall_best_box

#Parameters: accepts a box's coordinates, current agent position, and the board.
#Returns the path needed to push the box into the position.
def is_left_potential_move(box, board, cur_position, box_positions):
    #check if to the left is a wall
    if board[box[0], box[1] - 1] != FLOOR or [box[0], box[1] - 1] in box_positions:
        return [INVALID]
    #check that agent can reach the location to the right of the box using pathfinder.
    if board[box[0], box[1] + 1] == WALL or [box[0], box[1] + 1] in box_positions:
        return [INVALID]
    path = shortest_path_actions(board, [box[0], box[1] + 1], cur_position, box_positions)
    #print("Left path: ", path)
    if path != [INVALID]:
        path.append(LEFT)
        return path
    return [INVALID]

# Parameters: accepts a box's coordinates, current agent position, and the board.
def is_right_potential_move(box, board, cur_position, box_positions):
    # check if to the right is a wall
    if board[box[0], box[1] + 1] != FLOOR or [box[0], box[1] + 1] in box_positions:
        return [INVALID]
    # check if to the left is an empty space (so the agent can occupy it)
    if board[box[0], box[1] - 1] == WALL or [box[0], box[1] - 1] in box_positions:
        return [INVALID]
    path = shortest_path_actions(board, [box[0], box[1] - 1], cur_position, box_positions)
    if path != [INVALID]:
        path.append(RIGHT)
        return path
    return [INVALID]

# Parameters: accepts a box's coordinates, current agent position, and the board.
def is_up_potential_move(box, board, cur_position, box_positions):
    # check if up is a wall
    if board[box[0] - 1, box[1]] != FLOOR or [box[0] - 1, box[1]] in box_positions:
        return [INVALID]
    # check if down is an empty space (so the agent can occupy it)
    if board[box[0] + 1, box[1]] == WALL or [box[0] + 1, box[1]] in box_positions:
        return [INVALID]
    path = shortest_path_actions(board, [box[0] + 1, box[1]], cur_position, box_positions)
    if path != [INVALID]:
        path.append(UP)
        return path
    return [INVALID]

# Parameters: accepts a box's coordinates, current agent position, and the board.
def is_down_potential_move(box, board, cur_position, box_positions):
    # check if down is a wall
    if board[box[0] + 1, box[1]] != FLOOR or [box[0] + 1, box[1]] in box_positions:
        return [INVALID]
    # check if up is an empty space (so the agent can occupy it)
    if board[box[0] - 1, box[1]] == WALL or [box[0] - 1, box[1]] in box_positions: 
        return [INVALID]
    path = shortest_path_actions(board, [box[0] - 1, box[1]], cur_position, box_positions)
    if path != [INVALID]:
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

def next_location(cur_position, actions, board, boxes, goals):
    reward = 0
    for action in actions:
        if action == UP:                                 #move_row, move_column
            reward, deadlock = move_box(cur_position, board, boxes, -1, 0, goals)
            cur_position[0] += -1
        elif action == DOWN:
            reward, deadlock = move_box(cur_position, board, boxes, 1, 0, goals)
            cur_position[0] += 1
        elif action == LEFT:
            reward, deadlock = move_box(cur_position, board, boxes, 0, -1, goals)
            cur_position[1] += -1
        else:
            reward, deadlock = move_box(cur_position, board, boxes, 0, 1, goals)
            cur_position[1] += 1
    return reward, deadlock
        
# If there is a box, checks if the box can be pushed
# Cannot push box into wall or into another box 
def can_move_box(cur_position, board, boxes, move_row, move_col): 
    is_box = True if [cur_position[0] + move_row, cur_position[1] + move_col] in boxes else False
    can_push = True if is_box and board[cur_position[0] + move_row * 2, cur_position[1] + move_col * 2] != WALL and [cur_position[0] + move_row * 2, cur_position[1] + move_col * 2] not in boxes else False

    return is_box, can_push

# Check if we push the box into a deadlock
def detect_deadlock(cur_position, board, boxes, move_row, move_col, goals, idx):
    if idx == -1:
        idx = boxes.index([cur_position[0] + move_row * 2, cur_position[1] + move_col * 2])
    curBoxRow = boxes[idx][0]
    curBoxCol = boxes[idx][1]
    isDeadlock = False
    if boxes[idx] in goals:
        return False

    # if in a preprocessed deadlock state (non-goal corners), it's a deadlock
    if board[curBoxRow, curBoxCol] == DEADLOCK:
        return True

    # TODO:
    # non-goal corners are preprocessed into reward board
    # check deadlocks caused by walls and boxes in 3x3 grid
    # ex   #$_                                        _$_           
    #      _$#                                        _$#
    #      ___ is a deadlock for box just pushed but  ___ is not a deadlock
    # check up, down, left, right of just pushed box for boxes
    # if any of the boxes return deadlock, it's a deadlock, if not, check current box
    # return if deadlock
    
    # check first if any moves are free
    if check_surrounding_deadlock([curBoxRow, curBoxCol], board, boxes) == False:
        return isDeadlock

    # each direction is dead if agent can't occupy the other side
    dOccupied = board[curBoxRow + 1, curBoxCol] == WALL
    uOccupied = board[curBoxRow - 1, curBoxCol] == WALL
    rOccupied = board[curBoxRow, curBoxCol + 1] == WALL
    lOccupied = board[curBoxRow, curBoxCol - 1] == WALL
    
    uDeadlock = False
    dDeadlock = False
    lDeadlock = False
    rDeadlock = False
    if not dOccupied and [curBoxRow - 1, curBoxCol] in boxes:
        # _$_ _$_
        # #$_ $$_
        # ___ _#_ won't be checked
        #print('check udeadlock')
        uDeadlock = check_surrounding_deadlock([curBoxRow - 1, curBoxCol], board, boxes)
    if not uOccupied and [curBoxRow + 1, curBoxCol] in boxes:
        #print('check ddeadlock')
        dDeadlock = check_surrounding_deadlock([curBoxRow + 1, curBoxCol], board, boxes)
    if not rOccupied and [curBoxRow, curBoxCol - 1] in boxes:
        #print('check ldeadlock')
        lDeadlock = check_surrounding_deadlock([curBoxRow, curBoxCol - 1], board, boxes)
    if not lOccupied and [curBoxRow, curBoxCol + 1] in boxes:
        #print('check rdeadlock')
        rDeadlock = check_surrounding_deadlock([curBoxRow, curBoxCol + 1], board, boxes)
    
    #print(f'Deadlock states {uDeadlock} {dDeadlock} {lDeadlock} {rDeadlock}')
    udDeadlock = (dOccupied or uDeadlock) or (uOccupied or dDeadlock)
    lrDeadlock = (lOccupied or rDeadlock) or (rOccupied or lDeadlock)

    #print(f'Final Deadlock states {lrDeadlock} {udDeadlock}')

    isDeadlock = udDeadlock and lrDeadlock
    return isDeadlock

def check_surrounding_deadlock(box, board, boxes):
    # check if any moves are valid
    # can't use potential moves straight up because of pseudo-corral deadlocks, ie, the agent is trapped in a subspace
    lFree = (board[box[0], box[1] - 1] == FLOOR) and [box[0], box[1] - 1] not in boxes
    rFree = (board[box[0], box[1] + 1] == FLOOR) and [box[0], box[1] + 1] not in boxes
    lrFree = lFree and rFree
    
    uFree = (board[box[0] - 1, box[1]] == FLOOR) and [box[0] - 1, box[1]] not in boxes
    dFree = (board[box[0] + 1, box[1]] == FLOOR) and [box[0] + 1, box[1]] not in boxes
    udFree = uFree and dFree
    #print(f'{box}, {lrFree}, {udFree}')
    # returns true if there are no valid moves
    return not lrFree and not udFree

# If there is a pushable box, changes the position of that box
# If box is moved onto a goal, return GOAL reward. If not, return no reward
def move_box(cur_position, board, boxes, move_row, move_col, goals):
    boxReward = 0
    isDeadLock = False

    is_box, can_push = can_move_box(cur_position, board, boxes, move_row, move_col)

    if is_box and can_push:
        idx = boxes.index([cur_position[0] + move_row, cur_position[1] + move_col])
        boxes[idx] = [cur_position[0] + move_row * 2, cur_position[1] + move_col * 2]
        # move box, then check if the move produces a deadlock
        isDeadLock = detect_deadlock(cur_position,board, boxes, move_row, move_col, goals, idx)

    if is_box and can_push:
        if boxes[idx] in goals:
            boxReward = GOAL
        elif isDeadLock:
            boxReward = DEADLOCK

    return boxReward, isDeadLock


