import argparse
from preprocess import *
from q_learning import *
from pathfinder import *
class Sokoban_Board:
	def __init__(self):
		self.board = None
		self.sizeH = 0
		self.sizeV = 0
		self.walls = []
		self.boxes = []
		self.storage = []
		self.playerX = 0
		self.playerY = 0
# Reads command line and expects the input file as type .txt
def parse_args():
	parser = argparse.ArgumentParser(description='Solves a Sokoban game using an RL algorithm.')
	parser.add_argument('input_file', metavar='FILE', nargs=1,
	                    help='input file of type txt')
	args = parser.parse_args()
	if args.input_file[0][-4:] != '.txt':
		print("Input file must be a txt file.")
		args = None
	return args

# Reads the input file, line by line, and stores the formatted results in the board object
def read_input(args):
	with open(args.input_file[0], 'r') as file:
		s_board = Sokoban_Board()
		size_line = file.readline().split()
		s_board.sizeH, s_board.sizeV = int(size_line[0]), int(size_line[1])
		s_board.board = [[' ' for _ in range(s_board.sizeV)] for _ in range(s_board.sizeH)]
		s_board.walls = parse_input_line(file.readline().split(), s_board.board, '#')
		s_board.boxes = parse_input_line(file.readline().split(), s_board.board, '$')
		s_board.storage = parse_input_line(file.readline().split(), s_board.board, '.')
		player_line = file.readline().split()
		s_board.playerX, s_board.playerY = int(player_line[0]), int(player_line[1])
		s_board.board[s_board.playerX - 1][s_board.playerY - 1] = '@'
		return s_board
# For each line in file, parses the list of values and returns the coordinate pairs as a list
# Also "draws" sokoban board
def parse_input_line(line, board, symbol):
	line_len = int(line[0]) * 2
	coordinates = []
	for i in range(0, line_len, 2):
		row = int(line[i + 1])
		col = int(line[i + 2])
		board[row - 1][col - 1] = symbol
		coordinate = [row, col]
		coordinates.append(coordinate)
	return coordinates
# Executes input reading
args = parse_args()
if args:
	s_board = read_input(args)
	print("board_size: ", s_board.sizeH, " ", s_board.sizeV)
	print("walls: ", s_board.walls)
	print("boxes: ", s_board.boxes)
	print("storages: ", s_board.storage)
	print("player: ", s_board.playerX, " ", s_board.playerY)
	print()
	# Prints out sokoban board
	for i in range(s_board.sizeH):
		print(''.join(s_board.board[i]))

	print("")
	start_position = [s_board.playerX - 1, s_board.playerY - 1]
	reward_board = preprocess(s_board.sizeH, s_board.sizeV, s_board.walls, s_board.boxes, s_board.storage)
	print(reward_board)
	print("Goal locations: ", s_board.storage)
	#reward(s_board.sizeH, s_board.sizeV, rewards)
	q_learn(reward_board, s_board.boxes, start_position, s_board.storage, s_board.sizeH, s_board.sizeV)

