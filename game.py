import random
import copy
import time
import math
from math import inf
import numpy as np
INF = 99999999
class GameError(AttributeError):
	pass

class Game:

	def __init__(self, n):
		self.size = n
		self.half_the_size = int(n/2)
		self.reset()

	def reset(self):
		self.board = []
		value = 'B'
		for i in range(self.size):
			row = []
			for j in range(self.size):
				row.append(value)
				value = self.opponent(value)
			self.board.append(row)
			if self.size%2 == 0:
				value = self.opponent(value)

	def __str__(self):
		result = "  "
		for i in range(self.size):
			result += str(i) + " "
		result += "\n"
		for i in range(self.size):
			result += str(i) + " "
			for j in range(self.size):
				result += str(self.board[i][j]) + " "
			result += "\n"
		return result

	def valid(self, row, col):
		return row >= 0 and col >= 0 and row < self.size and col < self.size

	def contains(self, board, row, col, symbol):
		return self.valid(row,col) and board[row][col]==symbol

	def countSymbol(self, board, symbol):
		count = 0
		for r in range(self.size):
			for c in range(self.size):
				if board[r][c] == symbol:
					count += 1
		return count

	def opponent(self, player):
		if player == 'B':
			return 'W'
		else:
			return 'B'

	def distance(self, r1, c1, r2, c2):
		return abs(r1-r2 + c1-c2)

	def makeMove(self, player, move):
		self.board = self.nextBoard(self.board, player, move)

	def nextBoard(self, board, player, move):
		r1 = move[0]
		c1 = move[1]
		r2 = move[2]
		c2 = move[3]
		next = copy.deepcopy(board)
		if not (self.valid(r1, c1) and self.valid(r2, c2)):
			raise GameError
		if next[r1][c1] != player:
			raise GameError
		dist = self.distance(r1, c1, r2, c2)
		if dist == 0:
			if self.openingMove(board):
				next[r1][c1] = "."
				return next
			raise GameError
		if next[r2][c2] != ".":
			raise GameError
		jumps = int(dist/2)
		dr = int((r2 - r1)/dist)
		dc = int((c2 - c1)/dist)
		for i in range(jumps):
			if next[r1+dr][c1+dc] != self.opponent(player):
				raise GameError
			next[r1][c1] = "."
			next[r1+dr][c1+dc] = "."
			r1 += 2*dr
			c1 += 2*dc
			next[r1][c1] = player
		return next

	def openingMove(self, board):
		return self.countSymbol(board, ".") <= 1

	def generateFirstMoves(self, board):
		moves = []
		moves.append([0]*4)
		moves.append([self.size-1]*4)
		moves.append([self.half_the_size]*4)
		moves.append([(self.half_the_size)-1]*4)
		return moves

	def generateSecondMoves(self, board):
		moves = []
		if board[0][0] == ".":
			moves.append([0,1]*2)
			moves.append([1,0]*2)
			return moves
		elif board[self.size-1][self.size-1] == ".":
			moves.append([self.size-1,self.size-2]*2)
			moves.append([self.size-2,self.size-1]*2)
			return moves
		elif board[self.half_the_size-1][self.half_the_size-1] == ".":
			pos = self.half_the_size -1
		else:
			pos = self.half_the_size
		moves.append([pos,pos-1]*2)
		moves.append([pos+1,pos]*2)
		moves.append([pos,pos+1]*2)
		moves.append([pos-1,pos]*2)
		return moves

	def check(self, board, r, c, rd, cd, factor, opponent):
		if self.contains(board,r+factor*rd,c+factor*cd,opponent) and \
		   self.contains(board,r+(factor+1)*rd,c+(factor+1)*cd,'.'):
			return [[r,c,r+(factor+1)*rd,c+(factor+1)*cd]] + \
				   self.check(board,r,c,rd,cd,factor+2,opponent)
		else:
			return []

	def generateMoves(self, board, player):
		if self.openingMove(board):
			if player=='B':
				return self.generateFirstMoves(board)
			else:
				return self.generateSecondMoves(board)
		else:
			moves = []
			rd = [-1,0,1,0]
			cd = [0,1,0,-1]
			for r in range(self.size):
				for c in range(self.size):
					if board[r][c] == player:
						for i in range(len(rd)):
							moves += self.check(board,r,c,rd[i],cd[i],1,
												self.opponent(player))
			return moves

	def playOneGame(self, p1, p2, show):
		self.reset()
		while True:
			if show:
				print(self)
				print("player B's turn")
			move = p1.getMove(self.board)
			if move == []:
				print("Game over: W won.")
				return 'W'
			try:
				self.makeMove('B', move)
			except GameError:
				print("Game over: Invalid move by", p1.name)
				print(move)
				print(self)
				return 'W'
			if show:
				print(move)
				print(self)
				print("player W's turn")
			move = p2.getMove(self.board)
			if move == []:
				print("Game over: B won")
				return 'B'
			try:
				self.makeMove('W', move)
			except GameError:
				print("Game over: Invalid move by", p2.name)
				print(move)
				print(self)
				return 'B'
			if show:
				print(move)

	def playNGames(self, n, p1, p2, show):
		first = p1
		second = p2
		for i in range(n):
			print("Game", i)
			winner = self.playOneGame(first, second, show)
			if winner == 'B':
				first.won()
				second.lost()
				print(first.name, "wins")
			else:
				first.lost()
				second.won()
				print(second.name, "wins")
			first, second = second, first


class Player:
	name = "Player"
	wins = 0
	losses = 0
	def results(self):
		result = self.name
		result += " Wins:" + str(self.wins)
		result += " Losses:" + str(self.losses)
		return result
	def lost(self):
		self.losses += 1
	def won(self):
		self.wins += 1
	def reset(self):
		self.wins = 0
		self.losses = 0

	def initialize(self, side):
		abstract()

	def getMove(self, board):
		abstract()


class SimplePlayer(Game, Player):
	def initialize(self, side):
		self.side = side
		self.name = "Simple"
	def getMove(self, board):
		moves = self.generateMoves(board, self.side)
		n = len(moves)
		if n == 0:
			return []
		else:
			return moves[0]

class RandomPlayer(Game, Player):
	def initialize(self, side):
		self.side = side
		self.name = "Random"
	def getMove(self, board):
		moves = self.generateMoves(board, self.side)
		n = len(moves)
		if n == 0:
			return []
		else:
			return moves[random.randrange(0, n)]

class HumanPlayer(Game, Player):
	def initialize(self, side):
		self.side = side
		self.name = "Human"
	def getMove(self, board):
		moves = self.generateMoves(board, self.side)
		while True:
			print("Possible moves:", moves)
			n = len(moves)
			if n == 0:
				print("You must concede")
				return []
			index = input("Enter index of chosen move (0-"+ str(n-1) +
						  ") or -1 to concede: ")
			try:
				index = int(index)
				if index == -1:
					return []
				if 0 <= index <= (n-1):
					print("returning", moves[index])
					return moves[index]
				else:
					print("Invalid choice, try again.")
			except Exception as e:
				print("Invalid choice, try again.")


class MinimaxPlayer(Game, Player):
    def initialize(self, side,  depthLimit):
        self.side = side
        self.name = "Minimax Agent"
        self.limit = depthLimit
    def getMove(self, board):
        start = time.time()
        moves = self.generateMoves(board, self.side)
        n = len(moves)
        if n == 0:
            end = time.time()
            # print("ElapsedTime : ", end-start)
            return []
        else:
            values = []
            alpha = -float("inf")
            beta = float("inf")
            for move in moves:
                values.append(self.minimax(self.nextBoard(board, self.side, move), 1, alpha, beta))

                if max(values) > alpha:
                    alpha = max(values)

            end = time.time()
            # print("ElapsedTime : ", end-start)
            return moves[values.index(max(values))]
    def eval(self, board):
    	return self.piecesCount(board, self.side) - self.piecesCount(board, self.side)
        # return self.movablePieces(board, self.side) - (1.5 * self.movablePieces(board, self.opponent(self.side))) + (self.movesCount(board, self.side) - self.movesCount(board, self.opponent(self.side)))

    def piecesCount(self, board, player):
    	count = 0
    	for i in range(len(board)):
    		for j in range(len(board[0])):
    			if board[i][j] == player:
    				count += 1
    	return count
    def movesCount(self, board, player):
        return len(self.generateMoves(board, player))
    def movablePieces(self, board, player):
        moves = self.generateMoves(board, player)
        counter = 0
        pieces = []
        for move in moves:
            if [moves[counter][0], moves[counter][1]] not in pieces:
                pieces.append([moves[counter][0], moves[counter][1]])
            counter += 1
        return len(pieces)
    def extendPath(self, board, side):
        moves = self.generateMoves(board, side)
        boards = []
        for move in moves:
            boards.append(self.nextBoard(board, side, move))

        return boards

    def minimax(self, board, depth, alpha, beta):
        if depth >= self.limit:
            return self.eval(board)

        isMax = depth % 2 == 0

        if isMax:
            nextBoards = self.extendPath(board, self.side)
        else:
            nextBoards = self.extendPath(board, self.opponent(self.side))

        if not nextBoards:
            if isMax:
                return -float("inf")
            else:
                return float("inf")

        if isMax:
            maxEval = -float("inf")
            for nextBoard in nextBoards:
                thisEval = self.minimax(nextBoard, depth + 1, alpha, beta)
                maxEval = max(thisEval, maxEval)
                alpha = max(alpha, thisEval)
                if beta <= alpha:
                    break
            return maxEval
        else:
            minEval = float("inf")
            for nextBoard in nextBoards:
                thisEval = self.minimax(nextBoard, depth + 1, alpha, beta)
                minEval = min(thisEval, minEval)
                beta = min(beta, thisEval)
                if beta <= alpha:
                    break
            return minEval


class  Student(Game, Player):
	def evaluationFunction(self, state):
		movesB = self.generateMoves(state, 'B')
		movesW = self.generateMoves(state, 'W')
		if(self.side == 'W'):
			if(len(movesW) == 0):
				return -1*INF
			if(len(movesB) == 0):
				return INF
			return -1*len(movesB) + len(movesW)
		elif(self.side == 'B'):
			if(len(movesB) == 0):
				return -1*INF
			if(len(movesW) == 0):
				return INF
			return -1*len(movesW) + len(movesB)

	def initialize(self, side, depth):
		self.side = side
		self.depth = depth
		self.name = "Negin"

	def maxValue(self, state, N):
		moves = self.generateMoves(state, self.side)
		if(N == self.depth or len(moves) == 0):
			result = []
			result.append(self.evaluationFunction(state))
			result.append(0)
			return result
		v = -1*INF
		index = 0
		for i in range(0,len(moves)):
			M = max(v, self.minValue(self.nextBoard(state, self.side, moves[i]), N+1)[0])
			if(M > v):
				index = i
				v = M
		result = []
		result.append(v)
		result.append(index)
		return result

	def minValue(self, state, N):
		moves = self.generateMoves(state, self.side)
		if(N == self.depth or len(moves) == 0):
			result = []
			result.append(self.evaluationFunction(state))
			result.append(0)
			return result
		v = INF
		index = 0
		for i in range(0,len(moves)):
			M = min(v, self.maxValue(self.nextBoard(state, self.side, moves[i]), N+1)[0])
			if(M < v):
				index = i
				v = M
		result = []
		result.append(v)
		result.append(index)
		return result

	def maxValuePruning(self, state, A, B, N):
		moves = self.generateMoves(state, self.side)
		if(N == self.depth or len(moves) == 0):
			result = []
			result.append(self.evaluationFunction(state))
			result.append(0)
			return result
		v = -1*INF
		index = 0
		for i in range(0,len(moves)):
			M = max(v, self.minValuePruning(self.nextBoard(state, self.side, moves[i]), A, B, N+1)[0])
			if(M > v):
				index = i
				v = M
			if(v >= B):
				result = []
				result.append(v)
				result.append(index)
				return result
			A = max(A, v)
		result = []
		result.append(v)
		result.append(index)
		return result

	def minValuePruning(self, state, A, B, N):
		moves = self.generateMoves(state, self.opponent(self.side))
		if(N == self.depth or len(moves) == 0):
			result = []
			result.append(self.evaluationFunction(state))
			result.append(0)
			return result
		v = INF
		index = 0
		for i in range(0,len(moves)):
			M = min(v, self.maxValuePruning(self.nextBoard(state, self.opponent(self.side), moves[i]), A, B, N+1)[0])
			if(M < v):
				index = i
				v = M
			if(v <= A):
				result = []
				result.append(v)
				result.append(index)
				return result
			B = min(B, v)
		result = []
		result.append(v)
		result.append(index)
		return result

	def getMove(self, board):
		B = INF
		A = -1*INF
		result = self.maxValuePruning(board, A, B, 0)
		#result = self.maxValue(board, 0)
		index = result[1]
		if(len(self.generateMoves(board, self.side)) != 0):
			return self.generateMoves(board, self.side)[index]
		return []




if __name__ == '__main__':
    game = Game(8)

    student_w = Student(8)
    student_w.initialize('W',4)


    student_b = Student(8)
    student_b.initialize('B',4)





    minimax_b = MinimaxPlayer(8)
    minimax_b.initialize('B', 4)

    minimax_w = MinimaxPlayer(8)
    minimax_w.initialize('W', 4)

    print('Game1) B:student W:test_agent')
    game.playOneGame(student_b, minimax_w, False)

    print('Game2) W:student B:test_agent')
    game.playOneGame(minimax_b, student_w, False)

    print('Game3) B:student W:test_agent')
    game.playOneGame(student_b, minimax_w, False)

    print('Game4) W:student B:test_agent')
    game.playOneGame(minimax_b, student_w, False)

    print('Game5) B:student W:test_agent')
    game.playOneGame(student_b, minimax_w, False)

    print('Game6) W:student B:test_agent')
    game.playOneGame(minimax_b, student_w, False)
