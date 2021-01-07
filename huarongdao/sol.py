import queue
import sys
from queue import PriorityQueue as PQ


class State:
    def __init__(self):
        self.board = []
        self.posAvailable = [0, 0]
        self.cost = 0
        self.parentState = None
        self.h = 0
        self.prevMove = -1, "None"

    def copyParent(self, other):
        self.board = []
        for i in range(0, 5):
            col = []
            for j in range(0, 4):
                col.append(other.board[i][j])
            self.board.append(col)

        self.posAvailable[0] = other.posAvailable[0]
        self.posAvailable[1] = other.posAvailable[1]

        self.cost = other.cost + 1  # increment cost by 1
        self.parentState = other  # set parent to the current state

    # find the position relative to a piece
    # input: position of a piece, direction (left,right,up,down)
    # output: the position of the neighboring piece (row,col)
    def findPos(self, pos, direction):
        if direction == "left":
            row = pos[0]
            col = pos[1] - 1
        elif direction == "right":
            row = pos[0]
            col = pos[1] + 1
        elif direction == "up":
            row = pos[0] - 1
            col = pos[1]
        elif direction == "down":
            row = pos[0] + 1
            col = pos[1]
        else:
            (col, row) = (-1, -1)
        return row, col

    # readFile given and process the input to the state
    # input: file containing the puzzle
    # output: None, modified state
    def readFile(self, file):
        file1 = open(file, 'r')
        Lines = file1.readlines()
        row = 0
        col = 0
        count = 0
        cols = []
        for line in Lines:
            for i in range(0, 4):
                z = int(line[i])
                cols.append(z)
                if z == 0:
                    self.posAvailable[count] = (row, col)
                    count += 1
                col += 1
            self.board.append(cols)
            cols = []
            col = 0
            row += 1

    # move a piece to the available position
    # input: space, which space to move, direction (left,right,up,down)
    # output: newState, after the move
    def move(self, space, direction):
        newState = State()
        newState.copyParent(self)  # create a child state by copying from the info of its parent,
        # we will modify the child's info later

        row1, col1 = self.posAvailable[0]
        row2, col2 = self.posAvailable[1]

        pos = self.posAvailable[space]
        r, c = pos
        (row, col) = self.findPos(pos, direction)
        piece = int(self.board[row][col])

        if piece == 1:
            if direction == "left":
                newState.board[row1][col1 - 2] = 0
                newState.board[row2][col2 - 2] = 0
                newState.posAvailable = [(row1, col1 - 2), (row2, col2 - 2)]
            elif direction == "right":
                newState.board[row1][col1 + 2] = 0
                newState.board[row2][col2 + 2] = 0
                newState.posAvailable = [(row1, col1 + 2), (row2, col2 + 2)]
            elif direction == "up":
                newState.board[row1 - 2][col1] = 0
                newState.board[row2 - 2][col2] = 0
                newState.posAvailable = [(row1 - 2, col1), (row2 - 2, col2)]
            elif direction == "down":
                newState.board[row1 + 2][col1] = 0
                newState.board[row2 + 2][col2] = 0
                newState.posAvailable = [(row1 + 2, col1), (row2 + 2, col2)]

            newState.board[row1][col1] = 1
            newState.board[row2][col2] = 1

        elif piece == 7:
            newState.board[r][c] = 7
            newState.posAvailable[space] = (row, col)
            newState.board[row][col] = 0
        elif 2 <= piece <= 6:
            if direction == "left":
                if c - 2 >= 0 and newState.board[r][c - 2] == piece:
                    # horizontal
                    newState.board[r][c] = piece
                    newState.board[r][c - 2] = 0
                    newState.posAvailable[space] = (r, c - 2)
                else:
                    # vertical
                    newState.board[row1][col1 - 1] = 0
                    newState.board[row2][col2 - 1] = 0
                    newState.board[row1][col1] = piece
                    newState.board[row2][col2] = piece
                    newState.posAvailable = [(row1, col1 - 1), (row2, col2 - 1)]

            elif direction == "right":
                if c + 2 < 4 and newState.board[r][c + 2] == piece:
                    # horizontal
                    newState.board[r][c] = piece
                    newState.board[r][c + 2] = 0
                    newState.posAvailable[space] = (r, c + 2)
                else:
                    # vertical
                    newState.board[row1][col1 + 1] = 0
                    newState.board[row2][col2 + 1] = 0
                    newState.board[row1][col1] = piece
                    newState.board[row2][col2] = piece
                    newState.posAvailable = [(row1, col1 + 1), (row2, col2 + 1)]

            elif direction == "up":
                if r - 2 >= 0 and newState.board[r - 2][c] == piece:
                    # vertical
                    newState.board[r - 2][c] = 0
                    newState.board[r][c] = piece
                    newState.posAvailable[space] = (r - 2, c)
                else:  # horizontal
                    newState.board[row1][col1] = piece
                    newState.board[row2][col2] = piece
                    newState.board[row1 - 1][col1] = 0
                    newState.board[row2 - 1][col2] = 0
                    newState.posAvailable = [(row1 - 1, col1), (row2 - 1, col2)]

            elif direction == "down":
                if r + 2 < 5 and newState.board[r + 2][c] == piece:
                    # vertical
                    newState.board[r + 2][c] = 0
                    newState.board[r][c] = piece
                    newState.posAvailable[space] = (r + 2, c)
                else:  # horizontal
                    newState.board[row1][col1] = piece
                    newState.board[row2][col2] = piece
                    newState.board[row1 + 1][col1] = 0
                    newState.board[row2 + 1][col2] = 0
                    newState.posAvailable = [(row1 + 1, col1), (row2 + 1, col2)]
        newState.h = find_Val(newState)
        newState.prevMove = (space, direction)
        return newState

    def __str__(self):
        bb = ""
        for i in range(0, 5):
            line = ""
            for j in range(0, 4):
                line += str(self.board[i][j])
            bb += line + "\n"
        return bb

    def toString(self):
        s = ""
        for i in range(5):
            for j in range(4):
                s += str(self.board[i][j])
        return s

    # override the equal operator
    def __eq__(self, other):
        if self.board == other.board:
            return True
        else:
            return False

    # override the comparison operator <
    def __lt__(self, other):
        return self.h < other.h


# read_puzzle: which reads in an initial configuration of the puzzle
# from a file and store it as a state
# input: id, specific which text the function is trying to read
# output: a state
def read_puzzle(id):
    b = State()
    if id == "1":
        b.readFile("puzzle1.txt")
    elif id == "2":
        b.readFile("puzzle2.txt")
    return b


# get_successors: takes a state and returns a list of its successor states
# input: state, a state of huarongdao
# output: a list of its successor states
def get_successors(state, explored):
    successor = []

    space1 = state.posAvailable[0]
    space2 = state.posAvailable[1]

    row = space1[0]
    col = space1[1]
    space = 0

    flag = False

    direction = "right"
    if col != 3 and state.prevMove != (space, "left"):
        piece = state.board[row][col + 1]
        if piece == 1:
            if space2 == (row + 1, col) and state.board[row + 1][col + 1] == 1:
                flag = True
            elif space2 == (row - 1, col) and state.board[row - 1][col + 1] == 1:
                flag = True
        elif piece == 7:
            flag = True
        elif 2 <= piece <= 6:
            if col + 2 < 4 and state.board[row][col + 2] == piece:  # horizontal
                flag = True
            elif space2 == (row + 1, col) and state.board[row + 1][col + 1] == piece:
                flag = True
            elif space2 == (row - 1, col) and state.board[row - 1][col + 1] == piece:
                flag = True

    if flag:
        newState = state.move(space, direction)
        if newState.toString() not in explored:
            successor.append(newState)
    flag = False

    direction = "left"

    if col != 0 and state.prevMove != (space, "right"):
        piece = state.board[row][col - 1]
        if piece == 1:
            if space2 == (row - 1, col) and state.board[row - 1][col - 1] == 1:
                flag = True
            elif space2 == (row + 1, col) and state.board[row + 1][col - 1] == 1:
                flag = True
        elif piece == 7:
            flag = True
        elif 2 <= piece <= 6:
            if col - 2 >= 0 and state.board[row][col - 2] == piece:  # horizontal
                flag = True
            elif space2 == (row + 1, col) and state.board[row + 1][col - 1] == piece:
                flag = True
            elif space2 == (row - 1, col) and state.board[row - 1][col - 1] == piece:
                flag = True
    if flag:
        newState = state.move(space, direction)
        if newState.toString() not in explored:
            successor.append(newState)
    flag = False

    direction = "up"
    if row != 0 and state.prevMove != (space, "down"):
        piece = state.board[row - 1][col]
        if piece == 1:
            if space2 == (row, col + 1) and state.board[row - 1][col + 1] == 1:
                flag = True
            elif space2 == (row, col - 1) and state.board[row - 1][col - 1] == 1:
                flag = True
        elif piece == 7:
            flag = True
        elif 2 <= piece <= 6:
            # print("piece2-6")
            if row - 2 >= 0 and state.board[row - 2][col] == piece:  # vertical
                flag = True
            elif space2 == (row, col + 1) and state.board[row - 1][col + 1] == piece:
                flag = True
            elif space2 == (row, col - 1) and state.board[row - 1][col - 1] == piece:
                flag = True
    if flag:
        newState = state.move(space, direction)
        if newState.toString() not in explored:
            successor.append(newState)
    flag = False

    direction = "down"
    if row != 4 and state.prevMove != (space, "up"):
        piece = state.board[row + 1][col]
        if piece == 1:
            if space2 == (row, col + 1) and state.board[row + 1][col + 1] == 1:
                flag = True
            elif space2 == (row, col - 1) and state.board[row + 1][col - 1] == 1:
                flag = True
        elif piece == 7:
            flag = True
        elif 2 <= piece <= 6:
            if row + 2 < 5 and state.board[row + 2][col] == piece:  # vertical
                flag = True
            elif space2 == (row, col + 1) and state.board[row + 1][col + 1] == piece:
                flag = True
            elif space2 == (row, col - 1) and state.board[row + 1][col - 1] == piece:
                flag = True
    if flag:
        newState = state.move(space, direction)
        if newState.toString() not in explored:
            successor.append(newState)
    flag = False

    row = space2[0]
    col = space2[1]
    space = 1
    # left of space2
    direction = "left"
    if col != 0 and state.prevMove != (space, "right"):
        piece = state.board[row][col - 1]
        if piece == 7:
            flag = True
        elif 2 <= piece <= 6:
            if col - 2 >= 0:  # horizontal piece
                if state.board[row][col - 2] == piece:
                    flag = True
    if flag:
        newState = state.move(space, direction)
        if newState.toString() not in explored:
            successor.append(newState)
    flag = False

    direction = "right"
    if col != 3 and state.prevMove != (space, "left"):
        piece = state.board[row][col + 1]
        if piece == 7:
            flag = True
        elif 2 <= piece <= 6:
            if col + 2 < 4 and state.board[row][col + 2] == piece:  # horizontal piece
                flag = True
    if flag:
        newState = state.move(space, direction)
        if newState.toString() not in explored:
            successor.append(newState)
    flag = False

    direction = "down"
    if row != 4 and state.prevMove != (1, "up"):
        piece = state.board[row + 1][col]
        if piece == 7:
            flag = True
        elif 2 <= piece <= 6:
            if row + 2 < 5 and state.board[row + 2][col] == piece:  # vertical piece
                flag = True
    if flag:
        newState = state.move(space, direction)
        if newState.toString() not in explored:
            successor.append(newState)
    flag = False

    direction = "up"
    if row != 0 and state.prevMove != (1, "down"):
        piece = state.board[row - 1][col]
        if piece == 7:
            flag = True
        elif 2 <= piece <= 6:
            if row - 2 >= 0 and state.board[row - 2][col] == piece:  # vertical piece
                flag = True
    if flag:
        newState = state.move(space, direction)
        if newState.toString() not in explored:
            successor.append(newState)
    '''
    print("\nsuccessor:")
    for i in successor:
        print(i)
        print("") '''

    return successor


# get_cost: takes a path and returns the cost of the path
# input: state
# output: cost
def get_cost(state):
    # return 0
    return state.cost


# get_heuristic(state): takes a state and returns the heuristic estimate of the cost
# of the optimal path from the state to a goal state
# input: state
# output: heuristic value - h value
def get_heuristic(state):
    for i in range(5):
        for j in range(4):
            if state.board[i][j] == 1:
                return abs(i - 3) + abs(j - 1)


# find_val(state): finds the priority of a state for the A* search
# input: state
# output: the priority (an integer)
def find_Val(state):
    return get_heuristic(state) + get_cost(state)


# a_star: solves the Hua Rong Dao sliding puzzle using A* search given the
# initial configuration of the puzzle
# input: initial_state
# output: number of states expanded, the path of the solution
def a_star(initial_state):
    frontier = PQ()
    frontier.put((0, initial_state))
    explored = set()
    explored.add(initial_state.toString())

    expanded = 0

    while frontier:
        exploring = frontier.get()[1]
        s = get_successors(exploring, explored)

        for state in s:
            if state.board[3][1] == 1 and state.board[3][2] == 1 and state.board[4][1] == 1 and state.board[4][
                2] == 1:
                # backtrack path
                path = []
                while state != initial_state:
                    path.append(state)
                    state = state.parentState
                path.append(initial_state)
                print("done!")
                return path[::-1], expanded
            explored.add(state.toString())
            expanded += 1
            frontier.put((state.h, state))
    return (-1, -1)


# dfs: solves the Hua Rong Dao sliding puzzle using dfs search given the
# initial configuration of the puzzle
# input: initial_state (State)
# output: number of states expanded (int), the path of the solution (list of states)
def dfs(initial_state):
    # dfs is LIFO
    frontier = queue.LifoQueue()
    frontier.put(initial_state)
    explored = set()
    explored.add(initial_state.toString())

    expanded = 0

    while frontier:
        exploring = frontier.get()
        s = get_successors(exploring, explored)

        for state in s:
            # check if the current state is a goal state
            if state.board[3][1] == 1 and state.board[3][2] == 1 and state.board[4][1] == 1 and state.board[4][
                2] == 1:
                # backtrack path
                path = []
                while state != initial_state:
                    path.append(state)
                    state = state.parentState
                path.append(initial_state)
                print("done!")
                return path[::-1], expanded
            explored.add(state.toString())
            expanded += 1
            frontier.put(state)
    return (-1, -1)


# output the solution to specified files
# input: id - specify which puzzle to solve, method - specify which method to solve
# output: a file, using the specified approach to solve the puzzle given
def output(id, method):
    board = read_puzzle(id)
    if method == 1:
        p, g = a_star(board)
        if id == "1":
            f = open("puzzle1sol_astar.txt", "w")
        else:
            f = open("puzzle2sol_astar.txt", "w")
    else:
        if id == "1":
            f = open("puzzle1sol_dfs.txt", "w")
        else:
            f = open("puzzle2sol_dfs.txt", "w")
        p, g = dfs(board)

    f.write("Initial state:\n")
    f.write(p[0].__str__())
    f.write("\n")
    f.write("Cost of the solution: " + str(p[-1].cost))
    f.write("\n\n")
    f.write("Number of states expanded: " + str(g))
    f.write("\n\nSolution:\n\n")
    count = 0
    for i in p:
        f.write(str(count))
        f.write("\n")
        f.write(i.__str__())
        f.write("\n")
        count += 1


id = sys.argv[1]
output(id, 1)
output(id, 2)
