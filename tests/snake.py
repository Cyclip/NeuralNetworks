import math
import random
import copy
import sys
sys.path.insert(0, "../src")
from main import Layer, Network, Neuron, Connection, IONeuron

GRID_SIZE = 30
MAX_LIFE  = 10

MAX_DISTANCE = math.sqrt(2 * (GRID_SIZE ** 2))

def normalize(n, min_, max_):
    """Normalize n between a range.

    Args:
        n (float): Value to normalize
        min_ (float): Maximum value
        max_ (float): Minimum value

    Returns:
        float: Number between -1 and 1
    """
    return 2 * ((n - min_) / (max_ - min_)) - 1


class Cell:
    """Possible values for each cell:
    0   Empty cell
    1   Snake head
    2   Snake body
    3   Food cell

    Any value below 0:  Lifetime of snake body
                        Increments by 1 until it hits 0
    """
    def __init__(self):
        self.__value = 0
        self.__life = 0
    
    def get_val(self):
        return self.__value
    
    def get_life(self):
        return self.__life if self.__value == 2 else None

    def get_name(self):
        if self.__value == 0:
            return "empty"
        elif self.__value == 1:
            return "head"
        elif self.__value == 2:
            return "body"
        elif self.__value == 3:
            return "food"

    def set_val(self, val):
        # Do not use for body
        self.__value = val

    def set_body(self, life):
        self.__value = 2
        self.__life = life

    def update(self):
        if self.__value == 2:
            if self.__life <= 0:
                self.__value = 0
            else:
                self.__life -= 1


class Grid:
    def __init__(self, size):
        # gen empty grid
        self.__grid = []

        for i in range(size):
            row = []
            for cell in range(size):
                row.append(Cell())
            self.__grid.append(row)
    
    def get_pos(self, coords):
        # get cell at coords
        return self.__grid[coords[1]][coords[0]]
    
    def update(self):
        for row in self.__grid:
            for cell in row:
                cell.update()
    
    def __str__(self):
        rows = []
        for row in self.__grid:
            rows.append(
                " ".join(map(self.__dispCell, row))
            )
        
        return "\n".join(rows)
    
    def __dispCell(self, cell):
        if cell.get_val() == 0:
            return "-"
        elif cell.get_val()  == 1:
            return "■"
        elif cell.get_val()  == 2:
            return "□"
        elif cell.get_val()  == 3:
            return "◊"


class Game:
    def __init__(self, nn):
        self.grid = Grid(GRID_SIZE)

        # player
        self.headPos = [
            random.randint(0, GRID_SIZE - 1),
            random.randint(0, GRID_SIZE - 1),
        ]
        self.direction = "up"
        self.length = 4
        self.score = 0

        # ai
        self.movesAvailable = GRID_SIZE * 2
        self.movesMade = 0
        self.neuralNetwork = nn
        # fitness = 1000(score) + 1(movesMade)
        self.fitness = 0

        self.moveFood()
    
    def run(self):

        while (self.score < 10) and not (self.movesAvailable < 1):
            # make a move
            self.movesAvailable -= 1
            self.fitness += 1
            self.set_inputs()

            self.neuralNetwork.start()
            if not self.setDirection(self.neuralNetwork.get_highest_output()[0]):
                # print("Turned into itself")
                break
            
            # for output in self.neuralNetwork.get_outputs():
            #     print(f"{output.name:<8}: {round(output.value, 4):<8}" + '|' * round(40 * output.value))

            self.grid.update()
            
            # replace head with body
            self.grid.get_pos(self.headPos).set_body(self.length)

            # place head
            self.move_head()
            self.grid.get_pos(self.headPos).set_val(1)

            # if head touches food
            if self.headPos == self.foodPos:
                # print("Touched food")
                self.score += 1
                self.length += 1
                self.movesAvailable *= 1.5
                self.fitness += 1000
                self.moveFood()
            
            # if head touches body
            if self.grid.get_pos(self.headPos).get_name() == "body":
                # print("Ran into body")
                break

            # print(f"Fitness {self.fitness}")
            # print(f"Head {self.headPos}   Food {self.foodPos}   Score {self.score}   Moves {self.movesMade} ({self.movesAvailable} left)")
            # print(str(self.grid) + "\n\n")
            # time.sleep(0.2)
        
        return self.fitness

    def setDirection(self, newDir):
        combo = [newDir, self.direction]
        if "up" in combo and "down" in combo:
            return False
        if "left" in combo and "right" in combo:
            return False

        self.direction = newDir
        return True

    def moveFood(self):
        # dont spawn food in body/head
        self.foodPos = None

        while True:
            self.foodPos = [
                random.randint(0, GRID_SIZE - 1),
                random.randint(0, GRID_SIZE - 1),
            ]

            if self.grid.get_pos(self.foodPos).get_val() == 0:
                break

        self.grid.get_pos(self.foodPos).set_val(3)
    
    def move_head(self):
        # move in direction
        if self.direction == "down":
            self.headPos[1] += 1
        elif self.direction == "up":
            self.headPos[1] -= 1
        elif self.direction == "left":
            self.headPos[0] -= 1
        elif self.direction == "right":
            self.headPos[0] += 1
        
        self.headPos[0] = self.wrap(self.headPos[0], GRID_SIZE - 1)
        self.headPos[1] = self.wrap(self.headPos[1], GRID_SIZE - 1)
    

    def wrap(self, val, max):
        if val > max:
            return 0
        elif val < 0:
            return max
        
        return val
    
    def set_inputs(self):
        headx = normalize(self.headPos[0], 0, 30)
        heady = normalize(self.headPos[1], 0, 30)
        foodx = normalize(self.foodPos[0], 0, 30)
        foody = normalize(self.foodPos[1], 0, 30)

        if self.direction == "up":
            direction = 1
        elif self.direction == "right":
            direction = 1/3
        elif self.direction == "down":
            direction = -1/3
        else:
            direction = -1
        
        foodDistance = normalize(math.sqrt(
            (self.headPos[0] - self.foodPos[0])**2 + (self.headPos[1] - self.foodPos[1])**2
            ),
            0, MAX_DISTANCE
        )
        
        self.neuralNetwork.set_inputs((
            headx,
            heady,
            foodx,
            foody,
            direction,
            self.cell_to_input(self.get_relative(0, -1)),  # up1
			self.cell_to_input(self.get_relative(0, -2)),  # up2
			self.cell_to_input(self.get_relative(0, -3)),  # up3
			self.cell_to_input(self.get_relative(0, -4)),  # up4
			self.cell_to_input(self.get_relative(0, 1)),  # down1
			self.cell_to_input(self.get_relative(0, 2)),  # down2
			self.cell_to_input(self.get_relative(0, 3)),  # down3
			self.cell_to_input(self.get_relative(0, 4)),  # down4
			self.cell_to_input(self.get_relative(-1, 0)),  # left1
			self.cell_to_input(self.get_relative(-2, 0)),  # left2
			self.cell_to_input(self.get_relative(-3, 0)),  # left3
			self.cell_to_input(self.get_relative(-4, 0)),  # left4
			self.cell_to_input(self.get_relative(1, 0)),  # right1
			self.cell_to_input(self.get_relative(2, 0)),  # right2
			self.cell_to_input(self.get_relative(3, 0)),  # right3
			self.cell_to_input(self.get_relative(4, 0)),  # right4
			self.cell_to_input(self.get_relative(1, -1)),  # upright1
			self.cell_to_input(self.get_relative(2, -2)),  # upright2
			self.cell_to_input(self.get_relative(3, -3)),  # upright3
			self.cell_to_input(self.get_relative(4, -4)),  # upright4
			self.cell_to_input(self.get_relative(1, 1)),  # downright1
			self.cell_to_input(self.get_relative(2, 2)),  # downright2
			self.cell_to_input(self.get_relative(3, 3)),  # downright3
			self.cell_to_input(self.get_relative(4, 4)),  # downright4
			self.cell_to_input(self.get_relative(-1, 1)),  # downleft1
			self.cell_to_input(self.get_relative(-2, 2)),  # downleft2
			self.cell_to_input(self.get_relative(-3, 3)),  # downleft3
			self.cell_to_input(self.get_relative(-4, 4)),  # downleft4
			self.cell_to_input(self.get_relative(-1, -1)),  # upleft1
			self.cell_to_input(self.get_relative(-2, -2)),  # upleft2
			self.cell_to_input(self.get_relative(-3, -3)),  # upleft3
			self.cell_to_input(self.get_relative(-4, -4)),  # upleft4
        ))
    
    def get_relative(self, x, y):
        return self.grid.get_pos((
            self.wrap(self.headPos[0] + x, 29),
            self.wrap(self.headPos[1] + y, 29),
        ))
    
    def cell_to_input(self, cell):
        val = cell.get_val()

        if val == 2:
            return -1
        elif val == 0:
            return 0
        elif val == 3:
            return 1
         


def genNeurons(n):
    neurons = []
    for i in range(n):
        neurons.append(
            Neuron(random.uniform(-0.5, 0.5))
        )
    
    return neurons


"""
Inputs:
- Head x    ]
- Head y    ] Represented as normalized from
- Food x    ] bounds of grid
- Food y    ] 
- Direction
    up      1
    right   1/3
    down    -1/3
    left    -1
- Distance from food

- 4 cells up/down/left/right
- 4 cells in all diagonals

Value   Name    Input
2       Body    -1
0       Empty   0
3       Food    1

Outputs:
- Up
- Down
- Left
- Right
"""

# inputlayer
nn = Network((
    IONeuron("headX"),
    IONeuron("headY"),
    IONeuron("foodX"),
    IONeuron("foodY"),
    IONeuron("direction"),
    IONeuron("foodDistance"),
    IONeuron("up1"),
	IONeuron("up2"),
	IONeuron("up3"),
	IONeuron("up4"),
	IONeuron("down1"),
	IONeuron("down2"),
	IONeuron("down3"),
	IONeuron("down4"),
	IONeuron("left1"),
	IONeuron("left2"),
	IONeuron("left3"),
	IONeuron("left4"),
	IONeuron("right1"),
	IONeuron("right2"),
	IONeuron("right3"),
	IONeuron("right4"),
	IONeuron("upright1"),
	IONeuron("upright2"),
	IONeuron("upright3"),
	IONeuron("upright4"),
	IONeuron("downright1"),
	IONeuron("downright2"),
	IONeuron("downright3"),
	IONeuron("downright4"),
	IONeuron("downleft1"),
	IONeuron("downleft2"),
	IONeuron("downleft3"),
	IONeuron("downleft4"),
	IONeuron("upleft1"),
	IONeuron("upleft2"),
	IONeuron("upleft3"),
	IONeuron("upleft4"),
))

# hidden layer
nn.add_layer(
    Layer(genNeurons(24))
)


# hidden layer
nn.add_layer(
    Layer(genNeurons(16))
)

# output layer
nn.add_layer(
    Layer((
        IONeuron("up"),
        IONeuron("down"),
        IONeuron("left"),
        IONeuron("right"),
    ))
)

nn.build_connections()


def getTop(li, perc):
    """Get the top perc% performing networks.

    Args:
        li (list<Network>): List of networks
        perc (float): Top percentage threshold (0 - 1)

    Returns:
        list<Network>: Top performing networks
    """
    return li[len(li) - int(len(li) * perc):]


def doGeneration(population):
    # run all games
    results = []

    for p in population:
        game = Game(p)
        fitness = g.run()
        results.append((game, fitness))
    
    # identify top performing
    results = sorted(results, key=lambda x: x[1])
    topPerforming = getTop(results, 0.1)

    # get new population
    newPopulation = 

networks = []

while len(networks) < 200:
    nn.mutate(0.9, 0.5)

    g = Game(nn)
    r = g.run()

    networks.append((nn, r))