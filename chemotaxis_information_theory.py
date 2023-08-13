import math
import numpy as np

# Ideas:
#   Show tiles visited by the agent using transparent/smaller squares
#   Calculate information flow of the agent or similar (think of other cool information theory vis)

grid = np.zeros((11, 11))

seed = 128
np.random.seed(seed)


def set_concentration(x, y):
    # For every cell in the grid, set the concentration
    global grid
    for i, j in np.ndindex(grid.shape):
        c = calculate_concentration(i, j, x, y)
        grid[i][j] = c


def calculate_concentration(i, j, x, y):
    dist = math.sqrt(
        (x - i) ** 2 + (y - j) ** 2
    )  # euclidean distance between (i, j) and (x, y)
    center_exponent = 4
    start_exponent = 1
    origin_to_center = 0.5 * math.sqrt(grid.shape[0] ** 2 + grid.shape[1] ** 2)
    exponent = (1 - 2 * dist / origin_to_center) * (
        center_exponent - start_exponent
    ) + start_exponent
    return 10**exponent


def calc_gradient(x, y):
    # Return "gradient" as information on directions that should be chosen

    if x + 1 >= grid.shape[0]:
        right = -1
    else:
        right = grid[x + 1][y]
    if x - 1 < 0:
        left = -1
    else:
        left = grid[x - 1][y]
    if y + 1 >= grid.shape[1]:
        above = -1
    else:
        above = grid[x][y + 1]
    if y - 1 < 0:
        below = -1
    else:
        below = grid[x][y - 1]

    return [
        i
        for i, j in enumerate([above, below, right, left])
        if j == max([above, below, right, left])
    ]


set_concentration(grid.shape[0] // 2, grid.shape[1] // 2)


def calc_start_pos_entropy():  # H(X)
    H_X = 0
    for x, y in np.ndindex(grid.shape):
        H_X -= (1 / grid.size) * math.log2(1 / grid.size)  # -=p(x) * log(p(x))
    return H_X


# simulative
def calc_actor_entropy():  # H(Y); 6 steps
    sequences = {}  # U, D, R, L
    limit = grid.size * 4_096
    for i in range(limit):
        x = np.random.randint(0, grid.shape[0])
        y = np.random.randint(0, grid.shape[1])
        seq = simulate(x, y, stop=False, bear_trap=True)
        if seq in sequences:
            sequences[seq] += 1 / limit
        else:
            sequences[seq] = 1 / limit

    H_Y = 0
    for seq in sequences.keys():
        H_Y -= (sequences[seq]) * math.log2(sequences[seq])
    return H_Y


def calc_mutual_entropy():  # H(X,Y)
    # calculate p(actor_sequence|start_pos) ~ p(Y|X=x) * p(X) = p(X,Y)
    positions = {}
    for x, y in np.ndindex(grid.shape):
        start_x = x
        start_y = y
        sequences = {}  # U, D, R, L
        limit = int(4_096)
        for i in range(limit):
            seq = simulate(x, y, stop=False, bear_trap=True)
            if seq in sequences:
                sequences[seq] += 1 / limit
            else:
                sequences[seq] = 1 / limit
        positions[str(start_x) + "," + str(start_y)] = sequences

    H_XY = 0
    for x, y in np.ndindex(grid.shape):
        sequences = positions[str(x) + "," + str(y)]
        for seq in sequences.keys():
            H_XY -= (sequences[seq] * (1 / grid.size)) * math.log2(
                sequences[seq] * (1 / grid.size)
            )
    return H_XY


def simulate(x, y, stop=True, bear_trap=False):
    actions = ["U", "D", "R", "L", "S"]
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
    seq = ""
    indices = calc_gradient(x, y)
    index = np.random.choice(indices)
    for i in range(8):
        if bear_trap and x == grid.shape[0] // 2 and y == grid.shape[1] // 2:
            break
        if len(indices) < 4 or not stop:
            seq += actions[index]
            x += dirs[index][0]
            y += dirs[index][1]
        if len(indices) == 4 and stop:
            seq += "S"
        indices = calc_gradient(x, y)
        index = np.random.choice(indices)
    return seq


if __name__ == "__main__":
    print("Calcluating H(X)")
    H_X = calc_start_pos_entropy()
    print(H_X)

    print("Calculating H(Y)")
    H_Y = calc_actor_entropy()
    print(H_Y)

    print("Calculating H(X,Y)")
    H_XY = calc_mutual_entropy()
    print(H_XY)

    print("I(X;Y) is")
    print(H_X + H_Y - H_XY)
