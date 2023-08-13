import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np

# Ideas:
#   Show tiles visited by the agent using transparent/smaller squares
#   Calculate information flow of the agent or similar (think of other cool information theory vis)

# Create figure and axis
fig, ax = plt.subplots()

# Initialize stuff
agent_row = 5
agent_col = 5

grid = np.zeros((11, 11))

seed = 128
np.random.seed(seed)


def draw():
    ax.clear()

    # Draw grid
    for i in range(grid.shape[0]):
        ax.axvline(x=i, color="black", linewidth=1)
    for i in range(grid.shape[1]):
        ax.axhline(y=i, color="black", linewidth=1)

    # Show concentration
    max_concentration = np.amax(grid)
    if max_concentration > 0:
        norm = mcolors.Normalize(vmin=0, vmax=math.log2(max_concentration))
        cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])
        for i, j in np.ndindex(grid.shape):
            value = math.log2(grid[i][j])
            color = cmap(norm(value))
            rect = patches.Rectangle(
                (i, j), 1, 1, linewidth=1, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)

    # Mark the agent's position
    rect = patches.Rectangle(
        (agent_col + 0.17, agent_row + 0.17),
        0.66,
        0.66,
        linewidth=1,
        edgecolor="orange",
        facecolor="orange",
    )
    ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, grid.shape[0])
    ax.set_ylim(0, grid.shape[1])
    ax.set_aspect("equal", adjustable="box")

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.draw()


def tumble():
    # Choose a random direction for tumbling
    # Chosen with a uniform distribution as of now
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    viable_dir_indices = calc_gradient()
    dir_index = np.random.choice(viable_dir_indices)
    return dirs[dir_index]


def move(direction):
    # Set the internal agent position according to its movement
    global agent_row, agent_col
    agent_col = clamp(agent_col + direction[0], 0, grid.shape[0] - 1)
    agent_row = clamp(agent_row + direction[1], 0, grid.shape[1] - 1)
    # Update the display afterwards
    draw()


def clamp(value, small, large):
    return max(small, min(value, large))


def on_click(event):
    if event.button == 1:  # Left-click
        set_concentration(int(event.xdata), int(event.ydata))
        draw()


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


def calc_gradient():
    # Return "gradient" as information on directions that should be chosen
    x = agent_col
    y = agent_row

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


fig.canvas.mpl_connect(
    "button_press_event", on_click
)  # Execute "on_click" when a button press event occurrs


def update(frame):  # Called every frame (i.e. every 500ms)
    move(tumble())


ani = animation.FuncAnimation(fig, update, frames=100, interval=500)

draw()
plt.show()
