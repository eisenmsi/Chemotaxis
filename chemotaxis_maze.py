import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
import heapq

# Ideas:
#   Show tiles visited by the agent using transparent/smaller squares
#   Calculate information flow of the agent or similar (think of other cool information theory vis)

# Create figure and axis
fig, ax = plt.subplots()

# Initialize stuff
agent_row = 0
agent_col = 0

grid_conc = np.zeros((17, 15))
grid_pass = np.ones(grid_conc.shape)

ligand_centers = []

seed = 3481
np.random.seed(seed)


def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def a_star(grid, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {pos: float("inf") for pos in np.ndindex(grid.shape)}
    g_score[start] = 0
    f_score = {pos: float("inf") for pos in np.ndindex(grid.shape)}
    f_score[start] = heuristic(start, goal)

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return len(path) - 1  # Subtract 1 to get the number of tiles

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_pos = current[0] + dx, current[1] + dy
            if not (
                0 <= next_pos[0] < grid.shape[0] and 0 <= next_pos[1] < grid.shape[1]
            ):
                continue
            if grid[next_pos] == 0:
                continue

            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score[next_pos]:
                came_from[next_pos] = current
                g_score[next_pos] = tentative_g_score
                f_score[next_pos] = tentative_g_score + heuristic(next_pos, goal)
                heapq.heappush(open_list, (f_score[next_pos], next_pos))

    return -1  # No path found


def generate_maze(width, height):
    maze = np.zeros((height, width), dtype=int)

    def is_valid(x, y):
        return 0 <= x < width and 0 <= y < height and maze[y, x] == 0

    def create_maze(x, y):
        maze[y, x] = 1
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        np.random.shuffle(directions)

        for dx, dy in directions:
            new_x, new_y = x + 2 * dx, y + 2 * dy
            if is_valid(new_x, new_y):
                maze[new_y, new_x] = 1
                maze[y + dy, x + dx] = 1
                create_maze(new_x, new_y)

    create_maze(0, 0)

    return maze.transpose()


def random_position(width, height):
    return np.random.randint(0, width - 1), np.random.randint(0, height - 1)


def make_reachable(maze):
    height, width = maze.shape
    for y in range(height):
        for x in range(width):
            if maze[y, x] == 1 and np.random.random() < 0.2:
                maze[y, x] = 0


maze_width = grid_pass.shape[0]
maze_height = grid_pass.shape[1]

grid_pass = generate_maze(maze_width, maze_height)
grid_pass[0][0] = 1  # Ensure the starting point is passable


def draw():
    ax.clear()

    # Draw grid
    for i in range(grid_conc.shape[0]):
        ax.axvline(x=i, color="black", linewidth=1)
    for i in range(grid_conc.shape[1]):
        ax.axhline(y=i, color="black", linewidth=1)

    # Show concentration
    max_concentration = np.amax(grid_conc)
    if max_concentration > 0:
        norm = mcolors.Normalize(vmin=0, vmax=max_concentration)
        cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "red"])
        for i, j in np.ndindex(grid_conc.shape):
            value = grid_conc[i][j]
            color = cmap(norm(value))
            if not grid_pass[i][j] == 0:
                rect = patches.Rectangle(
                    (i, j), 1, 1, linewidth=1, edgecolor="black", facecolor=color
                )
                ax.add_patch(rect)
    for i, j in np.ndindex(grid_pass.shape):
        if grid_pass[i][j] == 0:
            rect = patches.Rectangle(
                (i, j), 1, 1, linewidth=1, edgecolor="black", facecolor="black"
            )
            ax.add_patch(rect)

    # Mark the agent's position
    rect = patches.Rectangle(
        (agent_col + 0.17, agent_row + 0.17),
        0.66,
        0.66,
        linewidth=1,
        edgecolor="blue",
        facecolor="blue",
    )
    ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, grid_conc.shape[0])
    ax.set_ylim(0, grid_conc.shape[1])
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
    goal_col = clamp(agent_col + direction[0], 0, grid_conc.shape[0] - 1)
    goal_row = clamp(agent_row + direction[1], 0, grid_conc.shape[1] - 1)
    if grid_pass[goal_col][goal_row] == 1:  # Do not move into walls
        agent_row = goal_row
        agent_col = goal_col
    # Update the display afterwards
    draw()


def clamp(value, small, large):
    return max(small, min(value, large))


# Idea: update concentration depening on current maze state
# Idea: add concentration decay / removal  when agent "has done its job"
def on_click(event):
    if event.button == 1:  # Left-Click
        global ligand_centers
        ligand_centers.append((int(event.xdata), int(event.ydata)))
        add_concentration(int(event.xdata), int(event.ydata))
        draw()
    if event.button == 3:  # Right-Click
        grid_pass[int(event.xdata)][int(event.ydata)] += 1
        grid_pass[int(event.xdata)][int(event.ydata)] %= 2
        recalculate_concentration()
        draw()


def recalculate_concentration():
    global ligand_centers, grid_conc
    grid_conc = np.zeros(grid_conc.shape)
    for center in ligand_centers:
        add_concentration(center[0], center[1])


def set_concentration(x, y):
    # For every cell in the grid, set the concentration
    global grid_conc
    for i, j in np.ndindex(grid_conc.shape):
        c = calculate_concentration(i, j, x, y)
        grid_conc[i][j] = c


def add_concentration(x, y):
    # For every cell in the grid, add the concentration
    global grid_conc
    for i, j in np.ndindex(grid_conc.shape):
        c = calculate_concentration(i, j, x, y)
        grid_conc[i][j] += c


def calculate_concentration(i, j, x, y):
    # dist = math.sqrt((x - i) ** 2 + (y - j) ** 2)  # euclidean distance between (i, j) and (x, y)
    if grid_pass[i][j] == 0:
        return -1
    dist = a_star(grid_pass, (i, j), (x, y))
    if dist == -1:
        return -1
    # center_exponent = 4
    # start_exponent = 1
    # origin_to_center = 0.5 * math.sqrt(grid_conc.shape[0] ** 2 + grid_conc.shape[1] ** 2)
    # exponent = (1 - 2 * dist / origin_to_center) * (center_exponent - start_exponent) + start_exponent
    return 1 / (dist + 1)


def calc_gradient():
    # Return "gradient" as information on directions that should be chosen
    x = agent_col
    y = agent_row

    if x + 1 >= grid_conc.shape[0]:
        right = -1
    else:
        right = grid_conc[x + 1][y]
    if x - 1 < 0:
        left = -1
    else:
        left = grid_conc[x - 1][y]
    if y + 1 >= grid_conc.shape[1]:
        above = -1
    else:
        above = grid_conc[x][y + 1]
    if y - 1 < 0:
        below = -1
    else:
        below = grid_conc[x][y - 1]

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
    if (agent_col, agent_row) in ligand_centers:
        ligand_centers.remove((agent_col, agent_row))
        recalculate_concentration()
        draw()


ani = animation.FuncAnimation(fig, update, frames=100, interval=250)

draw()
plt.show()
