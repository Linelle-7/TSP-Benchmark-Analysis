import random
import math
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
GRID_SIZE = 1000

BASE_PROB = 0.002           # bruit urbain
CLUSTER_PROB = 0.015        # propagation locale
LAMBDA_H = 6.0              # portée horizontale
LAMBDA_V = 10.0             # portée verticale

NUM_BLOCKS = 12             # quartiers planifiés
BLOCK_SIZE = 80
BLOCK_SPACING = 120
BLOCK_JITTER = 3            # imperfection des grilles

points = []

# Tracking last points
last_point_in_row = [None] * GRID_SIZE
last_point_in_col = [None] * GRID_SIZE

prev_row = [0] * GRID_SIZE
curr_row = [0] * GRID_SIZE

# -----------------------------
# 1. Generate regular blocks
# -----------------------------
for _ in range(NUM_BLOCKS):
    bx = random.randint(50, GRID_SIZE - BLOCK_SIZE - 50)
    by = random.randint(50, GRID_SIZE - BLOCK_SIZE - 50)

    for y in range(by, by + BLOCK_SIZE, BLOCK_SPACING // 4):
        for x in range(bx, bx + BLOCK_SIZE, BLOCK_SPACING // 4):
            jx = x + random.randint(-BLOCK_JITTER, BLOCK_JITTER)
            jy = y + random.randint(-BLOCK_JITTER, BLOCK_JITTER)

            if 0 <= jx < GRID_SIZE and 0 <= jy < GRID_SIZE:
                points.append((jx, jy))
                last_point_in_row[jy] = jx
                last_point_in_col[jx] = jy

# -----------------------------
# 2. Scan grid for organic growth
# -----------------------------
for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):

        # Skip if already occupied by a block
        if last_point_in_row[y] == x or last_point_in_col[x] == y:
            curr_row[x] = 1
            continue

        # Horizontal influence
        if last_point_in_row[y] is None:
            h_mult = 1.0
        else:
            dist = max(1, x - last_point_in_row[y])
            h_mult = math.exp(-dist / LAMBDA_H)

        # Vertical influence
        if last_point_in_col[x] is None:
            v_mult = 1.0
        else:
            vdist = max(1, y - last_point_in_col[x])
            v_mult = math.exp(-vdist / LAMBDA_V)

        final_prob = BASE_PROB + CLUSTER_PROB * h_mult * v_mult
        final_prob = min(final_prob, 0.15)

        if random.random() < final_prob:
            curr_row[x] = 1
            points.append((x, y))
            last_point_in_row[y] = x
            last_point_in_col[x] = y
        else:
            curr_row[x] = 0

    prev_row = curr_row
    curr_row = [0] * GRID_SIZE

# -----------------------------
# Output
# -----------------------------
print(f"Number of points: {len(points)}")

px = [p[0] for p in points]
py = [p[1] for p in points]

plt.figure(figsize=(8, 8))
plt.scatter(px, py, s=6)
plt.gca().invert_yaxis()
plt.gca().set_aspect("equal", "box")
plt.title("Urban-like Point Distribution for TSP Analysis")
plt.savefig("tsp_urban_instance.png", dpi=200)
plt.show()
