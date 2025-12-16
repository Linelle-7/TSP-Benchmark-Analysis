
"""
generate_tsp.py

Erzeugt TSP-Instanzen in verschiedenen 'regulären' Mustern und schreibt TSPLIB .tsp Dateien.
Benötigt: numpy, matplotlib (optional für Visualisierung)
"""

import math
import random
from typing import List, Tuple
import os

try:
    import numpy as np
except Exception:
    raise ImportError("Dieses Skript benötigt numpy. Installiere mit: pip install numpy")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # Visualisierung optional

Coord = Tuple[float, float]


def regular_polygon(n: int, radius: float = 1.0, center: Coord = (0.0, 0.0), jitter: float = 0.0, seed: int | None = None) -> List[Coord]:
    """n Punkte gleichmäßig auf einem Kreis (Regelmäßiges n-Eck). jitter: max Zufallsverschiebung (als Anteil von radius)."""
    if seed is not None:
        random.seed(seed)
    cx, cy = center
    pts = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        if jitter:
            # zufällige Verschiebung radial
            rj = radius * jitter * (random.random() * 2 - 1)
            ang = random.random() * 2 * math.pi
            x += rj * math.cos(ang)
            y += rj * math.sin(ang)
        pts.append((x, y))
    return pts


def circle_points(n: int, radius: float = 1.0, center: Coord = (0.0, 0.0), jitter: float = 0.0, seed: int | None = None) -> List[Coord]:
    """Punkte auf Kreis (Winkel zufällig) - n Zufallswinkel gleichmäßig verteilt oder vollständig zufällig?
       Hier: zufällige Winkel, aber auf Abstand radius, mit optionalem radialen jitter."""
    if seed is not None:
        random.seed(seed)
    cx, cy = center
    thetas = [random.random() * 2 * math.pi for _ in range(n)]
    thetas.sort()
    pts = []
    for theta in thetas:
        r = radius
        if jitter:
            r += radius * jitter * (random.random() * 2 - 1)
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        pts.append((x, y))
    return pts


def grid_points(nx: int, ny: int, spacing: float = 1.0, center: Coord = (0.0, 0.0), jitter: float = 0.0, seed: int | None = None) -> List[Coord]:
    """Reguläres Gitter nx*ny (in row-major Reihenfolge)."""
    if seed is not None:
        random.seed(seed)
    cx, cy = center
    total_w = (nx - 1) * spacing
    total_h = (ny - 1) * spacing
    origin_x = cx - total_w / 2
    origin_y = cy - total_h / 2
    pts = []
    for j in range(ny):
        for i in range(nx):
            x = origin_x + i * spacing
            y = origin_y + j * spacing
            if jitter:
                x += spacing * jitter * (random.random() * 2 - 1)
                y += spacing * jitter * (random.random() * 2 - 1)
            pts.append((x, y))
    return pts


def hex_lattice(n_rows: int, n_cols: int, spacing: float = 1.0, center: Coord = (0.0, 0.0), seed: int | None = None) -> List[Coord]:
    """Hexagonales Gitter (approx)."""
    if seed is not None:
        random.seed(seed)
    pts = []
    w = spacing
    h = spacing * math.sqrt(3) / 2
    total_w = (n_cols - 1) * w + (w / 2 if n_rows > 1 else 0)
    total_h = (n_rows - 1) * h
    origin_x = center[0] - total_w / 2
    origin_y = center[1] - total_h / 2
    for r in range(n_rows):
        for c in range(n_cols):
            x = origin_x + c * w + (0.5 * w if r % 2 == 1 else 0)
            y = origin_y + r * h
            pts.append((x, y))
    return pts


def clustered_points(n: int, k: int = 3, spread: float = 0.1, box: float = 1.0, seed: int | None = None) -> List[Coord]:
    """k Cluster; spread ist Standardabweichung der Cluster in Box-Größe."""
    if seed is not None:
        random.seed(seed)
    centers = [(random.uniform(-box, box), random.uniform(-box, box)) for _ in range(k)]
    pts = []
    for i in range(n):
        c = centers[i % k]
        x = random.gauss(c[0], spread)
        y = random.gauss(c[1], spread)
        pts.append((x, y))
    return pts


def euclidean_distance(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def write_tsplib(filename: str, coords: List[Coord], name: str | None = None, comment: str | None = None, edge_weight_type: str = "EUC_2D"):
    """Schreibt eine .tsp Datei im TSPLIB-Format (EUC_2D Koordinaten)."""
    if name is None:
        name = os.path.splitext(os.path.basename(filename))[0]
    n = len(coords)
    with open(filename, "w") as f:
        f.write(f"NAME: {name}\n")
        if comment:
            f.write(f"COMMENT: {comment}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write(f"EDGE_WEIGHT_TYPE: {edge_weight_type}\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            # TSPLIB meist auf 6 Nachkommastellen
            f.write(f"{i} {x:.6f} {y:.6f}\n")
        f.write("EOF\n")


def visualize(coords: List[Coord], title: str = "TSP Instance", show_indices: bool = False, tour: List[int] | None = None):
    if plt is None:
        print("matplotlib nicht installiert — Visualisierung übersprungen.")
        return
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys)
    if show_indices:
        for i, (x,y) in enumerate(coords, start=1):
            plt.text(x, y, str(i), fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    if tour is not None:
        tx = [coords[i-1][0] for i in tour] + [coords[tour[0]-1][0]]
        ty = [coords[i-1][1] for i in tour] + [coords[tour[0]-1]-1] if False else None  # placeholder fix below
        # correct building of tour lines:
        tx = [coords[i-1][0] for i in tour] + [coords[tour[0]-1][0]]
        ty = [coords[i-1][1] for i in tour] + [coords[tour[0]-1][1]]
        plt.plot(tx, ty, linewidth=0.8)
    plt.title(title)
    plt.axis('equal')
    plt.show()


# ------- Convenience generator wrapper -------
def generate(kind: str = "regular_polygon", n: int = 20, **kwargs) -> List[Coord]:
    """kind in {'regular_polygon','circle','grid','hex','clusters','random'}"""
    if kind == "regular_polygon":
        return regular_polygon(n, **kwargs)
    elif kind == "circle":
        return circle_points(n, **kwargs)
    elif kind == "grid":
        # kwargs should contain nx, ny or will compute near-square grid
        nx = kwargs.get("nx")
        ny = kwargs.get("ny")
        spacing = kwargs.get("spacing", 1.0)
        if nx is None or ny is None:
            # choose near-square
            nx = int(math.ceil(math.sqrt(n)))
            ny = int(math.ceil(n / nx))
        pts = grid_points(nx, ny, spacing=spacing, **{k:v for k,v in kwargs.items() if k in ("center","jitter","seed")})
        return pts[:n]
    elif kind == "hex":
        # try near-square
        ncols = int(math.ceil(math.sqrt(n)))
        nrows = int(math.ceil(n / ncols))
        pts = hex_lattice(nrows, ncols, spacing=kwargs.get("spacing", 1.0), center=kwargs.get("center",(0,0)))
        return pts[:n]
    elif kind == "clusters":
        k = kwargs.get("k", max(2, int(math.sqrt(n))))
        return clustered_points(n, k=k, spread=kwargs.get("spread", 0.1), box=kwargs.get("box", 1.0), seed=kwargs.get("seed"))
    elif kind == "random":
        seed = kwargs.get("seed")
        rng = random.Random(seed)
        box = kwargs.get("box", 1.0)
        return [(rng.uniform(-box, box), rng.uniform(-box, box)) for _ in range(n)]
    else:
        raise ValueError("Unbekannter kind")


# ------- Beispiel / CLI -------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generiere TSP-Instanzen (regelmäßig/strukturiert) und speichere als .tsp")
    parser.add_argument("--kind", choices=["regular_polygon","circle","grid","hex","clusters","random"], default="regular_polygon")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", type=str, default="instance.tsp")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--radius", type=float, default=10.0, help="Radius für Kreis/Polygon")
    parser.add_argument("--jitter", type=float, default=0.0, help="Jitter (0..1)")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--clusters", type=int, default=None)
    args = parser.parse_args()

    extra = {}
    if args.kind in ("regular_polygon", "circle"):
        extra["radius"] = args.radius
        extra["jitter"] = args.jitter
        extra["center"] = (0.0, 0.0)
        extra["seed"] = args.seed
    if args.kind == "grid":
        if args.nx: extra["nx"] = args.nx
        if args.ny: extra["ny"] = args.ny
        extra["spacing"] = args.spacing
        extra["jitter"] = args.jitter
        extra["center"] = (0.0,0.0)
        extra["seed"] = args.seed
    if args.kind == "hex":
        extra["spacing"] = args.spacing
        extra["center"] = (0.0,0.0)
    if args.kind == "clusters":
        extra["k"] = args.clusters or max(2, int(math.sqrt(args.n)))
        extra["spread"] = args.jitter * args.radius if args.radius else args.jitter
        extra["box"] = args.radius
        extra["seed"] = args.seed
    if args.kind == "random":
        extra["box"] = args.radius
        extra["seed"] = args.seed

    coords = generate(kind=args.kind, n=args.n, **extra)
    write_tsplib(args.out, coords, name=os.path.splitext(os.path.basename(args.out))[0],
                 comment=f"{args.kind} n={args.n} seed={args.seed}")
    print(f"Geschrieben: {args.out} ({len(coords)} Knoten, art={args.kind})")
    if args.visualize:
        visualize(coords, title=f"{args.kind} n={args.n}", show_indices=True)
