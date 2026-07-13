"""optimizer.py — evolutionary shade placement along the nabta loop.

A genetic algorithm places a fixed budget of 24 shade elements
(12 pergola segments of 12 m + 12 tree clusters) along the 1.0 km
figure-eight loop so that the fraction of the path in shadow at
15:00 on 21 June is maximized, while a spacing term discourages
elements from piling up on one bend.

Everything is vectorized with numpy: one generation of 80 individuals
is evaluated as a single broadcast over (pop, elements, path-samples).

Run:  python optimizer.py
Outputs: assets/convergence.png, assets/before_after.png,
         assets/optimized_layout.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

from loop import SITE_H, SITE_W, garden_rooms, generate_loop
from solar import shadow_offset, solar_position

# ---------------------------------------------------------------------------
# Shade-element budget (fixed)
# ---------------------------------------------------------------------------
N_PERGOLA = 12
N_TREE = 12
N_ELEMENTS = N_PERGOLA + N_TREE

PERGOLA_LEN = 12.0      # m, along the path
PERGOLA_WID = 4.0       # m, across the path
PERGOLA_H = 3.2         # m, canopy height
TREE_RADIUS = 4.5       # m, cluster canopy radius
TREE_H = 5.0            # m, effective canopy height

# ---------------------------------------------------------------------------
# GA hyper-parameters
# ---------------------------------------------------------------------------
POP_SIZE = 80
GENERATIONS = 120
TOURNAMENT_K = 3
CROSSOVER_P = 0.9
MUTATION_P = 0.20       # per-gene probability
MUTATION_SIGMA = 0.02   # in loop-fraction units (= 20 m on a 1 km loop)
ELITES = 2

D_MIN = 15.0            # m, spacing below which clustering is penalized
W_CLUSTER = 0.5         # weight of the clustering penalty

# ---------------------------------------------------------------------------
# Scene (module-level so the fitness function stays flat and fast)
# ---------------------------------------------------------------------------
LOOP = generate_loop(n_points=1000)
P = LOOP["points"]          # (M,2) path samples, 1 m apart
T = LOOP["tangents"]        # (M,2)
NRM = LOOP["normals"]       # (M,2)
M = len(P)
ELEV, AZ = solar_position()
OFF_PERGOLA = shadow_offset(PERGOLA_H, ELEV, AZ)   # (2,)
OFF_TREE = shadow_offset(TREE_H, ELEV, AZ)


# ---------------------------------------------------------------------------
# Fitness (population-vectorized)
# ---------------------------------------------------------------------------
def shaded_mask(pop: np.ndarray) -> np.ndarray:
    """Boolean (pop, M): which path samples are shaded, per individual.

    Genome: (pop, 24) floats in [0,1) — arc-length fractions along the loop.
    Genes 0..11 are pergolas, 12..23 tree clusters. A path sample is shaded
    when it lies inside any element footprint displaced by the sun vector.
    """
    pop = np.atleast_2d(pop)
    idx = (pop * M).astype(int) % M                    # (n, 24)

    # --- pergolas: rectangles aligned with the local path tangent ----------
    ip = idx[:, :N_PERGOLA]
    c = P[ip] + OFF_PERGOLA                            # shadow centres (n,12,2)
    t = T[ip]                                          # orientations   (n,12,2)
    n = NRM[ip]
    d = P[None, None, :, :] - c[:, :, None, :]         # (n,12,M,2)
    u = np.einsum("nkmc,nkc->nkm", d, t)               # along-axis coord
    v = np.einsum("nkmc,nkc->nkm", d, n)               # across-axis coord
    perg = ((np.abs(u) <= PERGOLA_LEN / 2)
            & (np.abs(v) <= PERGOLA_WID / 2)).any(axis=1)   # (n, M)

    # --- tree clusters: displaced canopy disks -----------------------------
    it = idx[:, N_PERGOLA:]
    c = P[it] + OFF_TREE                               # (n,12,2)
    d = P[None, None, :, :] - c[:, :, None, :]
    tree = ((d ** 2).sum(-1) <= TREE_RADIUS ** 2).any(axis=1)

    return perg | tree


def clustering_penalty(pop: np.ndarray) -> np.ndarray:
    """(pop,) penalty in [0,1]: how badly consecutive arc-length gaps
    undercut the D_MIN spacing target (wrap-around included)."""
    pop = np.atleast_2d(pop)
    s = np.sort(pop % 1.0, axis=1) * LOOP["length"]
    gaps = np.diff(s, axis=1)
    wrap = (LOOP["length"] - s[:, -1] + s[:, 0])[:, None]
    gaps = np.concatenate([gaps, wrap], axis=1)
    short = np.clip((D_MIN - gaps) / D_MIN, 0.0, None)
    return (short ** 2).mean(axis=1)


def evaluate(pop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fitness, shaded_fraction), each (pop,)."""
    shade = shaded_mask(pop).mean(axis=1)
    return shade - W_CLUSTER * clustering_penalty(pop), shade


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
def run_ga(rng: np.random.Generator,
           pop_size: int = POP_SIZE,
           generations: int = GENERATIONS) -> dict:
    pop = rng.random((pop_size, N_ELEMENTS))
    fit, shade = evaluate(pop)
    hist_best, hist_mean, hist_best_shade = [], [], []

    for _ in range(generations):
        order = np.argsort(fit)[::-1]
        elites = pop[order[:ELITES]].copy()

        # tournament selection (vectorized): best of K random contestants
        contenders = rng.integers(0, pop_size, (2 * (pop_size - ELITES), TOURNAMENT_K))
        winners = contenders[np.arange(len(contenders)),
                             np.argmax(fit[contenders], axis=1)]
        pa, pb = pop[winners[::2]], pop[winners[1::2]]

        # uniform crossover
        swap = rng.random(pa.shape) < 0.5
        children = np.where(swap, pa, pb)
        no_x = rng.random(len(children)) >= CROSSOVER_P
        children[no_x] = pa[no_x]

        # gaussian mutation on the loop (wraps around)
        mut = rng.random(children.shape) < MUTATION_P
        children = np.where(
            mut, (children + rng.normal(0, MUTATION_SIGMA, children.shape)) % 1.0,
            children)

        pop = np.vstack([elites, children])
        fit, shade = evaluate(pop)
        b = int(np.argmax(fit))
        hist_best.append(fit[b])
        hist_mean.append(fit.mean())
        hist_best_shade.append(shade[b])

    b = int(np.argmax(fit))
    return {
        "best": pop[b].copy(),
        "best_fitness": float(fit[b]),
        "best_shade": float(shade[b]),
        "hist_best": np.array(hist_best),
        "hist_mean": np.array(hist_mean),
        "hist_best_shade": np.array(hist_best_shade),
    }


def random_baseline(rng: np.random.Generator, n: int = 300) -> dict:
    """Shaded fraction of n random layouts (the no-optimizer null model)."""
    pop = rng.random((n, N_ELEMENTS))
    shade = shaded_mask(pop).mean(axis=1)
    typical = int(np.argmin(np.abs(shade - shade.mean())))  # most-average layout
    return {"mean": float(shade.mean()), "std": float(shade.std()),
            "best": float(shade.max()), "typical_genome": pop[typical].copy(),
            "typical_shade": float(shade[typical])}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID = "#fcfcfb", "#e1e0d9"
BLUE, AQUA = "#2a78d6", "#1baf7a"       # series colors (convergence)
C_PERGOLA, C_TREE = "#4a3aa7", "#008300"
C_SHADOW = "#9a9890"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11,
})


def _pergola_corners(center: np.ndarray, tang: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    hl, hw = PERGOLA_LEN / 2, PERGOLA_WID / 2
    return np.array([center + hl * tang + hw * nrm, center + hl * tang - hw * nrm,
                     center - hl * tang - hw * nrm, center - hl * tang + hw * nrm])


def draw_layout(ax, genome: np.ndarray, title: str, shade_frac: float,
                rooms: np.ndarray | None = None) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-SITE_W / 2 - 6, SITE_W / 2 + 6)
    ax.set_ylim(-SITE_H / 2 - 6, SITE_H / 2 + 6)
    ax.add_patch(Rectangle((-SITE_W / 2, -SITE_H / 2), SITE_W, SITE_H,
                           fill=False, edgecolor=MUTED, linewidth=1.0))

    idx = (genome * M).astype(int) % M
    mask = shaded_mask(genome[None, :])[0]

    # shadows first (underneath everything)
    for k in range(N_PERGOLA):
        corners = _pergola_corners(P[idx[k]] + OFF_PERGOLA, T[idx[k]], NRM[idx[k]])
        ax.add_patch(Polygon(corners, closed=True, facecolor=C_SHADOW,
                             alpha=0.45, edgecolor="none"))
    for k in range(N_PERGOLA, N_ELEMENTS):
        ax.add_patch(Circle(P[idx[k]] + OFF_TREE, TREE_RADIUS,
                            facecolor=C_SHADOW, alpha=0.45, edgecolor="none"))

    # the path: muted where sunlit, blue where shaded
    ax.plot(P[:, 0], P[:, 1], color=GRID, linewidth=2.6,
            solid_capstyle="round", zorder=2)
    ax.scatter(P[mask, 0], P[mask, 1], s=2.4, color=BLUE, zorder=3)

    # elements on top
    for k in range(N_PERGOLA):
        corners = _pergola_corners(P[idx[k]], T[idx[k]], NRM[idx[k]])
        ax.add_patch(Polygon(corners, closed=True, facecolor="none",
                             edgecolor=C_PERGOLA, linewidth=1.4, zorder=4))
    for k in range(N_PERGOLA, N_ELEMENTS):
        ax.add_patch(Circle(P[idx[k]], TREE_RADIUS, facecolor="none",
                            edgecolor=C_TREE, linewidth=1.4, zorder=4))

    if rooms is not None:
        for i, r in enumerate(rooms):
            ax.add_patch(Circle(r, 5.0, facecolor="none", edgecolor=MUTED,
                                linewidth=1.1, linestyle=":", zorder=4))
            ax.annotate(f"R{i + 1:02d}", r, ha="center", va="center",
                        fontsize=7, color=INK2)

    # sun direction annotation (shadows fall along +OFF)
    a0 = np.array([SITE_W / 2 - 16, SITE_H / 2 - 4])
    d = OFF_TREE / np.linalg.norm(OFF_TREE) * 9
    ax.annotate("", xy=a0 + d, xytext=a0,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.annotate("shadow", a0 + d + [2, 0], fontsize=7.5, color=INK2, va="center")

    ax.set_title(f"{title} — {shade_frac * 100:.1f}% of loop shaded", color=INK)
    ax.set_xticks([]), ax.set_yticks([])


def plot_convergence(res: dict, baseline: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    g = np.arange(1, len(res["hist_best"]) + 1)
    ax.plot(g, res["hist_best"], color=BLUE, lw=2, label="best fitness")
    ax.plot(g, res["hist_mean"], color=AQUA, lw=2, label="mean fitness")
    ax.axhline(baseline["mean"], color=MUTED, lw=1.4, ls="--")
    ax.annotate(f"random baseline (shade {baseline['mean'] * 100:.1f}%)",
                (g[-1], baseline["mean"]), xytext=(0, -12),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
    ax.annotate(f"{res['hist_best'][-1]:.3f}", (g[-1], res["hist_best"][-1]),
                xytext=(4, 2), textcoords="offset points", fontsize=8.5, color=BLUE)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness  (shaded fraction − clustering penalty)")
    ax.set_title("GA convergence — shade coverage of the 1.0 km loop at 15:00, 21 Jun",
                 color=INK)
    ax.grid(True, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--generations", type=int, default=GENERATIONS)
    ap.add_argument("--pop", type=int, default=POP_SIZE)
    args = ap.parse_args()

    print(f"loop length {LOOP['length']:.1f} m | sun el {ELEV:.1f} deg, az {AZ:.1f} deg")
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    baseline = random_baseline(rng)
    res = run_ga(rng, args.pop, args.generations)
    dt = time.time() - t0

    print(f"random baseline : {baseline['mean'] * 100:.1f}% shaded "
          f"(±{baseline['std'] * 100:.1f}, best of 300 = {baseline['best'] * 100:.1f}%)")
    print(f"GA optimized    : {res['best_shade'] * 100:.1f}% shaded "
          f"(fitness {res['best_fitness']:.3f}) in {dt:.1f}s")

    import os
    os.makedirs("assets", exist_ok=True)

    plot_convergence(res, baseline, "assets/convergence.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    draw_layout(axes[0], baseline["typical_genome"], "Random placement",
                baseline["typical_shade"])
    draw_layout(axes[1], res["best"], "GA-optimized placement", res["best_shade"])
    fig.suptitle("24 shade elements (12 pergolas + 12 tree clusters), "
                 "shadows at 15:00 on 21 June", color=INK2, fontsize=10)
    fig.tight_layout()
    fig.savefig("assets/before_after.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 7))
    draw_layout(ax, res["best"], "Optimized layout", res["best_shade"],
                rooms=garden_rooms(LOOP))
    handles = [
        plt.Line2D([], [], color=C_PERGOLA, lw=1.6, label="pergola (12 m)"),
        plt.Line2D([], [], color=C_TREE, lw=1.6, label="tree cluster"),
        plt.Line2D([], [], color=BLUE, lw=2.4, label="shaded path"),
        plt.Line2D([], [], color=MUTED, lw=1.2, ls=":", label="garden room"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig("assets/optimized_layout.png", dpi=160)
    plt.close(fig)

    print("wrote assets/convergence.png, assets/before_after.png, "
          "assets/optimized_layout.png")


if __name__ == "__main__":
    main()
