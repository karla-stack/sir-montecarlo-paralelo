"""
SIR Sequential Simulation on 2D Grid
Modelo: Susceptible-Infectado-Recuperado
Grilla: N x N celdas, cada celda = una persona
Estados: 0=S (susceptible), 1=I (infectado), 2=R (recuperado), 3=D (muerto)
"""

import numpy as np
import time
import csv
import os
import sys

# ─── Parámetros del modelo ───────────────────────────────────────────────────
GRID_SIZE   = 1000        # 1000×1000 = 1 millón de personas
DAYS        = 365
BETA        = 0.3         # probabilidad de contagio (vecino infectado → susceptible)
GAMMA       = 0.05        # probabilidad de recuperación diaria
MU          = 0.01        # probabilidad de muerte (infectado)
INITIAL_INF = 100         # infectados iniciales (centro de la grilla)
SEED        = 42

# Estados
S, I, R, D = 0, 1, 2, 3


def initialize_grid(n=GRID_SIZE, n_infected=INITIAL_INF, seed=SEED):
    """Crea grilla NxN con todos susceptibles salvo n_infected en el centro."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((n, n), dtype=np.int8)

    # Infectados en el centro
    cx, cy = n // 2, n // 2
    count = 0
    r = 0
    while count < n_infected:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < n and 0 <= y < n and grid[x, y] == S:
                    grid[x, y] = I
                    count += 1
                    if count >= n_infected:
                        break
            if count >= n_infected:
                break
        r += 1

    return grid, rng


def count_stats(grid):
    """Cuenta S, I, R, D en la grilla."""
    s = int(np.sum(grid == S))
    i = int(np.sum(grid == I))
    r = int(np.sum(grid == R))
    d = int(np.sum(grid == D))
    return s, i, r, d


def step(grid, rng, beta=BETA, gamma=GAMMA, mu=MU):
    """
    Actualiza la grilla un día:
    - Cada infectado puede contagiar a sus 4 vecinos con prob beta.
    - Cada infectado se recupera con prob gamma o muere con prob mu.
    """
    n = grid.shape[0]
    new_grid = grid.copy()

    # Máscara de infectados
    inf_mask = (grid == I)
    inf_positions = np.argwhere(inf_mask)

    # Propagación: para cada infectado, intentar contagiar vecinos S
    rand_spread = rng.random((len(inf_positions), 4))
    for idx, (x, y) in enumerate(inf_positions):
        neighbors = [
            (x - 1, y), (x + 1, y),
            (x, y - 1), (x, y + 1),
        ]
        for k, (nx, ny) in enumerate(neighbors):
            if 0 <= nx < n and 0 <= ny < n:
                if grid[nx, ny] == S and rand_spread[idx, k] < beta:
                    new_grid[nx, ny] = I

    # Recuperación / muerte
    rand_rec  = rng.random(len(inf_positions))
    rand_die  = rng.random(len(inf_positions))
    for idx, (x, y) in enumerate(inf_positions):
        if rand_die[idx] < mu:
            new_grid[x, y] = D
        elif rand_rec[idx] < gamma:
            new_grid[x, y] = R

    return new_grid


def run_simulation(grid_size=GRID_SIZE, days=DAYS, beta=BETA, gamma=GAMMA,
                   mu=MU, seed=SEED, verbose=True, store_snapshots=False):
    """Ejecuta la simulación secuencial completa."""
    grid, rng = initialize_grid(grid_size, seed=seed)
    stats = []
    snapshots = []

    t0 = time.perf_counter()

    for day in range(days):
        s, i, r, d = count_stats(grid)
        stats.append((day, s, i, r, d))

        # R0 estimado: beta/gamma * (S/N)
        N = s + i + r  # excluye muertos del denominador vivo
        r0 = (beta / gamma) * (s / N) if N > 0 else 0.0

        if verbose and day % 30 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Día {day:3d} | S={s:>8,} I={i:>7,} R={r:>7,} D={d:>5,} "
                  f"R0={r0:.3f} | {elapsed:.2f}s")

        if store_snapshots and day % 30 == 0:
            snapshots.append((day, grid.copy()))

        if i == 0:
            if verbose:
                print(f"  → Epidemia extinguida en el día {day}")
            # Rellenar días restantes con último valor
            for remaining in range(day + 1, days):
                stats.append((remaining, s, 0, r, d))
            break

        grid = step(grid, rng, beta, gamma, mu)

    elapsed_total = time.perf_counter() - t0
    return stats, elapsed_total, snapshots


# ─── Caso pequeño para validación ────────────────────────────────────────────
def run_small_validation():
    """Corre simulación 50×50 para validar que el modelo es correcto."""
    print("\n=== VALIDACIÓN CASO PEQUEÑO (50×50, 30 días) ===")
    stats, elapsed, _ = run_simulation(
        grid_size=50, days=30, beta=0.3, gamma=0.1, mu=0.01, seed=0, verbose=True
    )
    day, s, i, r, d = stats[-1]
    total = s + i + r + d
    print(f"\nDía final: {day} | Total personas = {total} (esperado 2500)")
    assert total == 2500, f"Error: total={total} ≠ 2500"
    print("✓ Validación superada\n")
    return stats


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SIR Sequential 2D Grid")
    parser.add_argument("--validate",  action="store_true", help="Corre caso pequeño")
    parser.add_argument("--size",      type=int,   default=GRID_SIZE)
    parser.add_argument("--days",      type=int,   default=DAYS)
    parser.add_argument("--beta",      type=float, default=BETA)
    parser.add_argument("--gamma",     type=float, default=GAMMA)
    parser.add_argument("--mu",        type=float, default=MU)
    parser.add_argument("--seed",      type=int,   default=SEED)
    parser.add_argument("--out-csv",   type=str,   default="../results/stats_sequential.csv")
    parser.add_argument("--snapshots", action="store_true")
    args = parser.parse_args()

    if args.validate:
        run_small_validation()
        sys.exit(0)

    print(f"=== SIR SECUENCIAL | Grilla {args.size}×{args.size} | {args.days} días ===")
    stats, elapsed, snapshots = run_simulation(
        grid_size=args.size, days=args.days,
        beta=args.beta, gamma=args.gamma, mu=args.mu,
        seed=args.seed, verbose=True,
        store_snapshots=args.snapshots,
    )

    print(f"\nTiempo total: {elapsed:.3f} s")

    # Guardar CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "S", "I", "R", "D"])
        writer.writerows(stats)
    print(f"CSV guardado: {args.out_csv}")

    # Guardar snapshots si se pidió
    if args.snapshots and snapshots:
        import pickle
        snap_path = "../results/snapshots_sequential.pkl"
        with open(snap_path, "wb") as f:
            pickle.dump(snapshots, f)
        print(f"Snapshots guardados: {snap_path}")
