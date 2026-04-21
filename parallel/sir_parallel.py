"""
SIR Parallel Simulation on 2D Grid
Estrategia: Descomposición en bloques horizontales + ghost cells
Paralelismo: multiprocessing (Pool de procesos)

Cada proceso:
  - Maneja un bloque de filas de la grilla
  - Recibe ghost cells (1 fila) de sus vecinos arriba/abajo
  - Actualiza su bloque localmente
  - Devuelve el bloque actualizado y estadísticas locales
"""

import numpy as np
import time
import csv
import os
import sys
import multiprocessing as mp
from functools import partial
import pickle

# ─── Parámetros del modelo ───────────────────────────────────────────────────
GRID_SIZE   = 1000
DAYS        = 365
BETA        = 0.3
GAMMA       = 0.05
MU          = 0.01
INITIAL_INF = 100
SEED        = 42

S, I, R, D = 0, 1, 2, 3


# ─── Inicialización ──────────────────────────────────────────────────────────
def initialize_grid(n=GRID_SIZE, n_infected=INITIAL_INF, seed=SEED):
    rng = np.random.default_rng(seed)
    grid = np.zeros((n, n), dtype=np.int8)
    cx, cy = n // 2, n // 2
    count, r = 0, 0
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
    return grid


# ─── Worker: procesa un bloque con ghost cells ───────────────────────────────
def worker_step(args):
    """
    Recibe:
      block_with_ghosts : np.array shape (block_rows+2, N)
                         [ghost_top | bloque | ghost_bottom]
      global_seed       : int — base para rng
      day               : int — día actual
      worker_id         : int
      beta, gamma, mu   : floats

    Devuelve:
      updated_block     : np.array shape (block_rows, N)
      (s, i, r, d)      : estadísticas del bloque
    """
    (block_with_ghosts, global_seed, day, worker_id, beta, gamma, mu) = args

    # RNG reproducible por worker+día
    rng = np.random.default_rng(global_seed + worker_id * 10_000 + day)

    full = block_with_ghosts              # incluye ghost rows
    block_rows = full.shape[0] - 2       # filas reales (sin ghosts)
    n_cols = full.shape[1]

    new_full = full.copy()

    # Solo actualizamos filas 1..block_rows (índice en full)
    inf_positions = np.argwhere(full[1:-1] == I)  # coordenadas en bloque real

    rand_spread = rng.random((len(inf_positions), 4))
    rand_rec    = rng.random(len(inf_positions))
    rand_die    = rng.random(len(inf_positions))

    for idx, (local_r, c) in enumerate(inf_positions):
        row = local_r + 1  # índice en full (con ghost offset)

        neighbors = [
            (row - 1, c),   # arriba (puede ser ghost top → no escribir)
            (row + 1, c),   # abajo  (puede ser ghost bottom → no escribir)
            (row, c - 1),
            (row, c + 1),
        ]
        for k, (nr, nc) in enumerate(neighbors):
            if 0 <= nc < n_cols and 1 <= nr <= block_rows:
                # Solo escribimos dentro del bloque real (no en ghosts)
                if full[nr, nc] == S and rand_spread[idx, k] < beta:
                    new_full[nr, nc] = I

        # Recuperación / muerte (sólo la fila real)
        if rand_die[idx] < mu:
            new_full[row, c] = D
        elif rand_rec[idx] < gamma:
            new_full[row, c] = R

    # Extraer solo el bloque real (sin ghost rows)
    updated_block = new_full[1:-1, :]

    # Estadísticas locales del bloque
    s_count = int(np.sum(updated_block == S))
    i_count = int(np.sum(updated_block == I))
    r_count = int(np.sum(updated_block == R))
    d_count = int(np.sum(updated_block == D))

    return updated_block, (s_count, i_count, r_count, d_count)


# ─── Descomposición y ensamblado ─────────────────────────────────────────────
def split_grid(grid, n_workers):
    """Divide la grilla en n_workers bloques horizontales."""
    n = grid.shape[0]
    base = n // n_workers
    sizes = [base] * n_workers
    for i in range(n % n_workers):
        sizes[i] += 1  # distribuir filas sobrantes

    blocks = []
    row = 0
    for size in sizes:
        blocks.append(grid[row:row + size, :])
        row += size
    return blocks


def add_ghost_cells(blocks, full_grid):
    """
    Para cada bloque, añade ghost cells (1 fila arriba y 1 abajo)
    tomadas del estado actual de full_grid.
    """
    n = full_grid.shape[0]
    n_workers = len(blocks)
    row_starts = []
    r = 0
    for b in blocks:
        row_starts.append(r)
        r += b.shape[0]

    result = []
    for w, (block, start) in enumerate(zip(blocks, row_starts)):
        top_ghost    = full_grid[start - 1, :]   if start > 0  else np.zeros((1, n), dtype=np.int8)[0]
        bottom_start = start + block.shape[0]
        bottom_ghost = full_grid[bottom_start, :] if bottom_start < n else np.zeros((1, n), dtype=np.int8)[0]
        result.append(np.vstack([top_ghost, block, bottom_ghost]))
    return result


def assemble_grid(updated_blocks):
    return np.vstack(updated_blocks)


# ─── Simulación paralela ─────────────────────────────────────────────────────
def run_parallel(n_workers, grid_size=GRID_SIZE, days=DAYS, beta=BETA,
                 gamma=GAMMA, mu=MU, seed=SEED, verbose=True,
                 store_snapshots=False):
    grid = initialize_grid(grid_size, seed=seed)
    stats = []
    snapshots = []

    t0 = time.perf_counter()

    with mp.Pool(processes=n_workers) as pool:
        for day in range(days):
            blocks = split_grid(grid, n_workers)
            blocks_with_ghosts = add_ghost_cells(blocks, grid)

            args_list = [
                (bwg, seed, day, w_id, beta, gamma, mu)
                for w_id, bwg in enumerate(blocks_with_ghosts)
            ]

            results = pool.map(worker_step, args_list)

            updated_blocks = [r[0] for r in results]
            local_stats    = [r[1] for r in results]

            grid = assemble_grid(updated_blocks)

            # Reducción paralela: suma de estadísticas locales
            s_total = sum(ls[0] for ls in local_stats)
            i_total = sum(ls[1] for ls in local_stats)
            r_total = sum(ls[2] for ls in local_stats)
            d_total = sum(ls[3] for ls in local_stats)
            stats.append((day, s_total, i_total, r_total, d_total))

            N = s_total + i_total + r_total
            r0 = (beta / gamma) * (s_total / N) if N > 0 else 0.0

            if verbose and day % 30 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  Día {day:3d} | S={s_total:>8,} I={i_total:>7,} "
                      f"R={r_total:>7,} D={d_total:>5,} R0={r0:.3f} | {elapsed:.2f}s")

            if store_snapshots and day % 30 == 0:
                snapshots.append((day, grid.copy()))

            if i_total == 0:
                if verbose:
                    print(f"  → Epidemia extinguida en el día {day}")
                for remaining in range(day + 1, days):
                    stats.append((remaining, s_total, 0, r_total, d_total))
                break

    elapsed_total = time.perf_counter() - t0
    return stats, elapsed_total, snapshots


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SIR Parallel 2D Grid")
    parser.add_argument("--cores",     type=int,   default=4)
    parser.add_argument("--size",      type=int,   default=GRID_SIZE)
    parser.add_argument("--days",      type=int,   default=DAYS)
    parser.add_argument("--beta",      type=float, default=BETA)
    parser.add_argument("--gamma",     type=float, default=GAMMA)
    parser.add_argument("--mu",        type=float, default=MU)
    parser.add_argument("--seed",      type=int,   default=SEED)
    parser.add_argument("--out-csv",   type=str,   default="../results/stats_parallel.csv")
    parser.add_argument("--snapshots", action="store_true")
    args = parser.parse_args()

    max_cores = mp.cpu_count()
    cores = min(args.cores, max_cores)
    print(f"=== SIR PARALELO | {cores} cores | Grilla {args.size}×{args.size} | {args.days} días ===")

    stats, elapsed, snapshots = run_parallel(
        n_workers=cores,
        grid_size=args.size, days=args.days,
        beta=args.beta, gamma=args.gamma, mu=args.mu,
        seed=args.seed, verbose=True,
        store_snapshots=args.snapshots,
    )

    print(f"\nTiempo total: {elapsed:.3f} s  ({cores} cores)")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "S", "I", "R", "D"])
        writer.writerows(stats)
    print(f"CSV guardado: {args.out_csv}")

    if args.snapshots and snapshots:
        snap_path = f"../results/snapshots_parallel_{cores}cores.pkl"
        with open(snap_path, "wb") as f:
            pickle.dump(snapshots, f)
        print(f"Snapshots guardados: {snap_path}")
