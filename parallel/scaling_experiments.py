"""
Experimentos de Strong Scaling
Ejecuta la simulación con 1, 2, 4, 8 cores (usando grilla más pequeña para rapidez)
y registra tiempos + speed-up.
"""

import sys
import os
import csv
import time
import multiprocessing as mp

# Ajustar path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sequential"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "parallel"))

from sir_sequential import run_simulation as run_seq
from sir_parallel   import run_parallel

# ─── Configuración del experimento ───────────────────────────────────────────
EXPERIMENT_SIZE = 300      # 300×300 para experimentos (ejecutable en minutos)
EXPERIMENT_DAYS = 120
CORES_TO_TEST   = [1, 2, 4, 8]
N_RUNS          = 3        # promediar N_RUNS ejecuciones para reducir varianza
SEED            = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    max_cores = mp.cpu_count()
    print(f"CPUs disponibles: {max_cores}")
    cores_list = [c for c in CORES_TO_TEST if c <= max(max_cores, 1)]

    rows = []  # (cores, run, time)

    # ── Secuencial (baseline) ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"BASELINE SECUENCIAL | Grilla {EXPERIMENT_SIZE}×{EXPERIMENT_SIZE} | {EXPERIMENT_DAYS} días")
    seq_times = []
    for run in range(N_RUNS):
        print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
        _, elapsed, _ = run_seq(
            grid_size=EXPERIMENT_SIZE, days=EXPERIMENT_DAYS,
            seed=SEED + run, verbose=False,
        )
        seq_times.append(elapsed)
        print(f"{elapsed:.3f}s")
        rows.append({"cores": 1, "run": run+1, "time_s": elapsed, "type": "sequential"})

    t_seq_mean = sum(seq_times) / len(seq_times)
    print(f"  → Media secuencial: {t_seq_mean:.3f}s")

    # ── Paralelo: varios cores ───────────────────────────────────────────────
    scaling_results = [{"cores": 1, "time_mean": t_seq_mean, "speedup": 1.0, "efficiency": 1.0}]

    for cores in cores_list:
        if cores == 1:
            continue  # ya medimos secuencial
        print(f"\n{'='*60}")
        print(f"PARALELO | {cores} cores | Grilla {EXPERIMENT_SIZE}×{EXPERIMENT_SIZE}")
        par_times = []
        for run in range(N_RUNS):
            print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
            _, elapsed, _ = run_parallel(
                n_workers=cores, grid_size=EXPERIMENT_SIZE,
                days=EXPERIMENT_DAYS, seed=SEED + run, verbose=False,
            )
            par_times.append(elapsed)
            print(f"{elapsed:.3f}s")
            rows.append({"cores": cores, "run": run+1, "time_s": elapsed, "type": "parallel"})

        t_par_mean  = sum(par_times) / len(par_times)
        speedup     = t_seq_mean / t_par_mean
        efficiency  = speedup / cores
        scaling_results.append({
            "cores":      cores,
            "time_mean":  t_par_mean,
            "speedup":    speedup,
            "efficiency": efficiency,
        })
        print(f"  → Media: {t_par_mean:.3f}s | Speed-up: {speedup:.2f}x | Eficiencia: {efficiency:.2%}")

    # ── Guardar CSV detallado ────────────────────────────────────────────────
    raw_csv = os.path.join(RESULTS_DIR, "scaling_raw.csv")
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cores", "run", "time_s", "type"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV detallado guardado: {raw_csv}")

    # ── Guardar CSV resumen ─────────────────────────────────────────────────
    summary_csv = os.path.join(RESULTS_DIR, "scaling_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cores", "time_mean", "speedup", "efficiency"])
        writer.writeheader()
        writer.writerows(scaling_results)
    print(f"CSV resumen guardado:   {summary_csv}")

    return scaling_results


def generate_speedup_plot(scaling_results):
    """Genera gráfica de speed-up vs. cores."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib no disponible, saltando gráfica")
        return

    cores   = [r["cores"]   for r in scaling_results]
    speedup = [r["speedup"] for r in scaling_results]
    effic   = [r["efficiency"] for r in scaling_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Speed-up
    ax = axes[0]
    ax.plot(cores, cores,   "--", color="#8b949e", lw=1.5, label="Ideal")
    ax.plot(cores, speedup, "o-", color="#58a6ff", lw=2.5,
            markersize=8, markerfacecolor="#1f6feb", label="Medido")
    for c, sp in zip(cores, speedup):
        ax.annotate(f"{sp:.2f}x", (c, sp),
                    textcoords="offset points", xytext=(6, 4),
                    color="#79c0ff", fontsize=9)
    ax.set_xlabel("Cores")
    ax.set_ylabel("Speed-up")
    ax.set_title("Strong Scaling — Speed-up")
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, color="#21262d", linestyle="--", alpha=0.7)

    # Eficiencia
    ax = axes[1]
    ax.plot(cores, [1.0]*len(cores), "--", color="#8b949e", lw=1.5, label="Ideal")
    ax.plot(cores, effic, "s-", color="#3fb950", lw=2.5,
            markersize=8, markerfacecolor="#238636", label="Medida")
    for c, e in zip(cores, effic):
        ax.annotate(f"{e:.0%}", (c, e),
                    textcoords="offset points", xytext=(6, 4),
                    color="#56d364", fontsize=9)
    ax.set_xlabel("Cores")
    ax.set_ylabel("Eficiencia (Speed-up / Cores)")
    ax.set_title("Strong Scaling — Eficiencia")
    ax.set_ylim(0, 1.15)
    ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, color="#21262d", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "speedup_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Gráfica speed-up guardada: {out}")


if __name__ == "__main__":
    results = run_experiment()
    generate_speedup_plot(results)
    print("\n✓ Experimentos completados")
