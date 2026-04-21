"""
Generador de animación side-by-side: Secuencial vs Paralelo
Produce un GIF/MP4 con el mapa de la epidemia evolucionando día a día.
"""

import sys
import os
import numpy as np
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sequential"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "parallel"))

from sir_sequential import run_simulation as run_seq
from sir_parallel   import run_parallel

ANIM_SIZE = 200     # 200×200 para animación rápida
ANIM_DAYS = 150
SEED      = 42
FPS       = 8

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results")
ANIMATION_DIR = os.path.join(os.path.dirname(__file__), "..", "animation")

# Colores: S=azul pálido, I=rojo, R=verde, D=gris oscuro
COLORS = {
    0: [173, 216, 230],   # S: lightblue
    1: [220,  53,  69],   # I: rojo
    2: [ 40, 167,  69],   # R: verde
    3: [ 80,  80,  80],   # D: gris
}


def grid_to_rgb(grid):
    """Convierte grilla de estados a imagen RGB."""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for state, color in COLORS.items():
        mask = grid == state
        img[mask] = color
    return img


def generate_animation(format="gif"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    except ImportError:
        print("matplotlib no disponible")
        return

    os.makedirs(ANIMATION_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Generando snapshots (grilla {ANIM_SIZE}×{ANIM_SIZE}, {ANIM_DAYS} días)...")

    print("  Corriendo secuencial...")
    _, _, snaps_seq = run_seq(
        grid_size=ANIM_SIZE, days=ANIM_DAYS, seed=SEED,
        verbose=False, store_snapshots=True,
    )

    print("  Corriendo paralelo...")
    _, _, snaps_par = run_parallel(
        n_workers=4, grid_size=ANIM_SIZE, days=ANIM_DAYS, seed=SEED,
        verbose=False, store_snapshots=True,
    )

    # Alinear snapshots por día
    seq_dict = {day: grid for day, grid in snaps_seq}
    par_dict = {day: grid for day, grid in snaps_par}
    days = sorted(set(seq_dict.keys()) & set(par_dict.keys()))

    print(f"  Animando {len(days)} frames...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.axis("off")

    im_seq = axes[0].imshow(grid_to_rgb(seq_dict[days[0]]), interpolation="nearest")
    im_par = axes[1].imshow(grid_to_rgb(par_dict[days[0]]), interpolation="nearest")

    axes[0].set_title("Secuencial", color="#e6edf3", fontsize=14, pad=8)
    axes[1].set_title("Paralelo (4 cores)", color="#e6edf3", fontsize=14, pad=8)

    day_text = fig.text(0.5, 0.02, "Día 0", ha="center", va="bottom",
                        color="#c9d1d9", fontsize=12)

    # Leyenda
    patches = [
        mpatches.Patch(color=[c/255 for c in COLORS[0]], label="Susceptible"),
        mpatches.Patch(color=[c/255 for c in COLORS[1]], label="Infectado"),
        mpatches.Patch(color=[c/255 for c in COLORS[2]], label="Recuperado"),
        mpatches.Patch(color=[c/255 for c in COLORS[3]], label="Muerto"),
    ]
    fig.legend(handles=patches, loc="upper center", ncol=4,
               facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=10,
               bbox_to_anchor=(0.5, 0.97))

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    def update(frame_idx):
        day = days[frame_idx]
        im_seq.set_data(grid_to_rgb(seq_dict[day]))
        im_par.set_data(grid_to_rgb(par_dict[day]))
        day_text.set_text(f"Día {day}")
        return im_seq, im_par, day_text

    anim = FuncAnimation(fig, update, frames=len(days), interval=1000 // FPS, blit=True)

    if format == "gif":
        out = os.path.join(ANIMATION_DIR, "sir_animation.gif")
        writer = PillowWriter(fps=FPS)
        anim.save(out, writer=writer, dpi=100)
    else:
        out = os.path.join(ANIMATION_DIR, "sir_animation.mp4")
        writer = FFMpegWriter(fps=FPS, bitrate=2000)
        anim.save(out, writer=writer, dpi=100)

    plt.close()
    print(f"✓ Animación guardada: {out}")
    return out


def generate_static_comparison():
    """Genera una imagen estática con 4 snapshots side-by-side (días 0, 30, 60, 90)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib no disponible")
        return

    os.makedirs(ANIMATION_DIR, exist_ok=True)

    print("Generando imagen estática de comparación...")

    _, _, snaps_seq = run_seq(
        grid_size=ANIM_SIZE, days=ANIM_DAYS, seed=SEED,
        verbose=False, store_snapshots=True,
    )
    _, _, snaps_par = run_parallel(
        n_workers=4, grid_size=ANIM_SIZE, days=ANIM_DAYS, seed=SEED,
        verbose=False, store_snapshots=True,
    )

    seq_dict = {day: grid for day, grid in snaps_seq}
    par_dict = {day: grid for day, grid in snaps_par}

    target_days = [0, 30, 60, 90, 120]
    available   = sorted(set(seq_dict) & set(par_dict))
    show_days   = [d for d in target_days if d in available][:4]
    if not show_days:
        show_days = available[:4]

    fig = plt.figure(figsize=(16, 7), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, len(show_days), figure=fig,
                            hspace=0.05, wspace=0.05)

    for col, day in enumerate(show_days):
        for row, (label, snap_dict) in enumerate([("Secuencial", seq_dict),
                                                   ("Paralelo",   par_dict)]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(grid_to_rgb(snap_dict[day]), interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Día {day}", color="#e6edf3", fontsize=11)
            if col == 0:
                ax.set_ylabel(label, color="#e6edf3", fontsize=11, rotation=90,
                              labelpad=8)

    patches = [
        mpatches.Patch(color=[c/255 for c in COLORS[0]], label="Susceptible"),
        mpatches.Patch(color=[c/255 for c in COLORS[1]], label="Infectado"),
        mpatches.Patch(color=[c/255 for c in COLORS[2]], label="Recuperado"),
        mpatches.Patch(color=[c/255 for c in COLORS[3]], label="Muerto"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("Comparación Secuencial vs. Paralelo — Modelo SIR 2D",
                 color="#e6edf3", fontsize=14, y=1.01)

    out = os.path.join(ANIMATION_DIR, "sir_comparison_static.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"✓ Imagen estática guardada: {out}")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    if args.static_only:
        generate_static_comparison()
    else:
        generate_animation(args.format)
        generate_static_comparison()
