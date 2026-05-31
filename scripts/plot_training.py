"""Generate thesis-quality training plots from a modelzoo run's train_metrics.csv.

Reads `<run_dir>/logs/metrics/train_metrics.csv` and writes four PDF + PNG figure
pairs into `thesis_figures/training/<run_timestamp>/`:

  1. loss_curve         train_loss + val_loss vs epoch (linear and log-y variants)
  2. val_map_curve      val_map vs epoch with best-epoch annotation
  3. lr_schedule        learning-rate schedule vs epoch (log-y)
  4. precision_recall   val_mpre and val_mrec vs epoch

Usage:
  python scripts/plot_training.py                       # picks the most recent run
  python scripts/plot_training.py --run <run_dir>       # explicit run dir
  python scripts/plot_training.py --usetex              # require a LaTeX install

PDF outputs are vector and embed Computer Modern via mathtext by default; pass
--usetex to switch to a real LaTeX backend if MikTeX/TeXLive is on PATH.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "object_detection" / "tf" / "src" / "experiments_outputs"
DEFAULT_FIG_ROOT = REPO_ROOT / "thesis_figures" / "training"

# Okabe-Ito colorblind-safe palette (Wong, 2011, Nature Methods)
C_TRAIN = "#0072B2"   # blue
C_VAL = "#D55E00"     # vermillion
C_BEST = "#009E73"    # green (best-epoch marker)
C_PRE = "#0072B2"     # blue
C_REC = "#CC79A7"     # reddish-purple
C_LR = "#000000"      # black for the schedule


def configure_matplotlib(usetex: bool) -> None:
    """Apply academic styling. `usetex=True` requires a working LaTeX install."""
    mpl.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # Math rendering
        "text.usetex": usetex,
        "mathtext.fontset": "cm",
        # Spines / grid
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.color": "#999999",
        "grid.alpha": 0.6,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        # Legend
        "legend.frameon": False,
        # Save defaults
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,    # TrueType in PDF (embedded, editable)
        "ps.fonttype": 42,
    })


def load_metrics(csv_path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for name in reader.fieldnames or []:
            cols[name] = []
        for row in reader:
            for k, v in row.items():
                cols[k].append(float(v))
    return cols


def find_latest_run(runs_dir: Path) -> Path:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "logs" / "metrics" / "train_metrics.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No runs with train_metrics.csv under {runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


def plot_loss(metrics: dict, out_dir: Path) -> None:
    epochs = metrics["epoch"]
    train_loss = metrics["loss"]
    val_loss = metrics["val_loss"]
    best_epoch = int(epochs[val_loss.index(min(val_loss))])

    for variant, yscale in (("linear", "linear"), ("logy", "log")):
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot(epochs, train_loss, color=C_TRAIN, label="train loss")
        ax.plot(epochs, val_loss, color=C_VAL, label="val loss")
        ax.axvline(best_epoch, color=C_BEST, linewidth=0.8, linestyle="--",
                   label=f"best epoch ({best_epoch})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_yscale(yscale)
        ax.legend(loc="best")
        save_figure(fig, out_dir, f"loss_curve_{variant}")


def plot_val_map(metrics: dict, out_dir: Path) -> None:
    epochs = metrics["epoch"]
    val_map = metrics["val_map"]
    best_epoch = int(epochs[val_map.index(max(val_map))])
    best_map = max(val_map)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(epochs, val_map, color=C_VAL, label=r"val mAP@0.5")
    ax.axvline(best_epoch, color=C_BEST, linewidth=0.8, linestyle="--",
               label=f"peak ({best_map:.3f} @ epoch {best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"mAP@0.5")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    save_figure(fig, out_dir, "val_map_curve")


def plot_lr(metrics: dict, out_dir: Path) -> None:
    epochs = metrics["epoch"]
    lr = metrics["lr"]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(epochs, lr, color=C_LR)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    save_figure(fig, out_dir, "lr_schedule")


def plot_precision_recall(metrics: dict, out_dir: Path) -> None:
    epochs = metrics["epoch"]
    pre = metrics["val_mpre"]
    rec = metrics["val_mrec"]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(epochs, pre, color=C_PRE, label="precision")
    ax.plot(epochs, rec, color=C_REC, label="recall")
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    save_figure(fig, out_dir, "precision_recall")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=None,
                        help="Path to a run dir (default: most recent under experiments_outputs/)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"Output dir (default: {DEFAULT_FIG_ROOT}/<run_name>)")
    parser.add_argument("--usetex", action="store_true",
                        help="Use a real LaTeX install (must be on PATH) instead of mathtext")
    args = parser.parse_args()

    run_dir = args.run if args.run is not None else find_latest_run(DEFAULT_RUNS_DIR)
    csv_path = run_dir / "logs" / "metrics" / "train_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = args.out if args.out is not None else (DEFAULT_FIG_ROOT / run_dir.name)
    print(f"Run dir : {run_dir}")
    print(f"CSV     : {csv_path}")
    print(f"Out dir : {out_dir}")

    configure_matplotlib(usetex=args.usetex)
    metrics = load_metrics(csv_path)

    plot_loss(metrics, out_dir)
    plot_val_map(metrics, out_dir)
    plot_lr(metrics, out_dir)
    plot_precision_recall(metrics, out_dir)

    print(f"Wrote {len(list(out_dir.glob('*.pdf')))} PDFs and {len(list(out_dir.glob('*.png')))} PNGs to {out_dir}")


if __name__ == "__main__":
    main()
