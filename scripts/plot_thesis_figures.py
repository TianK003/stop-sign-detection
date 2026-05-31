"""Generate all §4 plots from the per-run CSVs gathered into thesis_figures/plots/data/
and from the threshold-sweep outputs under
object_detection/tf/src/experiments_outputs/threshold_sweep/.

Produced PNGs (written to thesis_figures/plots/):
  lr-schedule.png            train LR vs epoch, both models on one axis
  loss-curves-256k2.png      train + val loss vs epoch for the 256 champion
  loss-curves-192k0.png      train + val loss vs epoch for the 192 compact alt
  val-map-curves.png         val_map vs epoch, both models on one axis
  f1-vs-threshold.png        F1 vs confidence threshold, four (model, test-set) curves
  k-sweep-256.png            float vs int8 mAP across k for the 256 family
  k-sweep-192.png            float vs int8 mAP across k for the 192 family
  k-sweep-pareto.png         float-vs-int8 scatter colored by k, both families

Run from repo root:
    python scripts/plot_thesis_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "thesis_figures" / "plots" / "data"
OUT_DIR = REPO_ROOT / "thesis_figures" / "plots"
THRESH_DIR = REPO_ROOT / "object_detection" / "tf" / "src" / "experiments_outputs" / "threshold_sweep"

DPI = 140

# Output as vector PDF so the plots stay sharp at any zoom / print resolution
# (these are line/scatter plots, not photos). pdf.fonttype=42 embeds TrueType
# (Type 42) fonts, which is required for the thesis's PDF/A-4 target -- matplotlib's
# default Type 3 fonts are forbidden in PDF/A.
FMT = "pdf"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def out_path(name: str) -> Path:
    """Resolve an output path under OUT_DIR, forcing the configured format (FMT)."""
    return OUT_DIR / f"{Path(name).stem}.{FMT}"


# Consistent colors so figures cross-reference cleanly in the thesis
COLOR_256 = "#1f77b4"   # matplotlib default blue
COLOR_192 = "#ff7f0e"   # matplotlib default orange
COLOR_TRAIN = "#1f77b4"
COLOR_VAL = "#d62728"   # red for validation so train/val are visually distinct


def load_tb(name: str) -> pd.DataFrame:
    """Load a TensorBoard CSV (`Wall time, Step, Value`) into a DataFrame."""
    return pd.read_csv(DATA_DIR / name)


def load_k_sweep(model: str) -> pd.DataFrame:
    """Load the k-sweep CSV for one model, filter to status=OK rows, sort by k."""
    df = pd.read_csv(DATA_DIR / f"{model}_k_sweep_results.csv")
    df = df[df["status"] == "OK"].copy()
    df["k"] = df["k"].astype(int)
    df["float_map"] = df["float_map"].astype(float)
    df["int8_map"] = df["int8_map"].astype(float)
    return df.sort_values("k").reset_index(drop=True)


def load_threshold(combo: str) -> pd.DataFrame:
    """Load a per-combo threshold-sweep CSV from object_detection/.../threshold_sweep/<combo>/."""
    return pd.read_csv(THRESH_DIR / combo / "sweep.csv")


def plot_lr() -> None:
    a = load_tb("tb-256k2-train-lr.csv")
    b = load_tb("tb-192k0-train-lr.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(a["Step"], a["Value"], color=COLOR_256, label="256_val_loss")
    ax.plot(b["Step"], b["Value"], color=COLOR_192, label="192_val_loss")
    ax.set_xlabel("Epoha")
    ax.set_ylabel("Učna hitrost")
    ax.set_title("Razpored učne hitrosti (ogrevanje + kosinusno zmanjševanje)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = out_path("lr-schedule")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_loss_one(model_tag: str, title: str, out_name: str) -> None:
    tr = load_tb(f"tb-{model_tag}-train-loss.csv")
    va = load_tb(f"tb-{model_tag}-val-loss.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tr["Step"], tr["Value"], color=COLOR_TRAIN, label="učenje")
    ax.plot(va["Step"], va["Value"], color=COLOR_VAL, label="validacija")
    ax.set_xlabel("Epoha")
    ax.set_ylabel("Izguba")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = out_path(out_name)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_val_map() -> None:
    a = load_tb("tb-256k2-val-map.csv")
    b = load_tb("tb-192k0-val-map.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(a["Step"], a["Value"], color=COLOR_256, label="256_val_loss")
    ax.plot(b["Step"], b["Value"], color=COLOR_192, label="192_val_loss")
    ax.set_xlabel("Epoha")
    ax.set_ylabel("Validacijska mAP@0,5")
    ax.set_title("Validacijska mAP skozi učenje")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = out_path("val-map-curves")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_f1_vs_threshold() -> None:
    combos = [
        ("256_default_k2__full",    "256k2 · celotna množica",     "#1f77b4", "o"),
        ("256_default_k2__nearmid", "256k2 · near+mid",      "#1f77b4", "s"),
        ("192v1_k0__full",          "192k1 · celotna množica",     "#ff7f0e", "o"),
        ("192v1_k0__nearmid",       "192k1 · near+mid",      "#ff7f0e", "s"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for combo, label, color, marker in combos:
        df = load_threshold(combo)
        # near+mid as dashed so it's visually paired with its full-set sibling
        ls = "-" if "full" in combo else "--"
        ax.plot(df["threshold"], df["f1"], color=color, linestyle=ls, marker=marker,
                markersize=4, label=label)
        best_idx = int(df["f1"].idxmax())
        ax.scatter([df["threshold"].iloc[best_idx]], [df["f1"].iloc[best_idx]],
                   s=140, facecolor="none", edgecolor=color, linewidth=2, zorder=5)
    ax.set_xlabel("Prag zaupanja")
    ax.set_ylabel("Ocena F1")
    ax.set_title("Ocena F1 v odvisnosti od praga zaupanja (najvišja F1 obkrožena)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = out_path("f1-vs-threshold")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_k_sweep_one(model: str, title: str, out_name: str) -> None:
    df = load_k_sweep(model)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["k"], df["float_map"], color=COLOR_TRAIN, marker="o", label="FP32 mAP")
    ax.plot(df["k"], df["int8_map"], color=COLOR_VAL, marker="s", label="INT8 mAP")
    # Highlight peak of each curve
    best_f = df.loc[df["float_map"].idxmax()]
    best_q = df.loc[df["int8_map"].idxmax()]
    ax.scatter([best_f["k"]], [best_f["float_map"]], s=160, facecolor="none",
               edgecolor=COLOR_TRAIN, linewidth=2.2, zorder=5,
               label=f"vrh FP32: {best_f['float_map']:.1f} pri k={int(best_f['k'])}")
    ax.scatter([best_q["k"]], [best_q["int8_map"]], s=160, facecolor="none",
               edgecolor=COLOR_VAL, linewidth=2.2, zorder=5,
               label=f"vrh INT8: {best_q['int8_map']:.1f} pri k={int(best_q['k'])}")
    ax.set_xlabel("Skalirni faktor obj-logit  k  (k=1 = brez skaliranja)")
    ax.set_ylabel("mAP@0,5 na validaciji COCO (%)")
    ax.set_title(title)
    ax.set_xticks(df["k"].tolist())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = out_path(out_name)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_k_sweep_pareto() -> None:
    d256 = load_k_sweep("256")
    d192 = load_k_sweep("192")
    fig, ax = plt.subplots(figsize=(8, 6))
    for df, color, marker, label in [
        (d256, COLOR_256, "o", "256k2"),
        (d192, COLOR_192, "s", "192k1"),
    ]:
        sc = ax.scatter(df["float_map"], df["int8_map"], c=df["k"], cmap="viridis",
                        s=120, marker=marker, edgecolor=color, linewidth=2, label=label)
        for _, row in df.iterrows():
            ax.annotate(f"k={int(row['k'])}",
                        xy=(row["float_map"], row["int8_map"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
    # y=x line for "perfect PTQ retention" reference
    lo = min(d256["float_map"].min(), d192["float_map"].min(),
             d256["int8_map"].min(), d192["int8_map"].min()) - 2
    hi = max(d256["float_map"].max(), d192["float_map"].max(),
             d256["int8_map"].max(), d192["int8_map"].max()) + 2
    ax.plot([lo, hi], [lo, hi], color="#999999", linestyle=":", linewidth=1,
            label="INT8 mAP = FP32 mAP (popolna ohranitev)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("k (skalirni faktor)")
    ax.set_xlabel("FP32 mAP@0,5 na validaciji COCO (%)")
    ax.set_ylabel("INT8 mAP@0,5 na validaciji COCO (%)")
    ax.set_title("Skaliranje obj-logit: FP32 proti INT8 mAP po k")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = out_path("k-sweep-pareto")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading data from {DATA_DIR.relative_to(REPO_ROOT)}/")
    print(f"Writing plots to  {OUT_DIR.relative_to(REPO_ROOT)}/\n")

    plot_lr()
    plot_loss_one("256k2", "Izguba pri učenju — 256_val_loss", "loss-curves-256k2.png")
    plot_loss_one("192k0", "Izguba pri učenju — 192_val_loss", "loss-curves-192k0.png")
    plot_val_map()
    plot_f1_vs_threshold()
    plot_k_sweep_one("256", "Pregled skaliranja obj-logit — 256_val_loss",
                     "k-sweep-256.png")
    plot_k_sweep_one("192", "Pregled skaliranja obj-logit — 192_val_loss",
                     "k-sweep-192.png")
    plot_k_sweep_pareto()

    print(f"\nAll plots written to {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
