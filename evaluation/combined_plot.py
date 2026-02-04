#!/usr/bin/env python3
"""
Combined evaluation figure: loss curve + force distribution + force components.

Layout (2 rows x 3 cols via GridSpec):
  Top-left  (1 col) : training & validation loss trajectory
  Top-mid/right (2 cols) : force-error distribution  (histogram + per-component)
  Bottom row (3 cols) : force component scatter      (pred vs ref x, y, z)

Usage:
    python rand_plots/combined_plot.py \
        <params.pkl> <config.yaml> \
        --log <train_*.log> \
        [--frames N] [--output DIR]
"""

import sys
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Path setup  –  clean_code_base must be importable
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
CLEAN_BASE   = SCRIPT_DIR.parent          # clean_code_base/
if str(CLEAN_BASE) not in sys.path:
    sys.path.insert(0, str(CLEAN_BASE))

# ---------------------------------------------------------------------------
# JAX / jax_md compatibility patch (before any jax_md import)
# ---------------------------------------------------------------------------
import jax                                # noqa: E402
jax.random.KeyArray = jax.Array

_to_uncache = [m for m in sys.modules if m.startswith("jax.random")]
for _m in _to_uncache:
    del sys.modules[_m]

# ---------------------------------------------------------------------------
import argparse                           # noqa: E402
import pickle                             # noqa: E402
import numpy as np                        # noqa: E402
import jax.numpy as jnp                   # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from tqdm import tqdm                     # noqa: E402
from contextlib import redirect_stdout    # noqa: E402
import io                                 # noqa: E402

from config.manager          import ConfigManager         # noqa: E402
from data.loader             import DatasetLoader         # noqa: E402
from data.preprocessor       import CoordinatePreprocessor # noqa: E402
from models.combined_model   import CombinedModel         # noqa: E402
from evaluation.evaluator    import Evaluator             # noqa: E402
from evaluation.visualizer   import LossPlotter           # noqa: E402


# =============================================================================
# Plotting helpers  –  draw onto caller-supplied axes (no figure creation)
# =============================================================================

def _plot_loss(ax, log_path: str):
    """Loss curve: legend only, no title, no config text box."""
    plotter = LossPlotter(log_path)
    plotter.parse_log()

    epochs = np.arange(len(plotter.train_losses))
    ax.plot(epochs, plotter.train_losses, label="Train",      linewidth=2)
    ax.plot(epochs, plotter.val_losses,   label="Validation", linewidth=2)

    ax.set_xlabel("Epoch",  fontsize=15)
    ax.set_ylabel("Loss",   fontsize=15)
    ax.legend(loc="best", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_force_distribution(ax_hist, ax_comp, F_pred, F_ref):
    """Force-error distributions onto two axes (histogram + per-component)."""
    F_error     = F_pred - F_ref
    F_error_mag = np.linalg.norm(F_error, axis=-1)

    # --- error magnitude histogram ---
    ax_hist.hist(F_error_mag, bins=50, alpha=0.7, edgecolor="black")
    ax_hist.axvline(np.mean(F_error_mag),   color="r", linestyle="--",
                    linewidth=2, label=f"Mean: {np.mean(F_error_mag):.3f}")
    ax_hist.axvline(np.median(F_error_mag), color="g", linestyle="--",
                    linewidth=2, label=f"Median: {np.median(F_error_mag):.3f}")
    ax_hist.set_xlabel("Force Error Magnitude", fontsize=15)
    ax_hist.set_ylabel("Count",                 fontsize=15)
    ax_hist.legend(fontsize=13)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    # --- per-component error histograms ---
    for i, name in enumerate(["x", "y", "z"]):
        ax_comp.hist(F_error[:, i], bins=30, alpha=0.5, label=f"{name}-component")
    ax_comp.set_xlabel("Force Component Error", fontsize=15)
    ax_comp.set_ylabel("Count",                 fontsize=15)
    ax_comp.legend(fontsize=13)
    ax_comp.grid(True, alpha=0.3)
    ax_comp.spines["top"].set_visible(False)
    ax_comp.spines["right"].set_visible(False)


def _plot_force_components(axes, F_pred, F_ref):
    """Pred-vs-ref scatter for x, y, z onto three axes."""
    for i, (ax, name) in enumerate(zip(axes, ["x", "y", "z"])):
        ax.scatter(F_ref[:, i], F_pred[:, i], alpha=0.5, s=15)

        lims = [
            min(F_ref[:, i].min(), F_pred[:, i].min()),
            max(F_ref[:, i].max(), F_pred[:, i].max()),
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1.5, label="Perfect")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel(f"Reference {name}-force",  fontsize=15)
        ax.set_ylabel(f"Predicted {name}-force",  fontsize=15)
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined loss + force-eval figure."
    )
    parser.add_argument("params",   help="Trained parameters .pkl file")
    parser.add_argument("config",   help="Config YAML file")
    parser.add_argument("--log",    required=True,
                        help="Training log file for loss curve")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of frames to evaluate (default: 10)")
    parser.add_argument("--output", default="./rand_plots",
                        help="Output directory")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load config + parameters
    # ------------------------------------------------------------------
    config = ConfigManager(args.config)
    model_id = config.get("model_id", default="model")

    print(f"Model ID : {model_id}")
    print(f"Log file : {args.log}")

    with open(args.params, "rb") as f:
        params = pickle.load(f)

    # Exported pkl contains raw Haiku params; CombinedModel expects {'allegro': ...}
    if "allegro" not in params:
        params = {"allegro": params}

    # ------------------------------------------------------------------
    # Load + preprocess dataset
    # ------------------------------------------------------------------
    data_path = Path(config.get_data_path())
    if not data_path.is_absolute():
        data_path = CLEAN_BASE / data_path

    loader = DatasetLoader(str(data_path), max_frames=None, seed=config.get_seed())
    print(f"Total frames : {len(loader)}   N_max : {loader.N_max}")

    cutoff = config.get_cutoff()
    preprocessor = CoordinatePreprocessor(
        cutoff=cutoff,
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
    )

    extent, R_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    dataset = loader.get_all()
    dataset["R"] = preprocessor.center_and_park(
        dataset["R"], dataset["mask"], extent, R_shift
    )
    box = extent

    # ------------------------------------------------------------------
    # Initialize model (suppress Allegro noise)
    # ------------------------------------------------------------------
    print("Initializing model …")
    with redirect_stdout(io.StringIO()):
        model = CombinedModel(
            config=config,
            R0=dataset["R"][0],
            box=box,
            species=dataset["species"][0],
            N_max=loader.N_max,
        )

    evaluator = Evaluator(model, params, config)

    # ------------------------------------------------------------------
    # Evaluate frames  →  collect F_pred / F_ref (real atoms only)
    # ------------------------------------------------------------------
    np.random.seed(args.seed)
    n_frames    = min(args.frames, len(loader))
    frame_indices = np.sort(
        np.random.choice(len(loader), size=n_frames, replace=False)
    )

    print(f"Evaluating {n_frames} frames …")

    all_F_pred, all_F_ref, all_masks = [], [], []

    for idx in tqdm(frame_indices, desc="Eval", unit="frame"):
        R       = jnp.asarray(dataset["R"][idx])
        F_ref   = jnp.asarray(dataset["F"][idx])
        mask    = jnp.asarray(dataset["mask"][idx])
        species = jnp.asarray(dataset["species"][idx])

        with redirect_stdout(io.StringIO()):
            result = evaluator.evaluate_frame(R, F_ref, mask, species)

        all_F_pred.append(np.asarray(result["forces"]))
        all_F_ref.append(np.asarray(F_ref))
        all_masks.append(np.asarray(mask))

    F_pred_all = np.concatenate(all_F_pred, axis=0)
    F_ref_all  = np.concatenate(all_F_ref,  axis=0)
    mask_all   = np.concatenate(all_masks,  axis=0)

    real_mask  = mask_all > 0
    F_pred_real = F_pred_all[real_mask]
    F_ref_real  = F_ref_all[real_mask]

    # quick metrics to stdout
    F_err  = F_pred_real - F_ref_real
    rmse   = np.sqrt(np.mean(F_err ** 2))
    mae    = np.mean(np.abs(F_err))
    print(f"Force RMSE : {rmse:.4f}   MAE : {mae:.4f}")

    # ------------------------------------------------------------------
    # Build combined figure
    # ------------------------------------------------------------------
    # 2 rows x 3 cols.
    #   row 0 : [loss (1 col)] [dist_hist (1 col)] [dist_comp (1 col)]
    #   row 1 : [comp_x]       [comp_y]            [comp_z]
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.18, wspace=0.18)

    ax_loss      = fig.add_subplot(gs[0, 0])
    ax_dist_hist = fig.add_subplot(gs[0, 1])
    ax_dist_comp = fig.add_subplot(gs[0, 2])
    ax_comp_x    = fig.add_subplot(gs[1, 0])
    ax_comp_y    = fig.add_subplot(gs[1, 1])
    ax_comp_z    = fig.add_subplot(gs[1, 2])

    # --- fill panels ---
    _plot_loss(ax_loss, args.log)
    _plot_force_distribution(ax_dist_hist, ax_dist_comp, F_pred_real, F_ref_real)
    _plot_force_components([ax_comp_x, ax_comp_y, ax_comp_z], F_pred_real, F_ref_real)

    # --- panel labels (a)–(f) top-left ---
    for ax, label in zip(
        [ax_loss, ax_dist_hist, ax_dist_comp, ax_comp_x, ax_comp_y, ax_comp_z],
        ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"],
    ):
        ax.set_title(label, loc="left", fontsize=16, fontweight="bold", pad=4)

    # --- save ---
    png_path = output_dir / f"{model_id}_combined.png"
    pdf_path = output_dir / f"{model_id}_combined.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
