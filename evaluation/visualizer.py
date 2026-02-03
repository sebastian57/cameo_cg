"""
Visualization Utilities for Training and Evaluation

Creates plots for loss curves, force analysis, and model diagnostics.

Extracted from:
- extract_and_plot_loss.py
- compute_single_multi.py
"""

import re
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import yaml

# Type alias for paths - accepts both str and Path objects
PathLike = Union[str, Path]

def as_path(p: PathLike) -> Path:
    """Convert str or Path to Path object."""
    return Path(p) if not isinstance(p, Path) else p

# Setup logger - use module logger or fall back to root logger
eval_logger = logging.getLogger(__name__)


class LossPlotter:
    """
    Parses training logs and plots loss curves.

    Example:
        >>> plotter = LossPlotter("train_allegro_12345.log", config)
        >>> plotter.parse_log()
        >>> plotter.plot("loss_curve.png")
    """

    def __init__(self, log_file: PathLike, config=None):
        """
        Initialize loss plotter.

        Args:
            log_file: Path to training log file
            config: ConfigManager instance (optional, for annotations)
        """
        self.log_file = as_path(log_file)
        self.config = config

        self.epochs = []
        self.train_losses = []
        self.val_losses = []

    def parse_log(self) -> Tuple[List[int], List[float], List[float]]:
        """
        Parse training log file to extract losses.

        Handles multi-node training where each node logs its own loss values.
        Deduplicates by only keeping one loss value per epoch (consecutive
        duplicates from multiple nodes are skipped).

        Returns:
            epochs, train_losses, val_losses

        Raises:
            FileNotFoundError: If log file doesn't exist
        """
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_file}")

        self.epochs = []
        self.train_losses = []
        self.val_losses = []

        # Track last recorded epoch to deduplicate multi-node logs
        # With N nodes, each epoch appears N times consecutively in the log
        last_recorded_epoch = None

        with open(self.log_file, "r") as f:
            epoch = None
            pending_train_loss = None

            for line in f:
                # Match epoch number
                m_epoch = re.search(r"\[Epoch (\d+)\]", line)
                if m_epoch:
                    epoch = int(m_epoch.group(1))

                # Match training loss
                m_train = re.search(r"Average train loss:\s*([0-9.eE+-]+)", line)
                if m_train and epoch is not None:
                    train_loss = float(m_train.group(1))

                    # Only record if this is a new epoch (not a duplicate from another node)
                    # Duplicates are consecutive, so we just check against last recorded
                    if last_recorded_epoch is None or epoch != last_recorded_epoch:
                        self.train_losses.append(train_loss)
                        self.epochs.append(epoch)
                        last_recorded_epoch = epoch
                        pending_train_loss = train_loss

                # Match validation loss - only record if we recorded the corresponding train loss
                m_val = re.search(r"Average val loss:\s*([0-9.eE+-]+)", line)
                if m_val and pending_train_loss is not None:
                    self.val_losses.append(float(m_val.group(1)))
                    pending_train_loss = None  # Reset to avoid double-counting

        eval_logger.info(f"Parsed {len(self.epochs)} unique epochs from {self.log_file.name}")
        return self.epochs, self.train_losses, self.val_losses

    def plot(self, output_path: PathLike, title: Optional[str] = None):
        """
        Create and save loss curve plot.

        Args:
            output_path: Path to save plot
            title: Custom title (optional)
        """
        if not self.epochs:
            eval_logger.warning("No data to plot. Run parse_log() first.")
            return

        output_path = as_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Use continuous epoch numbering (stages may restart from 0)
        # This ensures stage 2 continues from where stage 1 ended
        continuous_epochs = np.arange(len(self.train_losses))

        # Plot losses with continuous x-axis
        ax.plot(continuous_epochs, self.train_losses, label="Train", linewidth=2)
        ax.plot(continuous_epochs, self.val_losses, label="Validation", linewidth=2)

        # Add stage separator if config available
        if self.config:
            stage1_epochs = self.config.get_epochs(self.config.get_stage1_optimizer())
            total_epochs = len(self.train_losses)
            print(total_epochs)
            if stage1_epochs > 0 and stage1_epochs < total_epochs:
                ax.axvline(
                    x=stage1_epochs,
                    linestyle="--",
                    color="gray",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Stage Transition"
                )

            # Add config annotations
            config_text = self._format_config_text()
            ax.text(
                0.02, 0.98, config_text,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.9
                )
            )

        # Labels and title
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        elif self.config:
            dataset_name = Path(self.config.get_data_path()).stem
            ax.set_title(f"Training Loss - {dataset_name}", fontsize=14)
        else:
            ax.set_title("Training and Validation Loss", fontsize=14)

        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        eval_logger.info(f"Saved plot to: {output_path}")

    def _format_config_text(self) -> str:
        """Format config parameters for plot annotation."""
        if not self.config:
            return ""

        lines = []
        lines.append(f"Frames: {self.config.get_max_frames()}")
        lines.append(f"Cutoff: {self.config.get_cutoff()}")
        lines.append(f"Batch/device: {self.config.get_batch_per_device()}")
        lines.append(f"Val fraction: {self.config.get_val_fraction()}")

        stage1 = self.config.get_stage1_optimizer()
        stage2 = self.config.get_stage2_optimizer()
        stage1_cfg = self.config.get_optimizer_config(stage1)
        stage2_cfg = self.config.get_optimizer_config(stage2)

        lines.append(f"Opt1 ({stage1}): lr={stage1_cfg.get('peak_lr', 'N/A')}")
        lines.append(f"Opt2 ({stage2}): lr={stage2_cfg.get('peak_lr', 'N/A')}")

        return "\n".join(lines)

    def save_loss_data(self, output_path: PathLike):
        """
        Save loss data to text file.

        Args:
            output_path: Path to save data
        """
        if not self.train_losses:
            eval_logger.warning("No data to save. Run parse_log() first.")
            return

        output_path = as_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use continuous epoch numbering
        continuous_epochs = range(len(self.train_losses))

        with open(output_path, "w") as f:
            f.write("epoch train_loss val_loss\n")
            for e, t, v in zip(continuous_epochs, self.train_losses, self.val_losses):
                f.write(f"{e} {t:.6f} {v:.6f}\n")

        eval_logger.info(f"Saved loss data to: {output_path}")


class ForceAnalyzer:
    """
    Creates diagnostic plots for force predictions.

    Example:
        >>> analyzer = ForceAnalyzer()
        >>> analyzer.plot_force_components(F_pred, F_ref, "force_comp.png")
        >>> analyzer.plot_force_magnitude(F_pred, F_ref, R, "force_mag.png")
    """

    @staticmethod
    def plot_force_components(
        F_pred: np.ndarray,
        F_ref: np.ndarray,
        output_path: PathLike,
        mask: Optional[np.ndarray] = None
    ):
        """
        Plot predicted vs reference forces for each component (x, y, z).

        Args:
            F_pred: Predicted forces, shape (n_atoms, 3)
            F_ref: Reference forces, shape (n_atoms, 3)
            output_path: Path to save plot
            mask: Optional mask for real atoms, shape (n_atoms,)
        """
        output_path = as_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply mask if provided
        if mask is not None:
            real_mask = mask > 0
            F_pred = F_pred[real_mask]
            F_ref = F_ref[real_mask]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        component_names = ['x', 'y', 'z']

        for i, (ax, name) in enumerate(zip(axes, component_names)):
            fi_pred = F_pred[:, i]
            fi_ref = F_ref[:, i]

            # Scatter plot
            ax.scatter(fi_ref, fi_pred, alpha=0.6, s=20, label=f'{name}-component')

            # Ideal line
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect')

            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel(f'Reference {name}-force', fontsize=11)
            ax.set_ylabel(f'Predicted {name}-force', fontsize=11)
            ax.set_title(f'{name.upper()}-Component', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

        eval_logger.info(f"Saved force component plot to: {output_path}")

    @staticmethod
    def plot_force_magnitude(
        F_pred: np.ndarray,
        F_ref: np.ndarray,
        R: np.ndarray,
        output_path: PathLike,
        mask: Optional[np.ndarray] = None
    ):
        """
        Plot force magnitude comparison and error vs distance from center.

        Args:
            F_pred: Predicted forces, shape (n_atoms, 3)
            F_ref: Reference forces, shape (n_atoms, 3)
            R: Coordinates, shape (n_atoms, 3)
            output_path: Path to save plot
            mask: Optional mask for real atoms, shape (n_atoms,)
        """
        output_path = as_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply mask if provided
        if mask is not None:
            real_mask = mask > 0
            F_pred = F_pred[real_mask]
            F_ref = F_ref[real_mask]
            R = R[real_mask]

        # Compute force magnitudes
        F_mag_pred = np.linalg.norm(F_pred, axis=-1)
        F_mag_ref = np.linalg.norm(F_ref, axis=-1)
        F_mag_error = F_mag_pred - F_mag_ref

        # Distance from center
        center = np.mean(R, axis=0)
        R_dist = np.linalg.norm(R - center, axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Magnitude comparison
        axes[0].scatter(F_mag_ref, F_mag_pred, alpha=0.6, s=20)
        lims = [0, max(np.max(F_mag_ref), np.max(F_mag_pred))]
        axes[0].plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect')
        axes[0].set_xlabel("Reference Force Magnitude", fontsize=11)
        axes[0].set_ylabel("Predicted Force Magnitude", fontsize=11)
        axes[0].set_title("Force Magnitude Comparison", fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: Error vs distance
        axes[1].scatter(R_dist, F_mag_error, alpha=0.6, s=20, c=F_mag_error, cmap='RdBu_r')
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=2)
        axes[1].set_xlabel("Distance from Center (Å)", fontsize=11)
        axes[1].set_ylabel("Force Magnitude Error", fontsize=11)
        axes[1].set_title("Force Error vs Position", fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

        eval_logger.info(f"Saved force magnitude plot to: {output_path}")

    @staticmethod
    def plot_force_distribution(
        F_pred: np.ndarray,
        F_ref: np.ndarray,
        output_path: PathLike,
        mask: Optional[np.ndarray] = None
    ):
        """
        Plot distribution of force errors.

        Args:
            F_pred: Predicted forces, shape (n_atoms, 3)
            F_ref: Reference forces, shape (n_atoms, 3)
            output_path: Path to save plot
            mask: Optional mask for real atoms, shape (n_atoms,)
        """
        output_path = as_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Apply mask if provided
        if mask is not None:
            real_mask = mask > 0
            F_pred = F_pred[real_mask]
            F_ref = F_ref[real_mask]

        # Compute errors
        F_error = F_pred - F_ref
        F_error_magnitude = np.linalg.norm(F_error, axis=-1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Error magnitude histogram
        axes[0].hist(F_error_magnitude, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(F_error_magnitude), color='r', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(F_error_magnitude):.3f}')
        axes[0].axvline(np.median(F_error_magnitude), color='g', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(F_error_magnitude):.3f}')
        axes[0].set_xlabel("Force Error Magnitude", fontsize=11)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].set_title("Distribution of Force Errors", fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: Component-wise error
        for i, name in enumerate(['x', 'y', 'z']):
            axes[1].hist(F_error[:, i], bins=30, alpha=0.5, label=f'{name}-component')
        axes[1].set_xlabel("Force Component Error", fontsize=11)
        axes[1].set_ylabel("Count", fontsize=11)
        axes[1].set_title("Error Distribution by Component", fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

        eval_logger.info(f"Saved force distribution plot to: {output_path}")


# =============================================================================
# Standalone CLI for plotting from log files
# =============================================================================

def plot_from_log(log_file: str, config_file: str = None, output: str = None):
    """
    Create loss plot from a training log file.

    Args:
        log_file: Path to training log file
        config_file: Path to config YAML (optional, for annotations)
        output: Output path for plot (optional, defaults to loss_curve.png)
    """
    from pathlib import Path

    # Load config if provided
    config = None
    if config_file:
        try:
            from config.manager import ConfigManager
            config = ConfigManager(config_file)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")

    # Create plotter
    plotter = LossPlotter(log_file, config=config)
    plotter.parse_log()

    # Determine output path
    if output is None:
        log_path = Path(log_file)
        output = log_path.parent / f"loss_curve_{log_path.stem}.png"

    # Generate plot
    plotter.plot(output)
    print(f"Saved plot to: {output}")

    # Also save loss data
    data_path = Path(output).with_suffix(".txt")
    plotter.save_loss_data(data_path)
    print(f"Saved loss data to: {data_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add clean_code_base to path for imports when running standalone
    script_dir = Path(__file__).parent
    clean_code_base = script_dir.parent
    if str(clean_code_base) not in sys.path:
        sys.path.insert(0, str(clean_code_base))

    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <log_file> [config.yaml] [output.png]")
        print("")
        print("Examples:")
        print("  python evaluation/visualizer.py train_allegro_12345.log")
        print("  python evaluation/visualizer.py train_allegro_12345.log config.yaml")
        print("  python evaluation/visualizer.py train_allegro_12345.log config.yaml my_plot.png")
        sys.exit(1)

    log_file = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else None
    output = sys.argv[3] if len(sys.argv) > 3 else None

    plot_from_log(log_file, config_file, output)
