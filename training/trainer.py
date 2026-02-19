"""
Trainer for Force Matching

Orchestrates training of combined Prior + Allegro models using chemtrain.
Supports multi-stage training, prior pre-training, and checkpointing.

Extracted from:
- train_fm_multiple_proteins.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pickle
import time

from chemtrain.trainers.trainers import ForceMatching
from jax_sgmc.data.numpy_loader import NumpyDataLoader
from chemtrain.data.data_loaders import DataLoaders

from config.types import PretrainResult, TrainingResults, StageResult
from .optimizers import create_optimizer_from_config
from utils.logging import training_logger


class Trainer:
    """
    Trainer for force matching with Prior + Allegro models.

    Supports:
    - Multi-stage training with different optimizers
    - Optional prior pre-training (LBFGS or gradient-based)
    - Checkpointing and model export
    - Multi-GPU training
    - Single-node and multi-node distributed training

    Example:
        >>> trainer = Trainer(model, config, train_loader, val_loader)
        >>> # Optional prior pre-training
        >>> if config.get("training", "pretrain_prior"):
        ...     trainer.pretrain_prior(epochs=50)
        >>> # Main training
        >>> trainer.train_stage("adabelief", epochs=100)
        >>> trainer.train_stage("yogi", epochs=50)
        >>> # Export
        >>> trainer.export_model("model.mlir")
    """

    def __init__(
        self,
        model,  # CombinedModel instance
        config,  # ConfigManager instance
        train_loader,  # DatasetLoader or NumpyDataLoader
        val_loader: Optional[Any] = None,
        train_data: Optional[Dict[str, jax.Array]] = None,
        seed: Optional[int] = None,  # Optional seed override for ensemble training
    ):
        """
        Initialize trainer.

        Args:
            model: CombinedModel instance
            config: ConfigManager instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            train_data: Optional dict with R, F, mask for prior pre-training
            seed: Optional seed override (for ensemble training). If None, uses config seed.
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training parameters
        self.batch_per_device = config.get_batch_per_device()
        self.batch_cache = config.get_batch_cache()
        self.gammas = config.get_gammas()
        self.checkpoint_path = Path(config.get_checkpoint_path())
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self._rank = jax.process_index()
        self._world_size = jax.process_count()

        # Optional JAX profiler configuration (controlled by YAML)
        profiling_cfg = config.get_profiling_config()
        self._profiling_enabled = bool(profiling_cfg.get("enabled", False))
        self._profiling_trace_dir = Path(str(profiling_cfg.get("trace_dir", "./profiles")))
        self._profiling_trace_rank0_only = bool(profiling_cfg.get("trace_rank0_only", True))
        self._profiling_log_compiles = bool(profiling_cfg.get("log_compiles", False))
        self._batch_profiler_enabled = bool(profiling_cfg.get("batch_profiler_enabled", False))
        self._batch_profiler_warmup = int(profiling_cfg.get("batch_profiler_warmup", 5))
        self._batch_profiler_samples = int(profiling_cfg.get("batch_profiler_samples", 50))

        if self._profiling_log_compiles:
            try:
                jax.config.update("jax_log_compiles", True)
                training_logger.info("[Profiling] Enabled jax_log_compiles=True")
            except Exception as e:
                training_logger.warning(f"[Profiling] Could not enable jax_log_compiles: {e}")

        if self._profiling_enabled:
            self._profiling_trace_dir.mkdir(parents=True, exist_ok=True)
            if self._profiling_trace_rank0_only and self._rank != 0:
                training_logger.info(
                    f"[Profiling] Enabled in config, but rank {self._rank} tracing is disabled "
                    "(trace_rank0_only=true)"
                )
            else:
                training_logger.info(
                    f"[Profiling] JAX tracing enabled (rank={self._rank}, "
                    f"output={self._profiling_trace_dir})"
                )

        # Store training data for prior pre-training
        if train_data is not None:
            self._train_data = train_data
            if "species" not in self._train_data:
                self._train_data["species"] = jnp.zeros_like(
                    self._train_data["mask"], dtype=jnp.int32
                )
        else:
            # Try to extract from loader (for backwards compatibility)
            # NumpyDataLoader stores data in _chains internally
            try:
                if hasattr(train_loader, '_chains') and len(train_loader._chains) > 0:
                    chain_data = train_loader._chains[0]
                    self._train_data = {
                        "R": jnp.asarray(chain_data["R"]),
                        "F": jnp.asarray(chain_data["F"]),
                        "mask": jnp.asarray(chain_data["mask"]),
                        "species": jnp.asarray(
                            chain_data["species"]
                            if "species" in chain_data
                            else np.zeros_like(chain_data["mask"], dtype=np.int32)
                        ),
                    }
                else:
                    training_logger.warning("Could not extract training data from loader. Prior pre-training may not work.")
                    self._train_data = None
            except Exception as e:
                training_logger.warning(f"Could not extract training data: {e}. Prior pre-training may not work.")
                self._train_data = None

        # Initialize model parameters
        # Use provided seed or fall back to config seed
        if seed is not None:
            self._seed = seed
        else:
            self._seed = config.get_seed()
        self.params = model.initialize_params(jax.random.PRNGKey(self._seed))
        training_logger.info(f"Initialized model with seed={self._seed}")
        self.best_params = None

        # Current trainer instance (will be set during training)
        self._chemtrain_trainer = None

        # Optimizer state to restore on next train_stage call (set by load_chemtrain_checkpoint)
        self._resume_opt_state = None

        # Apply NumpyDataLoader patch if needed
        self._apply_dataloader_patch()

    def _apply_dataloader_patch(self):
        """
        Apply patch to NumpyDataLoader to fix cache_size issue.

        This is needed for chemtrain compatibility.
        """
        from jax_sgmc.data.numpy_loader import NumpyDataLoader as _NDL

        if not hasattr(_NDL, '_original_get_indices'):
            _orig_get_indices = _NDL._get_indices

            def _patched_get_indices(self, chain_id: int):
                chain = self._chains[chain_id]
                if chain.get("cache_size", 0) <= 0:
                    chain["cache_size"] = 1
                return _orig_get_indices(self, chain_id)

            _NDL._get_indices = _patched_get_indices
            _NDL._original_get_indices = _orig_get_indices
            training_logger.info("Applied NumpyDataLoader patch")

    def _should_trace_this_rank(self) -> bool:
        """Return True if JAX tracing should run on this rank."""
        if not self._profiling_enabled:
            return False
        if self._profiling_trace_rank0_only and self._rank != 0:
            return False
        return True

    def _build_trace_dir(
        self, optimizer_name: str, start_epoch: int, remaining_epochs: int
    ) -> Path:
        """Build a unique trace output directory for one stage."""
        stage_end = start_epoch + remaining_epochs
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = (
            f"stage_{optimizer_name}_rank{self._rank}_"
            f"epoch{start_epoch:04d}_to_{stage_end:04d}_{timestamp}"
        )
        return self._profiling_trace_dir / run_name

    def _start_jax_trace(
        self, optimizer_name: str, start_epoch: int, remaining_epochs: int
    ) -> Optional[Path]:
        """Start JAX profiler tracing for a stage."""
        if not self._should_trace_this_rank():
            return None

        trace_dir = self._build_trace_dir(optimizer_name, start_epoch, remaining_epochs)
        trace_dir.mkdir(parents=True, exist_ok=True)
        try:
            jax.profiler.start_trace(str(trace_dir))
            training_logger.info(f"[Profiling] Started JAX trace: {trace_dir}")
            return trace_dir
        except Exception as e:
            training_logger.warning(f"[Profiling] Failed to start JAX trace: {e}")
            return None

    def _stop_jax_trace(self, trace_dir: Optional[Path]) -> None:
        """Stop JAX profiler tracing if active."""
        if trace_dir is None:
            return
        try:
            jax.profiler.stop_trace()
            training_logger.info(f"[Profiling] Saved JAX trace to: {trace_dir}")
        except Exception as e:
            training_logger.warning(f"[Profiling] Failed to stop JAX trace cleanly: {e}")

    def _attach_batch_profiler(
        self, trainer, n_warmup: int = 5, n_samples: int = 50
    ) -> None:
        """
        Monkey-patch trainer._update_fn to record per-batch timing.

        Three timestamps are captured around each profiled _update_fn call:
          t0  — entry into _update_fn (dispatch start)
          t1  — return from _update_fn (dispatch end; GPU work is async at this point)
          t2  — return from jax.effects_barrier() (GPU compute complete)

        Derived metrics
        ---------------
        dispatch_ms  = t1 - t0   ~ time to queue the JIT work (should be ~0 if async)
        barrier_ms   = t2 - t1   ~ pure GPU compute time per batch
        gap_ms       = t0[i+1] - t0[i]  ~ wall time between batch starts

        Key diagnostic — gap / barrier ratio:
          ≈ 1.0  CPU blocks every step (current code: onp.asarray forces a sync per batch)
          ≈ 0.0  GPU is fully pipelined with data loading (target after async-sync fix)
        """
        call_ts: list = []
        dispatch_ts: list = []
        barrier_ts: list = []
        step = [0]
        original_fn = trainer._update_fn

        # Prefer effects_barrier (precise); fall back to block_until_ready on a dummy op.
        if hasattr(jax, "effects_barrier"):
            _barrier = jax.effects_barrier
        else:
            def _barrier():
                jax.block_until_ready(jnp.zeros(()))

        def _timed_update_fn(params, opt_state, batch, per_target=False):
            idx = step[0]
            step[0] += 1

            if idx < n_warmup or idx >= n_warmup + n_samples:
                return original_fn(params, opt_state, batch, per_target=per_target)

            t0 = time.perf_counter()
            result = original_fn(params, opt_state, batch, per_target=per_target)
            t1 = time.perf_counter()
            _barrier()
            t2 = time.perf_counter()

            call_ts.append(t0)
            dispatch_ts.append(t1)
            barrier_ts.append(t2)
            return result

        trainer._update_fn = _timed_update_fn
        # Store on self so _report_batch_profiler can access after train() returns.
        self._batch_profiler_data = (call_ts, dispatch_ts, barrier_ts, n_warmup, n_samples)

    def _report_batch_profiler(self) -> None:
        """Log batch-profiler statistics collected by _attach_batch_profiler."""
        if not hasattr(self, "_batch_profiler_data"):
            return

        call_ts, dispatch_ts, barrier_ts, n_warmup, n_samples = self._batch_profiler_data
        del self._batch_profiler_data  # clean up

        n = len(call_ts)
        if n < 2:
            training_logger.warning("[BatchProfiler] Too few samples collected (n=%d).", n)
            return

        dispatch_ms = np.array([(dispatch_ts[i] - call_ts[i]) * 1e3 for i in range(n)])
        barrier_ms  = np.array([(barrier_ts[i]  - dispatch_ts[i]) * 1e3 for i in range(n)])
        gap_ms      = np.array([(call_ts[i + 1] - call_ts[i]) * 1e3 for i in range(n - 1)])

        def _fmt(arr):
            return (
                f"mean={arr.mean():.2f} ± {arr.std():.2f} ms  "
                f"p50={np.median(arr):.2f}  p95={np.percentile(arr, 95):.2f}"
            )

        ratio = float(np.mean(gap_ms) / max(float(np.mean(barrier_ms)), 1e-6))

        training_logger.info(
            "\n[BatchProfiler] Per-batch timing (%d samples, %d warmup skipped):",
            n, n_warmup,
        )
        training_logger.info("  dispatch_fn  : %s", _fmt(dispatch_ms))
        training_logger.info("  gpu_barrier  : %s", _fmt(barrier_ms))
        training_logger.info("  inter-batch gap: %s", _fmt(gap_ms))
        training_logger.info(
            "  gap / barrier ratio: %.3f  "
            "(1.0 = CPU blocks each step; 0.0 = GPU fully pipelined)",
            ratio,
        )

        if ratio > 0.8:
            training_logger.warning(
                "  [!!] CPU is BLOCKING on every batch step. "
                "The onp.asarray() syncs in chemtrain._update are the dominant overhead. "
                "Deferred batch-sync fix will reduce this gap."
            )
        elif ratio < 0.3:
            training_logger.info(
                "  [OK] GPU is well-pipelined. Async dispatch is working."
            )
        else:
            training_logger.info(
                "  [~] Partial pipelining — some async benefit but host overhead visible."
            )

    @staticmethod
    def _block_until_ready(tree: Any) -> None:
        """Block host until device work for a pytree is complete."""
        try:
            jax.block_until_ready(tree)
            return
        except Exception:
            pass

        # Fallback: block every leaf individually
        leaves = jax.tree_util.tree_leaves(tree)
        for leaf in leaves:
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()

    def _create_chemtrain_loaders(self) -> DataLoaders:
        """
        Create chemtrain DataLoaders from our loaders.

        Returns:
            chemtrain.data.data_loaders.DataLoaders instance
        """
        # Convert our DatasetLoader to NumpyDataLoader if needed.
        # DatasetLoader stores NumPy arrays, so no device transfer is required.
        if not isinstance(self.train_loader, NumpyDataLoader):
            train_np_loader = NumpyDataLoader(
                R=self.train_loader.R,
                F=self.train_loader.F,
                mask=self.train_loader.mask,
                species=self.train_loader.species,
                copy=False
            )
        else:
            train_np_loader = self.train_loader

        if self.val_loader is not None:
            if not isinstance(self.val_loader, NumpyDataLoader):
                val_np_loader = NumpyDataLoader(
                    R=self.val_loader.R,
                    F=self.val_loader.F,
                    mask=self.val_loader.mask,
                    species=self.val_loader.species,
                    copy=False
                )
            else:
                val_np_loader = self.val_loader
        else:
            val_np_loader = train_np_loader  # Use training data for validation

        return DataLoaders(
            train_loader=train_np_loader,
            val_loader=val_np_loader,
            test_loader=None
        )

    def train_stage(
        self,
        optimizer_name: str,
        epochs: int,
        start_epoch: int = 0,
        checkpoint_freq: int = 0
    ) -> StageResult:
        """
        Train for a single stage with a specific optimizer.

        Args:
            optimizer_name: Name of optimizer (e.g., "adabelief", "yogi")
            epochs: Total number of epochs for this stage
            start_epoch: Epoch to start from (for resume, default 0)
            checkpoint_freq: Save checkpoint every N epochs (0 = only at end)

        Returns:
            Dictionary with final losses
        """
        remaining_epochs = epochs - start_epoch
        if remaining_epochs <= 0:
            training_logger.info(f"Stage {optimizer_name} already complete (epoch {start_epoch}/{epochs})")
            return {"train_loss": 0.0, "val_loss": 0.0, "skipped": True}

        training_logger.info(f"\n{'='*60}")
        training_logger.info(f"Training Stage: {optimizer_name.upper()} ({remaining_epochs} epochs, starting from {start_epoch})")
        training_logger.info(f"{'='*60}")

        # Create optimizer
        optimizer = create_optimizer_from_config(self.config, optimizer_name)

        # Create chemtrain loaders
        loaders = self._create_chemtrain_loaders()

        # Create energy function template
        energy_fn_template = self.model.energy_fn_template

        # Create ForceMatching trainer
        trainer = ForceMatching(
            init_params=self.params,
            optimizer=optimizer,
            energy_fn_template=energy_fn_template,
            nbrs_init=self.model.initial_neighbors,
            gammas=self.gammas,
            checkpoint_path=str(self.checkpoint_path),
            batch_per_device=self.batch_per_device,
            batch_cache=self.batch_cache,
        )

        # Set loaders
        trainer.set_loader(loaders.train_loader, stage="training")
        trainer.set_loader(loaders.val_loader, stage="validation")

        # Restore optimizer state from checkpoint if available.
        # This ensures the LR schedule continues from where it left off instead of
        # restarting from step 0, which would cause the LR to jump to its initial value.
        if self._resume_opt_state is not None:
            try:
                restored_opt_state = jax.tree_util.tree_map(jnp.asarray, self._resume_opt_state)
                # TrainerState is a NamedTuple; use _replace to create an updated copy
                trainer.state = trainer.state._replace(opt_state=restored_opt_state)
                training_logger.info("Restored optimizer state from checkpoint (LR schedule continues)")
            except Exception as e:
                training_logger.warning(
                    f"Could not restore optimizer state: {e}. "
                    "LR schedule will restart from step 0."
                )
            finally:
                self._resume_opt_state = None  # Only restore once per resume

        # Attach per-batch timing profiler if requested (non-invasive monkey-patch).
        # Must be done BEFORE trainer.train() so it intercepts from step 0.
        if self._batch_profiler_enabled and self._should_trace_this_rank():
            training_logger.info(
                "[BatchProfiler] Attaching to _update_fn "
                f"(warmup={self._batch_profiler_warmup}, "
                f"samples={self._batch_profiler_samples})"
            )
            self._attach_batch_profiler(
                trainer,
                n_warmup=self._batch_profiler_warmup,
                n_samples=self._batch_profiler_samples,
            )

        # Train with periodic checkpointing
        stage_start_time = time.perf_counter()
        trace_dir = self._start_jax_trace(optimizer_name, start_epoch, remaining_epochs)
        trace_annotation = getattr(jax.profiler, "TraceAnnotation", None)
        try:
            if trace_annotation is not None:
                with trace_annotation(f"train_stage_{optimizer_name}"):
                    trainer.train(
                        remaining_epochs,
                        checkpoint_freq=checkpoint_freq if checkpoint_freq > 0 else None
                    )
            else:
                trainer.train(
                    remaining_epochs,
                    checkpoint_freq=checkpoint_freq if checkpoint_freq > 0 else None
                )
            self._block_until_ready(trainer.params)
        finally:
            self._stop_jax_trace(trace_dir)
        stage_wall_seconds = time.perf_counter() - stage_start_time

        # Report batch profiler results (clears internal state).
        if self._batch_profiler_enabled:
            self._report_batch_profiler()

        # Update parameters
        self.params = trainer.params
        self.best_params = trainer.best_inference_params
        self._chemtrain_trainer = trainer
        if self.model.use_priors and getattr(self.model, "train_priors", False) and "prior" in self.params:
            self.model.prior.params = self.params["prior"]

        # Save stage checkpoint with metadata for resume capability
        if checkpoint_freq > 0:
            self._save_stage_checkpoint(optimizer_name, epochs)

        # Extract gradient norm history (per-step, logged by chemtrain internally)
        grad_norms = list(getattr(trainer, 'gradient_norm_history', []))
        if grad_norms:
            training_logger.info(
                f"Gradient norms — mean: {np.mean(grad_norms):.4e}, "
                f"max: {max(grad_norms):.4e}, "
                f"final: {grad_norms[-1]:.4e}"
            )
        # Store on self so _save_stage_checkpoint can include it in metadata
        self._last_gradient_norms = grad_norms

        # Compute total parameter L2 norm on-device (single scalar transfer)
        total_param_norm = float(jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda acc, v: acc + jnp.sum(v * v),
                self.params,
                initializer=jnp.float32(0.0),
            )
        ))
        training_logger.info(f"Total parameter L2 norm: {total_param_norm:.4e}")

        # Get final losses
        final_losses = {
            "train_loss": float(trainer.train_losses[-1]) if trainer.train_losses else 0.0,
            "val_loss": float(trainer.val_losses[-1]) if trainer.val_losses else 0.0,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "grad_norm_final": float(grad_norms[-1]) if grad_norms else 0.0,
            "param_norm": total_param_norm,
            "stage_wall_seconds": stage_wall_seconds,
            "stage_wall_minutes": stage_wall_seconds / 60.0,
            "epoch_wall_seconds_est": stage_wall_seconds / max(remaining_epochs, 1),
        }
        if trace_dir is not None:
            final_losses["jax_trace_dir"] = str(trace_dir)

        training_logger.info(
            f"Stage wall time: {stage_wall_seconds:.2f} s "
            f"({stage_wall_seconds / 60.0:.2f} min), "
            f"~{final_losses['epoch_wall_seconds_est']:.2f} s/epoch"
        )

        training_logger.info(f"\nStage complete: train_loss={final_losses['train_loss']:.6f}, "
                           f"val_loss={final_losses['val_loss']:.6f}")

        return final_losses

    def _save_stage_checkpoint(self, stage_name: str, completed_epochs: int):
        """
        Save checkpoint with stage metadata for resume capability.

        Args:
            stage_name: Name of the completed stage (e.g., "adabelief", "yogi")
            completed_epochs: Total epochs completed in this stage
        """
        import time

        checkpoint_file = self.checkpoint_path / f"stage_{stage_name}_epoch{completed_epochs}.pkl"
        meta_file = checkpoint_file.with_suffix(".meta.pkl")

        # Use chemtrain's save_trainer which includes optimizer state
        if self._chemtrain_trainer is not None:
            self._chemtrain_trainer.save_trainer(checkpoint_file)
            training_logger.info(f"Saved stage checkpoint: {checkpoint_file}")

            # Save metadata separately for resume logic
            metadata = {
                "stage": stage_name,
                "completed_epochs": completed_epochs,
                "timestamp": time.time(),
                "train_losses": list(self._chemtrain_trainer.train_losses),
                "val_losses": list(self._chemtrain_trainer.val_losses),
                "gradient_norm_history": getattr(self, '_last_gradient_norms', []),
            }
            with open(meta_file, 'wb') as f:
                pickle.dump(metadata, f)
            training_logger.info(f"Saved stage metadata: {meta_file}")

    def pretrain_prior(
        self,
        max_steps: int = None,
        tol_grad: float = None
    ) -> PretrainResult:
        """
        Pre-train prior energy parameters using LBFGS force matching.

        This optimizes ONLY the prior parameters to match reference forces,
        using the LBFGS optimizer as in the original implementation.

        Args:
            max_steps: Maximum LBFGS iterations (default from config)
            tol_grad: Gradient tolerance for convergence (default from config)

        Returns:
            Dictionary with keys: train_loss, val_loss, steps, converged,
            grad_norm, loss_history, fitted_params

        Note:
            Only works if model.use_priors is True.
            Uses LBFGS optimization with jax.lax.while_loop for convergence.
            Always uses LBFGS optimizer (not configurable).
            In multi-node mode, runs LBFGS on rank 0 only and broadcasts results.
        """
        # Read defaults from config
        if max_steps is None:
            max_steps = self.config.get_pretrain_prior_max_steps()
        if tol_grad is None:
            tol_grad = self.config.get_pretrain_prior_tol_grad()

        # Minimum steps before convergence check (from config)
        min_steps = self.config.get_pretrain_prior_min_steps()
        if not self.model.use_priors:
            training_logger.info("Skipping prior pre-training (use_priors=False)")
            return {"train_loss": 0.0, "val_loss": 0.0, "converged": True}

        prior = self.model.prior
        if prior is not None and getattr(prior, "uses_splines", False):
            training_logger.info(
                "Spline priors detected - skipping LBFGS prior pre-training "
                "(no parametric prior parameters to optimize)."
            )
            return {"train_loss": 0.0, "val_loss": 0.0, "converged": True}

        # Check if we're in multi-node distributed mode
        is_distributed = jax.process_count() > 1
        rank = jax.process_index()

        training_logger.info(f"\n{'='*60}")
        training_logger.info(f"Prior Pre-Training (LBFGS, max_steps={max_steps})")
        if is_distributed:
            training_logger.info(f"[Distributed] Running on rank 0, broadcasting to {jax.process_count()} processes")
        training_logger.info(f"{'='*60}")

        from typing import NamedTuple

        # Get training data (stored in __init__ to avoid _chains[0] access)
        if self._train_data is None:
            training_logger.error("Training data not available. Cannot perform prior pre-training.")
            training_logger.error("Ensure train_data parameter is passed to Trainer.__init__")
            raise ValueError("Training data required for prior pre-training")

        train_data = self._train_data

        # Get prior components
        displacement = prior.displacement
        bonds = prior.bonds
        angles = prior.angles
        rep_pairs = prior.rep_pairs

        # Initial prior parameters
        params0 = prior.params

        # Define prior force computation
        def prior_forces(params, R, mask, species):
            """Compute forces from prior energy only."""
            def energy_of_R(R_):
                return prior.compute_total_energy_from_params(
                    params, R_, mask, species=species
                )
            return -jax.grad(energy_of_R)(R)

        # Define force matching loss
        def force_matching_loss(params):
            """Compute L2 loss between predicted and reference forces."""
            R = train_data["R"]
            F_ref = train_data["F"]
            mask = train_data["mask"]
            species = train_data["species"]

            # Vectorized force prediction over batch
            F_pred = jax.vmap(
                lambda R_f, m_f, s_f: prior_forces(params, R_f, m_f, s_f)
            )(R, mask, species)

            # Masked squared error
            m3 = mask[..., None]  # Broadcast mask to (batch, atoms, 3)
            diff = (F_pred - F_ref) * m3

            # Normalize by number of real atoms
            denom = jnp.maximum(jnp.sum(m3), 1.0)
            return jnp.sum(diff * diff) / denom

        # Create LBFGS optimizer
        opt = optax.lbfgs(learning_rate=1.0)
        value_and_grad = optax.value_and_grad_from_state(force_matching_loss)

        # LBFGS state
        class FitState(NamedTuple):
            params: Dict[str, jax.Array]
            opt_state: optax.OptState
            step: jax.Array
            loss: jax.Array
            loss_hist: jax.Array

        # Initialize state
        def init_state(p0):
            opt_state = opt.init(p0)
            value0, grad0 = value_and_grad(p0, state=opt_state)
            loss_hist = jnp.full((max_steps,), jnp.nan, dtype=jnp.float32)
            loss_hist = loss_hist.at[0].set(value0.astype(jnp.float32))
            return FitState(
                params=p0,
                opt_state=opt_state,
                step=jnp.array(0, dtype=jnp.int32),
                loss=value0,
                loss_hist=loss_hist,
            )

        # Convergence condition
        def cond_fn(st: FitState):
            not_done = st.step < max_steps

            # Check gradient norm
            grad = optax.tree.get(st.opt_state, "grad")
            grad_norm = optax.tree.norm(grad)
            not_converged_grad = jnp.logical_or(st.step < min_steps, grad_norm >= tol_grad)

            return jnp.logical_and(not_done, not_converged_grad)

        # LBFGS update step
        def body_fn(st: FitState):
            p, s, k = st.params, st.opt_state, st.step

            # Compute value and gradient
            value, grad = value_and_grad(p, state=s)

            # LBFGS update
            updates, s_new = opt.update(
                grad, s, p,
                value=value,
                grad=grad,
                value_fn=force_matching_loss,
            )
            p_new = optax.apply_updates(p, updates)

            # Compute new loss
            value_new = force_matching_loss(p_new)

            # Record loss
            loss_hist = st.loss_hist.at[k].set(value.astype(jnp.float32))

            return FitState(
                params=p_new,
                opt_state=s_new,
                step=k + 1,
                loss=value_new,
                loss_hist=loss_hist,
            )

        # Run LBFGS optimization (only on rank 0 in distributed mode)
        if is_distributed:
            if rank == 0:
                training_logger.info("[LBFGS] Starting optimization on rank 0...")
                st0 = init_state(params0)
                stF = jax.lax.while_loop(cond_fn, body_fn, st0)

                # Extract results on rank 0
                fitted_params = stF.params
                loss_hist = stF.loss_hist
                final_step = int(stF.step)
                final_loss = float(stF.loss)

                # Get gradient norm for convergence check
                grad_final = optax.tree.get(stF.opt_state, "grad")
                grad_norm_final = float(optax.tree.norm(grad_final))
                converged = grad_norm_final < tol_grad

                training_logger.info(f"[LBFGS] Completed: {final_step} steps")
                training_logger.info(f"[LBFGS] Final loss: {final_loss:.6e}")
                training_logger.info(f"[LBFGS] Grad norm: {grad_norm_final:.6e} (tol={tol_grad:.6e})")
                training_logger.info(f"[LBFGS] Converged: {converged}")
            else:
                # Other ranks wait and will receive broadcasted params
                training_logger.info(f"[LBFGS] Rank {rank} waiting for broadcast from rank 0...")
                fitted_params = None
                final_step = 0
                final_loss = 0.0
                grad_norm_final = 0.0
                converged = False
                loss_hist = None

            # Broadcast fitted parameters from rank 0 to all other ranks
            # Use jax.experimental.multihost_utils for multi-process broadcast
            from jax.experimental import multihost_utils

            # Broadcast each parameter array individually
            if rank == 0:
                broadcast_params = fitted_params
            else:
                # Create placeholder with same structure as params0
                broadcast_params = jax.tree.map(lambda x: jnp.zeros_like(x), params0)

            # Synchronize parameters across all processes
            fitted_params = multihost_utils.broadcast_one_to_all(broadcast_params, is_source=(rank == 0))

            # Also broadcast scalar results
            final_step = int(multihost_utils.broadcast_one_to_all(
                jnp.array(final_step, dtype=jnp.int32), is_source=(rank == 0)
            ))
            final_loss = float(multihost_utils.broadcast_one_to_all(
                jnp.array(final_loss, dtype=jnp.float32), is_source=(rank == 0)
            ))
            grad_norm_final = float(multihost_utils.broadcast_one_to_all(
                jnp.array(grad_norm_final, dtype=jnp.float32), is_source=(rank == 0)
            ))
            converged = bool(multihost_utils.broadcast_one_to_all(
                jnp.array(converged, dtype=jnp.bool_), is_source=(rank == 0)
            ))

            training_logger.info(f"[LBFGS] Rank {rank} received broadcasted parameters")
        else:
            # Single-node mode: run LBFGS directly
            training_logger.info("[LBFGS] Starting optimization...")
            st0 = init_state(params0)
            stF = jax.lax.while_loop(cond_fn, body_fn, st0)

            # Extract results
            fitted_params = stF.params
            loss_hist = stF.loss_hist
            final_step = int(stF.step)
            final_loss = float(stF.loss)

            # Get gradient norm for convergence check
            grad_final = optax.tree.get(stF.opt_state, "grad")
            grad_norm_final = float(optax.tree.norm(grad_final))
            converged = grad_norm_final < tol_grad

            training_logger.info(f"[LBFGS] Completed: {final_step} steps")
            training_logger.info(f"[LBFGS] Final loss: {final_loss:.6e}")
            training_logger.info(f"[LBFGS] Grad norm: {grad_norm_final:.6e} (tol={tol_grad:.6e})")
            training_logger.info(f"[LBFGS] Converged: {converged}")

        # Update model parameters (all ranks now have the same fitted_params)
        self.model.prior.params = fitted_params
        if 'prior' in self.params:
            self.params['prior'] = fitted_params

        # Print fitted parameters (transfer only scalars/small arrays)
        training_logger.info("\n[LBFGS] Fitted parameters:")
        for key, val in fitted_params.items():
            if jnp.ndim(val) == 0:
                training_logger.info(f"  {key}: {float(val):.6f}")
            else:
                val_np = np.asarray(val)
                if val_np.size <= 10:
                    training_logger.info(f"  {key}: {val_np}")
                else:
                    training_logger.info(f"  {key}: shape={val_np.shape}, norm={np.linalg.norm(val_np):.6f}")

        # Prepare loss history (may be None for non-rank-0 in distributed mode)
        if loss_hist is not None:
            loss_history = np.array(loss_hist[:final_step])
        else:
            loss_history = np.array([])

        return {
            "train_loss": final_loss,
            "val_loss": final_loss,  # No separate validation in LBFGS
            "steps": final_step,
            "converged": converged,
            "grad_norm": grad_norm_final,
            "loss_history": loss_history,
            "fitted_params": {k: np.array(v) for k, v in fitted_params.items()},
        }

    def train_full_pipeline(
        self,
        resume_from: Optional[str] = None,
        checkpoint_freq: Optional[int] = None
    ) -> TrainingResults:
        """
        Run full training pipeline as configured in YAML.

        Reads training configuration and runs:
        1. Optional prior pre-training
        2. Stage 1 optimizer (e.g., AdaBelief)
        3. Stage 2 optimizer (e.g., Yogi)

        Args:
            resume_from: Path to checkpoint to resume from (optional)
            checkpoint_freq: Override checkpoint frequency from config (optional)

        Returns:
            Dictionary with training results
        """
        results = {}

        # Get checkpoint frequency from config if not overridden
        if checkpoint_freq is None:
            checkpoint_freq = self.config.get_checkpoint_freq()

        # Resume state tracking
        resume_stage = None
        resume_epoch = 0

        if resume_from:
            metadata = self.load_chemtrain_checkpoint(resume_from)
            resume_stage = metadata.get("stage", "unknown")
            resume_epoch = metadata.get("completed_epochs", 0)
            training_logger.info(f"Resuming from stage '{resume_stage}' at epoch {resume_epoch}")

        # Get optimizer names for stage comparison
        stage1_opt = self.config.get_stage1_optimizer()
        stage2_opt = self.config.get_stage2_optimizer()
        stage1_epochs = self.config.get_epochs(stage1_opt)
        stage2_epochs = self.config.get_epochs(stage2_opt)

        # Check if prior pre-training is enabled (skip if resuming from later stage)
        pretrain_prior = self.config.pretrain_prior_enabled()
        skip_pretrain = resume_stage in [stage1_opt, stage2_opt, "stage1", "stage2"]

        if pretrain_prior and self.model.use_priors and not skip_pretrain:
            max_steps = self.config.get_pretrain_prior_max_steps()
            tol_grad = self.config.get_pretrain_prior_tol_grad()
            results["prior_pretrain"] = self.pretrain_prior(
                max_steps=max_steps,
                tol_grad=tol_grad
            )

        # Stage 1: AdaBelief (or configured optimizer)
        skip_stage1 = resume_stage in [stage2_opt, "stage2"]
        stage1_start_epoch = resume_epoch if resume_stage == stage1_opt else 0

        if stage1_epochs > 0 and not skip_stage1:
            results["stage1"] = self.train_stage(
                stage1_opt,
                stage1_epochs,
                start_epoch=stage1_start_epoch,
                checkpoint_freq=checkpoint_freq
            )

        # Stage 2: Yogi (or configured optimizer)
        stage2_start_epoch = resume_epoch if resume_stage == stage2_opt else 0

        if stage2_epochs > 0:
            results["stage2"] = self.train_stage(
                stage2_opt,
                stage2_epochs,
                start_epoch=stage2_start_epoch,
                checkpoint_freq=checkpoint_freq
            )

        return results

    def evaluate_frame(self, frame_idx: int = 0) -> Dict[str, Any]:
        """
        Evaluate model on a single frame.

        Args:
            frame_idx: Frame index to evaluate

        Returns:
            Dictionary with energy components and force errors
        """
        R = self.train_loader.R[frame_idx]
        F_ref = self.train_loader.F[frame_idx]
        mask = self.train_loader.mask[frame_idx]
        species = self.train_loader.species[frame_idx]

        # Compute energy components
        components = self.model.compute_components(
            self.best_params or self.params,
            R, mask, species
        )

        # Compute forces
        def energy_fn(R_):
            return self.model.compute_energy(
                self.best_params or self.params,
                R_, mask, species
            )

        F_pred = -jax.grad(energy_fn)(R)

        # Compute errors (only for real atoms)
        real_mask = mask > 0
        F_pred_real = F_pred[real_mask]
        F_ref_real = F_ref[real_mask]

        rmse = float(jnp.sqrt(jnp.mean((F_pred_real - F_ref_real) ** 2)))
        mae = float(jnp.mean(jnp.abs(F_pred_real - F_ref_real)))

        return {
            "energy_components": {k: float(v) for k, v in components.items()},
            "force_rmse": rmse,
            "force_mae": mae,
        }

    def save_params(self, output_path: str):
        """
        Save model parameters to pickle file.

        Args:
            output_path: Path to save parameters
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        params_to_save = self.best_params if self.best_params is not None else self.params

        with open(output_path, 'wb') as f:
            pickle.dump(params_to_save, f)

        training_logger.info(f"Saved parameters to: {output_path}")

    def load_params(self, input_path: str):
        """
        Load model parameters from pickle file.

        Args:
            input_path: Path to load parameters from
        """
        input_path = Path(input_path)

        with open(input_path, 'rb') as f:
            self.params = pickle.load(f)

        self.best_params = self.params
        training_logger.info(f"Loaded parameters from: {input_path}")

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from training."""
        return self.best_params if self.best_params is not None else self.params

    def save_checkpoint(self, output_path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save full training checkpoint for resume capability.

        Args:
            output_path: Path to save checkpoint
            metadata: Optional metadata dict (e.g., current epoch, stage info)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "params": self.params,
            "best_params": self.best_params,
            "metadata": metadata or {},
        }

        with open(output_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        training_logger.info(f"Saved checkpoint to: {output_path}")

    def load_checkpoint(self, input_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint to resume training.

        Args:
            input_path: Path to checkpoint file

        Returns:
            Checkpoint metadata (e.g., epoch info)
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {input_path}")

        with open(input_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint["params"]
        self.best_params = checkpoint.get("best_params", checkpoint["params"])
        metadata = checkpoint.get("metadata", {})

        training_logger.info(f"Loaded checkpoint from: {input_path}")
        if metadata:
            training_logger.info(f"Checkpoint metadata: {metadata}")

        return metadata

    def load_chemtrain_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a chemtrain trainer checkpoint for resumption.

        Supports two formats saved by chemtrain's save_trainer():
        1. Dict format (full_checkpoint=False, default): keys are 'trainer_state',
           'best_params', 'train_losses', 'val_losses', etc.
        2. Full trainer object (full_checkpoint=True): has .params, .opt_state, etc.

        Also restores the optimizer state so the LR schedule continues seamlessly.

        Args:
            checkpoint_path: Path to chemtrain checkpoint file (.pkl)

        Returns:
            Dictionary with resume metadata:
                - stage: Stage name (e.g., "adabelief", "yogi")
                - completed_epochs: Number of epochs completed
                - train_losses: Training loss history
                - val_losses: Validation loss history
        """
        import re
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, 'rb') as f:
            saved = pickle.load(f)

        # chemtrain's save_trainer() (full_checkpoint=False, the default) saves a plain
        # dict with keys: 'trainer_state' (containing 'params' and 'opt_state'),
        # 'best_params', 'train_losses', 'val_losses', etc.
        if isinstance(saved, dict):
            trainer_state = saved.get('trainer_state', {})

            if 'params' in trainer_state:
                self.params = jax.tree_util.tree_map(jnp.asarray, trainer_state['params'])
                training_logger.info("Restored model params from trainer_state['params']")
            else:
                training_logger.warning("No 'params' key in trainer_state — params not restored!")

            if 'opt_state' in trainer_state:
                # Store for restoration in the next train_stage call
                self._resume_opt_state = trainer_state['opt_state']
                training_logger.info("Saved optimizer state for restoration in train_stage")

            if 'best_params' in saved:
                self.best_params = jax.tree_util.tree_map(jnp.asarray, saved['best_params'])
                training_logger.info("Restored best_params from checkpoint")
            else:
                self.best_params = self.params

            train_losses = list(saved.get('train_losses', []))
            val_losses = list(saved.get('val_losses', []))

        else:
            # Full trainer object (full_checkpoint=True) — less common
            if hasattr(saved, 'params'):
                self.params = saved.params
            if hasattr(saved, 'best_inference_params'):
                self.best_params = saved.best_inference_params
            elif hasattr(saved, 'best_params'):
                self.best_params = saved.best_params
            else:
                self.best_params = self.params
            # Try to extract opt_state from the full trainer object
            if hasattr(saved, 'state') and hasattr(saved.state, 'opt_state'):
                self._resume_opt_state = saved.state.opt_state
            train_losses = list(getattr(saved, 'train_losses', []))
            val_losses = list(getattr(saved, 'val_losses', []))

        training_logger.info(f"Loaded chemtrain checkpoint from: {checkpoint_path}")

        # Try to load metadata from companion .meta.pkl file
        meta_path = checkpoint_path.with_suffix(".meta.pkl")
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            training_logger.info(f"Loaded metadata: stage={metadata.get('stage')}, "
                               f"epochs={metadata.get('completed_epochs')}")
        else:
            # Infer epoch count from filename (e.g. epoch00040.pkl → 40)
            epoch_match = re.match(r'epoch0*(\d+)', checkpoint_path.stem)
            inferred_epoch = int(epoch_match.group(1)) if epoch_match else 0

            # Default stage to stage1 optimizer since epoch*.pkl files are only written
            # during stage 1 (stage checkpoints have a different naming convention)
            inferred_stage = self.config.get_stage1_optimizer()

            metadata = {
                "stage": inferred_stage,
                "completed_epochs": inferred_epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }
            training_logger.info(
                f"No metadata file found — inferred from filename: "
                f"stage='{inferred_stage}', completed_epochs={inferred_epoch}"
            )

        return metadata

    def __repr__(self) -> str:
        return (
            f"Trainer(model={self.model}, batch_per_device={self.batch_per_device}, "
            f"checkpoint_path={self.checkpoint_path})"
        )
