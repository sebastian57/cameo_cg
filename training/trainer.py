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

    def _create_chemtrain_loaders(self) -> DataLoaders:
        """
        Create chemtrain DataLoaders from our loaders.

        Returns:
            chemtrain.data.data_loaders.DataLoaders instance
        """
        # Convert our DatasetLoader to NumpyDataLoader if needed
        if not isinstance(self.train_loader, NumpyDataLoader):
            train_np_loader = NumpyDataLoader(
                R=np.array(self.train_loader.R),
                F=np.array(self.train_loader.F),
                mask=np.array(self.train_loader.mask),
                species=np.array(self.train_loader.species),
                copy=False
            )
        else:
            train_np_loader = self.train_loader

        if self.val_loader is not None:
            if not isinstance(self.val_loader, NumpyDataLoader):
                val_np_loader = NumpyDataLoader(
                    R=np.array(self.val_loader.R),
                    F=np.array(self.val_loader.F),
                    mask=np.array(self.val_loader.mask),
                    species=np.array(self.val_loader.species),
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

        # Train with periodic checkpointing
        trainer.train(remaining_epochs, checkpoint_freq=checkpoint_freq if checkpoint_freq > 0 else None)

        # Update parameters
        self.params = trainer.params
        self.best_params = trainer.best_inference_params
        self._chemtrain_trainer = trainer
        if self.model.use_priors and getattr(self.model, "train_priors", False) and "prior" in self.params:
            self.model.prior.params = self.params["prior"]

        # Save stage checkpoint with metadata for resume capability
        if checkpoint_freq > 0:
            self._save_stage_checkpoint(optimizer_name, epochs)

        # Get final losses
        final_losses = {
            "train_loss": float(trainer.train_losses[-1]) if trainer.train_losses else 0.0,
            "val_loss": float(trainer.val_losses[-1]) if trainer.val_losses else 0.0,
        }

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

        # Print fitted parameters
        training_logger.info("\n[LBFGS] Fitted parameters:")
        for key, val in fitted_params.items():
            if jnp.ndim(val) == 0:
                training_logger.info(f"  {key}: {float(val):.6f}")
            else:
                training_logger.info(f"  {key}: {val}")

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

        This loads checkpoints saved by chemtrain's save_trainer() method,
        which includes the full trainer state (params, optimizer state, losses).

        Args:
            checkpoint_path: Path to chemtrain checkpoint file (.pkl)

        Returns:
            Dictionary with resume metadata:
                - stage: Stage name (e.g., "adabelief", "yogi")
                - completed_epochs: Number of epochs completed
                - train_losses: Training loss history
                - val_losses: Validation loss history
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load the pickled trainer
        with open(checkpoint_path, 'rb') as f:
            saved_trainer = pickle.load(f)

        # Extract parameters from saved trainer
        # chemtrain trainers have .params and .best_inference_params attributes
        if hasattr(saved_trainer, 'params'):
            self.params = saved_trainer.params
        if hasattr(saved_trainer, 'best_inference_params'):
            self.best_params = saved_trainer.best_inference_params
        elif hasattr(saved_trainer, 'best_params'):
            self.best_params = saved_trainer.best_params
        else:
            self.best_params = self.params

        training_logger.info(f"Loaded chemtrain checkpoint from: {checkpoint_path}")

        # Try to load metadata from companion .meta.pkl file
        meta_path = checkpoint_path.with_suffix(".meta.pkl")
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            training_logger.info(f"Loaded metadata: stage={metadata.get('stage')}, "
                               f"epochs={metadata.get('completed_epochs')}")
        else:
            # Fallback: extract what we can from the trainer
            metadata = {
                "stage": "unknown",
                "completed_epochs": getattr(saved_trainer, '_epoch', 0),
                "train_losses": list(getattr(saved_trainer, 'train_losses', [])),
                "val_losses": list(getattr(saved_trainer, 'val_losses', [])),
            }
            training_logger.info(f"No metadata file found, extracted from trainer: "
                               f"epochs={metadata['completed_epochs']}")

        return metadata

    def __repr__(self) -> str:
        return (
            f"Trainer(model={self.model}, batch_per_device={self.batch_per_device}, "
            f"checkpoint_path={self.checkpoint_path})"
        )
