#!/usr/bin/env python3
"""
Summarize training runs into per-input analysis directories.

Training run layout:
    outputs/YYYYMMDD_run-name/
        config_input.yaml
        config_runtime.yaml
        train_<jobid>.log
        exports/
        checkpoints/

Analysis layout:
    outputs/YYYYMMDD_run-name_analysis/
        summary.csv
        tail_loss_plots/
        force_eval_plots/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.manager import ConfigManager
from data.loader import _COORD_ALIASES, _resolve_key

FIELDNAMES = [
    'run_dir_name',
    'run_date',
    'run_name',
    'config_name',
    'config_path',
    'input_config_path',
    'model_context',
    'model_id',
    'status',
    'is_re_run',
    'job_id',
    'run_dir',
    'analysis_dir',
    'summary_csv_path',
    'tail_plots_dir',
    'tail_fit_plot_path',
    'tail_fit_plot_error',
    'force_eval_plots_dir',
    'force_components_plot_path',
    'force_distribution_plot_path',
    'force_magnitude_plot_path',
    'force_vs_position_plot_path',
    'force_gaussian_plot_path',
    'force_gaussian_csv_path',
    'force_eval_plot_error',
    'export_dir',
    'log_path',
    'slurm_path',
    'params_path',
    'checkpoint_path',
    'initial_train_loss',
    'final_train_loss',
    'final_val_loss',
    'tail_loss_slope_last_third',
    'tail_loss_intercept_last_third',
    'n_logged_epochs',
    'epoch_wall_seconds',
    'batch_per_device',
    'global_batch_size',
    'optimizer_steps_per_epoch',
    'n_train_structures',
    'n_train_items',
    'force_eval_frames',
    'force_rmse',
    'force_mae',
    'force_error_magnitude_mean',
    'force_error_magnitude_std',
    'force_magnitude_diff_mean',
    'force_magnitude_diff_std',
    'force_magnitude_abs_diff_mean',
    'force_eval_per_frame_csv_path',
    'force_eval_worst_frames_json_path',
    'force_eval_error',
    'detailed_force_eval_dir',
    'detailed_metrics_json_path',
    'detailed_metrics_csv_path',
    'detailed_shuffle_csv_path',
    'detailed_baseline_plot_path',
    'detailed_shuffle_plot_path',
    'detailed_cosine_plot_path',
    'detailed_eval_error',
    'detailed_split_name',
    'detailed_n_train_samples',
    'detailed_n_eval_samples',
    'detailed_rmse_model',
    'detailed_rmse_zero',
    'detailed_rmse_mean',
    'detailed_rmse_shuffle_mean',
    'detailed_rmse_shuffle_std',
    'detailed_shuffle_gap_rmse',
    'detailed_pearson_global',
    'detailed_mean_cosine_similarity',
    'detailed_r2_explained_variance',
    'detailed_variance_ratio_pred_to_ref',
    'detailed_validation_per_frame_csv_path',
    'detailed_validation_worst_frames_json_path',
    'detailed_noised_rmse_model',
    'detailed_noised_rmse_zero',
    'detailed_noised_mean_cosine_similarity',
    'detailed_noised_n_eval_samples',
    'detailed_noised_per_frame_csv_path',
    'detailed_noised_worst_frames_json_path',
    'detailed_noised_eval_error',
    'complete_eval_dir',
    'complete_eval_metrics_json_path',
    'complete_eval_arrays_npz_path',
    'complete_eval_error',
    'complete_eval_n_val_frames',
    'complete_eval_n_valid_beads',
    'complete_eval_calibration_slope',
    'complete_eval_calibration_intercept',
    'complete_eval_calibration_pearson',
    'complete_eval_smoothness_mean',
    'complete_eval_smoothness_p95',
    'complete_eval_smoothness_max',
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float('nan')


def _fit_tail_line(losses: List[float]) -> Optional[Dict[str, Any]]:
    finite_losses = np.asarray([x for x in losses if np.isfinite(x)], dtype=np.float64)
    if finite_losses.size < 2:
        return None
    tail_len = max(2, int(math.ceil(finite_losses.size / 3.0)))
    tail_start = finite_losses.size - tail_len
    x_all = np.arange(finite_losses.size, dtype=np.float64)
    y_all = finite_losses
    x_tail = x_all[tail_start:]
    y_tail = y_all[tail_start:]
    slope, intercept = np.polyfit(x_tail, y_tail, 1)
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'tail_start': int(tail_start),
        'tail_len': int(tail_len),
        'x_all': x_all,
        'y_all': y_all,
        'x_tail': x_tail,
        'y_tail': y_tail,
    }


def _tail_slope(losses: List[float]) -> float:
    fit = _fit_tail_line(losses)
    if fit is None:
        return float('nan')
    return float(fit['slope'])


def _load_loss_history(log_path: Path) -> tuple[List[float], List[float]]:
    train_losses: List[float] = []
    val_losses: List[float] = []
    last_recorded_epoch: Optional[int] = None
    epoch: Optional[int] = None
    pending_train_loss = False

    with open(log_path, 'r') as f:
        for line in f:
            m_epoch = re.search(r"\[Epoch (\d+)\]", line)
            if m_epoch:
                epoch = int(m_epoch.group(1))

            m_train = re.search(
                r'Average train loss:\s*([0-9.eE+-]+|nan|inf|-inf)',
                line,
                re.IGNORECASE,
            )
            if m_train and epoch is not None:
                train_loss = float(m_train.group(1))
                if last_recorded_epoch is None or epoch != last_recorded_epoch:
                    train_losses.append(train_loss)
                    last_recorded_epoch = epoch
                    pending_train_loss = True

            m_val = re.search(
                r'Average val loss:\s*([0-9.eE+-]+|nan|inf|-inf)',
                line,
                re.IGNORECASE,
            )
            if m_val and pending_train_loss:
                val_losses.append(float(m_val.group(1)))
                pending_train_loss = False

    while len(val_losses) < len(train_losses):
        val_losses.append(float('nan'))

    return train_losses, val_losses


def _load_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    with open(checkpoint_path, 'rb') as f:
        payload = pickle.load(f)
    return dict(payload.get('metadata', {}))


def _mean_epoch_time_from_results(results: Dict[str, Any]) -> float:
    epoch_times = []
    for result in results.values():
        if isinstance(result, dict) and 'epoch_wall_seconds_est' in result:
            value = _safe_float(result.get('epoch_wall_seconds_est'))
            if np.isfinite(value):
                epoch_times.append(value)
    if not epoch_times:
        return float('nan')
    return float(np.mean(epoch_times))


def _select_eval_venv(config_path: Path) -> tuple[Path, str]:
    config = ConfigManager(str(config_path))
    ml_model = config.get_ml_model_type()

    if ml_model.startswith("allegro_cueq"):
        preferred_env = "CAMEO_CUEQ_VENV"
    else:
        preferred_env = "CAMEO_STANDARD_VENV"

    venv_raw = os.environ.get(preferred_env) or os.environ.get("CAMEO_ACTIVE_VENV")
    if not venv_raw:
        raise RuntimeError(
            f"Missing virtual environment path. Set {preferred_env} or CAMEO_ACTIVE_VENV."
        )

    venv_dir = Path(venv_raw).expanduser().resolve()
    python_bin = venv_dir / "bin" / "python"
    if not python_bin.exists():
        raise FileNotFoundError(f"Python executable not found in venv: {python_bin}")

    return python_bin, venv_dir.name


def _subprocess_env_for_config(config_path: Path) -> tuple[Path, str, Dict[str, str]]:
    python_bin, venv_name = _select_eval_venv(config_path)
    env = os.environ.copy()
    env['VIRTUAL_ENV'] = str(python_bin.parent.parent)
    env['PATH'] = f"{python_bin.parent}:{env.get('PATH', '')}" if env.get('PATH') else str(python_bin.parent)
    return python_bin, venv_name, env


def _resolve_data_path(config: ConfigManager) -> Path:
    data_path = Path(config.get_data_path())
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    return data_path


def _load_mask(data_path: Path) -> np.ndarray:
    def _mask_from_npz(data: Any, source: Path) -> np.ndarray:
        if 'mask' in data:
            return np.asarray(data['mask'], dtype=np.float32)
        r_key = _resolve_key(data, 'R', _COORD_ALIASES, str(source))
        print(
            f"[analyze_suite] WARNING: dataset {source} has no 'mask'; "
            "using all-ones mask derived from R.shape[:2].",
            flush=True,
        )
        return np.ones(np.asarray(data[r_key]).shape[:2], dtype=np.float32)

    if data_path.is_dir():
        masks = []
        for npz_file in sorted(data_path.glob('*.npz')):
            with np.load(npz_file, allow_pickle=True) as data:
                masks.append(_mask_from_npz(data, npz_file))
        if not masks:
            raise FileNotFoundError(f'No .npz files found in directory: {data_path}')
        return np.concatenate(masks, axis=0)

    with np.load(data_path, allow_pickle=True) as data:
        return _mask_from_npz(data, data_path)


def _dataset_frame_indices(n_total: int, max_frames: Optional[int], seed: int) -> np.ndarray:
    if max_frames is not None and int(max_frames) < n_total:
        rng = np.random.RandomState(seed)
        return rng.permutation(n_total)[: int(max_frames)]
    return np.arange(n_total, dtype=np.int32)


def _split_train_indices(n_frames: int, val_fraction: float, min_val_samples: int, seed: int) -> tuple[np.ndarray, int, int]:
    indices = np.arange(n_frames, dtype=np.int32)
    if n_frames > 1:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_frames)

    n_train = int(np.round(n_frames * (1.0 - val_fraction)))
    n_val = n_frames - n_train
    if n_val < min_val_samples:
        n_train = n_frames
        n_val = 0
    return indices[:n_train], n_train, n_val


def _pack_structures_greedy(
    sorted_indices: np.ndarray,
    valid_counts: np.ndarray,
    target_beads: int,
    drop_incomplete: bool,
) -> list[list[int]]:
    tiles: list[list[int]] = []
    current: list[int] = []
    current_beads = 0

    for idx in sorted_indices:
        n_valid = int(valid_counts[idx])
        if n_valid <= 0:
            continue
        if current and (current_beads + n_valid > target_beads):
            tiles.append(current)
            current = []
            current_beads = 0
        current.append(int(idx))
        current_beads += n_valid

    if current:
        if not drop_incomplete or current_beads >= target_beads or not tiles:
            tiles.append(current)

    return tiles


def _build_train_mask(config: ConfigManager, *, devices_per_run: int) -> tuple[np.ndarray, int, int]:
    mask = _load_mask(_resolve_data_path(config))
    selected = _dataset_frame_indices(mask.shape[0], config.get_max_frames(), int(config.get_seed()))
    mask = mask[selected]

    train_indices, n_train, n_val = _split_train_indices(
        mask.shape[0],
        val_fraction=float(config.get_val_fraction()),
        min_val_samples=int(config.get_batch_per_device()) * int(devices_per_run),
        seed=int(config.get_seed()),
    )
    train_mask = np.asarray(mask[train_indices], dtype=np.float32)
    return train_mask, n_train, n_val


def _estimate_optimizer_steps_per_epoch(config: ConfigManager, devices_per_run: int) -> Dict[str, int]:
    train_mask, n_train, _ = _build_train_mask(config, devices_per_run=devices_per_run)

    if config.get_batch_mode() == 'tiled':
        valid_counts = np.asarray(np.sum(train_mask > 0, axis=1), dtype=np.int32)
        order = np.arange(n_train, dtype=np.int32)
        if config.tile_shuffle_structures_enabled():
            rng = np.random.RandomState(int(config.get_seed()))
            rng.shuffle(order)
        if config.tile_sort_by_size_enabled():
            order = order[np.argsort(valid_counts[order])[::-1]]
        tiles = _pack_structures_greedy(
            order,
            valid_counts,
            target_beads=config.get_tile_target_beads(),
            drop_incomplete=config.tile_drop_incomplete_enabled(),
        )
        n_train_items = int(len(tiles))
    else:
        n_train_items = n_train

    batch_per_device = int(config.get_batch_per_device())
    global_batch_size = batch_per_device * int(devices_per_run)
    optimizer_steps_per_epoch = int(math.ceil(n_train_items / max(global_batch_size, 1))) if n_train_items > 0 else 0

    return {
        'n_train_structures': int(n_train),
        'n_train_items': int(n_train_items),
        'batch_per_device': batch_per_device,
        'global_batch_size': global_batch_size,
        'optimizer_steps_per_epoch': optimizer_steps_per_epoch,
    }


def _load_params(params_path: Path) -> Dict[str, Any]:
    with open(params_path, 'rb') as f:
        payload = pickle.load(f)

    params = payload
    if isinstance(payload, dict):
        if isinstance(payload.get('params'), dict):
            params = payload['params']
        elif isinstance(payload.get('trainer_state'), dict) and isinstance(payload['trainer_state'].get('params'), dict):
            params = payload['trainer_state']['params']

    if isinstance(params, dict) and 'ml' not in params and 'allegro' in params:
        params = dict(params)
        params['ml'] = params['allegro']
    return params


def _resolve_spline_path_if_needed(config: ConfigManager) -> None:
    if not config.use_spline_priors_enabled():
        return
    spline_path = Path(config.get_spline_file_path())
    if spline_path.is_absolute():
        resolved = spline_path
    else:
        candidates = [
            Path.cwd() / spline_path,
            PROJECT_ROOT / spline_path,
            config.config_path.parent / spline_path,
        ]
        resolved = spline_path
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                break
    config.set('model', 'priors', 'spline_file', str(resolved.resolve()))


def _apply_jax_compat_shims(jax_module: Any) -> None:
    if not hasattr(jax_module.random, 'KeyArray'):
        jax_module.random.KeyArray = jax_module.Array
    if not hasattr(jax_module, 'tree_map'):
        jax_module.tree_map = jax_module.tree_util.tree_map
    if not hasattr(jax_module, 'tree_leaves'):
        jax_module.tree_leaves = jax_module.tree_util.tree_leaves
    if not hasattr(jax_module, 'tree_flatten'):
        jax_module.tree_flatten = jax_module.tree_util.tree_flatten
    if not hasattr(jax_module, 'tree_unflatten'):
        jax_module.tree_unflatten = jax_module.tree_util.tree_unflatten
    if not hasattr(jax_module.lib, 'xla_bridge'):
        from jax._src import xla_bridge as _xla_bridge
        jax_module.lib.xla_bridge = _xla_bridge


def _collect_force_eval_data(
    config_path: Path,
    params_path: Path,
    *,
    n_frames: int,
    seed: int,
    devices_per_run: int,
) -> Dict[str, Any]:
    import jax
    import jax.numpy as jnp

    _apply_jax_compat_shims(jax)

    from data.loader import DatasetLoader
    from data.preprocessor import CoordinatePreprocessor
    from analysis_tests.evaluator import Evaluator
    from models.combined_model import CombinedModel

    config = ConfigManager(str(config_path))
    _resolve_spline_path_if_needed(config)

    data_path = _resolve_data_path(config)
    loader = DatasetLoader(str(data_path), max_frames=config.get_max_frames(), seed=config.get_seed())
    preprocessor = CoordinatePreprocessor(
        cutoff=config.get_cutoff(),
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
    )
    extent, r_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    dataset = loader.get_all()
    dataset['R'] = preprocessor.center_and_park(dataset['R'], dataset['mask'], extent, r_shift)

    train_indices, n_train, _ = _split_train_indices(
        dataset['R'].shape[0],
        val_fraction=float(config.get_val_fraction()),
        min_val_samples=int(config.get_batch_per_device()) * int(devices_per_run),
        seed=int(config.get_seed()),
    )
    train_subset = {
        'R': np.asarray(dataset['R'][train_indices], dtype=np.float32),
        'F': np.asarray(dataset['F'][train_indices], dtype=np.float32),
        'mask': np.asarray(dataset['mask'][train_indices], dtype=np.float32),
        'species': np.asarray(dataset['species'][train_indices], dtype=np.int32),
    }
    box = np.asarray(extent, dtype=np.float32)

    n_frames = min(int(n_frames), int(n_train))
    if n_frames <= 0:
        raise ValueError('No training frames available for force evaluation.')

    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(n_train, size=n_frames, replace=False))

    config.set('model', 'use_priors', bool(config.export_combined_ml_priors_enabled()))
    config.set('model', 'train_priors', False)

    params = _load_params(params_path)
    model = CombinedModel(
        config=config,
        R0=train_subset['R'][0],
        box=box,
        species=train_subset['species'][0],
        N_max=loader.N_max,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )
    evaluator = Evaluator(model, params, config)

    all_pred = []
    all_ref = []
    all_mask = []
    all_R = []
    per_frame_errors = []
    for idx in indices:
        result = evaluator.evaluate_frame(
            jnp.asarray(train_subset['R'][idx]),
            jnp.asarray(train_subset['F'][idx]),
            jnp.asarray(train_subset['mask'][idx]),
            jnp.asarray(train_subset['species'][idx]),
        )
        pred_i = np.asarray(result['forces'], dtype=np.float32)
        ref_i = np.asarray(train_subset['F'][idx], dtype=np.float32)
        mask_i = np.asarray(train_subset['mask'][idx], dtype=np.float32)
        valid_i = mask_i > 0
        diff_i = pred_i[valid_i] - ref_i[valid_i]
        denom_i = float(np.linalg.norm(pred_i[valid_i].reshape(-1)) * np.linalg.norm(ref_i[valid_i].reshape(-1)))
        per_frame_errors.append({
            'frame_id': int(train_indices[idx]),
            'local_train_index': int(idx),
            'is_noised_frame': int(train_subset.get('is_noised_frame', np.zeros((n_train,), dtype=np.int32))[idx]) if 'is_noised_frame' in train_subset else 0,
            'noise_level_id': int(train_subset.get('noise_level_id', np.full((n_train,), -1, dtype=np.int32))[idx]) if 'noise_level_id' in train_subset else -1,
            'rmse': float(np.sqrt(np.mean(diff_i ** 2))) if diff_i.size else float('nan'),
            'mae': float(np.mean(np.abs(diff_i))) if diff_i.size else float('nan'),
            'cosine': float(np.dot(pred_i[valid_i].reshape(-1), ref_i[valid_i].reshape(-1)) / denom_i) if denom_i > 1e-12 else float('nan'),
            'max_error_norm': float(np.max(np.linalg.norm(diff_i, axis=-1))) if diff_i.size else float('nan'),
            'n_valid': int(np.sum(valid_i)),
        })
        all_pred.append(pred_i)
        all_ref.append(ref_i)
        all_mask.append(mask_i)
        all_R.append(np.asarray(train_subset['R'][idx], dtype=np.float32))

    f_pred = np.concatenate(all_pred, axis=0)
    f_ref = np.concatenate(all_ref, axis=0)
    mask = np.concatenate(all_mask, axis=0) > 0
    R = np.concatenate(all_R, axis=0)
    f_pred = f_pred[mask]
    f_ref = f_ref[mask]
    R = R[mask]

    diff = f_pred - f_ref
    diff_mag = np.linalg.norm(diff, axis=-1)
    pred_mag = np.linalg.norm(f_pred, axis=-1)
    ref_mag = np.linalg.norm(f_ref, axis=-1)
    mag_diff = pred_mag - ref_mag

    return {
        'force_eval_frames': float(n_frames),
        'force_rmse': float(np.sqrt(np.mean(diff ** 2))),
        'force_mae': float(np.mean(np.abs(diff))),
        'force_error_magnitude_mean': float(np.mean(diff_mag)),
        'force_error_magnitude_std': float(np.std(diff_mag)),
        'force_magnitude_diff_mean': float(np.mean(mag_diff)),
        'force_magnitude_diff_std': float(np.std(mag_diff)),
        'force_magnitude_abs_diff_mean': float(np.mean(np.abs(mag_diff))),
        'F_pred_real': f_pred,
        'F_ref_real': f_ref,
        'R_real': R,
        'per_frame_errors': sorted(per_frame_errors, key=lambda r: (not np.isfinite(r['rmse']), -r['rmse'])),
    }


def _is_run_dir(path: Path) -> bool:
    return (path / 'config_runtime.yaml').exists() or (path / 'config_input.yaml').exists() or (path / 'config_runtime_re.yaml').exists() or (path / 'config_input_re.yaml').exists()


def _is_re_run_dir(path: Path) -> bool:
    re_dir = path / 'relative_entropy'
    if not re_dir.is_dir():
        return False
    re_checkpoint = re_dir / 'relative_entropy_final_checkpoint.pkl'
    if not re_checkpoint.exists():
        re_checkpoints = list(re_dir.glob('relative_entropy_iter*.pkl'))
        return len(re_checkpoints) > 0
    return True


def _re_checkpoint_path(run_dir: Path, *, include_incomplete: bool = False) -> Path | None:
    re_dir = run_dir / 'relative_entropy'
    if not re_dir.is_dir():
        return None
    final_checkpoint = re_dir / 'relative_entropy_final_checkpoint.pkl'
    if final_checkpoint.exists():
        return final_checkpoint
    if not include_incomplete:
        return None
    checkpoints = sorted(re_dir.glob('relative_entropy_iter*.pkl'))
    if checkpoints:
        return checkpoints[-1]
    return None


def _load_re_training_history(run_dir: Path) -> List[Dict[str, Any]]:
    re_dir = run_dir / 'relative_entropy'
    if not re_dir.is_dir():
        return []
    history_path = re_dir / 'relative_entropy_history.csv'
    if not history_path.exists():
        return []
    history = []
    with history_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append(dict(row))
    return history


def _extract_re_params_from_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    with open(checkpoint_path, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        if 'params' in payload:
            return payload['params']
        if 'best_params' in payload:
            return payload['best_params']
        raise ValueError(f'RE checkpoint has no params/best_params. Keys: {list(payload.keys())}')
    raise TypeError(f'cannot extract params from RE checkpoint of type {type(payload)}')


def _write_re_params_from_checkpoint(checkpoint_path: Path, params_path: Path) -> None:
    params = _extract_re_params_from_checkpoint(checkpoint_path)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = params_path.with_suffix(params_path.suffix + '.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(params, f)
    os.replace(tmp_path, params_path)


def _iter_run_dirs(outputs_root: Path) -> Iterable[Path]:
    run_dirs = []
    for child in sorted(outputs_root.iterdir()):
        if not child.is_dir():
            continue
        if _is_run_dir(child):
            run_dirs.append(child)
    return run_dirs


def _parse_run_dir_name(name: str) -> tuple[str, str]:
    match = re.match(r'^(\d{8})_(.+)$', name)
    if match:
        return match.group(1), match.group(2)
    return '', name


def _latest_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _job_tag_from_log_path(log_path: Optional[Path]) -> str:
    if log_path is None:
        return ''
    match = re.match(r'^train_(.+)\.log$', log_path.name)
    return match.group(1) if match else ''


def _resolve_slurm_path(run_dir: Path, *, include_incomplete: bool = False) -> Path | None:
    slurm_path = _latest_file(run_dir, 'slurm-*.out')
    if slurm_path is not None and slurm_path.exists():
        return slurm_path
    if not include_incomplete:
        return None

    log_path = _latest_file(run_dir, 'train_*.log')
    job_tag = _job_tag_from_log_path(log_path)
    if not job_tag:
        return None

    bootstrap_path = run_dir.parent / f'slurm-bootstrap-{job_tag}.out'
    if bootstrap_path.exists():
        return bootstrap_path
    return None


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob('epoch*.pkl')) + list(checkpoint_dir.glob('stage_*.pkl'))
    checkpoints = [p for p in checkpoints if not p.name.endswith('.meta.pkl')]
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def _extract_params_from_checkpoint_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        if 'best_params' in payload:
            return payload['best_params']
        if 'trainer_state' in payload and 'params' in payload['trainer_state']:
            return payload['trainer_state']['params']
        if 'params' in payload:
            return payload['params']
        raise ValueError(f'unexpected checkpoint structure. Keys: {list(payload.keys())}')

    if hasattr(payload, 'best_inference_params'):
        return payload.best_inference_params
    if hasattr(payload, 'params'):
        return payload.params

    raise TypeError(f'cannot extract params from checkpoint of type {type(payload)}')


def _write_params_from_checkpoint(checkpoint_path: Path, params_path: Path) -> None:
    with open(checkpoint_path, 'rb') as f:
        payload = pickle.load(f)

    params = _extract_params_from_checkpoint_payload(payload)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = params_path.with_suffix(params_path.suffix + '.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(params, f)
    os.replace(tmp_path, params_path)


def _resolve_checkpoint_path(
    config: ConfigManager,
    run_dir: Path,
    *,
    include_incomplete: bool = False,
) -> Path | None:
    model_name = f'{config.get_model_context()}_{config.get_model_id()}'
    export_checkpoint = run_dir / 'exports' / f'{model_name}_checkpoint.pkl'
    if export_checkpoint.exists():
        return export_checkpoint
    if not include_incomplete:
        return export_checkpoint

    checkpoint_dir = Path(config.get_checkpoint_path())
    latest_checkpoint = _find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        return latest_checkpoint
    return export_checkpoint


def _ensure_params_path(
    config: ConfigManager,
    run_dir: Path,
    *,
    include_incomplete: bool = False,
) -> tuple[Path, Optional[str]]:
    model_name = f'{config.get_model_context()}_{config.get_model_id()}'
    params_path = run_dir / 'exports' / f'{model_name}_params.pkl'
    if params_path.exists():
        return params_path, None
    if not include_incomplete:
        return params_path, None

    checkpoint_path = _resolve_checkpoint_path(config, run_dir, include_incomplete=True)
    if checkpoint_path is None or not checkpoint_path.exists():
        return params_path, 'missing checkpoint for automatic params extraction'

    try:
        _write_params_from_checkpoint(checkpoint_path, params_path)
    except Exception as exc:
        return params_path, f'auto-extract failed from {checkpoint_path.name}: {exc}'

    return params_path, None


def _ensure_re_params_path(
    run_dir: Path,
    config: ConfigManager,
    *,
    include_incomplete: bool = False,
) -> tuple[Path, Optional[str]]:
    model_name = f'{config.get_model_context()}_{config.get_model_id()}'
    params_path = run_dir / 'relative_entropy' / f'{model_name}_params.pkl'
    if params_path.exists():
        return params_path, None
    if not include_incomplete:
        return params_path, None

    checkpoint_path = _re_checkpoint_path(run_dir, include_incomplete=True)
    if checkpoint_path is None or not checkpoint_path.exists():
        return params_path, 'missing RE checkpoint for automatic params extraction'

    try:
        _write_re_params_from_checkpoint(checkpoint_path, params_path)
    except Exception as exc:
        return params_path, f'auto-extract failed from {checkpoint_path.name}: {exc}'

    return params_path, None


def _resolve_input_run_dirs(input_path: Path) -> List[Path]:
    if _is_run_dir(input_path):
        return [input_path]

    if (input_path / 'outputs').is_dir():
        return list(_iter_run_dirs(input_path / 'outputs'))

    return list(_iter_run_dirs(input_path))


def _analysis_root_for_input(input_path: Path) -> Path:
    if _is_run_dir(input_path):
        return input_path.parent / f'{input_path.name}_analysis'
    if (input_path / 'outputs').is_dir():
        outputs_root = input_path / 'outputs'
        return outputs_root.parent / f'{outputs_root.name}_analysis'
    return input_path.parent / f'{input_path.name}_analysis'


def _read_text_if_exists(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ''
    try:
        return path.read_text(errors='replace')
    except Exception:
        return ''


def _completed_run_reason(run_dir: Path, include_incomplete: bool = False) -> tuple[bool, str]:
    log_path = _latest_file(run_dir, 'train_*.log')
    slurm_path = _resolve_slurm_path(run_dir, include_incomplete=include_incomplete)
    runtime_config = run_dir / 'config_runtime.yaml'
    input_config = run_dir / 'config_input.yaml'
    config_path = runtime_config if runtime_config.exists() else input_config

    if not config_path.exists():
        return False, 'missing config_runtime.yaml/config_input.yaml'
    if log_path is None or not log_path.exists():
        return False, 'missing train log'
    if (slurm_path is None or not slurm_path.exists()) and not include_incomplete:
        return False, 'missing slurm copy'

    if not include_incomplete:
        log_text = _read_text_if_exists(log_path)
        slurm_text = _read_text_if_exists(slurm_path)
        combined = '\n'.join([log_text, slurm_text])

        if 'Training has been unsuccessful' in combined:
            return False, 'training marked unsuccessful in log'
        if 'Traceback (most recent call last):' in combined:
            return False, 'traceback found in log'
        if 'Exited with exit code' in combined:
            return False, 'srun reported nonzero exit'
        if 'Complete!' not in combined:
            return False, 'run does not show final Complete! marker'

    config = ConfigManager(str(config_path))
    params_path, params_error = _ensure_params_path(config, run_dir, include_incomplete=include_incomplete)
    if params_error is not None:
        return False, params_error
    if not params_path.exists():
        return False, f'missing params export: {params_path.name}'

    return True, 'completed'


def _completed_re_run_reason(run_dir: Path, include_incomplete: bool = False) -> tuple[bool, str]:
    re_dir = run_dir / 'relative_entropy'
    if not re_dir.is_dir():
        return False, 'no relative_entropy/ subdirectory'

    re_checkpoint = _re_checkpoint_path(run_dir, include_incomplete=include_incomplete)
    if re_checkpoint is None or not re_checkpoint.exists():
        return False, 'missing RE checkpoint'

    re_history = re_dir / 'relative_entropy_history.csv'
    re_log = re_dir / 'relative_entropy_loss.log'

    if not include_incomplete:
        if not re_history.exists() and not re_log.exists():
            return False, 'missing RE history CSV or log'
        log_text = _read_text_if_exists(re_log)
        slurm_path = _resolve_slurm_path(run_dir, include_incomplete=False)
        slurm_text = _read_text_if_exists(slurm_path)
        combined = '\n'.join([log_text, slurm_text])
        if 'Traceback (most recent call last):' in combined:
            return False, 'traceback found in log'
        if 'Exited with exit code' in combined:
            return False, 'srun reported nonzero exit'

    return True, 'completed'


def _filter_completed_run_dirs(run_dirs: List[Path], include_incomplete: bool = False) -> tuple[List[Path], List[tuple[Path, str]]]:
    completed: List[Path] = []
    excluded: List[tuple[Path, str]] = []
    for run_dir in run_dirs:
        if _is_re_run_dir(run_dir):
            is_completed, reason = _completed_re_run_reason(run_dir, include_incomplete=include_incomplete)
        else:
            is_completed, reason = _completed_run_reason(run_dir, include_incomplete=include_incomplete)
        if is_completed:
            completed.append(run_dir)
        else:
            excluded.append((run_dir, reason))
    return completed, excluded


def _write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_tail_fit_plot(train_losses: List[float], val_losses: Optional[List[float]], output_path: Path, *, title: str) -> str:
    fit = _fit_tail_line(train_losses)
    if fit is None:
        return 'Not enough finite training-loss points for a linear tail fit.'

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f'Plotting import failed: {exc}'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_all = fit['x_all']
    y_all = fit['y_all']
    x_tail = fit['x_tail']
    y_fit = fit['slope'] * x_tail + fit['intercept']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_all, y_all, color='tab:blue', linewidth=1.8, label='Train loss')
    if val_losses:
        y_val = np.asarray(val_losses, dtype=np.float64)
        if y_val.size > 0:
            n_val = min(y_val.size, x_all.size)
            x_val = x_all[:n_val]
            y_val = y_val[:n_val]
            mask = np.isfinite(y_val)
            if np.any(mask):
                ax.plot(x_val[mask], y_val[mask], color='tab:green', linewidth=1.6, label='Val loss')
    ax.plot(x_tail, fit['y_tail'], color='tab:orange', linewidth=2.2, label='Tail segment')
    ax.plot(x_tail, y_fit, color='tab:red', linestyle='--', linewidth=2.0, label=f"Linear fit (slope={fit['slope']:.4g})")
    ax.axvline(float(fit['tail_start']), color='gray', linestyle=':', linewidth=1.2, label='Tail start')
    ax.set_xlabel('Logged epoch index')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return ''


def _save_re_loss_plot(re_history: List[Dict[str, Any]], output_path: Path, *, title: str) -> str:
    if not re_history:
        return 'Empty RE history'

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f'Plotting import failed: {exc}'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    iterations = []
    objectives = []
    gaps = []
    grad_norms = []
    for row in re_history:
        try:
            iterations.append(int(row.get('iteration', 0)))
            objectives.append(float(row.get('objective', float('nan'))))
            gaps.append(float(row.get('re_energy_gap', float('nan'))))
            grad_norms.append(float(row.get('grad_norm', float('nan'))))
        except (ValueError, TypeError):
            continue

    if not iterations:
        return 'No numeric data in RE history'

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(iterations, objectives, marker='o', linewidth=1.5, markersize=4, label='Objective (abs gap)')
    axes[0].set_ylabel('Objective')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, gaps, marker='o', linewidth=1.5, markersize=4, label='RE Energy Gap', color='tab:orange')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel('Gap')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iterations, grad_norms, marker='o', linewidth=1.5, markersize=4, label='Grad Norm', color='tab:green')
    axes[2].set_ylabel('Grad Norm')
    axes[2].set_xlabel('Iteration')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return ''


def _collect_re_force_eval_data(
    config_path: Path,
    params_path: Path,
    run_dir: Path,
    *,
    n_frames: int,
    seed: int,
    devices_per_run: int,
) -> Dict[str, Any]:
    import jax
    import jax.numpy as jnp

    _apply_jax_compat_shims(jax)

    from data.loader import DatasetLoader
    from data.preprocessor import CoordinatePreprocessor
    from analysis_tests.evaluator import Evaluator
    from models.combined_model import CombinedModel
    from training.relative_entropy import relative_entropy_config

    config = ConfigManager(str(config_path))
    _resolve_spline_path_if_needed(config)

    re_cfg = relative_entropy_config(config)
    if not re_cfg.reference_data_path:
        raise ValueError('RE config missing reference_data_path')

    from training.path_utils import repo_root_from_file, resolve_from_config_or_repo

    data_path = resolve_from_config_or_repo(
        re_cfg.reference_data_path,
        config.config_path,
        repo_root_from_file(__file__),
    )

    loader = DatasetLoader(str(data_path), max_frames=config.get_max_frames(), seed=config.get_seed())
    dataset = loader.get_all()

    use_pbc = config.use_pbc_enabled()
    if use_pbc:
        if loader.box is None:
            raise ValueError('model.pbc=true requires a box in the RE reference dataset')
        box = jnp.asarray(loader.box, dtype=jnp.float32)
        dataset['R'] = jnp.mod(jnp.asarray(dataset['R'], dtype=jnp.float32), box[None, None, :])
    else:
        preprocessor = CoordinatePreprocessor(
            cutoff=config.get_cutoff(),
            buffer_multiplier=config.get_buffer_multiplier(),
            park_multiplier=config.get_park_multiplier(),
        )
        box, shift = preprocessor.compute_box_extent(loader.R, loader.mask)
        dataset['R'] = preprocessor.center_and_park(
            jnp.asarray(dataset['R'], dtype=jnp.float32),
            jnp.asarray(dataset['mask'], dtype=jnp.float32),
            box, shift,
        )

    params = _load_params(params_path)
    model = CombinedModel(
        config=config,
        R0=dataset['R'][0],
        box=box,
        species=dataset['species'][0] if dataset.get('species') is not None else jnp.zeros_like(dataset['mask'][0], dtype=jnp.int32),
        N_max=loader.N_max,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )
    evaluator = Evaluator(model, params, config)

    n_total = int(dataset['R'].shape[0])
    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(n_total, size=min(n_frames, n_total), replace=False))

    all_pred = []
    all_ref = []
    all_mask = []
    all_R = []
    per_frame_errors = []

    for idx in indices:
        result = evaluator.evaluate_frame(
            jnp.asarray(dataset['R'][idx]),
            jnp.asarray(dataset['F'][idx]),
            jnp.asarray(dataset['mask'][idx]),
            jnp.asarray(dataset['species'][idx]) if dataset.get('species') is not None else jnp.zeros_like(dataset['mask'][idx], dtype=jnp.int32),
        )
        pred_i = np.asarray(result['forces'], dtype=np.float32)
        ref_i = np.asarray(dataset['F'][idx], dtype=np.float32)
        mask_i = np.asarray(dataset['mask'][idx], dtype=np.float32)
        valid_i = mask_i > 0
        diff_i = pred_i[valid_i] - ref_i[valid_i]
        denom_i = float(np.linalg.norm(pred_i[valid_i].reshape(-1)) * np.linalg.norm(ref_i[valid_i].reshape(-1)))
        per_frame_errors.append({
            'frame_id': int(idx),
            'local_train_index': int(idx),
            'is_noised_frame': 0,
            'noise_level_id': -1,
            'rmse': float(np.sqrt(np.mean(diff_i ** 2))) if diff_i.size else float('nan'),
            'mae': float(np.mean(np.abs(diff_i))) if diff_i.size else float('nan'),
            'cosine': float(np.dot(pred_i[valid_i].reshape(-1), ref_i[valid_i].reshape(-1)) / denom_i) if denom_i > 1e-12 else float('nan'),
            'max_error_norm': float(np.max(np.linalg.norm(diff_i, axis=-1))) if diff_i.size else float('nan'),
            'n_valid': int(np.sum(valid_i)),
        })
        all_pred.append(pred_i)
        all_ref.append(ref_i)
        all_mask.append(mask_i)
        all_R.append(np.asarray(dataset['R'][idx], dtype=np.float32))

    f_pred = np.concatenate(all_pred, axis=0)
    f_ref = np.concatenate(all_ref, axis=0)
    mask = np.concatenate(all_mask, axis=0) > 0
    R = np.concatenate(all_R, axis=0)
    f_pred = f_pred[mask]
    f_ref = f_ref[mask]
    R = R[mask]

    diff = f_pred - f_ref
    diff_mag = np.linalg.norm(diff, axis=-1)
    pred_mag = np.linalg.norm(f_pred, axis=-1)
    ref_mag = np.linalg.norm(f_ref, axis=-1)
    mag_diff = pred_mag - ref_mag

    return {
        'force_eval_frames': float(len(indices)),
        'force_rmse': float(np.sqrt(np.mean(diff ** 2))),
        'force_mae': float(np.mean(np.abs(diff))),
        'force_error_magnitude_mean': float(np.mean(diff_mag)),
        'force_error_magnitude_std': float(np.std(diff_mag)),
        'force_magnitude_diff_mean': float(np.mean(mag_diff)),
        'force_magnitude_diff_std': float(np.std(mag_diff)),
        'force_magnitude_abs_diff_mean': float(np.mean(np.abs(mag_diff))),
        'F_pred_real': f_pred,
        'F_ref_real': f_ref,
        'R_real': R,
        'per_frame_errors': sorted(per_frame_errors, key=lambda r: (not np.isfinite(r['rmse']), -r['rmse'])),
    }


def _save_force_eval_plots(force_eval_data: Dict[str, Any], plot_prefix: str, force_eval_plots_dir: Path) -> Dict[str, str]:
    from analysis_tests.visualizer import ForceAnalyzer

    force_eval_plots_dir.mkdir(parents=True, exist_ok=True)
    f_pred = force_eval_data['F_pred_real']
    f_ref = force_eval_data['F_ref_real']
    R_real = force_eval_data['R_real']

    components = force_eval_plots_dir / f'{plot_prefix}_force_components.png'
    distribution = force_eval_plots_dir / f'{plot_prefix}_force_distribution.png'
    magnitude = force_eval_plots_dir / f'{plot_prefix}_force_magnitude.png'
    vs_position = force_eval_plots_dir / f'{plot_prefix}_force_vs_position.png'
    gaussian = force_eval_plots_dir / f'{plot_prefix}_force_gaussian_distribution.png'
    gaussian_csv = force_eval_plots_dir / f'{plot_prefix}_force_gaussian_distribution_values.csv'

    ForceAnalyzer.plot_force_components(f_pred, f_ref, components)
    ForceAnalyzer.plot_force_distribution(f_pred, f_ref, distribution)
    ForceAnalyzer.plot_force_magnitude(f_pred, f_ref, R_real, magnitude)
    ForceAnalyzer.plot_force_vs_position(f_pred, f_ref, R_real, vs_position)
    ForceAnalyzer.plot_force_gaussian_distribution(f_pred, f_ref, gaussian)
    ForceAnalyzer.save_force_gaussian_data_csv(f_pred, f_ref, gaussian_csv)

    return {
        'force_components_plot_path': str(components),
        'force_distribution_plot_path': str(distribution),
        'force_magnitude_plot_path': str(magnitude),
        'force_vs_position_plot_path': str(vs_position),
        'force_gaussian_plot_path': str(gaussian),
        'force_gaussian_csv_path': str(gaussian_csv),
    }


def _run_basic_force_eval_subprocess(
    config_path: Path,
    params_path: Path,
    force_eval_plots_dir: Path,
    *,
    plot_prefix: str,
    devices_per_run: int,
    n_frames: int,
    seed: int,
) -> Dict[str, Any]:
    import json

    script_path = ANALYSIS_DIR / 'basic_force_eval.py'
    metrics_json_path = force_eval_plots_dir / f'{plot_prefix}_force_eval_metrics.json'
    force_eval_plots_dir.mkdir(parents=True, exist_ok=True)

    python_bin, venv_name, env = _subprocess_env_for_config(config_path)
    print(f'[analyze_suite] basic_force_eval: using {venv_name} for {config_path.name}', flush=True)
    cmd = [
        str(python_bin),
        str(script_path),
        str(config_path),
        str(params_path),
        str(force_eval_plots_dir),
        '--plot-prefix', str(plot_prefix),
        '--devices-per-run', str(devices_per_run),
        '--frames', str(n_frames),
        '--seed', str(seed),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        signal_suffix = f' (signal {-proc.returncode})' if proc.returncode < 0 else ''
        stderr_tail = proc.stderr.strip()[-1500:]
        stdout_tail = proc.stdout.strip()[-1500:]
        details = '\n'.join(part for part in [stderr_tail, stdout_tail] if part)
        raise RuntimeError(
            f'basic_force_eval subprocess failed with exit code {proc.returncode}{signal_suffix}'
            + (f':\n{details}' if details else '')
        )
    if not metrics_json_path.exists():
        raise FileNotFoundError(f'Basic force evaluation completed but did not write {metrics_json_path}')

    with metrics_json_path.open() as f:
        return json.load(f)


def _run_detailed_force_eval_subprocess(
    config_path: Path,
    params_path: Path,
    output_dir: Path,
    *,
    devices_per_run: int,
    shuffle_repeats: int,
    shuffle_seed: int,
    max_val_frames: Optional[int],
    batch_size: Optional[int],
    use_train: bool = False,
) -> Dict[str, Any]:
    import json

    script_path = ANALYSIS_DIR / 'detailed_force_eval.py'
    metrics_json_path = output_dir / 'metrics.json'
    output_dir.mkdir(parents=True, exist_ok=True)

    python_bin, venv_name, env = _subprocess_env_for_config(config_path)
    print(f'[analyze_suite] detailed_force_eval: using {venv_name} for {config_path.name}', flush=True)
    cmd = [
        str(python_bin),
        str(script_path),
        str(config_path),
        str(params_path),
        str(output_dir),
        '--devices-per-run',
        str(devices_per_run),
        '--shuffle-repeats',
        str(shuffle_repeats),
        '--shuffle-seed',
        str(shuffle_seed),
    ]
    if max_val_frames is not None:
        cmd.extend(['--max-val-frames', str(max_val_frames)])
    if batch_size is not None:
        cmd.extend(['--batch-size', str(batch_size)])
    if use_train:
        cmd.append('--use-train')

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        signal_suffix = f' (signal {-proc.returncode})' if proc.returncode < 0 else ''
        stderr_tail = proc.stderr.strip()[-1500:]
        stdout_tail = proc.stdout.strip()[-1500:]
        details = '\n'.join(part for part in [stderr_tail, stdout_tail] if part)
        raise RuntimeError(
            f'detailed_force_eval subprocess failed with exit code {proc.returncode}{signal_suffix}'
            + (f':\n{details}' if details else '')
        )
    if not metrics_json_path.exists():
        raise FileNotFoundError(f'Detailed force evaluation completed but did not write {metrics_json_path}')

    with metrics_json_path.open() as f:
        return json.load(f)


def _run_complete_eval_subprocess(
    config_path: Path,
    params_path: Path,
    output_dir: Path,
    *,
    devices_per_run: int,
    max_val_frames: Optional[int],
    batch_size: Optional[int],
    smoothness_frames: int = 5,
    smoothness_perturbations: int = 20,
    smoothness_sigma: float = 0.01,
    use_train: bool = False,
) -> Dict[str, Any]:
    import json as _json

    script_path = ANALYSIS_DIR / 'complete_eval.py'
    metrics_json_path = output_dir / 'complete_eval_metrics.json'
    output_dir.mkdir(parents=True, exist_ok=True)

    python_bin, venv_name, env = _subprocess_env_for_config(config_path)
    print(f'[analyze_suite] complete_eval: using {venv_name} for {config_path.name}', flush=True)
    cmd = [
        str(python_bin),
        str(script_path),
        str(config_path),
        str(params_path),
        str(output_dir),
        '--devices-per-run', str(devices_per_run),
        '--smoothness-frames', str(smoothness_frames),
        '--smoothness-perturbations', str(smoothness_perturbations),
        '--smoothness-sigma', str(smoothness_sigma),
    ]
    if max_val_frames is not None:
        cmd.extend(['--max-val-frames', str(max_val_frames)])
    if batch_size is not None:
        cmd.extend(['--batch-size', str(batch_size)])
    if use_train:
        cmd.append('--use-train')

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        signal_suffix = f' (signal {-proc.returncode})' if proc.returncode < 0 else ''
        stderr_tail = proc.stderr.strip()[-1500:]
        stdout_tail = proc.stdout.strip()[-1500:]
        details = '\n'.join(part for part in [stderr_tail, stdout_tail] if part)
        raise RuntimeError(
            f'complete_eval subprocess failed with exit code {proc.returncode}{signal_suffix}'
            + (f':\n{details}' if details else '')
        )
    if not metrics_json_path.exists():
        raise FileNotFoundError(f'Complete eval completed but did not write {metrics_json_path}')

    with metrics_json_path.open() as f:
        return _json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize ML training testing-suite runs into a CSV and analysis plots.')
    parser.add_argument('input_dir', type=str, help='Single run directory or a directory containing multiple run directories.')
    parser.add_argument('--output-csv', type=str, default=None, help='Output CSV path. Defaults to <input>_analysis/summary.csv')
    parser.add_argument('--analysis-dir', type=str, default=None, help='Directory for analysis artifacts. Defaults to input-derived <input>_analysis path.')
    parser.add_argument('--force-eval-frames', type=int, default=32, help='Random training frames to use for force evaluation.')
    parser.add_argument('--force-eval-seed', type=int, default=42, help='Seed for the force-eval subset.')
    parser.add_argument('--devices-per-run', type=int, default=4, help='Devices used during training; used for global batch size and optimizer steps/epoch.')
    parser.add_argument('--skip-force-eval', action='store_true', help='Skip post-training force evaluation and leave those CSV fields empty.')
    parser.add_argument('--detailed-force-eval', action='store_true', help='Run the detailed held-out baseline/correlation evaluation suite and save its outputs.')
    parser.add_argument('--detailed-shuffle-repeats', type=int, default=8, help='Number of shuffled-baseline repeats for detailed force evaluation.')
    parser.add_argument('--detailed-max-val-frames', type=int, default=None, help='Optional cap on held-out validation frames for detailed force evaluation.')
    parser.add_argument('--detailed-batch-size', type=int, default=None, help='Frames per vmap chunk for detailed force evaluation. Defaults to all val frames at once.')
    parser.add_argument('--include-incomplete', action='store_true', help='Include runs that did not finish training. Auto-extracts params from the latest checkpoint when needed and accepts suite-level bootstrap slurm logs for active runs.')
    parser.add_argument('--complete-eval', action='store_true', help='Run the complete offline diagnostic suite (7 modules: force magnitude, calibration, strain, environment, bead type, top-k, smoothness).')
    parser.add_argument('--complete-eval-max-val-frames', type=int, default=None, help='Cap on held-out val frames for complete eval.')
    parser.add_argument('--complete-eval-batch-size', type=int, default=None, help='Frames per vmap chunk for complete eval.')
    parser.add_argument('--smoothness-frames', type=int, default=5, help='Frames for smoothness test in complete eval.')
    parser.add_argument('--smoothness-perturbations', type=int, default=20, help='Perturbations per frame for smoothness test.')
    parser.add_argument('--smoothness-sigma', type=float, default=0.01, help='Gaussian noise std for smoothness test.')
    parser.add_argument('--re-only', action='store_true', help='Analyze only relative-entropy training runs.')
    parser.add_argument('--no-re', action='store_true', help='Exclude relative-entropy training runs from analysis.')
    args = parser.parse_args()

    input_path = Path(args.input_dir).resolve()
    run_dirs = _resolve_input_run_dirs(input_path)
    if not run_dirs:
        raise SystemExit(f'No run directories found under: {input_path}')

    if args.re_only:
        run_dirs = [d for d in run_dirs if _is_re_run_dir(d)]
        if not run_dirs:
            raise SystemExit('No relative-entropy runs found.')
    elif args.no_re:
        run_dirs = [d for d in run_dirs if not _is_re_run_dir(d)]
        if not run_dirs:
            raise SystemExit('All runs are relative-entropy runs and --no-re was specified.')

    run_dirs, excluded_runs = _filter_completed_run_dirs(run_dirs, include_incomplete=args.include_incomplete)
    if excluded_runs:
        print('excluded_noncompleted_runs:')
        for run_dir, reason in excluded_runs:
            print(f'  {run_dir.name} | {reason}')
    if not run_dirs:
        raise SystemExit('No completed run directories found after excluding non-completed runs.')

    analysis_root = Path(args.analysis_dir).resolve() if args.analysis_dir else _analysis_root_for_input(input_path)
    tail_plots_dir = analysis_root / 'tail_loss_plots'
    force_eval_plots_dir = analysis_root / 'force_eval_plots'
    detailed_force_eval_root = analysis_root / 'detailed_force_eval'
    complete_eval_root = analysis_root / 'complete_eval'
    analysis_root.mkdir(parents=True, exist_ok=True)
    tail_plots_dir.mkdir(parents=True, exist_ok=True)
    force_eval_plots_dir.mkdir(parents=True, exist_ok=True)
    if args.detailed_force_eval:
        detailed_force_eval_root.mkdir(parents=True, exist_ok=True)
    if args.complete_eval:
        complete_eval_root.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output_csv).resolve() if args.output_csv else analysis_root / 'summary.csv'

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        runtime_config = run_dir / 'config_runtime.yaml'
        input_config = run_dir / 'config_input.yaml'
        runtime_re_config = run_dir / 'config_runtime_re.yaml'
        input_re_config = run_dir / 'config_input_re.yaml'
        runtime_config = runtime_config if runtime_config.exists() else runtime_re_config
        input_config = input_config if input_config.exists() else input_re_config
        config_path = runtime_config if runtime_config.exists() else input_config
        if not config_path.exists():
            continue

        is_re_run = _is_re_run_dir(run_dir)
        config = ConfigManager(str(config_path))
        model_name = f'{config.get_model_context()}_{config.get_model_id()}'
        export_dir = run_dir / 'exports' if not is_re_run else run_dir / 'relative_entropy'
        re_checkpoint_path = _re_checkpoint_path(run_dir, include_incomplete=args.include_incomplete)
        params_path, params_error = _ensure_re_params_path(run_dir, config, include_incomplete=args.include_incomplete) if is_re_run else _ensure_params_path(config, run_dir, include_incomplete=args.include_incomplete)
        checkpoint_path = re_checkpoint_path if is_re_run else _resolve_checkpoint_path(config, run_dir, include_incomplete=args.include_incomplete)
        log_path = _latest_file(run_dir / 'relative_entropy' if is_re_run else run_dir, 'relative_entropy_loss.log' if is_re_run else 'train_*.log')
        slurm_path = _resolve_slurm_path(run_dir, include_incomplete=args.include_incomplete)
        run_date, run_name = _parse_run_dir_name(run_dir.name)
        plot_prefix = run_dir.name
        tail_plot_path = tail_plots_dir / f'{plot_prefix}_tail_loss_linear_fit.png'

        row: Dict[str, Any] = {
            'run_dir_name': run_dir.name,
            'run_date': run_date,
            'run_name': run_name,
            'config_name': input_config.stem if input_config.exists() else config_path.stem,
            'config_path': str(config_path),
            'input_config_path': str(input_config) if input_config.exists() else '',
            'model_context': config.get_model_context(),
            'model_id': config.get_model_id(),
            'status': 'ok',
            'is_re_run': int(is_re_run),
            'job_id': '',
            'run_dir': str(run_dir),
            'analysis_dir': str(analysis_root),
            'summary_csv_path': str(output_csv),
            'tail_plots_dir': str(tail_plots_dir),
            'tail_fit_plot_path': '',
            'tail_fit_plot_error': '',
            'force_eval_plots_dir': str(force_eval_plots_dir),
            'force_components_plot_path': '',
            'force_distribution_plot_path': '',
            'force_magnitude_plot_path': '',
            'force_vs_position_plot_path': '',
            'force_gaussian_plot_path': '',
            'force_gaussian_csv_path': '',
            'force_eval_per_frame_csv_path': '',
            'force_eval_worst_frames_json_path': '',
            'force_eval_plot_error': '',
            'export_dir': str(export_dir),
            'log_path': str(log_path) if log_path is not None else '',
            'slurm_path': str(slurm_path) if slurm_path is not None else '',
            'params_path': str(params_path),
            'checkpoint_path': str(checkpoint_path) if checkpoint_path is not None else '',
            'force_eval_error': '',
            'detailed_force_eval_dir': '',
            'detailed_metrics_json_path': '',
            'detailed_metrics_csv_path': '',
            'detailed_shuffle_csv_path': '',
            'detailed_baseline_plot_path': '',
            'detailed_shuffle_plot_path': '',
            'detailed_cosine_plot_path': '',
            'detailed_eval_error': '',
            'detailed_split_name': '',
            'detailed_n_train_samples': float('nan'),
            'detailed_n_eval_samples': float('nan'),
            'detailed_rmse_model': float('nan'),
            'detailed_rmse_zero': float('nan'),
            'detailed_rmse_mean': float('nan'),
            'detailed_rmse_shuffle_mean': float('nan'),
            'detailed_rmse_shuffle_std': float('nan'),
            'detailed_shuffle_gap_rmse': float('nan'),
            'detailed_pearson_global': float('nan'),
            'detailed_mean_cosine_similarity': float('nan'),
            'detailed_r2_explained_variance': float('nan'),
            'detailed_variance_ratio_pred_to_ref': float('nan'),
            'detailed_validation_per_frame_csv_path': '',
            'detailed_validation_worst_frames_json_path': '',
            'detailed_noised_rmse_model': float('nan'),
            'detailed_noised_rmse_zero': float('nan'),
            'detailed_noised_mean_cosine_similarity': float('nan'),
            'detailed_noised_n_eval_samples': float('nan'),
            'detailed_noised_per_frame_csv_path': '',
            'detailed_noised_worst_frames_json_path': '',
            'detailed_noised_eval_error': '',
            'tail_loss_intercept_last_third': float('nan'),
            'complete_eval_dir': '',
            'complete_eval_metrics_json_path': '',
            'complete_eval_arrays_npz_path': '',
            'complete_eval_error': '',
            'complete_eval_n_val_frames': float('nan'),
            'complete_eval_n_valid_beads': float('nan'),
            'complete_eval_calibration_slope': float('nan'),
            'complete_eval_calibration_intercept': float('nan'),
            'complete_eval_calibration_pearson': float('nan'),
            'complete_eval_smoothness_mean': float('nan'),
            'complete_eval_smoothness_p95': float('nan'),
            'complete_eval_smoothness_max': float('nan'),
        }

        row.update(_estimate_optimizer_steps_per_epoch(config, devices_per_run=int(args.devices_per_run)))

        if params_error and row['status'] == 'ok':
            row['status'] = 'missing_params'
            row['force_eval_error'] = params_error
            row['force_eval_plot_error'] = params_error

        train_losses: List[float] = []
        val_losses: List[float] = []

        if is_re_run:
            re_history = _load_re_training_history(run_dir)
            if re_history:
                objective_vals = []
                for row_h in re_history:
                    try:
                        objective_vals.append(float(row_h.get('objective', float('nan'))))
                    except (ValueError, TypeError):
                        objective_vals.append(float('nan'))
                train_losses = objective_vals
                row['n_logged_epochs'] = len(re_history)
                row['final_train_loss'] = objective_vals[-1] if objective_vals else float('nan')
                row['initial_train_loss'] = objective_vals[0] if objective_vals else float('nan')
                if len(objective_vals) >= 3:
                    tail_start = len(objective_vals) // 3
                    tail_vals = [v for v in objective_vals[tail_start:] if np.isfinite(v)]
                    if tail_vals:
                        row['tail_loss_slope_last_third'] = float(np.polyfit(range(len(tail_vals)), tail_vals, 1)[0])
                    else:
                        row['tail_loss_slope_last_third'] = float('nan')
                else:
                    row['tail_loss_slope_last_third'] = float('nan')
                row['tail_loss_intercept_last_third'] = float('nan')
                re_plot_error = _save_re_loss_plot(re_history, tail_plot_path, title=f'{run_dir.name}: RE training loss')
                if re_plot_error:
                    row['tail_fit_plot_error'] = re_plot_error
                else:
                    row['tail_fit_plot_path'] = str(tail_plot_path)
            else:
                row['status'] = 'missing_re_history'
                row['tail_fit_plot_error'] = 'RE history CSV not found'
                row['initial_train_loss'] = float('nan')
                row['final_train_loss'] = float('nan')
                row['tail_loss_slope_last_third'] = float('nan')
                row['tail_loss_intercept_last_third'] = float('nan')
                row['n_logged_epochs'] = 0
                row['final_val_loss'] = float('nan')
        else:
            if log_path is not None and log_path.exists():
                train_losses, val_losses = _load_loss_history(log_path)

            if train_losses:
                tail_fit = _fit_tail_line(train_losses)
                row['initial_train_loss'] = _safe_float(train_losses[0])
                row['final_train_loss'] = _safe_float(train_losses[-1])
                row['tail_loss_slope_last_third'] = _tail_slope(train_losses)
                row['tail_loss_intercept_last_third'] = float(tail_fit['intercept']) if tail_fit is not None else float('nan')
                row['n_logged_epochs'] = len(train_losses)
                row['final_val_loss'] = _safe_float(val_losses[-1]) if val_losses else float('nan')
                tail_plot_error = _save_tail_fit_plot(train_losses, val_losses, tail_plot_path, title=f'{run_dir.name}: tail loss linear fit')
                if tail_plot_error:
                    row['tail_fit_plot_error'] = tail_plot_error
                else:
                    row['tail_fit_plot_path'] = str(tail_plot_path)
            else:
                row['initial_train_loss'] = float('nan')
                row['final_train_loss'] = float('nan')
                row['tail_loss_slope_last_third'] = float('nan')
                row['n_logged_epochs'] = 0
                row['final_val_loss'] = float('nan')
                row['status'] = 'missing_log'
                row['tail_fit_plot_error'] = 'Training log missing or no parseable training-loss history found.'

            if checkpoint_path is not None and checkpoint_path.exists():
                metadata = _load_checkpoint_metadata(checkpoint_path)
                row['job_id'] = metadata.get('job_id', '')
                row['epoch_wall_seconds'] = _mean_epoch_time_from_results(metadata.get('results', {}))
                if not train_losses:
                    for result in metadata.get('results', {}).values():
                        if isinstance(result, dict) and 'train_loss' in result:
                            row['final_train_loss'] = _safe_float(result.get('train_loss'))
                        if isinstance(result, dict) and 'val_loss' in result:
                            row['final_val_loss'] = _safe_float(result.get('val_loss'))
            else:
                row['epoch_wall_seconds'] = float('nan')
                if row['status'] == 'ok':
                    row['status'] = 'missing_checkpoint'

        if is_re_run:
            row['epoch_wall_seconds'] = float('nan')

        if not args.skip_force_eval and params_path.exists():
            try:
                if is_re_run:
                    force_eval_data = _collect_re_force_eval_data(
                        config_path=config_path,
                        params_path=params_path,
                        run_dir=run_dir,
                        n_frames=int(args.force_eval_frames),
                        seed=int(args.force_eval_seed),
                        devices_per_run=int(args.devices_per_run),
                    )
                    plot_paths = _save_force_eval_plots(force_eval_data, plot_prefix, force_eval_plots_dir)

                    import json as _json
                    per_frame_rows = force_eval_data.get('per_frame_errors', [])
                    for rank, r in enumerate(per_frame_rows, start=1):
                        r['rank_by_rmse'] = rank
                    bad_csv_path = force_eval_plots_dir / f'{plot_prefix}_force_eval_per_frame_errors.csv'
                    bad_json_path = force_eval_plots_dir / f'{plot_prefix}_force_eval_worst_frames_top50.json'
                    if per_frame_rows:
                        with bad_csv_path.open('w', newline='') as f:
                            fieldnames = ['rank_by_rmse'] + [k for k in per_frame_rows[0].keys() if k != 'rank_by_rmse']
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(per_frame_rows)
                    else:
                        bad_csv_path.write_text('')
                    bad_json_path.write_text(_json.dumps(per_frame_rows[:50], indent=2, sort_keys=True))

                    force_eval_metrics = {
                        'force_eval_frames': force_eval_data['force_eval_frames'],
                        'force_rmse': force_eval_data['force_rmse'],
                        'force_mae': force_eval_data['force_mae'],
                        'force_error_magnitude_mean': force_eval_data['force_error_magnitude_mean'],
                        'force_error_magnitude_std': force_eval_data['force_error_magnitude_std'],
                        'force_magnitude_diff_mean': force_eval_data['force_magnitude_diff_mean'],
                        'force_magnitude_diff_std': force_eval_data['force_magnitude_diff_std'],
                        'force_magnitude_abs_diff_mean': force_eval_data['force_magnitude_abs_diff_mean'],
                        'force_eval_per_frame_csv_path': str(bad_csv_path),
                        'force_eval_worst_frames_json_path': str(bad_json_path),
                    }
                    force_eval_metrics.update(plot_paths)

                    for metric_key in (
                        'force_eval_frames',
                        'force_rmse',
                        'force_mae',
                        'force_error_magnitude_mean',
                        'force_error_magnitude_std',
                        'force_magnitude_diff_mean',
                        'force_magnitude_diff_std',
                        'force_magnitude_abs_diff_mean',
                    ):
                        row[metric_key] = force_eval_metrics[metric_key]
                    row.update({
                        'force_components_plot_path': force_eval_metrics.get('force_components_plot_path', ''),
                        'force_distribution_plot_path': force_eval_metrics.get('force_distribution_plot_path', ''),
                        'force_magnitude_plot_path': force_eval_metrics.get('force_magnitude_plot_path', ''),
                        'force_vs_position_plot_path': force_eval_metrics.get('force_vs_position_plot_path', ''),
                        'force_gaussian_plot_path': force_eval_metrics.get('force_gaussian_plot_path', ''),
                        'force_gaussian_csv_path': force_eval_metrics.get('force_gaussian_csv_path', ''),
                        'force_eval_per_frame_csv_path': force_eval_metrics.get('force_eval_per_frame_csv_path', ''),
                        'force_eval_worst_frames_json_path': force_eval_metrics.get('force_eval_worst_frames_json_path', ''),
                    })
                else:
                    force_eval_metrics = _run_basic_force_eval_subprocess(
                        config_path=config_path,
                        params_path=params_path,
                        force_eval_plots_dir=force_eval_plots_dir,
                        plot_prefix=plot_prefix,
                        devices_per_run=int(args.devices_per_run),
                        n_frames=int(args.force_eval_frames),
                        seed=int(args.force_eval_seed),
                    )
                    for metric_key in (
                        'force_eval_frames',
                        'force_rmse',
                        'force_mae',
                        'force_error_magnitude_mean',
                        'force_error_magnitude_std',
                        'force_magnitude_diff_mean',
                        'force_magnitude_diff_std',
                        'force_magnitude_abs_diff_mean',
                    ):
                        row[metric_key] = force_eval_metrics[metric_key]
                    row.update({
                        'force_components_plot_path': force_eval_metrics['force_components_plot_path'],
                        'force_distribution_plot_path': force_eval_metrics['force_distribution_plot_path'],
                        'force_magnitude_plot_path': force_eval_metrics['force_magnitude_plot_path'],
                        'force_vs_position_plot_path': force_eval_metrics['force_vs_position_plot_path'],
                        'force_gaussian_plot_path': force_eval_metrics['force_gaussian_plot_path'],
                        'force_gaussian_csv_path': force_eval_metrics['force_gaussian_csv_path'],
                        'force_eval_per_frame_csv_path': force_eval_metrics.get('force_eval_per_frame_csv_path', ''),
                        'force_eval_worst_frames_json_path': force_eval_metrics.get('force_eval_worst_frames_json_path', ''),
                    })
            except Exception as exc:
                for key in (
                    'force_eval_frames',
                    'force_rmse',
                    'force_mae',
                    'force_error_magnitude_mean',
                    'force_error_magnitude_std',
                    'force_magnitude_diff_mean',
                    'force_magnitude_diff_std',
                    'force_magnitude_abs_diff_mean',
                ):
                    row[key] = float('nan')
                row['force_eval_error'] = str(exc)
                row['force_eval_plot_error'] = str(exc)
                if row['status'] == 'ok':
                    row['status'] = 'force_eval_error'
        else:
            for key in (
                'force_eval_frames',
                'force_rmse',
                'force_mae',
                'force_error_magnitude_mean',
                'force_error_magnitude_std',
                'force_magnitude_diff_mean',
                'force_magnitude_diff_std',
                'force_magnitude_abs_diff_mean',
            ):
                row[key] = float('nan')
            if not args.skip_force_eval and not params_path.exists() and row['status'] == 'ok':
                row['status'] = 'missing_params'

        if args.detailed_force_eval and params_path.exists():
            try:
                detailed_output_dir = detailed_force_eval_root / plot_prefix
                row['detailed_force_eval_dir'] = str(detailed_output_dir)
                detailed_metrics = _run_detailed_force_eval_subprocess(
                    config_path=config_path,
                    params_path=params_path,
                    output_dir=detailed_output_dir,
                    devices_per_run=int(args.devices_per_run),
                    shuffle_repeats=int(args.detailed_shuffle_repeats),
                    shuffle_seed=int(args.force_eval_seed),
                    max_val_frames=args.detailed_max_val_frames,
                    batch_size=args.detailed_batch_size,
                    use_train=bool(is_re_run),
                )
                row['detailed_metrics_json_path'] = str(detailed_metrics.get('metrics_json_path', ''))
                row['detailed_metrics_csv_path'] = str(detailed_metrics.get('metrics_csv_path', ''))
                row['detailed_shuffle_csv_path'] = str(detailed_metrics.get('shuffle_csv_path', ''))
                row['detailed_baseline_plot_path'] = str(detailed_metrics.get('baseline_plot_path', ''))
                row['detailed_shuffle_plot_path'] = str(detailed_metrics.get('shuffle_plot_path', ''))
                row['detailed_cosine_plot_path'] = str(detailed_metrics.get('cosine_plot_path', ''))
                row['detailed_split_name'] = str(detailed_metrics.get('split_name', ''))
                row['detailed_n_train_samples'] = float(detailed_metrics.get('n_train_samples', float('nan')))
                row['detailed_n_eval_samples'] = float(detailed_metrics.get('n_eval_samples', float('nan')))
                row['detailed_rmse_model'] = float(detailed_metrics.get('rmse_model', float('nan')))
                row['detailed_rmse_zero'] = float(detailed_metrics.get('rmse_zero', float('nan')))
                row['detailed_rmse_mean'] = float(detailed_metrics.get('rmse_mean', float('nan')))
                row['detailed_rmse_shuffle_mean'] = float(detailed_metrics.get('rmse_shuffle_mean', float('nan')))
                row['detailed_rmse_shuffle_std'] = float(detailed_metrics.get('rmse_shuffle_std', float('nan')))
                row['detailed_shuffle_gap_rmse'] = float(detailed_metrics.get('shuffle_gap_rmse', float('nan')))
                row['detailed_pearson_global'] = float(detailed_metrics.get('pearson_global', float('nan')))
                row['detailed_mean_cosine_similarity'] = float(detailed_metrics.get('mean_cosine_similarity', float('nan')))
                row['detailed_r2_explained_variance'] = float(detailed_metrics.get('r2_explained_variance', float('nan')))
                row['detailed_variance_ratio_pred_to_ref'] = float(detailed_metrics.get('variance_ratio_pred_to_ref', float('nan')))
                row['detailed_validation_per_frame_csv_path'] = str(detailed_metrics.get('validation_per_frame_csv_path', ''))
                row['detailed_validation_worst_frames_json_path'] = str(detailed_metrics.get('validation_worst_frames_json_path', ''))
                row['detailed_noised_rmse_model'] = float(detailed_metrics.get('noised_rmse_model', float('nan')))
                row['detailed_noised_rmse_zero'] = float(detailed_metrics.get('noised_rmse_zero', float('nan')))
                row['detailed_noised_mean_cosine_similarity'] = float(detailed_metrics.get('noised_mean_cosine_similarity', float('nan')))
                row['detailed_noised_n_eval_samples'] = float(detailed_metrics.get('noised_n_eval_samples', float('nan')))
                row['detailed_noised_per_frame_csv_path'] = str(detailed_metrics.get('noised_per_frame_csv_path', ''))
                row['detailed_noised_worst_frames_json_path'] = str(detailed_metrics.get('noised_worst_frames_json_path', ''))
                row['detailed_noised_eval_error'] = str(detailed_metrics.get('noised_eval_error', ''))
            except Exception as exc:
                row['detailed_eval_error'] = str(exc)

        if args.complete_eval and params_path.exists():
            try:
                ce_output_dir = complete_eval_root / plot_prefix
                row['complete_eval_dir'] = str(ce_output_dir)
                ce_metrics = _run_complete_eval_subprocess(
                    config_path=config_path,
                    params_path=params_path,
                    output_dir=ce_output_dir,
                    devices_per_run=int(args.devices_per_run),
                    max_val_frames=args.complete_eval_max_val_frames,
                    batch_size=args.complete_eval_batch_size,
                    smoothness_frames=int(args.smoothness_frames),
                    smoothness_perturbations=int(args.smoothness_perturbations),
                    smoothness_sigma=float(args.smoothness_sigma),
                    use_train=bool(is_re_run),
                )
                row['complete_eval_metrics_json_path'] = str(ce_metrics.get('metrics_json_path', ''))
                row['complete_eval_arrays_npz_path'] = str(ce_metrics.get('arrays_npz_path', ''))
                row['complete_eval_n_val_frames'] = float(ce_metrics.get('n_val_frames', float('nan')))
                row['complete_eval_n_valid_beads'] = float(ce_metrics.get('n_valid_beads', float('nan')))
                row['complete_eval_calibration_slope'] = float(ce_metrics.get('calibration_slope', float('nan')))
                row['complete_eval_calibration_intercept'] = float(ce_metrics.get('calibration_intercept', float('nan')))
                row['complete_eval_calibration_pearson'] = float(ce_metrics.get('calibration_pearson', float('nan')))
                row['complete_eval_smoothness_mean'] = float(ce_metrics.get('smoothness_mean_sensitivity', float('nan')))
                row['complete_eval_smoothness_p95'] = float(ce_metrics.get('smoothness_p95_sensitivity', float('nan')))
                row['complete_eval_smoothness_max'] = float(ce_metrics.get('smoothness_max_sensitivity', float('nan')))
            except Exception as exc:
                row['complete_eval_error'] = str(exc)

        rows.append(row)

    _write_csv(rows, output_csv)
    print(f'wrote_csv:            {output_csv}')
    print(f'n_runs:               {len(rows)}')
    print(f'analysis_dir:         {analysis_root}')
    print(f'tail_plots_dir:       {tail_plots_dir}')
    print(f'force_eval_plots_dir: {force_eval_plots_dir}')


if __name__ == '__main__':
    main()
