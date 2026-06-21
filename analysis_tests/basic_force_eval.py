#!/usr/bin/env python3
"""Run the lightweight suite force eval in the correct per-model environment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_tests.analyze_suite import _collect_force_eval_data, _save_force_eval_plots


def main() -> None:
    parser = argparse.ArgumentParser(description='Run basic force evaluation and write metrics/plots.')
    parser.add_argument('config_path', type=str)
    parser.add_argument('params_path', type=str)
    parser.add_argument('force_eval_plots_dir', type=str)
    parser.add_argument('--plot-prefix', type=str, required=True)
    parser.add_argument('--devices-per-run', type=int, required=True)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    params_path = Path(args.params_path).resolve()
    force_eval_plots_dir = Path(args.force_eval_plots_dir).resolve()

    force_eval_data = _collect_force_eval_data(
        config_path=config_path,
        params_path=params_path,
        n_frames=int(args.frames),
        seed=int(args.seed),
        devices_per_run=int(args.devices_per_run),
    )
    plot_paths = _save_force_eval_plots(
        force_eval_data,
        plot_prefix=str(args.plot_prefix),
        force_eval_plots_dir=force_eval_plots_dir,
    )

    per_frame_rows = force_eval_data.get('per_frame_errors', [])
    for rank, row in enumerate(per_frame_rows, start=1):
        row['rank_by_rmse'] = rank
    bad_csv_path = force_eval_plots_dir / f'{args.plot_prefix}_force_eval_per_frame_errors.csv'
    bad_json_path = force_eval_plots_dir / f'{args.plot_prefix}_force_eval_worst_frames_top50.json'
    if per_frame_rows:
        with bad_csv_path.open('w', newline='') as f:
            fieldnames = ['rank_by_rmse'] + [k for k in per_frame_rows[0].keys() if k != 'rank_by_rmse']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_frame_rows)
    else:
        bad_csv_path.write_text('')
    bad_json_path.write_text(json.dumps(per_frame_rows[:50], indent=2, sort_keys=True))

    metrics = {
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
    metrics.update(plot_paths)

    metrics_path = force_eval_plots_dir / f'{args.plot_prefix}_force_eval_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open('w') as f:
        json.dump(metrics, f, indent=2)

    print(metrics_path)


if __name__ == '__main__':
    main()
