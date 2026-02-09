#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class AmdahlFit:
    p: float
    s: float


@dataclass
class PowerLawFit:
    alpha: float
    a: float
    b: float


@dataclass
class KarpFlattFit:
    c0: float
    c1: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--devices-col", default="GPUs")
    ap.add_argument("--time-col", default="Steady-State Time B=1 (min)")
    ap.add_argument("--ref-devices", type=int, default=1)
    ap.add_argument("--benchmark-frames", type=float, default=10000.0)
    ap.add_argument("--full-frames", type=float, default=13495000.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--predict", type=int, nargs="*", default=[100, 200, 500, 1000, 1500, 2000])
    ap.add_argument("--use-rows", default="all")
    ap.add_argument("--title", default="Scaling Extrapolation")
    ap.add_argument("--out-prefix", default="scaling")
    ap.add_argument("--show", action="store_true")
    return ap.parse_args()


def choose_rows(df: pd.DataFrame, use_rows: str, devices_col: str) -> pd.DataFrame:
    if use_rows.strip().lower() == "all":
        return df.copy()
    wanted = [int(x.strip()) for x in use_rows.split(",") if x.strip()]
    return df[df[devices_col].isin(wanted)].copy()


def compute_measured_metrics(df: pd.DataFrame, devices_col: str, time_col: str, ref_devices: int) -> pd.DataFrame:
    df = df.sort_values(devices_col).reset_index(drop=True)
    if ref_devices not in set(df[devices_col].tolist()):
        raise ValueError(f"ref-devices={ref_devices} not found in CSV.")
    t_ref = float(df.loc[df[devices_col] == ref_devices, time_col].iloc[0])
    df["speedup"] = t_ref / df[time_col]
    df["ideal_speedup"] = df[devices_col] / ref_devices
    df["efficiency"] = df["speedup"] / df["ideal_speedup"]
    return df


def minutes_to_hms(minutes: float) -> str:
    total_seconds = int(round(minutes * 60))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fit_amdahl(devices: np.ndarray, speedup: np.ndarray) -> AmdahlFit:
    ps = np.linspace(0.80, 0.99995, 8000)
    Ns = devices.astype(float)
    best_p = None
    best_err = np.inf
    for p in ps:
        s = 1.0 - p
        pred = 1.0 / (s + p / Ns)
        err = np.mean((pred - speedup) ** 2)
        if err < best_err:
            best_err = err
            best_p = p
    if best_p is None:
        raise RuntimeError("Failed to fit Amdahl p.")
    p = float(best_p)
    return AmdahlFit(p=p, s=1.0 - p)


def amdahl_speedup(N: np.ndarray, p: float) -> np.ndarray:
    s = 1.0 - p
    N = N.astype(float)
    return 1.0 / (s + p / N)


def gustafson_speedup(N: np.ndarray, s: float) -> np.ndarray:
    N = N.astype(float)
    return N - s * (N - 1.0)


def fit_linear_speedup(devices: np.ndarray, speedup: np.ndarray) -> Tuple[float, float]:
    x = devices.astype(float)
    y = speedup.astype(float)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(b)


def linear_speedup(N: np.ndarray, m: float, b: float, eps: float = 1e-9) -> np.ndarray:
    s = m * N.astype(float) + b
    return np.maximum(s, eps)


def fit_powerlaw_time_nonneg_b(devices: np.ndarray, times: np.ndarray) -> PowerLawFit:
    xN = devices.astype(float)
    yT = times.astype(float)
    alphas = np.linspace(0.1, 1.6, 3000)
    best = None
    best_err = np.inf
    for alpha in alphas:
        x1 = xN ** (-alpha)
        A = np.vstack([x1, np.ones_like(x1)]).T
        a, b = np.linalg.lstsq(A, yT, rcond=None)[0]
        if b < 0:
            continue
        pred = a * x1 + b
        err = np.mean((pred - yT) ** 2)
        if err < best_err:
            best_err = err
            best = (alpha, a, b)
    if best is None:
        alpha = 1.0
        x1 = xN ** (-alpha)
        A = np.vstack([x1, np.ones_like(x1)]).T
        a, b = np.linalg.lstsq(A, yT, rcond=None)[0]
        b = max(float(b), 0.0)
        return PowerLawFit(alpha=float(alpha), a=float(a), b=float(b))
    alpha, a, b = best
    return PowerLawFit(alpha=float(alpha), a=float(a), b=float(b))


def powerlaw_time(N: np.ndarray, fit: PowerLawFit, eps: float = 1e-9) -> np.ndarray:
    Nf = N.astype(float)
    pred = fit.a * (Nf ** (-fit.alpha)) + fit.b
    return np.maximum(pred, eps)


def karp_flatt_epsilon(N: np.ndarray, S: np.ndarray) -> np.ndarray:
    N = N.astype(float)
    S = S.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = (1.0 / S - 1.0 / N) / (1.0 - 1.0 / N)
    return eps


def fit_karp_flatt(devices: np.ndarray, speedup: np.ndarray) -> KarpFlattFit:
    N = devices.astype(float)
    S = speedup.astype(float)
    mask = (N > 1) & np.isfinite(S) & (S > 0)
    N = N[mask]
    S = S[mask]
    eps = karp_flatt_epsilon(N, S)
    eps = np.clip(eps, 0.0, 0.999)
    x = np.log(N)
    A = np.vstack([np.ones_like(x), x]).T
    c0, c1 = np.linalg.lstsq(A, eps, rcond=None)[0]
    return KarpFlattFit(c0=float(c0), c1=float(c1))


def karp_flatt_speedup(N: np.ndarray, fit: KarpFlattFit) -> np.ndarray:
    Nf = N.astype(float)
    eps = fit.c0 + fit.c1 * np.log(np.maximum(Nf, 1.0))
    eps = np.clip(eps, 0.0, 0.999)
    return 1.0 / (eps + (1.0 - eps) / Nf)


def relative_speedup(model_speedup: np.ndarray, ref_speedup_scalar: float) -> np.ndarray:
    return model_speedup / float(ref_speedup_scalar)


def gpu_hours(minutes_full_epoch: np.ndarray, devices: np.ndarray, epochs: int) -> np.ndarray:
    return (minutes_full_epoch / 60.0) * devices.astype(float) * float(epochs)


def band_low_high(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return lo, hi


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)
    if args.devices_col not in df.columns:
        raise ValueError(f"Column '{args.devices_col}' not found in CSV columns: {list(df.columns)}")
    if args.time_col not in df.columns:
        raise ValueError(f"Column '{args.time_col}' not found in CSV columns: {list(df.columns)}")

    df = df[[args.devices_col, args.time_col]].dropna().copy()
    df[args.devices_col] = df[args.devices_col].astype(int)
    df[args.time_col] = df[args.time_col].astype(float)

    df_metrics = compute_measured_metrics(df, args.devices_col, args.time_col, args.ref_devices)
    df_fit = choose_rows(df_metrics, args.use_rows, args.devices_col)

    scale_factor = float(args.full_frames) / float(args.benchmark_frames)
    t_ref_bench = float(df_metrics.loc[df_metrics[args.devices_col] == args.ref_devices, args.time_col].iloc[0])

    pred_N = np.array(sorted(set(args.predict)), dtype=int)

    amd = fit_amdahl(df_fit[args.devices_col].to_numpy(), df_fit["speedup"].to_numpy())
    S_ref_amd = amdahl_speedup(np.array([args.ref_devices]), amd.p)[0]
    S_pred_amd = amdahl_speedup(pred_N, amd.p)
    sp_rel_amd = relative_speedup(S_pred_amd, S_ref_amd)
    t_bench_amd = t_ref_bench / sp_rel_amd

    S_ref_gus = gustafson_speedup(np.array([args.ref_devices]), amd.s)[0]
    S_pred_gus = gustafson_speedup(pred_N, amd.s)
    sp_rel_gus = relative_speedup(S_pred_gus, S_ref_gus)
    t_bench_gus = t_ref_bench / sp_rel_gus

    m_lin, b_lin = fit_linear_speedup(df_fit[args.devices_col].to_numpy(), df_fit["speedup"].to_numpy())
    S_ref_lin = linear_speedup(np.array([args.ref_devices]), m_lin, b_lin)[0]
    S_pred_lin = linear_speedup(pred_N, m_lin, b_lin)
    sp_rel_lin = relative_speedup(S_pred_lin, S_ref_lin)
    t_bench_lin = t_ref_bench / sp_rel_lin

    pl = fit_powerlaw_time_nonneg_b(df_fit[args.devices_col].to_numpy(), df_fit[args.time_col].to_numpy())
    t_bench_pl = powerlaw_time(pred_N, pl)
    sp_rel_pl = t_ref_bench / t_bench_pl

    kf = fit_karp_flatt(df_fit[args.devices_col].to_numpy(), df_fit["speedup"].to_numpy())
    S_ref_kf = karp_flatt_speedup(np.array([args.ref_devices]), kf)[0]
    S_pred_kf = karp_flatt_speedup(pred_N, kf)
    sp_rel_kf = relative_speedup(S_pred_kf, S_ref_kf)
    t_bench_kf = t_ref_bench / sp_rel_kf

    ideal_speedup_rel = (pred_N / args.ref_devices).astype(float)
    t_bench_ideal = t_ref_bench / ideal_speedup_rel

    t_full_amd = t_bench_amd * scale_factor
    t_full_gus = t_bench_gus * scale_factor
    t_full_pl = t_bench_pl * scale_factor
    t_full_kf = t_bench_kf * scale_factor
    t_full_lin = t_bench_lin * scale_factor
    t_full_ideal = t_bench_ideal * scale_factor

    weights = {"KarpFlatt": 0.40, "Amdahl": 0.30, "PowerLaw": 0.20, "Gustafson": 0.10}
    wsum = sum(weights.values())
    weights = {k: v / wsum for k, v in weights.items()}

    models4_full = np.vstack([t_full_amd, t_full_gus, t_full_pl, t_full_kf])
    models4_bench = np.vstack([t_bench_amd, t_bench_gus, t_bench_pl, t_bench_kf])

    mean4_full = np.mean(models4_full, axis=0)
    mean4_bench = np.mean(models4_bench, axis=0)
    mean4_speedup_rel = t_ref_bench / mean4_bench

    wmean4_full = (
        weights["Amdahl"] * t_full_amd
        + weights["Gustafson"] * t_full_gus
        + weights["PowerLaw"] * t_full_pl
        + weights["KarpFlatt"] * t_full_kf
    )
    wmean4_bench = (
        weights["Amdahl"] * t_bench_amd
        + weights["Gustafson"] * t_bench_gus
        + weights["PowerLaw"] * t_bench_pl
        + weights["KarpFlatt"] * t_bench_kf
    )
    wmean4_speedup_rel = t_ref_bench / wmean4_bench

    gh_amd = gpu_hours(t_full_amd, pred_N, args.epochs)
    gh_gus = gpu_hours(t_full_gus, pred_N, args.epochs)
    gh_pl = gpu_hours(t_full_pl, pred_N, args.epochs)
    gh_kf = gpu_hours(t_full_kf, pred_N, args.epochs)
    gh_lin = gpu_hours(t_full_lin, pred_N, args.epochs)
    gh_mean4 = gpu_hours(mean4_full, pred_N, args.epochs)
    gh_wmean4 = gpu_hours(wmean4_full, pred_N, args.epochs)
    gh_ideal = gpu_hours(t_full_ideal, pred_N, args.epochs)

    print("\n=== Input summary ===")
    print(f"CSV: {args.csv}")
    print(f"Reference: {args.ref_devices} device(s), T_ref(benchmark) = {t_ref_bench:.3f} min")
    print(f"Benchmark frames = {args.benchmark_frames:g}")
    print(f"Full frames      = {args.full_frames:g}")
    print(f"Scale factor     = {scale_factor:.6f}")
    print(f"Epochs           = {args.epochs}")

    print("\n=== Fits ===")
    print(f"Amdahl: p = {amd.p:.6f}, serial = {amd.s:.6f}, max speedup ~ {1.0/amd.s:.2f}x")
    print(f"Gustafson: uses serial = {amd.s:.6f} from Amdahl")
    print(f"Power-law time (b>=0): T(N) = a*N^(-alpha) + b, alpha = {pl.alpha:.6f}, a = {pl.a:.6f}, b = {pl.b:.6f}")
    print(f"Karp–Flatt epsilon fit: eps(N) = c0 + c1*log(N), c0 = {kf.c0:.6f}, c1 = {kf.c1:.6f}")
    print(f"Linear speedup: S(N) = m*N + b, m = {m_lin:.6f}, b = {b_lin:.6f}")
    print(f"Weighted mean (4-model): {weights}")

    print("\n=== Predictions (full epoch) ===")
    print("Devices |  Amdahl time | Gustafson time | PowerLaw time | KarpFlatt time | Linear-fit time | Mean(4) time | WMean(4) time | Ideal time")
    print("------------------------------------------------------------------------------------------------------------------------------------------")
    for i, N in enumerate(pred_N):
        print(
            f"{N:7d} |"
            f" {minutes_to_hms(float(t_full_amd[i])):>11} |"
            f" {minutes_to_hms(float(t_full_gus[i])):>13} |"
            f" {minutes_to_hms(float(t_full_pl[i])):>12} |"
            f" {minutes_to_hms(float(t_full_kf[i])):>13} |"
            f" {minutes_to_hms(float(t_full_lin[i])):>14} |"
            f" {minutes_to_hms(float(mean4_full[i])):>11} |"
            f" {minutes_to_hms(float(wmean4_full[i])):>12} |"
            f" {minutes_to_hms(float(t_full_ideal[i])):>9}"
        )

    print(f"\n=== GPU-hours for {args.epochs} epochs (full dataset) ===")
    print("Devices |   Amdahl GH | Gustafson GH |  PowerLaw GH | KarpFlatt GH | Linear-fit GH |  Mean(4) GH | WMean(4) GH |   Ideal GH")
    print("----------------------------------------------------------------------------------------------------------------------------------")
    for i, N in enumerate(pred_N):
        print(
            f"{N:7d} |"
            f" {gh_amd[i]:11,.0f} |"
            f" {gh_gus[i]:12,.0f} |"
            f" {gh_pl[i]:11,.0f} |"
            f" {gh_kf[i]:11,.0f} |"
            f" {gh_lin[i]:12,.0f} |"
            f" {gh_mean4[i]:11,.0f} |"
            f" {gh_wmean4[i]:12,.0f} |"
            f" {gh_ideal[i]:10,.0f}"
        )

    Ns_meas = df_metrics[args.devices_col].to_numpy()
    sp_meas = df_metrics["speedup"].to_numpy()

    Ns_dense = np.unique(np.concatenate([Ns_meas, pred_N]))
    Ns_dense = np.array(sorted(Ns_dense), dtype=int)

    sp_dense_amd_rel = amdahl_speedup(Ns_dense, amd.p) / amdahl_speedup(np.array([args.ref_devices]), amd.p)[0]
    sp_dense_gus_rel = gustafson_speedup(Ns_dense, amd.s) / gustafson_speedup(np.array([args.ref_devices]), amd.s)[0]
    sp_dense_kf_rel = karp_flatt_speedup(Ns_dense, kf) / karp_flatt_speedup(np.array([args.ref_devices]), kf)[0]
    sp_dense_lin_rel = linear_speedup(Ns_dense, m_lin, b_lin) / linear_speedup(np.array([args.ref_devices]), m_lin, b_lin)[0]
    t_dense_pl = powerlaw_time(Ns_dense, pl)
    sp_dense_pl_rel = t_ref_bench / t_dense_pl
    ideal_dense = Ns_dense / args.ref_devices

    t_dense_amd_bench = t_ref_bench / sp_dense_amd_rel
    t_dense_gus_bench = t_ref_bench / sp_dense_gus_rel
    t_dense_kf_bench = t_ref_bench / sp_dense_kf_rel
    t_dense_pl_bench = t_dense_pl

    mean_dense_bench = np.mean(np.vstack([t_dense_amd_bench, t_dense_gus_bench, t_dense_pl_bench, t_dense_kf_bench]), axis=0)
    wmean_dense_bench = (
        weights["Amdahl"] * t_dense_amd_bench
        + weights["Gustafson"] * t_dense_gus_bench
        + weights["PowerLaw"] * t_dense_pl_bench
        + weights["KarpFlatt"] * t_dense_kf_bench
    )
    sp_dense_mean_rel = t_ref_bench / mean_dense_bench
    sp_dense_wmean_rel = t_ref_bench / wmean_dense_bench

    sp_band_lo, sp_band_hi = band_low_high(sp_dense_amd_rel, sp_dense_gus_rel)

    t_full_band_lo, t_full_band_hi = band_low_high(t_full_gus, t_full_amd)
    gh_band_lo, gh_band_hi = band_low_high(gh_gus, gh_amd)

    colors = {
        "Measured": "black",
        "Amdahl": "#1f77b4",
        "Gustafson": "#ff7f0e",
        "PowerLaw": "#2ca02c",
        "KarpFlatt": "#d62728",
        "LinearFit": "#9467bd",
        "Mean4": "#8c564b",
        "WMean4": "#e377c2",
        "Ideal": "#7f7f7f",
        "Band": "#c7c7c7",
    }

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.fill_between(Ns_dense, sp_band_lo, sp_band_hi, alpha=0.18, color=colors["Band"])
    ax1.plot(Ns_meas, sp_meas, marker="o", linestyle="none", label="Measured speedup", color=colors["Measured"])
    ax1.plot(Ns_dense, sp_dense_amd_rel, linestyle="-", label="Amdahl", color=colors["Amdahl"])
    ax1.plot(Ns_dense, sp_dense_gus_rel, linestyle="-", label="Gustafson", color=colors["Gustafson"])
    ax1.plot(Ns_dense, sp_dense_pl_rel, linestyle="-", label="Power-law (time)", color=colors["PowerLaw"])
    ax1.plot(Ns_dense, sp_dense_kf_rel, linestyle="-", label="Karp–Flatt", color=colors["KarpFlatt"])
    ax1.plot(Ns_dense, sp_dense_lin_rel, linestyle="-.", label="Linear fit (speedup)", color=colors["LinearFit"])
    ax1.plot(Ns_dense, sp_dense_mean_rel, linestyle="--", label="Mean (4-model)", color=colors["Mean4"])
    ax1.plot(Ns_dense, sp_dense_wmean_rel, linestyle="--", label="Weighted mean (4-model)", color=colors["WMean4"])
    ax1.plot(Ns_dense, ideal_dense, linestyle=":", label="Ideal linear", color=colors["Ideal"])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Devices (log)")
    ax1.set_ylabel(f"Speedup vs {args.ref_devices} device(s) (log)")
    ax1.set_title(f"{args.title} — Speedup")
    ax1.grid(True, which="both", linestyle=":")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(f"{args.out_prefix}_speedup.png", dpi=200)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.fill_between(pred_N, t_full_band_lo / 60.0, t_full_band_hi / 60.0, alpha=0.18, color=colors["Band"])
    ax2.plot(pred_N, t_full_amd / 60.0, marker="o", label="Amdahl", color=colors["Amdahl"])
    ax2.plot(pred_N, t_full_gus / 60.0, marker="o", label="Gustafson", color=colors["Gustafson"])
    ax2.plot(pred_N, t_full_pl / 60.0, marker="o", label="Power-law (time)", color=colors["PowerLaw"])
    ax2.plot(pred_N, t_full_kf / 60.0, marker="o", label="Karp–Flatt", color=colors["KarpFlatt"])
    ax2.plot(pred_N, t_full_lin / 60.0, marker="o", linestyle="-.", label="Linear fit (speedup)", color=colors["LinearFit"])
    ax2.plot(pred_N, mean4_full / 60.0, marker="o", linestyle="--", label="Mean (4-model)", color=colors["Mean4"])
    ax2.plot(pred_N, wmean4_full / 60.0, marker="o", linestyle="--", label="Weighted mean (4-model)", color=colors["WMean4"])
    ax2.plot(pred_N, t_full_ideal / 60.0, marker="o", linestyle=":", label="Ideal linear", color=colors["Ideal"])
    ax2.set_xscale("log")
    ax2.set_xlabel("Devices (log)")
    ax2.set_ylabel("Full epoch time (hours)")
    ax2.set_title(f"{args.title} — Full epoch time (frames={args.full_frames:g})")
    ax2.grid(True, which="both", linestyle=":")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{args.out_prefix}_epoch_time.png", dpi=200)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.fill_between(pred_N, gh_band_lo, gh_band_hi, alpha=0.18, color=colors["Band"])
    ax3.plot(pred_N, gh_amd, marker="o", label="Amdahl", color=colors["Amdahl"])
    ax3.plot(pred_N, gh_gus, marker="o", label="Gustafson", color=colors["Gustafson"])
    ax3.plot(pred_N, gh_pl, marker="o", label="Power-law (time)", color=colors["PowerLaw"])
    ax3.plot(pred_N, gh_kf, marker="o", label="Karp–Flatt", color=colors["KarpFlatt"])
    ax3.plot(pred_N, gh_lin, marker="o", linestyle="-.", label="Linear fit (speedup)", color=colors["LinearFit"])
    ax3.plot(pred_N, gh_mean4, marker="o", linestyle="--", label="Mean (4-model)", color=colors["Mean4"])
    ax3.plot(pred_N, gh_wmean4, marker="o", linestyle="--", label="Weighted mean (4-model)", color=colors["WMean4"])
    ax3.plot(pred_N, gh_ideal, marker="o", linestyle=":", label="Ideal linear", color=colors["Ideal"])
    ax3.set_xscale("log")
    ax3.set_xlabel("Devices (log)")
    ax3.set_ylabel(f"Total GPU-hours for {args.epochs} epochs")
    ax3.set_title(f"{args.title} — GPU-hours (frames={args.full_frames:g}, \n epochs={args.epochs})")
    ax3.grid(True, which="both", linestyle=":")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(f"{args.out_prefix}_gpu_hours.png", dpi=200)

    if args.show:
        plt.show()

    print(f"\nSaved plots: {args.out_prefix}_speedup.png, {args.out_prefix}_epoch_time.png, {args.out_prefix}_gpu_hours.png")


if __name__ == "__main__":
    main()

