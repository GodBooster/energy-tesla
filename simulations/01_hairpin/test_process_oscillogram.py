"""
test_process_oscillogram.py — verifies process_oscillogram.py against known-truth
synthetic signals. We generate sinusoids with prescribed P, Q, S and check the
script recovers them within tolerance.

This is a metrology gate: if our analysis script gives wrong P_active on synthetic
data with known answer, every measurement we make later is suspect.

Usage:
    python3 test_process_oscillogram.py
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from process_oscillogram import load_oscillogram, power_metrics  # noqa: E402


def make_signal(
    f_Hz: float,
    V_peak: float,
    I_peak: float,
    phi_rad: float,
    duration_s: float,
    fs_Hz: float,
    noise_v: float = 0.0,
    noise_i: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic v(t) = V·sin(ωt), i(t) = I·sin(ωt − φ).

    For pure sinusoids the analytical answer is:
        P_active = (V·I/2)·cos(φ)
        S        = (V/√2)·(I/√2) = V·I/2
        cos(φ)   = P/S
    """
    n = int(round(duration_s * fs_Hz))
    t = np.arange(n) / fs_Hz
    omega = 2 * math.pi * f_Hz
    v = V_peak * np.sin(omega * t)
    i = I_peak * np.sin(omega * t - phi_rad)
    if noise_v > 0:
        v = v + np.random.normal(0, noise_v, n)
    if noise_i > 0:
        i = i + np.random.normal(0, noise_i, n)
    return t, v, i


def write_csv(path: Path, t: np.ndarray, v: np.ndarray, i: np.ndarray) -> None:
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_s', 'V_in', 'I_in'])
        for row in zip(t, v, i):
            w.writerow([f"{row[0]:.9e}", f"{row[1]:.6e}", f"{row[2]:.6e}"])


def case(label: str, expected: dict, computed: dict, tol_pct: float = 1.0) -> bool:
    """Compare computed metrics to expected within tolerance, print PASS/FAIL line."""
    ok = True
    msgs = []
    for k, v_exp in expected.items():
        v_got = computed.get(k)
        if v_got is None:
            ok = False
            msgs.append(f"    {k}: missing")
            continue
        if v_exp == 0:
            err = abs(v_got - v_exp)
            ok_k = err < tol_pct / 100.0
        else:
            err = abs(v_got - v_exp) / abs(v_exp) * 100
            ok_k = err < tol_pct
        status = '✓' if ok_k else '✗'
        msgs.append(f"    {status} {k}: expected {v_exp:.4g}, got {v_got:.4g} ({err:.2f}{'%' if v_exp != 0 else ''})")
        if not ok_k:
            ok = False
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    for m in msgs:
        print(m)
    return ok


def main() -> int:
    np.random.seed(20260507)
    tmp = Path(tempfile.mkdtemp(prefix='oscill_test_'))
    print(f"Temp dir: {tmp}\n")

    all_pass = True

    # ── Case 1: pure resistive load, cos(φ)=1 ───────────────────────────
    f, V, I = 88500.0, 100.0, 2.0  # 88.5 kHz, 100 V peak, 2 A peak
    t, v, i = make_signal(f, V, I, 0.0, duration_s=1e-3, fs_Hz=10e6)
    csv_path = tmp / 'case1_resistive.csv'
    write_csv(csv_path, t, v, i)
    data = load_oscillogram(csv_path)
    m = power_metrics(data['V_in'], data['I_in'], data['t'])
    expected = {
        'P_active_W': V * I / 2.0,                 # 100 W
        'S_apparent_VA': V * I / 2.0,              # 100 VA
        'cos_phi': 1.0,
        'freq_Hz': f,
    }
    all_pass &= case("Case 1: resistive (cos φ = 1) at 88.5 kHz", expected, m, tol_pct=2.0)
    print()

    # ── Case 2: purely reactive, φ=90°, P should be ≈ 0 ─────────────────
    t, v, i = make_signal(f, V, I, math.pi / 2, duration_s=1e-3, fs_Hz=10e6)
    csv_path = tmp / 'case2_reactive.csv'
    write_csv(csv_path, t, v, i)
    data = load_oscillogram(csv_path)
    m = power_metrics(data['V_in'], data['I_in'], data['t'])
    expected = {
        'P_active_W': 0.0,                         # ≈ 0
        'S_apparent_VA': V * I / 2.0,              # 100 VA (still!)
        'cos_phi': 0.0,
        'freq_Hz': f,
    }
    all_pass &= case("Case 2: purely reactive (φ = 90°)", expected, m, tol_pct=2.0)
    print()

    # ── Case 3: φ=60°, P = S·cos(60°) = S/2 ─────────────────────────────
    phi = math.pi / 3
    t, v, i = make_signal(f, V, I, phi, duration_s=1e-3, fs_Hz=10e6)
    csv_path = tmp / 'case3_phi60.csv'
    write_csv(csv_path, t, v, i)
    data = load_oscillogram(csv_path)
    m = power_metrics(data['V_in'], data['I_in'], data['t'])
    S = V * I / 2.0
    expected = {
        'P_active_W': S * math.cos(phi),           # 50 W
        'S_apparent_VA': S,
        'cos_phi': math.cos(phi),                  # 0.5
        'freq_Hz': f,
    }
    all_pass &= case("Case 3: φ = 60°, cos φ = 0.5", expected, m, tol_pct=2.0)
    print()

    # ── Case 4: very common Tesla-coil failure mode ──────────────────────
    # naïve V_rms · I_rms = 100 VA but P_active = 0.5 W (cos φ ≈ 0.005)
    # If we mistakenly reported S as P, we'd claim 200× the real power.
    phi = math.pi / 2 - 0.005   # 89.7°
    t, v, i = make_signal(f, V, I, phi, duration_s=1e-3, fs_Hz=10e6)
    csv_path = tmp / 'case4_tesla_trap.csv'
    write_csv(csv_path, t, v, i)
    data = load_oscillogram(csv_path)
    m = power_metrics(data['V_in'], data['I_in'], data['t'])
    S = V * I / 2.0
    expected = {
        'P_active_W': S * math.cos(phi),           # tiny
        'S_apparent_VA': S,
        'cos_phi': math.cos(phi),                  # ~0.005
    }
    all_pass &= case("Case 4: Tesla cos-phi trap (φ ≈ 89.7°)", expected, m, tol_pct=5.0)
    print()

    # ── Case 5: noisy signal — P_active should still be recoverable ─────
    t, v, i = make_signal(f, V, I, 0.0, duration_s=1e-3, fs_Hz=10e6,
                          noise_v=2.0, noise_i=0.05)
    csv_path = tmp / 'case5_noisy.csv'
    write_csv(csv_path, t, v, i)
    data = load_oscillogram(csv_path)
    m = power_metrics(data['V_in'], data['I_in'], data['t'])
    expected = {
        'P_active_W': V * I / 2.0,                 # 100 W
        'cos_phi': 1.0,
        'freq_Hz': f,
    }
    all_pass &= case("Case 5: resistive + 2% noise", expected, m, tol_pct=3.0)
    print()

    print("=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED ✓' if all_pass else 'FAILURES PRESENT ✗'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
