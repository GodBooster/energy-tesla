"""
process_oscillogram.py — обробка осцилограм для розрахунку активної потужності.

Очікуваний формат CSV (експорт з осцилографа Rigol/Siglent/тощо):
  Колонка 0: t [s]
  Колонка 1: V_input [V] (CH1, через дифф.щуп з відомим коефіцієнтом)
  Колонка 2: I_input [A] (CH2, через шунт або токовий щуп)
  Колонка 3: V_load  [V] (CH3) — опціонально
  Колонка 4: I_load  [A] (CH4) — опціонально

Усього 5 колонок мінімум; додаткові ігноруються.

Використання:
  python3 process_oscillogram.py path/to/file.csv
  python3 process_oscillogram.py path/to/file.csv --probe-v 1000 --shunt 0.098

Виходить:
  - Активна, реактивна, повна потужність на вході і навантаженні
  - cos(φ)
  - Спектр напруги і струму (FFT)
  - Графіки v(t), i(t), p(t) — зберігаються поряд з CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_oscillogram(csv_path: Path, skip_header: int = 1) -> dict:
    """Завантажити CSV. Очікується t,V_in,I_in[,V_load,I_load]."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=skip_header)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"CSV {csv_path} має {data.shape[1] if data.ndim == 2 else '?'} колонок, "
            "очікується щонайменше 3 (t, V, I)"
        )
    out = {'t': data[:, 0], 'V_in': data[:, 1], 'I_in': data[:, 2]}
    if data.shape[1] >= 5:
        out['V_load'] = data[:, 3]
        out['I_load'] = data[:, 4]
    return out


def power_metrics(v: np.ndarray, i: np.ndarray, t: np.ndarray) -> dict:
    """
    Активна, реактивна, повна потужність.

    P_active = <v(t) · i(t)>           — істинна активна (Вт)
    S_apparent = V_rms · I_rms          — повна (ВА)
    Q_reactive = sqrt(S² - P²)          — реактивна (ВАр)
    cos(φ) = P / S                     — фактор потужності

    Усереднення — точно за цілу кількість періодів сигналу. Якщо запис неперіодичний,
    відрізаємо хвости.
    """
    p_inst = v * i

    # Виявляємо період за переходами через нуль (на компонент, який ми очікуємо
    # ближчим до синусоїди — V_in зазвичай чистіший за I_in)
    v_centered = v - np.mean(v)
    crossings = np.where(np.diff(np.sign(v_centered)) > 0)[0]

    if len(crossings) >= 2:
        # Усереднюємо по цілих періодах для зменшення артефактів краю
        n_full = len(crossings) - 1
        if n_full >= 1:
            i0, i1 = crossings[0], crossings[n_full]
            v_w, i_w, p_w = v[i0:i1], i[i0:i1], p_inst[i0:i1]
            t_w = t[i0:i1]
        else:
            v_w, i_w, p_w, t_w = v, i, p_inst, t
    else:
        v_w, i_w, p_w, t_w = v, i, p_inst, t

    P_active = np.mean(p_w)
    V_rms = np.sqrt(np.mean(v_w ** 2))
    I_rms = np.sqrt(np.mean(i_w ** 2))
    S_apparent = V_rms * I_rms
    Q_reactive = np.sqrt(max(S_apparent ** 2 - P_active ** 2, 0.0))
    cos_phi = P_active / S_apparent if S_apparent > 0 else float('nan')

    # Період / частота
    if len(crossings) >= 2:
        period_samples = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
        dt = np.mean(np.diff(t_w))
        T = period_samples * dt
        freq = 1.0 / T if T > 0 else float('nan')
    else:
        T = float('nan')
        freq = float('nan')

    return {
        'P_active_W': P_active,
        'S_apparent_VA': S_apparent,
        'Q_reactive_VAR': Q_reactive,
        'cos_phi': cos_phi,
        'V_rms': V_rms,
        'I_rms': I_rms,
        'period_s': T,
        'freq_Hz': freq,
        'window_samples': len(v_w),
    }


def fft_spectrum(signal: np.ndarray, t: np.ndarray, top_n: int = 5) -> list:
    """Топ-N частотних компонентів сигналу."""
    n = len(signal)
    if n < 4:
        return []
    dt = np.mean(np.diff(t))
    if dt <= 0:
        return []
    sp = np.fft.rfft(signal - np.mean(signal))
    freqs = np.fft.rfftfreq(n, dt)
    amps = np.abs(sp) * 2.0 / n
    idx = np.argsort(amps)[::-1][:top_n]
    return [(float(freqs[k]), float(amps[k])) for k in idx]


def plot_summary(data: dict, metrics: dict, out_png: Path) -> None:
    """Графіки v(t), i(t), p_inst(t). Без matplotlib не виходить — пропускаємо."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed — skipping plot {out_png}")
        return

    t = data['t']
    v, i = data['V_in'], data['I_in']
    p = v * i

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, v, label='V_in')
    axes[0].set_ylabel('V (V)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, i, label='I_in', color='C1')
    axes[1].set_ylabel('I (A)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, p, label='p(t) = v·i', color='C2')
    axes[2].axhline(metrics['P_active_W'], color='red', linestyle='--',
                    label=f"<P> = {metrics['P_active_W']:.2f} W")
    axes[2].set_ylabel('Power (W)')
    axes[2].set_xlabel('t (s)')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Oscillogram analysis: f={metrics['freq_Hz']:.1f} Hz, "
                 f"cos(φ)={metrics['cos_phi']:.3f}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Plot saved: {out_png}")


def report(metrics_in: dict, metrics_load: dict | None) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("INPUT (NST/network side)")
    lines.append("=" * 60)
    lines.append(f"  P_active   = {metrics_in['P_active_W']:.3f} W")
    lines.append(f"  S_apparent = {metrics_in['S_apparent_VA']:.3f} VA")
    lines.append(f"  Q_reactive = {metrics_in['Q_reactive_VAR']:.3f} VAr")
    lines.append(f"  cos(phi)   = {metrics_in['cos_phi']:.4f}")
    lines.append(f"  V_rms      = {metrics_in['V_rms']:.3f} V")
    lines.append(f"  I_rms      = {metrics_in['I_rms']:.3f} A")
    lines.append(f"  freq       = {metrics_in['freq_Hz']:.2f} Hz")
    if metrics_load is not None:
        lines.append("")
        lines.append("=" * 60)
        lines.append("LOAD")
        lines.append("=" * 60)
        lines.append(f"  P_active   = {metrics_load['P_active_W']:.3f} W")
        lines.append(f"  S_apparent = {metrics_load['S_apparent_VA']:.3f} VA")
        lines.append(f"  Q_reactive = {metrics_load['Q_reactive_VAR']:.3f} VAr")
        lines.append(f"  cos(phi)   = {metrics_load['cos_phi']:.4f}")
        eta = metrics_load['P_active_W'] / metrics_in['P_active_W'] \
            if metrics_in['P_active_W'] > 0 else float('nan')
        lines.append("")
        lines.append("=" * 60)
        lines.append("EFFICIENCY")
        lines.append("=" * 60)
        lines.append(f"  eta_electrical = P_load / P_in = {eta:.4f} ({eta*100:.2f}%)")
        if eta > 1.0:
            lines.append("  *** ALERT: eta > 1 — suspect measurement error ***")
            lines.append("  *** Check: shunt R, probe phase delay, RMS vs avg(v*i) ***")
            lines.append("  *** Verify with calorimeter before any further claims ***")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process oscillogram CSV and report active/reactive/apparent power."
    )
    parser.add_argument('csv', type=Path, help="Path to CSV file")
    parser.add_argument('--probe-v', type=float, default=1.0,
                        help="V probe scaling (e.g. 1000 for 1:1000 HV probe)")
    parser.add_argument('--shunt', type=float, default=None,
                        help="Shunt resistance Ω if I channel is voltage across shunt")
    parser.add_argument('--skip-header', type=int, default=1,
                        help="Number of CSV header rows to skip")
    parser.add_argument('--no-plot', action='store_true',
                        help="Skip matplotlib output")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: file not found: {args.csv}", file=sys.stderr)
        return 1

    data = load_oscillogram(args.csv, skip_header=args.skip_header)

    # Apply probe / shunt scaling
    data['V_in'] *= args.probe_v
    if args.shunt is not None:
        data['I_in'] = data['I_in'] / args.shunt

    metrics_in = power_metrics(data['V_in'], data['I_in'], data['t'])
    metrics_load = None
    if 'V_load' in data:
        if args.shunt is not None and data['I_load'].max() < 10:  # likely shunt voltage
            data['I_load'] = data['I_load'] / args.shunt
        data['V_load'] *= args.probe_v  # assume same HV probe; adjust as needed
        metrics_load = power_metrics(data['V_load'], data['I_load'], data['t'])

    print(report(metrics_in, metrics_load))

    print("\nFFT top components (V_in):")
    for f, a in fft_spectrum(data['V_in'], data['t']):
        print(f"  f = {f:9.2f} Hz, amplitude = {a:.4f}")

    if not args.no_plot:
        out_png = args.csv.with_suffix('.png')
        plot_summary(data, metrics_in, out_png)

    return 0


if __name__ == '__main__':
    sys.exit(main())
