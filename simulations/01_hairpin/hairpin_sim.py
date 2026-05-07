"""
hairpin_sim.py — pure-Python симуляція Hairpin резонатора.

Альтернатива ngspice (на macOS Tier 3 ngspice вимагає компіляції gcc-15 з джерел —
надто довго). Використовує scipy.solve_ivp для RLC з гістерезисним розрядником.

Топологія (Kraakman / waveguide.blog):
  NST (15 кВ AC, 50 Гц) → MMC (2 нФ) → іскровий розрядник → дві паралельні мідні шини
  довжиною 220 см (U-форма). Шини утворюють резонатор з власною індуктивністю
  Lbus ~ 5 мкГн/гілка і паразитною ємністю Cbus між собою.

Ця модель не замінює реальне вимірювання, але дає базову лінію для порівняння:
  • очікувана резонансна частота
  • очікувана амплітуда напруги на шинах
  • час затухання Q
  • ВВ-форма (затухаюча синусоїда)

Використання:
  python3 hairpin_sim.py
  python3 hairpin_sim.py --c-mmc 2e-9 --l-bus 5e-6 --c-bus 100e-12
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def simulate_hairpin(
    Vnst_pk: float = 21213,    # 15 kV RMS → 21.2 kV peak
    f_nst: float = 50.0,
    Cmmc: float = 150e-9,      # ~150 нФ (з MMC 30×4.7нФ паралельно для роботи в kHz)
    Lbus: float = 5e-6,        # ~5 μH на гілку шин (мідь Ø22мм × 220 см)
    Cbus: float = 100e-12,     # паразитна ємність між шинами
    Rbus: float = 0.05,
    Vgap_on: float = 8000,
    Vgap_off: float = 1500,
    Rgap_on: float = 0.05,
    t_max: float = 0.1,           # 5 NST periods
    points_per_period_res: int = 50,
) -> dict:
    """Hybrid solver: lumped RLC with state-machine spark gap.

    State variables:
        V_C_mmc:  напруга на MMC
        V_C_bus:  напруга на ємності між шинами
        I_bus:    струм у шинах (резонансному контурі)

    DE (коли іскровий замкнений):
        L · dI/dt = V_C_mmc - V_C_bus - R·I
        C_mmc · dV_C_mmc/dt = -I + I_NST
        C_bus · dV_C_bus/dt = +I

    Коли іскровий відкритий — струм через нього I=0; MMC заряджається тільки від NST.
    """
    # Очікувана резонансна частота: тока через шини резонує з MMC
    # (Cbus між шинами — паразитний шунт, не послідовний з MMC у контурі)
    L_eff = 2 * Lbus            # дві гілки в серії = 2L
    C_eff = Cmmc                # MMC домінує (Cbus << Cmmc)
    f_res = 1.0 / (2 * math.pi * math.sqrt(L_eff * C_eff))
    T_res = 1.0 / f_res
    dt_max = T_res / points_per_period_res

    omega_nst = 2 * math.pi * f_nst

    # NST модель: ідеальне джерело SIN(t), обмежене внутрішнім R
    R_nst = 1000.0  # 1 кОм внутрішнього опору

    def vnst(t):
        return Vnst_pk * math.sin(omega_nst * t)

    # Стан розрядника
    spark_state = {'closed': False}

    def rhs(t, y):
        V_mmc, V_bus, I = y
        Vsource = vnst(t)

        # Заряджання MMC від NST через R_nst
        I_charge = (Vsource - V_mmc) / R_nst

        if spark_state['closed']:
            # Розряд через шини
            dI = (V_mmc - V_bus - Rbus * I) / L_eff
            dV_mmc = (I_charge - I) / Cmmc
            dV_bus = I / Cbus
        else:
            # Розрядник розімкнений; MMC заряджається; струм у шинах загасає
            dI = (-Rbus * I) / L_eff if abs(I) > 1e-6 else -I / 1e-6  # quench
            dV_mmc = I_charge / Cmmc
            dV_bus = 0.0

        return [dV_mmc, dV_bus, dI]

    # Подія: пробій (V_mmc досяг Vgap_on)
    def fire_event(t, y):
        return y[0] - Vgap_on
    fire_event.terminal = True
    fire_event.direction = +1

    # Подія: гасіння (струм досяг 0 повертаючись)
    def quench_event(t, y):
        return abs(y[2]) - 0.5  # струм впав нижче 0.5 А
    quench_event.terminal = True
    quench_event.direction = -1

    # Інтегруємо короткими сегментами зі зміною state-machine
    t_arr = [0.0]
    V_mmc_arr = [0.0]
    V_bus_arr = [0.0]
    I_arr = [0.0]

    t_now = 0.0
    state = [0.0, 0.0, 0.0]   # V_mmc, V_bus, I

    n_segments = 0
    max_segments = 500

    while t_now < t_max and n_segments < max_segments:
        n_segments += 1
        if not spark_state['closed']:
            # Чекаємо пробій
            sol = solve_ivp(
                rhs, (t_now, t_max), state, method='RK45',
                max_step=dt_max * 100,  # крупніше при заряджанні
                rtol=1e-7, atol=1e-9,
                events=fire_event,
                dense_output=False,
            )
        else:
            # Чекаємо гасіння
            sol = solve_ivp(
                rhs, (t_now, t_max), state, method='RK45',
                max_step=dt_max,
                rtol=1e-8, atol=1e-10,
                events=quench_event,
                dense_output=False,
            )

        t_arr.extend(sol.t[1:])
        V_mmc_arr.extend(sol.y[0][1:])
        V_bus_arr.extend(sol.y[1][1:])
        I_arr.extend(sol.y[2][1:])

        if sol.t_events[0].size > 0:
            # Подія спрацювала — змінюємо стан
            spark_state['closed'] = not spark_state['closed']
            t_now = sol.t[-1]
            state = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1]]
        else:
            break

    return {
        't': np.array(t_arr),
        'V_mmc': np.array(V_mmc_arr),
        'V_bus': np.array(V_bus_arr),
        'I': np.array(I_arr),
        'f_res_predicted_Hz': f_res,
        'L_eff_uH': L_eff * 1e6,
        'C_eff_pF': C_eff * 1e12,
        'n_sparks': n_segments // 2,
    }


def report(d: dict) -> None:
    print("=" * 60)
    print("HAIRPIN RESONATOR SIMULATION")
    print("=" * 60)
    print(f"\nДекомпозиція контуру:")
    print(f"  L_eff = {d['L_eff_uH']:.2f} мкГн (2 гілки шин в серії)")
    print(f"  C_eff = {d['C_eff_pF']/1000:.1f} нФ (MMC; Cbus паразитний)")
    print(f"\nПЕРЕДБАЧЕНА резонансна частота: {d['f_res_predicted_Hz']/1e3:.1f} кГц")
    print(f"  (Типова для Hairpin: 45–150 кГц залежно від MMC)")
    if d['f_res_predicted_Hz'] > 200e3:
        print(f"  ⚠ Висока частота. Збільш C_MMC або L_шин для роботи в кГц.")
    if d['f_res_predicted_Hz'] < 30e3:
        print(f"  ⚠ Низька частота. Можливо завелика C_MMC.")
    print(f"\nЗа час симуляції (0.1 с):")
    print(f"  Кількість іскор: {d['n_sparks']}")
    if d['n_sparks'] > 0:
        spark_freq = d['n_sparks'] / 0.1
        print(f"  Частота іскор:   {spark_freq:.0f} Гц")

    # FFT для перевірки реальної резонансної частоти
    if len(d['I']) > 100:
        # Беремо ділянку де є коливання
        I_centered = d['I'][len(d['I'])//4:] - np.mean(d['I'][len(d['I'])//4:])
        if len(I_centered) > 4 and np.std(I_centered) > 0.01:
            dt_avg = np.mean(np.diff(d['t'][len(d['t'])//4:]))
            sp = np.fft.rfft(I_centered)
            freqs = np.fft.rfftfreq(len(I_centered), dt_avg)
            peak_idx = np.argmax(np.abs(sp[1:])) + 1  # skip DC
            f_measured = freqs[peak_idx]
            err = (f_measured - d['f_res_predicted_Hz']) / d['f_res_predicted_Hz'] * 100
            print(f"\nFFT-виміряна f_res:  {f_measured/1e3:.1f} кГц (Δ {err:+.1f}% від теорії)")


def plot_traces(d: dict, out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не встановлено")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    t_ms = d['t'] * 1000

    axes[0].plot(t_ms, d['V_mmc'] / 1000, color='C0', linewidth=0.7)
    axes[0].set_ylabel('V_MMC (кВ)')
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Hairpin симуляція: NST → MMC → spark gap → bus bars")

    axes[1].plot(t_ms, d['V_bus'] / 1000, color='C1', linewidth=0.7)
    axes[1].set_ylabel('V_шин (кВ)')
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_ms, d['I'], color='C2', linewidth=0.7)
    axes[2].set_xlabel('t (мс)')
    axes[2].set_ylabel('I_шин (А)')
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nГрафік: {out_png}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--c-mmc', type=float, default=150e-9,
                   help='MMC capacitance [F]; ~150 nF gives ~50 kHz with 10 μH bus')
    p.add_argument('--l-bus', type=float, default=5e-6,
                   help='Per-arm bus bar inductance [H]')
    p.add_argument('--c-bus', type=float, default=100e-12)
    p.add_argument('--vnst', type=float, default=21213)
    p.add_argument('--t-max', type=float, default=0.05)
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--out-dir', type=Path, default=None)
    args = p.parse_args()

    d = simulate_hairpin(
        Vnst_pk=args.vnst,
        Cmmc=args.c_mmc,
        Lbus=args.l_bus,
        Cbus=args.c_bus,
        t_max=args.t_max,
    )
    report(d)

    if not args.no_plot:
        out = (args.out_dir / 'hairpin_predict.png') if args.out_dir else \
              Path(__file__).parent / 'hairpin_predict.png'
        plot_traces(d, str(out))

    return 0


if __name__ == '__main__':
    sys.exit(main())
