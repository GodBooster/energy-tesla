"""
spark_pump_model.py — модель накачки резонатора через іскровий розрядник.

Це Python-аналог Tesla_proc.xls (з SOFT/) адаптований під наші параметри
Stage 2: f_res = 88.5 кГц, MMC = 17.5 нФ, V_breakdown = 10 кВ.

Модель:
  1. NST заряджає MMC до V_breakdown.
  2. Іскровий розрядник пробиває → R_arc ≈ 0.05 Ом.
  3. LC-контур (Lp=Ctc) дзвенить на f_res із загасанням.
  4. Коли амплітуда падає нижче V_quench — розрядник гаситься.
  5. NST знову заряджає MMC.
  6. Повторити.

Вихід:
  • середня частота іскор (Hz)
  • енергія за іскру (J)
  • середня потужність накачки (W)
  • графік кількох циклів накачки
  • SPICE-сумісне передбачення для перевірки

Використання:
  python3 spark_pump_model.py
  python3 spark_pump_model.py --f-res 88500 --c-mmc 17.5e-9 --v-break 10000
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np


def simulate_one_burst(
    L: float, C: float, R_arc: float,
    V0: float, V_quench: float,
    dt: float, t_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RLC-розряд: V(t), I(t) у затухаючому LC.

    DE (послідовне RLC, V_C — єдине джерело при відсутності V_supply):
        L · dI/dt = V_C - R·I       (V_C тягне струм через L через R)
        C · dV_C/dt = -I            (позитивний струм розряджає C)

    Інтегрування — scipy.solve_ivp (RK45 адаптивний крок). Forward Euler
    розходиться при низькому загасанні (Q≈90 для нашої установки).
    """
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        I_, V_ = y
        dI = (V_ - R_arc * I_) / L
        dV = -I_ / C
        return [dI, dV]

    # Подія гасіння: енергія в контурі впала нижче E_quench
    E_quench = 0.5 * C * V_quench ** 2

    def quench_event(t, y):
        I_, V_ = y
        E = 0.5 * L * I_ ** 2 + 0.5 * C * V_ ** 2
        return E - E_quench

    quench_event.terminal = True
    quench_event.direction = -1  # коли E падає нижче E_quench

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_max),
        y0=[0.0, V0],
        method='RK45',
        max_step=dt,
        rtol=1e-8, atol=1e-10,
        events=quench_event,
        dense_output=False,
    )
    t_arr = sol.t
    I_arr = sol.y[0]
    V_arr = sol.y[1]
    return t_arr, V_arr, I_arr


def estimate_charge_time(C: float, V_target: float, I_charge: float) -> float:
    """Час заряду MMC від 0 до V_target через NST з обмеженням струму.

    Спрощено: NST = ідеальне джерело струму I_sc до V_open, тоді V на C
    зростає лінійно: t = V·C/I.
    """
    return V_target * C / I_charge


def simulate_pumping(
    f_res: float = 88500,
    Lp: float = None,
    C_mmc: float = 17.5e-9,
    V_break: float = 10000,
    V_quench: float = 2000,
    R_arc: float = 0.05,
    R_skin: float = 0.1,   # активний опір первинки на f_res
    NST_V_open: float = 15000,
    NST_I_sc: float = 0.030,   # 30 mA short-circuit
    n_bursts: int = 5,
) -> dict:
    """Повна симуляція кількох циклів накачки і розряду."""
    # Lp за резонансною умовою
    if Lp is None:
        omega = 2 * math.pi * f_res
        Lp = 1.0 / (omega ** 2 * C_mmc)

    # Період резонансу і шматок інтегрування
    T_res = 1.0 / f_res
    dt = T_res / 200.0  # 200 точок на період

    bursts: list[dict] = []
    t_global = 0.0

    for burst_n in range(n_bursts):
        # Заряджання MMC до V_break
        t_charge = estimate_charge_time(C_mmc, V_break, NST_I_sc)

        # Розряд через іскровий розрядник
        # Активний опір контуру = R_arc + R_skin
        R_total = R_arc + R_skin
        t_max = 30.0 * T_res  # на всякий випадок
        t_burst, V_burst, I_burst = simulate_one_burst(
            L=Lp, C=C_mmc, R_arc=R_total,
            V0=V_break, V_quench=V_quench,
            dt=dt, t_max=t_max,
        )

        # Енергія: початкова − залишок
        E_initial = 0.5 * C_mmc * V_break ** 2
        E_left = (0.5 * Lp * I_burst[-1] ** 2 +
                  0.5 * C_mmc * V_burst[-1] ** 2)
        E_per_spark = E_initial - E_left

        bursts.append({
            'burst_n': burst_n,
            't_charge_s': t_charge,
            't_discharge_s': t_burst[-1] - t_burst[0],
            't_global_start': t_global,
            'E_per_spark_J': E_per_spark,
            'V_peak': V_break,
            'I_peak': max(abs(I_burst.min()), abs(I_burst.max())),
            't': t_burst,
            'V': V_burst,
            'I': I_burst,
        })
        t_global += t_charge + (t_burst[-1] - t_burst[0])

    # Зведена статистика
    t_period = bursts[0]['t_charge_s'] + bursts[0]['t_discharge_s']
    spark_freq = 1.0 / t_period
    P_avg = bursts[0]['E_per_spark_J'] / t_period

    return {
        'Lp_uH': Lp * 1e6,
        'spark_freq_Hz': spark_freq,
        'E_per_spark_J': bursts[0]['E_per_spark_J'],
        'P_avg_W': P_avg,
        't_charge_us': bursts[0]['t_charge_s'] * 1e6,
        't_discharge_us': bursts[0]['t_discharge_s'] * 1e6,
        'I_peak_A': bursts[0]['I_peak'],
        'bursts': bursts,
        'f_res': f_res,
    }


def report(d: dict) -> None:
    print("=" * 60)
    print("SPARK GAP PUMP — STAGE 2 PARAMETERS")
    print("=" * 60)
    print(f"  f_резонансу         = {d['f_res']/1000:.2f} кГц")
    print(f"  Lp (з резонансу)    = {d['Lp_uH']:.2f} мкГн")
    print(f"  Енергія за іскру    = {d['E_per_spark_J']:.3f} Дж")
    print(f"  I пік (через розр.) = {d['I_peak_A']:.0f} А")
    print(f"  Час заряду MMC      = {d['t_charge_us']:.0f} мкс")
    print(f"  Час розряду         = {d['t_discharge_us']:.0f} мкс")
    print(f"  Частота іскор       = {d['spark_freq_Hz']:.1f} Гц")
    print(f"  Середня P накачки   = {d['P_avg_W']:.1f} Вт")
    print()
    print("Порівняння з історичними значеннями:")
    print(f"  Colorado Springs (Тесла): 800–1200 іскор/с")
    print(f"  Houston Street    (Тесла): 5000 іскор/с")
    print(f"  Наша установка:   {d['spark_freq_Hz']:.0f} іскор/с")

    if d['spark_freq_Hz'] < 50:
        print("\n  ⚠ Низька частота іскор: NST 30 мА надто слабкий")
        print("     для повного перезаряду на цій частоті.")
    elif d['spark_freq_Hz'] > 2000:
        print("\n  ⚠ Дуже висока частота — переконатись, що іскровик")
        print("     встигає гасити (інакше — короткозамкнений)")
    else:
        print("\n  ✓ Реалістичний робочий режим.")


def plot_bursts(d: dict, out_png: str = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не встановлено — пропускаю графік")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    burst = d['bursts'][0]
    t_us = burst['t'] * 1e6
    axes[0].plot(t_us, burst['V'] / 1000, color='C0')
    axes[0].set_ylabel('V_C (кВ)')
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"Іскрова накачка: f={d['f_res']/1000:.1f} кГц, "
                      f"E={d['E_per_spark_J']:.2f} Дж, I_peak={d['I_peak_A']:.0f} А")

    axes[1].plot(t_us, burst['I'], color='C1')
    axes[1].set_xlabel('t (мкс)')
    axes[1].set_ylabel('I (А)')
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = out_png or 'spark_pump_burst.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nГрафік збережено: {out}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--f-res', type=float, default=88500)
    p.add_argument('--c-mmc', type=float, default=17.5e-9)
    p.add_argument('--v-break', type=float, default=10000)
    p.add_argument('--v-quench', type=float, default=2000)
    p.add_argument('--nst-isc', type=float, default=0.030)
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()

    d = simulate_pumping(
        f_res=args.f_res,
        C_mmc=args.c_mmc,
        V_break=args.v_break,
        V_quench=args.v_quench,
        NST_I_sc=args.nst_isc,
    )
    report(d)
    if not args.no_plot:
        plot_bursts(d)
    return 0


if __name__ == '__main__':
    sys.exit(main())
