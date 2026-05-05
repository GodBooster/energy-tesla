"""
coil_geometry_optimizer.py — підбір параметрів для Тесла-коіла з потрійним резонансом.

Задача:
  Дано бажану резонансну частоту f_target і верхню напругу V_target,
  обчислити параметри вторинної котушки (Ns, Rs, Hs), top-load (Ctl),
  первинної котушки (Lp, Ctc), MMC (Ctc), щоб умова T=1 (тобто
  fp = fs) виконувалася і коефіцієнт зв'язку k відповідав цілочисельному
  (a,c) набору з таблиці де Кейроза.

Формули:
  Wheeler для вторинної (соленоїдна, одношарова):
    L_s [H] = 2.46e-7 * Ns^2 * Rs^2 / (9*Rs + 10*Hs)   [Rs, Hs у метрах]

  Wheeler для плоскої спіральної первинної:
    L_p [μH] = Np^2 * Rp^2 / (8*Rp + 11*Cp)            [Rp, Cp у см]

  Резонанс:
    f = 1 / (2π√(L*C))

  Ємність тороїдального top-load (Medhurst):
    C_tl ≈ 5.08 * D_tor (мідь) [пФ]      де D_tor — діаметр великого кола в см
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


# Таблиця де Кейроза для повного переносу за N циклів
DE_QUEIROZ_TABLE = [
    # (a, b, c, k, cycles_to_full_transfer)
    (1, 2, 3, 0.600, 1.0),
    (1, 3, 5, 0.882, 1.5),
    (3, 4, 5, 0.280, 2.0),
    (5, 6, 7, 0.180, 3.0),
    (7, 8, 9, 0.133, 4.0),
    (10, 11, 12, 0.095, 5.5),
]


@dataclass
class SecondaryCoilParams:
    Ns: int           # кількість витків
    Rs_m: float       # радіус каркаса (м)
    Hs_m: float       # висота обмотки (м)
    wire_d_mm: float  # діаметр проводу (з ізоляцією)

    @property
    def Ls_henry(self) -> float:
        """Wheeler formula для одношарового соленоїда"""
        # L [H] = μ₀ * π * R² * N² * F / Hs
        # для довгого соленоїда. Поправка Wheeler:
        # L [μH] = R² * N² / (9R + 10Hs), R, Hs in inches
        # Конвертуємо в метричні:
        Rs_in = self.Rs_m * 39.37
        Hs_in = self.Hs_m * 39.37
        L_uH = (Rs_in ** 2 * self.Ns ** 2) / (9 * Rs_in + 10 * Hs_in)
        return L_uH * 1e-6


@dataclass
class FlatPrimaryParams:
    Np: int           # кількість витків
    Rp_inner_m: float  # внутрішній радіус
    pitch_m: float    # відстань між витками
    tube_d_mm: float = 6.0  # діаметр трубки

    @property
    def Rp_avg_m(self) -> float:
        return self.Rp_inner_m + self.Np * self.pitch_m / 2.0

    @property
    def Cp_m(self) -> float:
        """Ширина обмотки (зовн.радіус - внутр.радіус)"""
        return self.Np * self.pitch_m

    @property
    def Lp_henry(self) -> float:
        """Wheeler для плоскої спіральної котушки"""
        # L [μH] = N² * R² / (8R + 11C), all in cm
        # R = середній радіус, C = ширина обмотки
        R_cm = self.Rp_avg_m * 100
        C_cm = self.Cp_m * 100
        if 8 * R_cm + 11 * C_cm == 0:
            return 0.0
        L_uH = (self.Np ** 2 * R_cm ** 2) / (8 * R_cm + 11 * C_cm)
        return L_uH * 1e-6


def toroid_capacitance_pF(D_outer_cm: float, d_minor_cm: float) -> float:
    """
    Ємність тороїда у вільному просторі (наближення Medhurst).
    D_outer — діаметр великого кола, d_minor — діаметр перерізу трубки.
    """
    # Медіурст: C [пФ] ≈ 1.4 * (1.2438 * D + 0.6438 * d) для тонкого тороїда
    # Більш точно — формула Кьорті, але для нашого використання достатньо:
    return 2.8 * (1.2438 * (D_outer_cm / 2.0) + 0.6438 * (d_minor_cm / 2.0))


def sphere_capacitance_pF(R_cm: float) -> float:
    """Ємність ізольованої сфери: C = 4πε₀R [Ф], для R у см: C[пФ] = R/9"""
    return R_cm / 0.899


def resonant_freq_Hz(L_henry: float, C_farad: float) -> float:
    if L_henry <= 0 or C_farad <= 0:
        return 0.0
    return 1.0 / (2 * math.pi * math.sqrt(L_henry * C_farad))


def select_coupling(target_cycles: int = 2) -> tuple[int, int, int, float, float]:
    """Обрати (a,b,c,k,cycles) з таблиці де Кейроза"""
    closest = min(DE_QUEIROZ_TABLE, key=lambda row: abs(row[4] - target_cycles))
    return closest


def design_tesla_coil(
    f_target_Hz: float = 76e3,
    Ns: int = 1600,
    Rs_m: float = 0.057,
    Hs_m: float = 0.570,
    wire_d_mm: float = 0.25,
    toroid_D_cm: float = 40.0,
    toroid_d_cm: float = 5.0,
    target_cycles: int = 2,
    Vc_max_kV: float = 10.0,
) -> dict:
    """Повний розрахунок параметрів Тесла-коіла"""

    sec = SecondaryCoilParams(Ns=Ns, Rs_m=Rs_m, Hs_m=Hs_m, wire_d_mm=wire_d_mm)
    Ls = sec.Ls_henry

    Ctl_pF = toroid_capacitance_pF(toroid_D_cm, toroid_d_cm)
    Ctl = Ctl_pF * 1e-12

    f_secondary = resonant_freq_Hz(Ls, Ctl)

    # Оберемо k за бажаним режимом
    a, b, c, k_target, cycles = select_coupling(target_cycles)

    # Для T=1 потрібно щоб первинна резонувала на тій же частоті.
    # Обираємо частоту первинки = f_secondary, а Ctc — згідно V_max і енергетичних міркувань.
    # Енергія в первинному конденсаторі ½ Ctc V² має бути достатньою
    # для роботи; типово 50–500 Дж для крупних котушок.
    # Конкретно для V_NST=15kV (Vc_max=10kV після ВВ-діода):
    Ctc_target_nF = 17.5  # типове значення для нашої установки
    Ctc = Ctc_target_nF * 1e-9

    # Lp = 1 / ((2π·f)² · Ctc)
    omega = 2 * math.pi * f_secondary
    Lp = 1.0 / (omega ** 2 * Ctc)

    # Знайдемо геометрію плоскої первинки, що дає Lp
    # Зворотна задача: підберемо Np, Rp_inner, pitch для заданої Lp
    # Зафіксуємо pitch=15mm, Rp_inner=80mm, шукаємо Np
    pitch_m = 0.015
    Rp_inner_m = 0.080
    Np_best = None
    for Np in range(4, 20):
        prim_test = FlatPrimaryParams(Np=Np, Rp_inner_m=Rp_inner_m, pitch_m=pitch_m)
        if abs(prim_test.Lp_henry - Lp) < abs(
                FlatPrimaryParams(Np=Np_best or 1,
                                  Rp_inner_m=Rp_inner_m,
                                  pitch_m=pitch_m).Lp_henry - Lp):
            Np_best = Np
    primary = FlatPrimaryParams(Np=Np_best, Rp_inner_m=Rp_inner_m, pitch_m=pitch_m)
    Lp_actual = primary.Lp_henry

    # Енергія в первинному конденсаторі
    E_primary_J = 0.5 * Ctc * (Vc_max_kV * 1000) ** 2

    return {
        'secondary': {
            'Ns': Ns,
            'Rs_m': Rs_m,
            'Hs_m': Hs_m,
            'Ls_H': Ls,
            'Ls_mH': Ls * 1000,
        },
        'top_load': {
            'D_outer_cm': toroid_D_cm,
            'd_minor_cm': toroid_d_cm,
            'Ctl_pF': Ctl_pF,
        },
        'secondary_resonance_Hz': f_secondary,
        'primary': {
            'Lp_target_H': Lp,
            'Lp_target_uH': Lp * 1e6,
            'Np_optimal': primary.Np,
            'Rp_inner_m': primary.Rp_inner_m,
            'Rp_avg_m': primary.Rp_avg_m,
            'pitch_m': primary.pitch_m,
            'Lp_actual_uH': Lp_actual * 1e6,
        },
        'mmc': {
            'Ctc_nF': Ctc * 1e9,
            'V_max_kV': Vc_max_kV,
            'E_stored_J': E_primary_J,
        },
        'coupling': {
            'a': a,
            'b': b,
            'c': c,
            'k_target': k_target,
            'cycles_to_transfer': cycles,
        },
        'tuning': {
            'T': 1.0,
            'frequency_match': abs(f_secondary - resonant_freq_Hz(Lp_actual, Ctc))
                / f_secondary,
        },
    }


def print_design(d: dict) -> None:
    print("=" * 60)
    print("TESLA COIL DESIGN — TRIPLE RESONANCE / DE QUEIROZ COUPLING")
    print("=" * 60)
    s = d['secondary']
    print(f"\nВТОРИННА КАТУШКА:")
    print(f"  Ns          = {s['Ns']} витків")
    print(f"  Rs          = {s['Rs_m']*1000:.0f} мм (діаметр {s['Rs_m']*2000:.0f} мм)")
    print(f"  Hs          = {s['Hs_m']*1000:.0f} мм")
    print(f"  Ls          = {s['Ls_mH']:.1f} мГн")

    t = d['top_load']
    print(f"\nTOP-LOAD (тороїд):")
    print(f"  D_outer     = {t['D_outer_cm']:.1f} см")
    print(f"  d_minor     = {t['d_minor_cm']:.1f} см")
    print(f"  Ctl         = {t['Ctl_pF']:.2f} пФ")

    print(f"\nРЕЗОНАНС вторинного контуру: {d['secondary_resonance_Hz']/1000:.2f} кГц")

    p = d['primary']
    print(f"\nПЕРВИННА КАТУШКА (плоска спіральна):")
    print(f"  Цільова Lp  = {p['Lp_target_uH']:.1f} мкГн")
    print(f"  Np          = {p['Np_optimal']} витків")
    print(f"  Rp_inner    = {p['Rp_inner_m']*1000:.0f} мм")
    print(f"  Rp_avg      = {p['Rp_avg_m']*1000:.0f} мм")
    print(f"  pitch       = {p['pitch_m']*1000:.1f} мм")
    print(f"  Реальна Lp  = {p['Lp_actual_uH']:.1f} мкГн")

    m = d['mmc']
    print(f"\nMMC (первинний конденсатор):")
    print(f"  Ctc         = {m['Ctc_nF']:.1f} нФ")
    print(f"  V_max       = {m['V_max_kV']} кВ")
    print(f"  E_stored    = {m['E_stored_J']:.2f} Дж")

    c = d['coupling']
    print(f"\nКОЕФІЦІЄНТ ЗВ'ЯЗКУ (де Кейроз):")
    print(f"  (a,b,c)     = ({c['a']}, {c['b']}, {c['c']})")
    print(f"  k_target    = {c['k_target']:.3f}")
    print(f"  Циклів      = {c['cycles_to_transfer']:.1f}")

    tn = d['tuning']
    print(f"\nT=1 ТЮНІНГ:")
    print(f"  Розбіжність частот: {tn['frequency_match']*100:.2f}%")
    print(f"  (підстроюй переміщенням відводу первинки до <0.5%)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Design Tesla coil with triple resonance and de Queiroz coupling."
    )
    parser.add_argument('--Ns', type=int, default=1600)
    parser.add_argument('--Rs', type=float, default=0.057, help="Secondary radius [m]")
    parser.add_argument('--Hs', type=float, default=0.570, help="Secondary height [m]")
    parser.add_argument('--toroid-D', type=float, default=40.0,
                        help="Toroid outer diameter [cm]")
    parser.add_argument('--toroid-d', type=float, default=5.0,
                        help="Toroid minor (tube) diameter [cm]")
    parser.add_argument('--cycles', type=int, default=2,
                        help="Target cycles for full energy transfer (1, 2, 3, 4)")
    parser.add_argument('--Vmax', type=float, default=10.0,
                        help="Max primary voltage [kV]")
    args = parser.parse_args()

    design = design_tesla_coil(
        Ns=args.Ns,
        Rs_m=args.Rs,
        Hs_m=args.Hs,
        toroid_D_cm=args.toroid_D,
        toroid_d_cm=args.toroid_d,
        target_cycles=args.cycles,
        Vc_max_kV=args.Vmax,
    )

    print_design(design)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
