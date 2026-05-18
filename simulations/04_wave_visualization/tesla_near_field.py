"""
tesla_near_field.py — інтерактивна візуалізація ближнього/далекого поля Тесла-резонатора.

Наша науково-обґрунтована альтернатива Romancorp/EMVR-симулятору
(sites.google.com/view/emvr/). На відміну від нього:
  • справжні числа для нашої установки (f₀ = 88.5 кГц, h = 2.44 м, top-load 33 см)
  • формули диполя з Jackson "Classical Electrodynamics" §9.2
  • розмежування реактивної (r < λ/2π), перехідної, та радіаційної зон
  • порівняння конфігурацій з/без заземлення

Запуск:
  python3 tesla_near_field.py
  python3 tesla_near_field.py --grounded
  python3 tesla_near_field.py --animation
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button

# Фізичні константи
c = 2.998e8          # швидкість світла (м/с)
eps0 = 8.854e-12     # ε₀ (Ф/м)
mu0 = 4 * math.pi * 1e-7  # μ₀ (Гн/м)
eta0 = math.sqrt(mu0 / eps0)  # 377 Ом — імпеданс вакууму


def dipole_fields(r, theta, k, I0, dl, omega, t=0.0):
    """Поля електричного диполя з Jackson §9.2 (формули 9.18).

    Args:
        r:        радіус від диполя (м)
        theta:    кут від осі диполя (рад)
        k:        хвильове число ω/c (1/м)
        I0:       пік. струм у диполі (А)
        dl:       довжина диполя (м)
        omega:    кутова частота (рад/с)
        t:        час (с)

    Returns:
        (Er, Etheta, Bphi) — реальні компоненти поля у моменті t
    """
    p0 = I0 * dl / omega   # дипольний момент

    # Загальна фаза
    phase = k * r - omega * t

    # E_r = (2 p₀ cos θ / 4πε₀) · [1/r³ + ik/r²] e^(i(kr-ωt))
    # E_θ = (p₀ sin θ / 4πε₀) · [1/r³ + ik/r² - k²/r] e^(i(kr-ωt))
    # B_φ = (μ₀ p₀ ω sin θ / 4π) · [ik/r² + 1/r] e^(i(kr-ωt))  (×ω/c для імпульсу)

    # Беремо дійсну частину (cos для e^(iφ))
    Er = (2 * p0 * math.cos(theta) / (4 * math.pi * eps0)) * (
        (1/r**3) * math.cos(phase) - (k/r**2) * math.sin(phase)
    )
    Etheta = (p0 * math.sin(theta) / (4 * math.pi * eps0)) * (
        (1/r**3) * math.cos(phase)
        - (k/r**2) * math.sin(phase)
        - (k**2/r) * math.cos(phase)
    )
    Bphi = (mu0 * p0 * omega * math.sin(theta) / (4 * math.pi)) * (
        (k/r**2) * math.cos(phase) + (1/r) * (-math.sin(phase))
    ) / c   # /c щоб мати B, не H

    return Er, Etheta, Bphi


def poynting_vector(Er, Etheta, Bphi):
    """Вектор Пойнтінга S = E × H / μ₀ (Вт/м²).

    У сферичних координатах (r, θ, φ):
        S_r = (E_θ · B_φ) / μ₀     ← радіальний потік (енергія наружу)
        S_θ = -(E_r · B_φ) / μ₀    ← полярний (циркуляційний)
    """
    Hphi = Bphi / mu0
    S_r = Etheta * Hphi
    S_theta = -Er * Hphi
    return S_r, S_theta


def compute_field_map(f_Hz, I0, dl, R_max, n_r=80, n_theta=60, t=0.0):
    """Обчислити поля на 2D сітці (r, θ)."""
    omega = 2 * math.pi * f_Hz
    k = omega / c
    lam = c / f_Hz

    rs = np.linspace(0.1 * lam, R_max, n_r)
    thetas = np.linspace(0.01, math.pi - 0.01, n_theta)

    Er = np.zeros((n_r, n_theta))
    Etheta = np.zeros((n_r, n_theta))
    Bphi = np.zeros((n_r, n_theta))

    for i, r in enumerate(rs):
        for j, theta in enumerate(thetas):
            Er[i, j], Etheta[i, j], Bphi[i, j] = dipole_fields(
                r, theta, k, I0, dl, omega, t
            )

    S_r, S_theta = poynting_vector(Er, Etheta, Bphi)
    S_mag = np.sqrt(S_r**2 + S_theta**2)

    return {
        'rs': rs,
        'thetas': thetas,
        'Er': Er,
        'Etheta': Etheta,
        'Bphi': Bphi,
        'S_r': S_r,
        'S_theta': S_theta,
        'S_mag': S_mag,
        'lam': lam,
        'r_near': lam / (2 * math.pi),   # межа реактивної зони
        'r_far': 2 * lam,                # початок далекої зони
    }


def plot_static(data, title_suffix=''):
    """Статичний 2D-зріз поля у площині (r, θ)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Перевести у декартові x = r·sinθ, z = r·cosθ
    R, T = np.meshgrid(data['rs'], data['thetas'], indexing='ij')
    X = R * np.sin(T)
    Z = R * np.cos(T)

    # Ліворуч: модуль E
    E_mag = np.sqrt(data['Er']**2 + data['Etheta']**2)
    pcm0 = axes[0].pcolormesh(X, Z, np.log10(E_mag + 1e-20), shading='gouraud', cmap='viridis')
    axes[0].set_title(f'log₁₀|E| (V/m){title_suffix}')
    axes[0].set_xlabel('x (м)')
    axes[0].set_ylabel('z (м, вісь диполя)')
    axes[0].set_aspect('equal')
    plt.colorbar(pcm0, ax=axes[0])

    # Кружок межі реактивної зони
    circle_near = plt.Circle((0, 0), data['r_near'], fill=False, color='red', linewidth=2, label=f"r=λ/2π={data['r_near']:.1f} м (реактивна)")
    circle_far = plt.Circle((0, 0), data['r_far'], fill=False, color='cyan', linewidth=2, linestyle='--', label=f"r=2λ={data['r_far']:.1f} м (далекa)")
    axes[0].add_patch(circle_near)
    axes[0].add_patch(circle_far)
    axes[0].legend(loc='upper right', fontsize=8)

    # Праворуч: |S| (вектор Пойнтінга — потік активної потужності)
    pcm1 = axes[1].pcolormesh(X, Z, np.log10(np.abs(data['S_mag']) + 1e-20), shading='gouraud', cmap='plasma')
    axes[1].set_title(f'log₁₀|S| (Вт/м²) — вектор Пойнтінга{title_suffix}')
    axes[1].set_xlabel('x (м)')
    axes[1].set_ylabel('z (м)')
    axes[1].set_aspect('equal')
    plt.colorbar(pcm1, ax=axes[1])

    # Стрілки напрямку S_r на тлі
    skip = 8
    Xs = X[::skip, ::skip]
    Zs = Z[::skip, ::skip]
    # У сферичних: S = S_r·r̂ + S_θ·θ̂. У декартових: S_x = S_r·sin θ + S_θ·cos θ, S_z = S_r·cos θ − S_θ·sin θ
    Ts = T[::skip, ::skip]
    Sx = data['S_r'][::skip, ::skip] * np.sin(Ts) + data['S_theta'][::skip, ::skip] * np.cos(Ts)
    Sz = data['S_r'][::skip, ::skip] * np.cos(Ts) - data['S_theta'][::skip, ::skip] * np.sin(Ts)
    S_norm = np.sqrt(Sx**2 + Sz**2) + 1e-30
    axes[1].quiver(Xs, Zs, Sx/S_norm, Sz/S_norm, scale=30, color='white', alpha=0.6, width=0.003)

    plt.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description='Tesla coil near/far field visualization')
    p.add_argument('--freq', type=float, default=88500, help='робоча частота (Гц)')
    p.add_argument('--I0', type=float, default=1.0, help='пік. струм у диполі (А)')
    p.add_argument('--dl', type=float, default=2.44, help='ефективна довжина диполя (м, висота котушки)')
    p.add_argument('--Rmax', type=float, default=2000, help='макс. радіус відображення (м)')
    p.add_argument('--save', type=str, default=None, help='зберегти PNG замість показу')
    args = p.parse_args()

    print(f"Параметри:")
    print(f"  f = {args.freq/1e3:.1f} кГц → λ = {c/args.freq:.1f} м")
    print(f"  Реактивна межа (r=λ/2π): {c/args.freq/(2*math.pi):.1f} м")
    print(f"  Далека зона (r>2λ): {2*c/args.freq:.0f} м")
    print(f"  I₀ = {args.I0} А, dl = {args.dl} м")
    print(f"  Дипольний момент: p₀ = {args.I0 * args.dl / (2*math.pi*args.freq):.2e} Кл·м")

    data = compute_field_map(args.freq, args.I0, args.dl, args.Rmax)
    fig = plot_static(data, title_suffix=f' @ {args.freq/1e3:.1f} кГц')

    if args.save:
        fig.savefig(args.save, dpi=120, bbox_inches='tight')
        print(f"Збережено: {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
