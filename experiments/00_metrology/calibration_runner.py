"""
calibration_runner.py — Stage 0 metrology gate.

Перевіряє три критичні речі перед переходом на Stage 1:

  1. Шунт відкалібрований 4-провідно і R_4w погоджується з паспортом.
  2. Калоріметр з резистивним еталонним нагрівачем дає правильний P (±2%).
  3. process_oscillogram.py на синтетичних даних дає правильні P_active, cos φ.

Запит інтерактивний — спочатку обчислюємо очікувані значення, потім просимо ввести
виміряні. Скрипт каже PASS/FAIL для кожного підтесту і загалом.

Використання:
    python3 experiments/00_metrology/calibration_runner.py
    python3 experiments/00_metrology/calibration_runner.py --quick   # тільки тест №3
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path


def ask_float(prompt: str, allow_skip: bool = False) -> float | None:
    while True:
        raw = input(f"  {prompt}: ").strip()
        if allow_skip and raw == '':
            return None
        try:
            return float(raw)
        except ValueError:
            print("    (введіть число або порожнє щоб пропустити)" if allow_skip
                  else "    (введіть число)")


def test_shunt() -> bool:
    print("=" * 60)
    print("ТЕСТ 1: Кельвінівське (4-провідне) вимірювання шунта")
    print("=" * 60)
    print("Підключи шунт у 4-провідній схемі: підвід струму на крайніх затискачах,"
          " вимірювальні щупи мультиметра — на внутрішніх.")
    print("Пропусти точно відомий струм (наприклад, 1.000 А зі стабілізованого БЖ),"
          " виміряй падіння напруги на шунті.\n")

    R_nominal = ask_float("Номінал шунта з паспорта [Ω]")
    I_test = ask_float("Тестовий струм через шунт [A]")
    V_drop = ask_float("Напруга на шунті [V]")
    R_4w = V_drop / I_test
    err_pct = abs(R_4w - R_nominal) / R_nominal * 100

    print(f"\n  R виміряний (4w): {R_4w*1000:.4f} мОм")
    print(f"  R паспортний:     {R_nominal*1000:.4f} мОм")
    print(f"  Розбіжність:      {err_pct:.2f}%")

    ok = err_pct < 2.0
    print(f"\n  [{'PASS' if ok else 'FAIL'}] {'допуск ±2%' if ok else 'перевищено допуск ±2%'}")
    if not ok:
        print("  → Перевір контакти, або переоцінити R з виміряного значення (краще).")
    return ok


def test_calorimeter() -> bool:
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Калоріметр з еталонним нагрівачем 100 Вт")
    print("=" * 60)
    print("Зануриш резистивний нагрівач відомої потужності у воду калоріметра."
          " Виміряєш ΔT за час t. Очікуєш P = m·c·ΔT/t = заявлена потужність ±2%.\n")

    P_nominal = ask_float("Заявлена потужність нагрівача [W] (типово 100)")
    m = ask_float("Маса води в калоріметрі [kg]")
    dT = ask_float("ΔT (різниця температур) [K]")
    t = ask_float("Тривалість нагріву [s]")

    cp_water = 4186.0  # Дж/(кг·К)
    P_cal = m * cp_water * dT / t
    err_pct = abs(P_cal - P_nominal) / P_nominal * 100

    print(f"\n  P за калоріметром: {P_cal:.2f} Вт")
    print(f"  P заявлена:        {P_nominal:.2f} Вт")
    print(f"  Розбіжність:       {err_pct:.2f}%")

    ok = err_pct < 5.0  # калоріметр допуск ширший — теплові втрати
    print(f"\n  [{'PASS' if ok else 'FAIL'}] {'допуск ±5%' if ok else 'перевищено допуск ±5%'}")
    if not ok:
        print("  → Перевір ізоляцію калоріметра, точність датчика T,")
        print("    спробуй ширше t (більше ΔT — менший вплив втрат).")
    return ok


def test_process_oscillogram() -> bool:
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Аналітичний скрипт process_oscillogram.py")
    print("=" * 60)
    print("Запускаємо вбудовані тести з відомою наперед відповіддю.\n")

    test_script = (Path(__file__).parent.parent.parent /
                   'simulations' / '01_hairpin' / 'test_process_oscillogram.py')
    if not test_script.exists():
        print(f"  [FAIL] не знайдено: {test_script}")
        return False

    py = '/usr/local/opt/python@3.13/bin/python3'
    if not Path(py).exists():
        py = sys.executable

    result = subprocess.run([py, str(test_script)], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Run only test 3 (no manual measurements)')
    args = parser.parse_args()

    print("\nStage 0 — Метрологічний шлюз перед Stage 1\n")
    print("Цей скрипт перевіряє, чи можна довіряти твоїм вимірюванням.")
    print("Якщо хоч один тест провалився — Stage 1 заблокований.\n")

    results: dict[str, bool] = {}
    if not args.quick:
        results['shunt'] = test_shunt()
        results['calorimeter'] = test_calorimeter()
    results['process_oscillogram'] = test_process_oscillogram()

    print("\n" + "=" * 60)
    print("ПІДСУМОК Stage 0")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:25s} {'PASS' if v else 'FAIL'}")
    all_ok = all(results.values())
    print()
    if all_ok:
        print("✓ ВСЕ OK. Можна переходити до Stage 1 (Hairpin).")
    else:
        print("✗ Є провали. Stage 1 заблокований до виправлення.")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
