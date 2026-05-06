# Сліпий аналіз — A/B/C тест заземлення

**Мета:** Запобігти підсвідомому зміщенню при аналізі даних.

---

## Принцип

Аналітик обчислює P_in, P_load, η для кожного запуску **не знаючи** конфігурацію (A/B/C) — тільки run_id.
Зв'язок run_id → конфігурація розкривається **після** того, як таблиця заповнена.

---

## Файлова структура

```
experiments/02_tesla_coil_grounding/data/
├── run_schedule_ENCRYPTED.csv      ← run_id → config (ЗАКРИТИ ДО КІНЦЯ)
├── run_001.csv                     ← сирі дані: V_in(t), I_in(t), V_load(t), I_load(t), T(t)
├── run_002.csv
├── ...
├── run_030.csv
└── analysis/
    ├── step1_power_calculations.py ← обчислення без конфігів
    ├── step2_anonymized_table.csv  ← run_id | P_in | P_load | eta
    └── step3_unblind_results.py   ← після розкриття run_schedule
```

---

## Крок 1 — Обчислення потужностей (сліпе)

Запустити `step1_power_calculations.py` для кожного run_*.csv.

Вхід: `data/run_<id>.csv` з колонками: `t_s, V_in_V, I_in_A, V_load_V, I_load_A, T_cal_C`

```python
import numpy as np
import pandas as pd
import os

def analyze_run(filepath):
    df = pd.read_csv(filepath)
    dt = df['t_s'].diff().mean()

    # Активна потужність = середнє миттєвого добутку
    P_in  = np.mean(df['V_in_V']   * df['I_in_A'])
    P_out = np.mean(df['V_load_V'] * df['I_load_A'])

    # Калориметрична потужність (якщо доступно)
    T_end  = df['T_cal_C'].iloc[-1]
    T_beg  = df['T_cal_C'].iloc[0]
    t_exp  = df['t_s'].iloc[-1] - df['t_s'].iloc[0]
    m_kg   = 1.0  # маса води в калориметрі, кг
    cp     = 4186 # Дж/(кг·К)
    P_cal  = m_kg * cp * (T_end - T_beg) / t_exp if t_exp > 0 else float('nan')

    eta_elec = P_out / P_in if P_in > 0 else float('nan')
    eta_cal  = P_cal / P_in if P_in > 0 else float('nan')

    return {
        'run_id': os.path.basename(filepath).replace('.csv',''),
        'P_in_W': round(P_in, 2),
        'P_out_elec_W': round(P_out, 2),
        'P_out_cal_W': round(P_cal, 2),
        'eta_elec': round(eta_elec, 4),
        'eta_cal':  round(eta_cal,  4),
    }

# Запустити для всіх run_*.csv
data_dir = 'data/'
results = [analyze_run(f'{data_dir}{f}') for f in sorted(os.listdir(data_dir)) if f.startswith('run_') and f.endswith('.csv')]
table = pd.DataFrame(results)
table.to_csv('data/analysis/step2_anonymized_table.csv', index=False)
print(table.to_string())
```

---

## Крок 2 — Анонімізована таблиця (заповнюється сліпо)

Перед тим як відкривати `run_schedule_ENCRYPTED.csv`, подивитись на таблицю і записати від руки:
- "Яке групування видно візуально?"
- "Чи є очевидний кластер з вищою/нижчою η?"
- Попереднє враження (без знання конфігів)

Записати це **письмово** (у файл `data/analysis/impressions_before_unblind.txt`) перед наступним кроком.

---

## Крок 3 — Розкриття (unblinding)

Відкрити `run_schedule_ENCRYPTED.csv`. Приєднати конфіги до таблиці:

```python
schedule = pd.read_csv('data/run_schedule_ENCRYPTED.csv')  # run_id, config
merged   = table.merge(schedule, on='run_id')

# Статистика по групах
summary = merged.groupby('config')['P_out_cal_W'].agg(['mean','std','count'])
print(summary)

# Welch t-test: A vs C
from scipy.stats import ttest_ind
A = merged[merged['config']=='A']['P_out_cal_W']
C = merged[merged['config']=='C']['P_out_cal_W']
t, p = ttest_ind(A, C, equal_var=False)
print(f"A vs C: ΔP = {A.mean() - C.mean():.2f} W, t = {t:.2f}, p = {p:.4f}")

delta = A.mean() - C.mean()
se    = np.sqrt(A.var()/len(A) + C.var()/len(C))
sigma = abs(delta / se)
print(f"Significance: {sigma:.1f}σ")
```

---

## Критерії рішення (нагадування з протоколу)

| σ | Інтерпретація | Дія |
|---|---|---|
| < 2 | Нема ефекту | H₀ прийнята, Капанадзе-напрямок закритий |
| 2–3 | Підозрілий | 10 додаткових прогонів |
| 3–5 | Значущий | Повна калориметрія Рівня 2 + корекція артефактів |
| ≥ 5 | Сильний | Кореляція з атмосферою + повтор через місяць + чорновик публікації |

---

## Архівація (після аналізу)

Всі сирі дані, скрипти, таблиці і висновки — в git-коміт з тегом `grounding-test-v1`.
Навіть при негативному результаті — все зберігається повністю. Негативні результати цінні.
