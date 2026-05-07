# Stage 2 — симуляційний стек

Інструменти для проектування і моделювання Tesla-котушки з потрійним резонансом.

## Файли

| Файл | Призначення |
|---|---|
| `coil_geometry_optimizer.py` | Дизайн вторинної + первинної + top-load. Підказка під f_target |
| `triple_resonance_de_queiroz.cir` | SPICE-нетлист з варіативним заземленням (rg=5 для A, 1 МОм для C) |
| `spark_pump_model.py` | Динаміка накачки контуру через іскровий розрядник (Python-аналог Tesla_proc.xls) |

## Швидкий старт

```bash
# 1. Дизайн котушки під f=88.5 кГц (Colorado Springs)
python3 coil_geometry_optimizer.py --f-target 88500

# 2. Симуляція накачки
python3 spark_pump_model.py --f-res 88500 --c-mmc 17.5e-9 --v-break 10000

# 3. SPICE-симуляція повного контуру (потрібен ngspice)
ngspice triple_resonance_de_queiroz.cir
```

## Відповідність XLS-інструментам

| Original XLS | Наш Python-аналог | Перевага |
|---|---|---|
| `Tesla_proc.xls` (Romancorp) | `spark_pump_model.py` | scipy RK45 замість Forward Euler; параметризація під 88.5 кГц |
| `Calculator.xls` (Ілчук) | `coil_geometry_optimizer.py` | додано інверсну задачу під f_target |
| `Iskra.xls` | (спрощено в spark_pump) | моделює лише R_arc=const; для R(t) — у наступній ітерації |

## Адаптація параметрів під наш Stage 2

Поточні значення (з плану):
- f_target = 88.5 кГц (Colorado Springs)
- C_MMC = 17.5 нФ
- V_breakdown = 10 кВ
- NST: 15 кВ / 30 мА (50 Гц залізний)
- Lp ≈ 184 мкГн (з резонансу)

Зауваги після першого прогону:
- При Ns=1600, Hs=570 мм, тороїд 40×5 см: f_secondary = 80.5 кГц (на 9% нижче за target)
- Варіанти підгонки: зменшити Ns до 1455, або тороїд до 33 см, або підняти top-load (Ctl менше)

Ось чому ми робимо T=1 (тюнінг через відвід первинки) — щоб не перенамотувати вторинку при невеликих розбіжностях.
