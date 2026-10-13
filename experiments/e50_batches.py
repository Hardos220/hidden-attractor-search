"""Батчи расчётов для статьи о методе (план Р1–Р6, 2026-10-09).

Каждый батч — список заданий «вариант системы × стратегия × R запусков». Запуск r использует
генератор default_rng(1000 + r), как e20_search_benchmark.py, поэтому первые запуски повторяют
прежние результаты (это проверяется автоматически, см. поле compare).

Вариант системы задаётся словарём:
    {"system": "se11"}                                   — система как есть (реестр или синтетика e20)
    {"system": "se11", "box": 8}                         — область B, увеличенная в k раз от центра
    {"system": "chua_II", "horizon": 2.0}                — T_tr и T_obs, умноженные на h
    {"coupled": "se11", "k": 3, "eps": 0.01, "coord": 1} — k связанных копий (hsearch.systems.coupled)

Оценка времени: мс на одно интегрирование и среднее число интегрирований на запуск взяты из
results/e20_search_*.json (time_s / суммарный бюджет) — это машинное время на этом ПК.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# доли бассейнов целей в исходных областях (e02 и табл. I статьи)
P_BASE = {"chua_I": 0.035, "chua_II": 0.0027, "se11": 0.0079, "le1": 0.0016, "jafari2026_lc": 0.009,
          "jafari2026_chaos": 3e-4, "huynh2019": 0.0091}
UNIFORM_BATCH = 500        # равномерный перебор: found_at не зависит от размера пачки (тот же поток чисел)


def _old(name):
    p = RES / f"e20_search_{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _measured(name, strat):
    """(мс на интегрирование, среднее число интегрирований на запуск) по старому файлу; None, если нет."""
    d = _old(name)
    if not d or strat not in d or "time_s" not in d[strat]:
        return None
    B = d["_meta"]["budget"]
    fa = d[strat]["found_at"]
    ev = sum(x if x is not None else B for x in fa)
    return 1e3 * d[strat]["time_s"] / max(ev, 1), ev / len(fa)


def _ms(system, strat, files=()):
    """Стоимость одного интегрирования для оценки: по старым результатам системы, иначе по похожей."""
    for f in list(files) + [f"{system}_R50", system]:
        m = _measured(f, strat)
        if m:
            return m[0]
    for f in list(files) + [f"{system}_R50", system]:
        m = _measured(f, "portfolio" if strat not in ("uniform",) else "uniform")
        if m:
            return m[0]
    return 1.0


def vname(var):
    if "coupled" in var:
        return (f"coupled_{var['coupled']}_k{var['k']}_eps{var.get('eps', 0.0):g}_c{var.get('coord', 0)}"
                + ("_prod" if var.get("products") else "") + ("_eq" if var.get("alleq") else ""))
    n = var["system"]
    if var.get("box"):
        n += f"_box{var['box']:g}"
    if var.get("horizon"):
        n += f"_h{var['horizon']:g}"
    if var.get("dt"):
        n += f"_dt{var['dt']:g}"
    if var.get("par"):
        n += f"_p{var['par'][0]}_{var['par'][1]:g}"
    return n


def job(var, strategy, R, budget, targets=None, est_evals=None, est_ms=None, compare=None, batch=None, upper_evals=None,
        screen=None, n_init=50, tag=None):
    """upper_evals — число интегрирований на запуск в худшем случае (метод не выигрывает); по умолчанию
    для повторов прежних конфигураций берётся 1.3 × оценка (разброс по запускам). screen — двухэтапная оценка
    (harness.Problem.screen: True, dict или "auto"); у таких заданий к имени добавляется __scr."""
    j = {"var": var, "strategy": strategy, "R": int(R), "budget": int(budget), "targets": targets,
         "batch": batch or (UNIFORM_BATCH if strategy == "uniform" else 20), "n_init": int(n_init),
         "est_evals": est_evals, "est_ms": est_ms, "compare": compare,
         "upper_evals": upper_evals if upper_evals is not None else (1.3 * est_evals if est_evals else None)}
    j["id"] = f"{vname(var)}__{strategy}"
    if screen:
        j["screen"] = screen
        j["id"] += "__scr"
    if tag:
        j["id"] += f"__{tag}"
    return j


def _replicate(old_name, strategies, R, var=None):
    """Повтор прежней конфигурации (бюджет и цели из старого файла) с R запусками."""
    d = _old(old_name)
    m = d["_meta"]
    base = old_name.replace("_R50", "").replace("_rob", "")
    var = var or {"system": base}
    out = []
    for s in strategies:
        meas = _measured(old_name, s)
        out.append(job(var, s, R, m["budget"], targets=m.get("targets"),
                       est_evals=meas[1] if meas else None, est_ms=meas[0] if meas else None,
                       compare={"file": old_name, "strategy": s}))
    return out


# ---------------------------------------------------------------------------- батчи
def batches():
    B = {}

    # B0 — проверка перед длинными расчётами (минуты): воспроизводимость прежних запусков,
    # новые варианты систем и новая стратегия на коротких бюджетах. Запускать первым.
    b0 = []
    b0 += [dict(j, R=3) for j in _replicate("se11_R50", ["uniform", "portfolio"], 3)]
    b0 += [dict(j, R=3) for j in _replicate("le1_R50", ["sus"], 3)]
    b0 += [dict(j, R=3) for j in _replicate("edge_se3_m3_p0.001", ["portfolio"], 3)]
    b0 += [dict(j, R=2) for j in _replicate("chua_I_R50", ["portfolio_cvs"], 2)]
    b0.append(dict(job({"system": "se11"}, "uniform", 3, 3000, targets=["HA"], batch=20,
                       compare={"file": "se11_R50", "strategy": "uniform"}), id="se11__uniform_batch20"))
    b0.append(job({"system": "se11", "box": 2}, "uniform", 2, 3000, targets=["HA"]))
    b0.append(job({"system": "se11", "horizon": 0.5}, "portfolio", 2, 3000, targets=["HA"]))
    b0.append(job({"coupled": "se11", "k": 2, "eps": 0.01, "coord": 1}, "uniform", 2, 2000, targets=["HA"]))
    b0.append(job({"system": "se11"}, "edge_bisect", 2, 3000, targets=["HA"]))
    b0.append(job({"system": "se11"}, "edge_bisect_u20", 2, 3000, targets=["HA"]))
    B["B0"] = {"title": "проверка: воспроизводимость, новые варианты систем и стратегия (минуты)", "jobs": b0}

    # B1 — Р2: все прежние сравнения заново с R = 100 (первые 50/20 запусков сверяются со старыми)
    b1 = []
    ab = ["uniform", "de_zelinka", "sus", "portfolio_no_sus", "portfolio", "portfolio_cvs"]
    for f in ("chua_I_R50", "chua_II_R50", "se11_R50", "le1_R50", "jafari2026_lc_R50"):
        b1 += _replicate(f, ab, 100)
    b1 += _replicate("huynh2019_R50", ["uniform", "de_zelinka", "sus", "portfolio_no_sus", "portfolio"], 100)
    b1 += _replicate("jafari2026_chaos", ["uniform", "de_zelinka", "sus", "portfolio_no_sus", "portfolio", "portfolio_cvs"], 100)
    for f in ("jiang_nfe_R50", "jiang_nff_R50", "jiang_nfg_R50", "jiang_sfa_R50", "jiang_sfb_R50"):
        b1 += _replicate(f, ["uniform", "de_zelinka", "sus", "portfolio_no_sus", "portfolio"], 100)
    # синтетика n = 6: скейлинг по p (D2 и D1) и абляция
    d2 = ["uniform", "sus_tesc", "sus_rarity", "sus", "portfolio_no_sus", "portfolio_no_ab", "portfolio"]
    for p in ("0.01", "0.001", "0.0001", "1e-05"):
        b1 += _replicate(f"edge_se3_m3_p{p}", d2, 100)
    for p in ("0.01", "0.001", "0.0001"):
        b1 += _replicate(f"emb_se3_m3_p{p}", ["uniform", "sus", "portfolio_no_sus", "portfolio"], 100)
    # зависимость от размерности (D2, p = 1e-4)
    for m in (1, 5, 7, 9, 11):
        b1 += _replicate(f"edge_se3_m{m}_p0.0001", d2, 100)
    # абляция на реальных системах (варианты, которые считались только с R = 20)
    for f in ("chua_I", "chua_II", "se11", "le1", "jafari2026_lc", "huynh2019"):
        old = _old(f)
        have = [s for s in ("sus_tesc", "sus_rarity", "portfolio_no_ab", "portfolio_eps0") if old and s in old]
        b1 += _replicate(f, have, 100)
    # робастность к гиперпараметрам: R = 50 вместо 20
    rob = ["portfolio_eps01", "portfolio_eps03", "portfolio_p01", "portfolio_p03", "portfolio_sig03", "portfolio_n100"]
    for f in ("chua_II_rob", "le1_rob", "edge_se3_m3_p0.0001_rob"):
        b1 += _replicate(f, rob, 50)
    syn = [j for j in b1 if j["var"].get("system", "").startswith(("edge_", "emb_")) and "_rob" not in (j.get("compare") or {}).get("file", "")
           and not j["strategy"].startswith("portfolio_eps0") and j["strategy"] not in rob]
    abl = [j for j in b1 if (j.get("compare") or {}).get("file") in ("chua_I", "chua_II", "se11", "le1", "jafari2026_lc", "huynh2019")
           or j["strategy"] in rob]
    ab_ = [j for j in b1 if j not in syn and j not in abl]
    B["B1a"] = {"title": "Р2: уровни A/B и отображения Цзяна, R = 100", "jobs": ab_}
    B["B1b"] = {"title": "Р2: синтетика — скейлинг D2/D1 по p и по размерности, R = 100", "jobs": syn}
    B["B1c"] = {"title": "Р2: абляция на реальных системах (R = 100) и робастность (R = 50)", "jobs": abl}

    # B2 — Р1: реальные системы в увеличенных областях B_k = k·B, p_k ≈ p/k³ (фактическое p меряет перебор)
    def box_jobs(system, ks, R=50):
        out = []
        files = [f"{system}_R50", system]
        G = {}
        for f in files:
            d = _old(f)
            if d and "uniform" in d:
                import numpy as np
                q = lambda fa: float(np.quantile([np.inf if x is None else x for x in fa], 0.9))  # noqa: E731
                nu = q(d["uniform"]["found_at"])
                G = {s: max(nu / q(d[s]["found_at"]), 1 / 3) for s in ("sus", "portfolio") if s in d}
                break
        for k in ks:
            p = P_BASE[system] / k ** 3
            var = {"system": system, "box": k}
            out.append(job(var, "uniform", R, math.ceil(6 / p), targets=None, est_evals=1 / p,
                           est_ms=_ms(system, "uniform", files), upper_evals=1.5 / p))
            for s in ("sus", "portfolio"):
                out.append(job(var, s, R, math.ceil(8 / p), est_evals=1 / (p * G.get(s, 1.0)),
                               est_ms=_ms(system, s, files) * (1.5 if s == "portfolio" else 1.0), upper_evals=2 / p))
        return out
    b2a = box_jobs("se11", [2, 4, 8]) + box_jobs("le1", [2, 4]) + box_jobs("chua_I", [2, 4]) + box_jobs("jafari2026_lc", [2])
    b2b = box_jobs("se11", [16]) + box_jobs("le1", [8]) + box_jobs("chua_I", [8]) + box_jobs("chua_II", [2, 4]) \
        + box_jobs("jafari2026_lc", [4])
    B["B2a"] = {"title": "Р1, быстрая часть: SE11 (k ≤ 8), LE1 и Чуа I (k ≤ 4), цикл Jafari (k = 2)", "jobs": b2a}
    part = lambda name, ks: [j for j in b2b if j["var"]["system"] == name and j["var"]["box"] in ks]  # noqa: E731
    B["B2b1"] = {"title": "Р1, длинная часть: SE11 k = 16", "jobs": part("se11", [16])}
    B["B2b2"] = {"title": "Р1, длинная часть: LE1 k = 8", "jobs": part("le1", [8])}
    B["B2b3"] = {"title": "Р1, длинная часть: Чуа I k = 8", "jobs": part("chua_I", [8])}
    B["B2b4"] = {"title": "Р1, длинная часть: Чуа II k = 2 и 4", "jobs": part("chua_II", [2, 4])}
    B["B2b5"] = {"title": "Р1, длинная часть: цикл Jafari k = 4 (контроль)", "jobs": part("jafari2026_lc", [4])}

    # B3 — Р3: асимптотика скейлинга D2 (n = 6) до p = 1e-7; D1 до 1e-5 — отдельно (дольше, сигнала нет)
    b3 = []
    cu = _measured("edge_se3_m3_p1e-05", "uniform")[0]
    for p, grow in ((1e-6, 3.0), (1e-7, 9.0)):
        var = {"system": f"edge_se3_m3_p{p:g}"}
        b3.append(job(var, "uniform", 100, math.ceil(6 / p), targets=["HA"], est_evals=1 / p, est_ms=cu))
        for s in ("sus", "sus_tesc", "portfolio"):
            meas = _measured("edge_se3_m3_p1e-05", s)
            bud = int(2e5 if p == 1e-6 else 1e6)
            b3.append(job(var, s, 100, bud, targets=["HA"], est_evals=meas[1] * grow, est_ms=meas[0], upper_evals=0.3 * bud))
    B["B3"] = {"title": "Р3: D2 (n = 6) при p = 1e-6 и 1e-7, R = 100", "jobs": b3}
    b3d = []
    var = {"system": "emb_se3_m3_p1e-05"}
    cu1 = _measured("emb_se3_m3_p0.0001", "uniform")[0]
    b3d.append(job(var, "uniform", 100, 600000, targets=["HA"], est_evals=1e5, est_ms=cu1))
    for s in ("sus", "portfolio_no_sus", "portfolio"):
        b3d.append(job(var, s, 100, 800000, targets=["HA"], est_evals=1e5, est_ms=_measured("emb_se3_m3_p0.0001", s)[0]))
    B["B3d"] = {"title": "Р3, дополнительно: D1 (n = 6) при p = 1e-5, R = 100", "jobs": b3d}

    # B4 — Р4: k связанных копий SE11, связь среднего поля по y с eps = 0.01. Скан 2026-10-09: HA^k выживает
    # 4 горизонта интегрирования при eps <= 0.003 по x и eps <= 0.01 по y и z (k = 2, 3); при eps = 0.02 по x
    # связанный HA^2 сваливается в равновесие
    p1 = P_BASE["se11"]
    ms_u, ms_p = _ms("se11", "uniform"), _ms("se11", "portfolio") * 1.5
    b4 = []
    v2 = {"coupled": "se11", "k": 2, "eps": 0.01, "coord": 1}
    b4.append(job(v2, "uniform", 50, math.ceil(6 / p1 ** 2), targets=["HA"], est_evals=1 / p1 ** 2, est_ms=2 * ms_u))
    for s in ("sus", "portfolio"):
        b4.append(job(v2, s, 50, math.ceil(8 / p1 ** 2), targets=["HA"], est_evals=1 / (10 * p1 ** 2), est_ms=2 * ms_p,
                      upper_evals=2 / p1 ** 2))
    v3 = {"coupled": "se11", "k": 3, "eps": 0.01, "coord": 1}
    b4.append(job(v3, "uniform", 20, math.ceil(6 / p1 ** 3), targets=["HA"], est_evals=1 / p1 ** 3, est_ms=3 * ms_u))
    for s in ("sus", "portfolio"):
        b4.append(job(v3, s, 20, 2_000_000, targets=["HA"], est_evals=1 / (30 * p1 ** 3), est_ms=3 * ms_p,
                      upper_evals=2_000_000))
    B["B4a"] = {"title": "Р4: 2 связанные копии SE11 (n = 6)", "jobs": [j for j in b4 if j["var"]["k"] == 2]}
    B["B4b"] = {"title": "Р4: 3 связанные копии SE11 (n = 9)", "jobs": [j for j in b4 if j["var"]["k"] == 3]}
    v4 = {"coupled": "se11", "k": 4, "eps": 0.0}
    b4c = [job(v4, s, 20, 1_000_000, targets=["HA"], est_evals=1e6, est_ms=4 * ms_p, upper_evals=1e6)
           for s in ("sus", "portfolio")]
    B["B4c"] = {"title": "Р4, дополнительно: 4 несвязанные копии SE11 (n = 12, p = p₁⁴ точно), только стратегии", "jobs": b4c}

    # B5 — Р5: чувствительность к горизонту интегрирования (T_tr и T_obs × 0.5 и × 2)
    b5 = []
    for f, base in (("chua_II_R50", "chua_II"), ("se11_R50", "se11"), ("le1_R50", "le1"), ("edge_se3_m3_p0.0001", "edge_se3_m3_p0.0001")):
        d = _old(f)
        for h in (0.5, 2.0):
            var = {"system": base, "horizon": h}
            for s in ("uniform", "sus", "portfolio"):
                meas = _measured(f, s)
                b5.append(job(var, s, 50, d["_meta"]["budget"], targets=d["_meta"].get("targets"),
                              est_evals=meas[1], est_ms=meas[0] * h))
    B["B5"] = {"title": "Р5: горизонт интегрирования × 0.5 и × 2 (Чуа II, SE11, LE1, D2 p = 1e-4)", "jobs": b5}

    # B6 — Р6: бисекция к границам бассейнов как базовая стратегия
    b6 = []
    for f in ("chua_I_R50", "chua_II_R50", "se11_R50", "le1_R50", "jafari2026_lc_R50", "huynh2019_R50",
              "jafari2026_chaos", "edge_se3_m3_p0.001", "edge_se3_m3_p0.0001", "emb_se3_m3_p0.001"):
        d = _old(f)
        base = f.replace("_R50", "")
        meas = _measured(f, "portfolio") or _measured(f, "uniform")
        for s in ("edge_bisect", "edge_bisect_u20"):
            b6.append(job({"system": base}, s, 50, d["_meta"]["budget"], targets=d["_meta"].get("targets"),
                          est_evals=_measured(f, "uniform")[1], est_ms=meas[0]))
    B["B6"] = {"title": "Р6: бисекция к границам бассейнов (edge tracking) на уровнях A/B и синтетике", "jobs": b6}

    # B7 — Р3, диагностика (2026-10-10): на D2 выигрыш обрывается при p <= 1e-6. Гипотеза: правило застоя
    # sigma < sigma_min = 0.005 ширины области перезапускает каскад, когда радиус бассейна цели rho меньше
    # нескольких минимальных шагов (5.0 шага при 1e-5, 2.3 при 1e-6, 1.1 при 1e-7). Порог в 10 раз меньше и
    # без правила по шагу; 1e-5 — контроль. Бюджет 2e5 одинаковый (перебор в нём: 86 %, 18 %, 2 %)
    b7 = []
    for p in ("1e-05", "1e-06", "1e-07"):
        var = {"system": f"edge_se3_m3_p{p}"}
        for s in ("sus_smin", "sus_nostall", "portfolio_smin"):
            base = "portfolio" if s.startswith("portfolio") else "sus"
            b7.append(job(var, s, 50, 200_000, targets=["HA"], est_evals=2e4,
                          est_ms=_measured("edge_se3_m3_p1e-05", base)[0], upper_evals=200_000))
    B["B7"] = {"title": "Р3, диагностика: D2 при 1e-5…1e-7 с меньшим порогом шага расщепления", "jobs": b7, "threads": 2}

    # B8 — Р4, расширение (2026-10-10): связанные копии дали G ~ 1e3 при n = 9 на R = 20. Для статьи:
    # n = 9 с R = 100; те же копии без связи (eps = 0, p = p1^k с точностью до общего радиуса ухода);
    # 4 копии без связи (n = 12, p = p1^4 ~ 4e-9 — только стратегии, перебор недостижим); бисекция
    # к границам на связанных копиях. Задания uniform считает видеокарта (e51)
    b8 = []
    v3 = {"coupled": "se11", "k": 3, "eps": 0.01, "coord": 1}
    v30 = {"coupled": "se11", "k": 3, "eps": 0.0, "coord": 1}
    v40 = {"coupled": "se11", "k": 4, "eps": 0.0, "coord": 1}
    for v, R in ((v3, 100), (v30, 50)):
        b8.append(job(v, "uniform", R, math.ceil(6 / p1 ** 3), targets=["HA"], est_evals=1 / p1 ** 3, est_ms=3 * ms_u))
        for s in ("sus", "portfolio"):
            b8.append(job(v, s, R, 2_000_000, targets=["HA"], est_evals=5e3, est_ms=3 * ms_p, upper_evals=2e5))
    for s in ("sus", "portfolio"):
        b8.append(job(v40, s, 50, 1_000_000, targets=["HA"], est_evals=5e4, est_ms=4 * ms_p, upper_evals=1e6))
    for k, bud in ((2, math.ceil(8 / p1 ** 2)), (3, 200_000)):
        v = {"coupled": "se11", "k": k, "eps": 0.01, "coord": 1}
        b8.append(job(v, "edge_bisect", 50, bud, targets=["HA"], est_evals=bud / 4, est_ms=k * ms_p, upper_evals=bud))
    B["B8"] = {"title": "Р4, расширение: связанные копии SE11 n = 9 (R = 100), без связи n = 9 и 12, бисекция", "jobs": b8}

    # B9 — Р3, диагностика 2 (2026-10-10): B7 показал, что порог шага ни при чём (с меньшим порогом и без него —
    # те же запуски). Гипотеза: изотропный шаг цепей не может одновременно разрешать тонкую координату (шар
    # |w| < rho) и исследовать координаты без градиента (бассейн по x). Проверка — ход с независимой перевыборкой
    # подмножества координат (sus_coord, portfolio_coord); контроль — реальные системы и связанные копии
    b9 = []
    for p in ("0.0001", "1e-05", "1e-06", "1e-07"):
        for s in ("sus_coord", "portfolio_coord"):
            b9.append(job({"system": f"edge_se3_m3_p{p}"}, s, 50, 200_000, targets=["HA"], est_evals=2e4,
                          est_ms=_measured("edge_se3_m3_p1e-05", "portfolio")[0], upper_evals=200_000))
    for s in ("sus_coord", "portfolio_coord"):
        b9.append(job({"system": "edge_se3_m1_p0.0001"}, s, 50, 200_000, targets=["HA"], est_evals=2e4,
                      est_ms=_measured("edge_se3_m1_p0.0001", "portfolio")[0], upper_evals=200_000))
    for f in ("chua_I_R50", "chua_II_R50", "se11_R50", "le1_R50"):
        m = _old(f)["_meta"]
        base = f.replace("_R50", "")
        for s in ("sus_coord", "portfolio_coord"):
            meas = _measured(f, "portfolio")
            b9.append(job({"system": base}, s, 50, m["budget"], targets=m.get("targets"), est_evals=meas[1], est_ms=meas[0]))
    v2 = {"coupled": "se11", "k": 2, "eps": 0.01, "coord": 1}
    for s in ("sus_coord", "portfolio_coord"):
        b9.append(job(v2, s, 50, math.ceil(8 / p1 ** 2), targets=["HA"], est_evals=3e3, est_ms=2 * ms_p, upper_evals=2e4))
    B["B9"] = {"title": "Р3, диагностика 2: ход с перевыборкой координат (D2 1e-4…1e-7, n = 4, контроль)", "jobs": b9, "threads": 2}

    # B10 — Р3, диагностика 3 (2026-10-10): B9 — перевыборка координат помогает лишь частично. Гипотеза — предел
    # дискретизации модели стоимости: вне зоны |w0| < rho / sqrt(2 kappa dt) уход быстрее шага, координата плоская;
    # доля зоны 2.3 % при 1e-5, 0.23 % при 1e-6, 0.023 % при 1e-7. Шаг dt / 4 (1e-6) и dt / 16 (1e-7) возвращает
    # долю зоны к уровню 1e-5; T_tr и T_obs прежние. Перебор не нужен: p не зависит от шага (шар |w0| < rho)
    b10 = []
    for p, f in (("1e-06", 0.25), ("1e-07", 0.0625), ("1e-07", 0.25)):
        var = {"system": f"edge_se3_m3_p{p}", "dt": f}
        for s in ("sus", "portfolio"):
            b10.append(job(var, s, 50, 200_000, targets=["HA"], est_evals=2e4,
                           est_ms=_measured("edge_se3_m3_p1e-05", s)[0] / f, upper_evals=200_000))
    B["B10"] = {"title": "Р3, диагностика 3: D2 при 1e-6 и 1e-7 с меньшим шагом интегрирования", "jobs": b10, "threads": 2}

    # B11 — пилот (2026-10-12): портфель с четвёртым рукавом (бисекция к границам), фиксированный и адаптивный
    # (доли по состоянию каскада расщепления, минимальная доля каждого рукава — четверть неравномерной части).
    # Задачи, где рукава ведут себя по-разному: выигрывает расщепление (Чуа II, D2), бисекция (SE11, D1),
    # сигнала нет (цикл Jafari), граница обманчива (LE1), обманчивая координата (хаос Jafari), связанные копии
    b11 = []
    for f in ("chua_II_R50", "se11_R50", "le1_R50", "jafari2026_lc_R50", "edge_se3_m3_p0.0001", "emb_se3_m3_p0.001",
              "jafari2026_chaos"):
        m = _old(f)["_meta"]
        base = f.replace("_R50", "")
        meas = _measured(f, "portfolio")
        for s in ("portfolio_bis", "portfolio_ad"):
            b11.append(job({"system": base}, s, 50, m["budget"], targets=m.get("targets"), est_evals=meas[1], est_ms=meas[0] * 1.3))
    v2 = {"coupled": "se11", "k": 2, "eps": 0.01, "coord": 1}
    for s in ("portfolio_bis", "portfolio_ad"):
        b11.append(job(v2, s, 50, math.ceil(8 / p1 ** 2), targets=["HA"], est_evals=1e3, est_ms=2 * ms_p, upper_evals=2e4))
    B["B11"] = {"title": "Пилот: портфель с рукавом бисекции, фиксированный и адаптивный", "jobs": b11, "threads": 2}

    # B12 — время счёта (2026-10-12): все стратегии в одинаковых условиях — запуски по одному в процессе с
    # фиксированным числом потоков, без соседних батчей; время до попадания и доля накладных расходов
    b12 = []
    for f in ("chua_II_R50", "se11_R50", "le1_R50", "edge_se3_m3_p0.0001"):
        m = _old(f)["_meta"]
        base = f.replace("_R50", "")
        for s in ("uniform", "sus", "portfolio", "edge_bisect", "portfolio_bis", "portfolio_ad"):
            meas = _measured(f, "portfolio" if s != "uniform" else "uniform") or _measured(f, "uniform")
            b12.append(job({"system": base}, s, 20, m["budget"], targets=m.get("targets"), est_evals=meas[1],
                           est_ms=meas[0] * 1.3, batch=(UNIFORM_BATCH if s == "uniform" else 20)))
    B["B12"] = {"title": "Время счёта: стратегии в одинаковых условиях (R = 20)", "jobs": b12, "threads": 4}

    # B13 — стоимость (2026-10-12): шаги интегратора, время интегрирования, сопоставления и накладные расходы
    # (поля steps, t_int, t_match записи запуска) у перебора, портфеля и портфеля с бисекцией — с полным счётом и с
    # двухэтапной оценкой (screen="auto": досрочная остановка траекторий, уверенно пришедших на известный режим;
    # правило выключает её, если она не окупается). Задачи: одиночные с дешёвым и дорогим интегрированием,
    # три связанные копии SE11 (редкая цель, траектории уходят) и две независимые копии Чуа II (редкая цель,
    # p = p1^2 ≈ 7e-6, траектории ограничены, интегрирование дорогое)
    b13 = []
    for f in ("chua_I_R50", "chua_II_R50", "se11_R50", "le1_R50", "edge_se3_m3_p0.0001"):
        m = _old(f)["_meta"]
        base = f.replace("_R50", "")
        for s in ("uniform", "portfolio", "portfolio_bis"):
            meas = _measured(f, "portfolio" if s != "uniform" else "uniform") or _measured(f, "uniform")
            for scr in (None, "auto"):
                b13.append(job({"system": base}, s, 20, m["budget"], targets=m.get("targets"), est_evals=meas[1],
                               est_ms=meas[0] * 1.3, screen=scr))
    v3 = {"coupled": "se11", "k": 3, "eps": 0.01, "coord": 1}
    for scr in (None, "auto"):
        b13.append(job(v3, "uniform", 20, 12_169_423, targets=["HA"], est_evals=4e6, est_ms=0.006, screen=scr))
        for s in ("portfolio", "portfolio_bis"):
            b13.append(job(v3, s, 20, 2_000_000, targets=["HA"], est_evals=5e3, est_ms=3.0, upper_evals=3e4, screen=scr))
    vc = {"coupled": "chua_II", "k": 2, "eps": 0.0, "coord": 0, "ha": ["HA_a", "HA_b"]}
    tc = ["HA_aa", "HA_ab", "HA_ba", "HA_bb"]
    for scr in (None, "auto"):
        # перебор здесь — только замер стоимости одной оценки (бюджет 2000, без попаданий): N_0.9 = ln 10 / p1^2
        b13.append(job(vc, "uniform", 2, 2000, targets=tc, est_evals=2000, est_ms=40.0, screen=scr, batch=100))
    for s in ("portfolio", "portfolio_bis"):
        b13.append(job(vc, s, 20, 150_000, targets=tc, est_evals=1e4, est_ms=12.0, upper_evals=6e4, screen="auto"))
    for j in b13:
        j["warmup"] = 300           # невременной прогрев процесса перед первым запуском задания (run_one)
    B["B13"] = {"title": "Стоимость: шаги, время интегрирования и накладные расходы; двухэтапная оценка", "jobs": b13,
                "threads": 4}

    # B14 (2026-10-12): (а) три независимые копии Чуа II — редкая ограниченная цель, p = p1^3 ≈ 1.9e-8, перебор
    # недоступен (N_0.9 ≈ 1.2e8 оценок) и оценивается по p и замеру стоимости; (б) ширина пачки основного портфеля —
    # n_init и batch в 2 и 4 раза больше (во столько же раз больше цепей расщепления и точек рукава границ): насколько
    # растёт число оценок при пачке, достаточной для многих ядер; (в) связанные копии SE11 с эталонами-произведениями
    # (products): смешанные состояния известны, двухэтапная оценка может их останавливать; (г) портфель с быстрым
    # kNN (portfolio_fast) там, где накладные расходы велики (B13: 22-62 %). Везде двухэтапная оценка (auto)
    b14 = []
    vc3 = {"coupled": "chua_II", "k": 3, "eps": 0.0, "coord": 0, "ha": ["HA_a", "HA_b"]}
    tc3 = [f"HA_{a}{b}{c}" for a in "ab" for b in "ab" for c in "ab"]
    # (а) вынесено в B17: запуски на трёх копиях Чуа II идут около часа и задерживали бы остальные задания
    vc = {"coupled": "chua_II", "k": 2, "eps": 0.0, "coord": 0, "ha": ["HA_a", "HA_b"]}
    tc = ["HA_aa", "HA_ab", "HA_ba", "HA_bb"]
    v2 = {"coupled": "se11", "k": 2, "eps": 0.01, "coord": 1}
    v3 = {"coupled": "se11", "k": 3, "eps": 0.01, "coord": 1}
    for var, tg, bud, ev, ms in ((vc, tc, 150_000, 2e3, 15.0), (v3, ["HA"], 2_000_000, 4e3, 2.5),
                                 (v2, ["HA"], 200_000, 1e3, 1.0), ({"system": "edge_se3_m3_p0.0001"}, ["HA"], 200_000, 1e3, 0.3),
                                 ({"system": "edge_se3_m3_p1e-05"}, ["HA"], 200_000, 5e3, 0.3)):
        for w in (2, 4):
            b14.append(job(var, "portfolio", 20, bud, targets=tg, est_evals=ev * w, est_ms=ms, upper_evals=bud / 4,
                           screen="auto", n_init=50 * w, batch=20 * w, tag=f"w{w}"))
    for var, tg, bud in ((v2, ["HA"], 200_000), ({"system": "edge_se3_m3_p1e-05"}, ["HA"], 200_000)):
        b14.append(job(var, "portfolio", 20, bud, targets=tg, est_evals=2e3, est_ms=1.0, upper_evals=bud / 4,
                       screen="auto"))                               # ширина 1 (в B13 этих задач нет)
        b14.append(job(var, "uniform", 20, bud * 10, targets=tg, est_evals=bud / 2, est_ms=0.01, screen="auto"))
    v3p = {"coupled": "se11", "k": 3, "eps": 0.01, "coord": 1, "products": True}
    b14.append(job(v3p, "uniform", 20, 12_169_423, targets=["HA"], est_evals=4e6, est_ms=0.006, screen="auto"))
    for s in ("portfolio", "portfolio_fast"):
        b14.append(job(v3p, s, 20, 2_000_000, targets=["HA"], est_evals=4e3, est_ms=2.5, upper_evals=3e4, screen="auto"))
    for f in ("se11_R50", "le1_R50", "edge_se3_m3_p0.0001"):
        m = _old(f)["_meta"]
        b14.append(job({"system": f.replace("_R50", "")}, "portfolio_fast", 20, m["budget"], targets=m.get("targets"),
                       est_evals=500, est_ms=0.4, screen="auto"))
    b14.append(job(v3, "portfolio_fast", 20, 2_000_000, targets=["HA"], est_evals=4e3, est_ms=2.0, upper_evals=3e4,
                   screen="auto"))
    for j in b14:
        j["warmup"] = 300
    B["B14"] = {"title": "Ширина пачки портфеля; эталоны-произведения; быстрый kNN", "jobs": b14, "threads": 4}

    # B17 (2026-10-12): три независимые копии Чуа II (p = p1^3 ≈ 1.9e-8; перебор — замер стоимости), R = 10
    b17 = [job(vc3, "uniform", 1, 1500, targets=tc3, est_evals=1500, est_ms=60.0, screen="auto", batch=100),
           job(vc3, "portfolio", 10, 400_000, targets=tc3, est_evals=1e5, est_ms=30.0, upper_evals=2e5, screen="auto")]
    for j in b17:
        j["warmup"] = 300
    B["B17"] = {"title": "Три копии Чуа II: редкая ограниченная цель (p ≈ 1.9e-8)", "jobs": b17, "threads": 4}

    # B18 (2026-10-12): итоговая конфигурация (портфель с быстрым kNN и двухэтапной оценкой; у связанных систем —
    # эталоны-произведения) там, где её ещё не мерили: Чуа I, Чуа II, две копии Чуа II (с произведениями и без)
    b18 = []
    for f in ("chua_I_R50", "chua_II_R50"):
        m = _old(f)["_meta"]
        b18.append(job({"system": f.replace("_R50", "")}, "portfolio_fast", 20, m["budget"], targets=m.get("targets"),
                       est_evals=100, est_ms=3.0, screen="auto"))
    vcp = dict(vc, products=True)
    b18.append(job(vcp, "uniform", 2, 2000, targets=tc, est_evals=2000, est_ms=15.0, screen="auto", batch=100))
    for var in (vc, vcp):
        b18.append(job(var, "portfolio_fast", 20, 150_000, targets=tc, est_evals=2e3, est_ms=15.0, upper_evals=6e4,
                       screen="auto"))
    b18.append(job({"system": "edge_se3_m3_p1e-05"}, "portfolio_fast", 20, 200_000, targets=["HA"], est_evals=2e3,
                   est_ms=0.3, upper_evals=5e4, screen="auto"))
    v2p = dict(v2, products=True)
    b18.append(job(v2p, "uniform", 20, 2_000_000, targets=["HA"], est_evals=4e4, est_ms=0.01, screen="auto"))
    for var in (v2, v2p):
        b18.append(job(var, "portfolio_fast", 20, 200_000, targets=["HA"], est_evals=1e3, est_ms=0.6, upper_evals=5e4,
                       screen="auto"))
    for j in b18:
        j["warmup"] = 300
    B["B18"] = {"title": "Итоговая конфигурация: Чуа I, Чуа II, две копии Чуа II", "jobs": b18, "threads": 4}

    # B19 (2026-10-13): метод на реальных системах с естественно редкими целями (сканы e56, e57 и перепись e55):
    # (а) LE1 при a = 21, где бассейн скрытого аттрактора сужается до p ≈ 2.4e-4 (продолжение по параметру);
    # (б) две цепи Чуа II со связью 0.1 по x (все равновесия, эталоны-произведения): редкое скрытое семейство «одна
    # цепь на скрытом хаосе HA, другая около равновесия», p ~ 1e-4 (двойного HA при этой связи нет);
    # (в) там же — новый скрытый хаотический аттрактор, рождённый связью (одна цепь на цикле LC, другая в хаосе,
    # которого нет у отдельной цепи), p ≈ 0.016. Итоговая конфигурация метода и перебор, оба с двухэтапной оценкой
    b19 = []
    v_le = {"system": "le1", "par": [0, 21.0], "known_from": "e57_scan_le1_p0_up"}
    b19.append(job(v_le, "uniform", 20, 200_000, targets=["HA"], est_evals=6e3, est_ms=0.02, screen="auto"))
    b19.append(job(v_le, "portfolio_fast", 20, 200_000, targets=["HA"], est_evals=2e3, est_ms=0.3, upper_evals=2e4,
                   screen="auto"))
    vc01 = {"coupled": "chua_II", "k": 2, "eps": 0.1, "coord": 0, "ha": ["HA_a", "HA_b"], "products": True, "alleq": True}
    t_rare = ["HA_aa", "HA_ab", "HA_ba", "HA_bb", "P_HA_a-E2", "P_E1-HA_a", "P_E2-HA_b"]
    b19.append(job(vc01, "uniform", 20, 300_000, targets=t_rare, est_evals=2e4, est_ms=9.0, upper_evals=3e4, screen="auto"))
    b19.append(job(vc01, "portfolio_fast", 20, 300_000, targets=t_rare, est_evals=5e3, est_ms=10.0, upper_evals=3e4,
                   screen="auto"))
    t_new = ["P_LC-HA_a", "P_HA_a-LC"]
    for s_ in ("uniform", "portfolio_fast"):
        b19.append(job(vc01, s_, 20, 20_000, targets=t_new, est_evals=200, est_ms=10.0, screen="auto", tag="newattr"))
    for j in b19:
        j["warmup"] = 300
    B["B19"] = {"title": "Естественно редкие цели: LE1 при a = 21, две цепи Чуа II со связью 0.1", "jobs": b19, "threads": 4}

    # B20 (2026-10-13): ограничение обучающей выборки рукава границ последними 20 000 точками (portfolio_cap) там, где
    # запуски длинные и kNN съедал до 57 % времени, — три копии Чуа II (сравнение с B17: portfolio__scr, R = 10)
    b20 = [job(vc3, "portfolio_cap", 10, 400_000, targets=tc3, est_evals=1e5, est_ms=20.0, upper_evals=2e5, screen="auto")]
    for j in b20:
        j["warmup"] = 300
    B["B20"] = {"title": "Три копии Чуа II: портфель с ограниченной выборкой kNN", "jobs": b20, "threads": 4}

    # B21 (2026-10-13): время на одном потоке для сравнения с рекуррентным методом Attractors.jl (e59): портфель и
    # перебор с двухэтапной оценкой, R = 20; перебор на связанных цепях Чуа (N_0.9 = 1e5) не считается — его время
    # берётся как N_0.9 × стоимость одной оценки на общих начальных условиях (e59_eval_cost.py cost)
    b21 = []
    for f in ("chua_II_R50", "se11_R50", "le1_R50"):
        m = _old(f)["_meta"]
        base = f.replace("_R50", "")
        for s_ in ("uniform", "portfolio_fast"):
            b21.append(job({"system": base}, s_, 20, m["budget"], targets=m.get("targets"), est_evals=1e3, est_ms=10.0,
                           screen="auto"))
    b21.append(job(vc01, "portfolio_fast", 20, 300_000, targets=t_rare, est_evals=2e3, est_ms=60.0, upper_evals=1e4,
                   screen="auto"))
    for j in b21:
        j["warmup"] = 300
    B["B21"] = {"title": "Время на одном потоке: портфель и перебор (сравнение с Attractors.jl)", "jobs": b21, "threads": 1}

    # B22 (2026-10-13): новый аттрактор «цикл + хаос» как цель, время на одном потоке (для таблицы в статье)
    b22 = []
    for s_ in ("uniform", "portfolio_fast"):
        b22.append(job(vc01, s_, 20, 20_000, targets=t_new, est_evals=200, est_ms=60.0, screen="auto", tag="newattr"))
    for j in b22:
        j["warmup"] = 300
    B["B22"] = {"title": "Новый аттрактор как цель: время на одном потоке", "jobs": b22, "threads": 1}

    # B15 (2026-10-12): время на всех потоках в одном процессе (как у пользователя на одной машине): две копии Чуа II,
    # замер стоимости перебора и портфель ширины 1, 2, 4; R = 10
    b15 = [job(vc, "uniform", 1, 4000, targets=tc, est_evals=4000, est_ms=3.0, screen="auto", batch=500)]
    for w in (1, 2, 4):
        b15.append(job(vc, "portfolio", 10, 150_000, targets=tc, est_evals=2e3 * w, est_ms=3.0, upper_evals=3e4,
                       screen="auto", n_init=50 * w, batch=20 * w, tag=f"w{w}"))
    for j in b15:
        j["warmup"] = 300
    B["B15"] = {"title": "Время на 20 потоках в одном процессе: две копии Чуа II", "jobs": b15, "threads": 20}

    # B16 (2026-10-12): варианты двухэтапной оценки на Чуа II (R = 50) — асимптотические статистики остановленных
    # траекторий по короткому окну этапа 1 (window, как в B13), по эталону опознанного режима (ref) и окно этапа 1
    # полной длины n_obs (m1 = n_obs); полный счёт — контроль; две копии Чуа II с ref (R = 20, сравнение с B13)
    m = _old("chua_II_R50")["_meta"]
    b16 = [job({"system": "chua_II"}, "portfolio", 50, m["budget"], targets=m.get("targets"), est_evals=100, est_ms=10.0)]
    for tg, scr in (("win", {"asym": "window"}), ("ref", {"asym": "ref"}), ("m1full", {"asym": "window", "m1": 40000})):
        b16.append(job({"system": "chua_II"}, "portfolio", 50, m["budget"], targets=m.get("targets"), est_evals=100,
                       est_ms=4.0, screen=scr, tag=tg))
    b16.append(job(vc, "portfolio", 20, 150_000, targets=tc, est_evals=2e3, est_ms=15.0, upper_evals=6e4,
                   screen={"asym": "ref"}, tag="ref"))
    for j in b16:
        j["warmup"] = 300
    B["B16"] = {"title": "Варианты двухэтапной оценки: статистики остановленных траекторий", "jobs": b16, "threads": 4}
    # потоков numba на процесс внутри задачи rmq (процессов = RMQ_CPU // threads): у коротких запусков
    # стратегий пачки по 10–30 точек — 2 потока на процесс; у перебора пачками по 500 и длинных запусков — 4
    for bid in B:
        B[bid].setdefault("threads", 2 if bid.startswith(("B0", "B1", "B5", "B6")) else 4)
    return B


# рекомендуемый порядок: проверки и короткие батчи сначала, длинные потом, дополнительные в конце
ORDER = ["B0", "B5", "B6", "B1a", "B1b", "B1c", "B3", "B2a", "B4a", "B4b", "B2b1", "B2b2", "B2b3", "B2b4", "B2b5", "B7", "B8", "B9", "B10",
         "B3d", "B4c"]
OPTIONAL = {"B3d", "B4c"}


def est_hours(j, upper=False):
    """Машинное время задания (ч): R × интегрирований на запуск × мс на интегрирование."""
    ev = j.get("upper_evals") if upper else j.get("est_evals")
    if j.get("est_ms") is None or ev is None:
        return float("nan")
    return j["R"] * min(j["budget"], ev) * j["est_ms"] / 1e3 / 3600.0
