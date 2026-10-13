"""Разбор стоимости поиска по записям запусков с полями steps, t_int, t_match (батчи B13 и далее).

Для каждого задания: N_0.9 (интегрирования), S_0.9 (шаги интегратора) и T_0.9 (время, с) — квантили уровня 0.9
по запускам; шаги и время до попадания пропорциональны доле пачки до попадания (steps · found_at / n_eval), чтобы
размер последней пачки не входил в стоимость; среднее число шагов и время на одну оценку; доля накладных расходов
стратегии (время запуска минус интегрирование и сопоставление). Выигрыши G_N, G_S, G_T — относительно перебора с
тем же видом оценки (полный счёт или двухэтапная) и относительно лучшего перебора (двухэтапная оценка).

Перебор, для которого задано только задание замера стоимости (бюджет без попаданий), оценивается как
N_0.9 = ln 10 / p, S_0.9 = N_0.9 · (шаги на оценку), T_0.9 = N_0.9 · (время на оценку) — у перебора оценки
независимы и одинаково распределены.

    python experiments/e54_cost_table.py B13
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import e50_batches as plan  # noqa: E402

OUT = ROOT / "results"
# точные доли цели для заданий-замеров перебора: две независимые копии Чуа II — p = (p_a + p_b)^2 (e02, 5·10^4 точек)
P_EXACT = {"coupled_chua_II_k2_eps0_c0": (0.00142 + 0.00126) ** 2, "coupled_chua_II_k3_eps0_c0": (0.00142 + 0.00126) ** 3,
           "coupled_chua_II_k2_eps0_c0_prod": (0.00142 + 0.00126) ** 2}


def load(bid):
    d = OUT / "e50" / bid
    jobs = {j["id"]: j for j in plan.batches()[bid]["jobs"]}
    recs = {}
    for jid in jobs:
        p = d / f"{jid}.jsonl"
        if not p.exists():
            continue
        rr = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r and "steps" in r:
                    rr[r["seed"]] = r
        if rr:
            recs[jid] = [rr[k] for k in sorted(rr)]
    return jobs, recs


def q09(x):
    x = np.asarray(x, float)
    return float(np.quantile(x, 0.9)) if len(x) else math.nan


def job_stats(recs):
    fa = np.array([r["found_at"] if r["found_at"] is not None else np.inf for r in recs], float)
    ne = np.array([r["n_eval"] for r in recs], float)
    st = np.array([r["steps"] for r in recs], float)
    ts = np.array([r["time_s"] for r in recs], float)
    ti = np.array([r["t_int"] for r in recs], float)
    tm = np.array([r["t_match"] for r in recs], float)
    frac = np.where(np.isfinite(fa), fa / ne, np.inf)
    out = {"R": len(recs), "success": float(np.isfinite(fa).mean()),
           "N09": q09(fa), "S09": q09(st * frac), "T09": q09(ts * frac),
           "steps_per_eval": float(st.sum() / ne.sum()), "ms_per_eval": float(1e3 * ts.sum() / ne.sum()),
           "int_ms_per_eval": float(1e3 * ti.sum() / ne.sum()),
           "overhead_frac": float(max(0.0, ts.sum() - ti.sum() - tm.sum()) / max(ts.sum(), 1e-12)),
           "match_frac": float(tm.sum() / max(ts.sum(), 1e-12))}
    if "n_stopped" in recs[0]:
        out["stopped_frac"] = float(sum(r["n_stopped"] for r in recs) / ne.sum())
        offs = [r.get("screen_off_at") for r in recs]
        out["screen_off_frac"] = float(np.mean([o is not None for o in offs]))
    return out


def main():
    bids = sys.argv[1:] or ["B13"]
    stats, jobs_all = {}, {}
    for bid in bids:
        jobs, recs = load(bid)
        for jid, r in recs.items():
            stats[(bid, jid)] = job_stats(r)
            jobs_all[(bid, jid)] = jobs[jid]
    # перебор без попаданий (замер стоимости) — по точной доле цели
    for key, st in stats.items():
        j = jobs_all[key]
        vn = plan.vname(j["var"])
        if j["strategy"] == "uniform" and st["success"] == 0 and vn in P_EXACT:
            n09 = math.log(10) / P_EXACT[vn]
            st.update({"N09": n09, "S09": n09 * st["steps_per_eval"], "T09": n09 * st["ms_per_eval"] / 1e3,
                       "uniform_from_p": P_EXACT[vn]})
    by_var = {}
    for (bid, jid), st in stats.items():
        j = jobs_all[(bid, jid)]
        by_var.setdefault(plan.vname(j["var"]), []).append((bid, jid.split("__", 1)[1], j, st))
    rows = []
    for vn, items in by_var.items():
        def uni(scr, bid):
            c = [it for it in items if it[2]["strategy"] == "uniform" and bool(it[2].get("screen")) == scr]
            same = [it for it in c if it[0] == bid]
            return (same or c or [None])[0]
        for bid, label, j, st in sorted(items, key=lambda t: (t[1], t[0])):
            scr = bool(j.get("screen"))
            ref = uni(scr, bid) or uni(not scr, bid)
            best = uni(True, bid) or ref
            ref, best = (ref[3] if ref else None), (best[3] if best else None)
            row = {"system": vn, "label": label, "batch": bid, "strategy": j["strategy"], "screen": scr, **st}
            for key, g in (("N09", "G_N"), ("S09", "G_S"), ("T09", "G_T")):
                row[g] = ref[key] / st[key] if ref and st[key] and np.isfinite(st[key]) else None
                row[g + "_best"] = best[key] / st[key] if best and st[key] and np.isfinite(st[key]) else None
            rows.append(row)
    tag = "_".join(bids)
    (OUT / f"e54_cost_{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["| система | вариант | батч | R | успех | N₀.₉ | шагов/оценку | мс/оценку | накл. | ост. | G_N | G_S | G_T | G_T* |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(x, nd=2):
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{nd}f}" if abs(x) < 1e4 else f"{x:.3g}")
    for r in rows:
        lines.append(f"| {r['system']} | {r['label']} | {r['batch']} | {r['R']} | {f(r['success'])} | {f(r['N09'], 0)} | "
                     f"{f(r['steps_per_eval'], 0)} | {f(r['ms_per_eval'], 3)} | {f(100 * r['overhead_frac'], 0)}% | "
                     f"{f(100 * r['stopped_frac'], 0) + '%' if 'stopped_frac' in r else '—'} | {f(r['G_N'])} | {f(r['G_S'])} | "
                     f"{f(r['G_T'])} | {f(r['G_T_best'])} |")
    md = "\n".join(lines)
    (OUT / f"e54_cost_{tag}.md").write_text(md + "\n\nG_T* — по времени относительно лучшего перебора (двухэтапная оценка).\n",
                                           encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
