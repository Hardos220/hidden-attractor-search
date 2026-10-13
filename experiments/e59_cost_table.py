"""Сводка сравнения по времени с рекуррентным методом Attractors.jl (2026-10-13).

Входы: results/e59/cost_python.json (полный счёт и двухэтапная оценка на одном потоке), results/e59/cost_julia.json
(Attractors.jl на тех же начальных условиях, один поток), сводки B21 (портфель и перебор на одном потоке, R = 20)
и N_0.9 перебора из B13/B19. Время перебора при любом способе оценки — N_0.9 × средняя стоимость оценки
(номер первой удачной точки от способа оценки не зависит). Пишет results/e59/cost_table.md.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e59"
N09_UNIFORM = {"chua_II": ("B13", "chua_II__uniform__scr"), "se11": ("B13", "se11__uniform__scr"),
               "le1": ("B13", "le1__uniform__scr"),
               "coupled_chua_II_eps0.1": ("B19", "coupled_chua_II_k2_eps0.1_c0_prod_eq__uniform__scr")}
PORTFOLIO_B21 = {"chua_II": "chua_II__portfolio_fast__scr", "se11": "se11__portfolio_fast__scr",
                 "le1": "le1__portfolio_fast__scr",
                 "coupled_chua_II_eps0.1": "coupled_chua_II_k2_eps0.1_c0_prod_eq__portfolio_fast__scr"}
UNIFORM_B21 = {"chua_II": "chua_II__uniform__scr", "se11": "se11__uniform__scr", "le1": "le1__uniform__scr"}
RU = {"chua_II": "Чуа II", "se11": "SE11", "le1": "LE1", "coupled_chua_II_eps0.1": "две цепи Чуа II, ε = 0.1"}


def q09(recs, key="found_at"):
    fa = np.array([r[key] if r[key] is not None else np.inf for r in recs], float)
    return float(np.quantile(fa, 0.9))


def t09(recs):
    v = np.array([r["time_s"] * r["found_at"] / r["n_eval"] if r["found_at"] else np.inf for r in recs], float)
    return float(np.quantile(v, 0.9))


def load_runs(bid, jid):
    p = ROOT / "results" / "e50" / bid / f"{jid}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and "error" not in l]


def main():
    py = json.loads((OUT / "cost_python.json").read_text(encoding="utf-8"))
    jl = json.loads((OUT / "cost_julia.json").read_text(encoding="utf-8")) if (OUT / "cost_julia.json").exists() else {}
    lines = ["| задача | p | N₀.₉ перебора | способ оценки | мс/оценку | T₀.₉ перебора, с | портфель: N₀.₉ | T₀.₉, с | G_T |",
             "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for name in py:
        bid, jid = N09_UNIFORM[name]
        n09 = q09(load_runs(bid, jid))
        pu = load_runs("B21", PORTFOLIO_B21[name])
        n09_p = q09(pu) if pu else math.nan
        t09_p = t09(pu) if pu else math.nan
        rows = []
        rows.append(("полный счёт (наш РК4)", py[name]["full"]["ms_per_ic"]))
        rows.append(("двухэтапная оценка (наш РК4)", py[name]["screened"]["ms_per_ic"]))
        uu = load_runs("B21", UNIFORM_B21.get(name, ""))
        if uu:
            rows.append(("двухэтапная, измерено в B21", 1e3 * t09(uu) / n09))
        for key, v in sorted(jl.get(name, {}).items()):
            rows.append((f"Attractors.jl {key} ({v['n_attractors']} аттр.)", v["ms_per_ic"]))
        summary[name] = {"N09_uniform": n09, "N09_portfolio": n09_p, "T09_portfolio_1thr": t09_p, "rows": rows}
        for i, (lab, ms) in enumerate(rows):
            T = n09 * ms / 1e3
            G = T / t09_p if t09_p and np.isfinite(t09_p) else math.nan
            lines.append(f"| {RU[name] if i == 0 else ''} | {py[name]['p_target_sample']:.1e} | {n09:.0f} | {lab} | {ms:.3f} | {T:.1f} | "
                         f"{n09_p:.0f} | {t09_p:.1f} | {G:.2f} |")
    md = "\n".join(lines)
    (OUT / "cost_table.md").write_text(md + "\n", encoding="utf-8")
    (OUT / "cost_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
