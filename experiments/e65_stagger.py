"""Базовая стратегия stagger-and-step (Sweet, Nusse, Yorke 2001) на задачах таблицы II статьи (2026-10-13).

Запускает e20_search_benchmark.py со стратегией stagger (R = 50, бюджеты таблицы II: Чуа II 6000, LE1 8000, SE11 3000,
D2 p = 1e-4 60000) с меткой _stagger (results/e20_search_<система>_stagger.json — ничего существующего не перезаписывается),
затем считает N_0.9 с бутстрап-интервалом и выигрыш G относительно перебора статьи (N0_0.9 = 818, 1617, 317, 17416;
интервал G — из интервала N_0.9 при фиксированном N0). Пишет results/e65_stagger.md и .json.

    rmq submit --cpu 1 --ram 2 --time 3h --name e65-stagger -- python experiments/e65_stagger.py
    python experiments/e65_stagger.py --report-only
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from hsearch.search.harness import n_delta_from_runs  # noqa: E402

JOBS = [("se11", 3000, 317, "SE11"), ("le1", 8000, 1617, "LE1"), ("edge_se3_m3_p0.0001", 60000, 17416, "D2, p = 1e-4"),
        ("chua_II", 6000, 818, "Chua II")]
R = 50


def fq(v):
    return "inf" if v is None or not np.isfinite(v) else f"{v:.0f}"


def fg(v):
    return "inf" if v is None or not np.isfinite(v) else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--systems", default=",".join(j[0] for j in JOBS))
    args = ap.parse_args()
    want = args.systems.split(",")
    if not args.report_only:
        for name, budget, _, _ in JOBS:
            if name not in want:
                continue
            p = ROOT / "results" / f"e20_search_{name}_stagger.json"
            if p.exists() and "stagger" in json.loads(p.read_text(encoding="utf-8")):
                print(f"{name}: уже посчитано", flush=True)
                continue
            t0 = time.time()
            cmd = [sys.executable, str(ROOT / "experiments" / "e20_search_benchmark.py"), name, "--strategies", "stagger",
                   "--R", str(R), "--budget", str(budget), "--tag", "_stagger"]
            print(">>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, cwd=ROOT)
            print(f"{name}: {time.time() - t0:.0f} с", flush=True)
    rows = []
    lines = ["# e65: stagger-and-step (Sweet, Nusse, Yorke 2001) на задачах таблицы II", "",
             "Одна цепь по координате «длительность переходного процесса» xi (как у расщепления): предложение x + delta, "
             "|delta| = m·(ширина области), m лог-равномерно в [1e-3, 1]; принимается при росте xi; после 30 отказов подряд — "
             "новая равномерная точка. Полный счёт траекторий (как в таблице II), R = 50, бюджеты таблицы II. "
             "G = N0_0.9 / N_0.9 относительно перебора статьи (N0 фиксировано; интервал G — из бутстрап-интервала N_0.9, "
             "2000 перевыборок запусков).", "",
             "| задача | бюджет | успех | медиана | N_0.9 [95 %] | N0_0.9 (статья) | G [95 %] |", "|---|---|---|---|---|---|---|"]
    for name, budget, n0, label in JOBS:
        p = ROOT / "results" / f"e20_search_{name}_stagger.json"
        st = json.loads(p.read_text(encoding="utf-8")).get("stagger") if p.exists() else None
        if not st:
            lines.append(f"| {label} | {budget} | — | — | не посчитано | {n0} | — |")
            continue
        found = st["found_at"]
        q, lo, hi = n_delta_from_runs(found, n_boot=2000)
        g = n0 / q if np.isfinite(q) else 0.0
        g_lo = n0 / hi if np.isfinite(hi) else 0.0
        g_hi = n0 / lo if np.isfinite(lo) and lo > 0 else float("inf")
        rows.append({"system": name, "label": label, "budget": budget, "R": len(found), "success": st["success_rate"],
                     "median": st["median"], "N09": None if not np.isfinite(q) else q,
                     "N09_ci": [None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi],
                     "N0_ref": n0, "G": g, "G_ci": [g_lo, None if not np.isfinite(g_hi) else g_hi], "time_s": st.get("time_s")})
        lines.append(f"| {label} | {budget} | {st['success_rate']:.2f} | {st['median']} | {fq(q)} [{fq(lo)}, {fq(hi)}] | {n0} | "
                     f"{fg(g)} [{fg(g_lo)}, {fg(g_hi)}] |")
    lines += ["", "Для сравнения (статья, таблица II; портфель): Чуа II 7.36 [6.09, 10.2], SE11 3.10 [2.46, 3.70], LE1 6.24 [3.76, 10.0]; "
                  "D2 p = 1e-4: 17.0 [13.6, 22.2]. Успех < 0.9 означает N_0.9 = inf (цель не найдена в бюджете у > 10 % запусков) и G = 0."]
    (ROOT / "results" / "e65_stagger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "results" / "e65_stagger.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
