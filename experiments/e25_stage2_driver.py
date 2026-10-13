"""Драйвер полного бенчмарка этапа 2 (последовательно, с возобновлением).

Конфигурации: (система, стратегии, бюджет). Пропускает уже посчитанные пары система/стратегия
(results/e20_search_<system>.json). Журнал: results/stage2_log.txt
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

S_FULL = ["uniform", "de_zelinka", "ams", "novelty_me", "sus_tesc", "sus_rarity", "sus",
          "portfolio_no_sus", "portfolio_no_ab", "portfolio_eps0", "portfolio", "portfolio_cvs"]
S_MAP = [s for s in S_FULL if not s.endswith("_cvs")]
S_SWEEP = ["uniform", "sus_tesc", "sus_rarity", "sus", "portfolio_no_sus", "portfolio_no_ab", "portfolio"]
S_SWEEP_D1 = ["uniform", "sus", "portfolio_no_sus", "portfolio"]

CONFIGS = [
    # уровень A / B: редкие цели
    ("chua_II", S_FULL, 6000), ("se11", S_FULL, 3000), ("le1", S_FULL, 8000),
    ("jafari2026_lc", S_FULL, 3000), ("huynh2019", S_MAP, 3000), ("chua_I", S_FULL, 1000),
    # D2 («краевое состояние»): зависимость от p при n = 6
    ("edge_se3_m3_p0.01", S_SWEEP, 2000), ("edge_se3_m3_p0.001", S_SWEEP, 10000),
    ("edge_se3_m3_p0.0001", S_SWEEP, 60000), ("edge_se3_m3_p1e-05", S_SWEEP, 400000),
    # D2: зависимость от размерности при p = 1e-4
    ("edge_se3_m1_p0.0001", S_SWEEP, 60000), ("edge_se3_m5_p0.0001", S_SWEEP, 60000),
    ("edge_se3_m7_p0.0001", S_SWEEP, 60000), ("edge_se3_m9_p0.0001", S_SWEEP, 60000),
    ("edge_se3_m11_p0.0001", S_SWEEP, 60000),
    # D1 («ворота»): зависимость от p и от размерности
    ("emb_se3_m3_p0.01", S_SWEEP_D1, 2000), ("emb_se3_m3_p0.001", S_SWEEP_D1, 12000),
    ("emb_se3_m3_p0.0001", S_SWEEP_D1, 80000), ("emb_se3_m1_p0.0001", S_SWEEP_D1, 80000),
    ("emb_se3_m7_p0.0001", S_SWEEP_D1, 80000),
    # самая редкая реальная цель
    ("jafari2026_chaos", S_FULL, 40000),
]


def done_strategies(system):
    p = ROOT / "results" / f"e20_search_{system}.json"
    if not p.exists():
        return set()
    d = json.loads(p.read_text(encoding="utf-8"))
    return {k for k in d if k != "_meta"}


def main(R=20):
    log = ROOT / "results" / "stage2_log.txt"
    for system, strategies, budget in CONFIGS:
        todo = [s for s in strategies if s not in done_strategies(system)]
        if not todo:
            continue
        t0 = time.time()
        cmd = [PY, str(ROOT / "experiments" / "e20_search_benchmark.py"), system, "--strategies", ",".join(todo),
               "--R", str(R), "--budget", str(budget)]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        with log.open("a", encoding="utf-8") as f:
            f.write(f"=== {system} {todo} budget={budget} ({time.time() - t0:.0f} s) rc={res.returncode}\n")
            f.write(res.stdout[-4000:])
            if res.returncode != 0:
                f.write(res.stderr[-4000:])
        print(system, "done", round(time.time() - t0), "s", "rc", res.returncode, flush=True)


if __name__ == "__main__":
    main()
