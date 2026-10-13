"""Этап 3: робастность портфеля к гиперпараметрам (ε, p0, σ0, n0) на трёх задачах с выигрышем.
Результаты: results/e20_search_<system>_rob.json (R = 20); журнал results/stage3_log.txt. С возобновлением.
Перебор и базовый портфель берутся из прогонов R = 50 (общие случайные числа: генератор запуска r
одинаков, поэтому первые 20 запусков сопоставимы).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TAG = "_rob"
VARIANTS = ["portfolio_eps01", "portfolio_eps03", "portfolio_p01", "portfolio_p03", "portfolio_sig03", "portfolio_n100"]
CONFIGS = [("chua_II", 6000), ("le1", 8000), ("edge_se3_m3_p0.0001", 20000)]


def done(system):
    p = ROOT / "results" / f"e20_search_{system}{TAG}.json"
    if not p.exists():
        return set()
    return {k for k in json.loads(p.read_text(encoding="utf-8")) if k != "_meta"}


def main(R=20):
    log = ROOT / "results" / "stage3_log.txt"
    for system, budget in CONFIGS:
        todo = [s for s in VARIANTS if s not in done(system)]
        if not todo:
            continue
        t0 = time.time()
        cmd = [PY, str(ROOT / "experiments" / "e20_search_benchmark.py"), system, "--strategies", ",".join(todo),
               "--R", str(R), "--budget", str(budget), "--tag", TAG]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        with log.open("a", encoding="utf-8") as f:
            f.write(f"=== robustness {system} {todo} budget={budget} R={R} ({time.time() - t0:.0f} s) rc={res.returncode}\n")
            f.write(res.stdout[-3000:])
            if res.returncode != 0:
                f.write(res.stderr[-3000:])
        print(system, "done", round(time.time() - t0), "s rc", res.returncode, flush=True)


if __name__ == "__main__":
    main()
