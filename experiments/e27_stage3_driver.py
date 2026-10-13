"""Этап 3, часть 1: повтор уровней A/B с R = 50 (статистика сравнения на p ~ 0.01–0.03) и новые
отображения Jiang et al. (2016). Результаты: results/e20_search_<system>_R50.json; журнал
results/stage3_log.txt. С возобновлением.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TAG = "_R50"
S_FLOW = ["uniform", "de_zelinka", "sus", "portfolio_no_sus", "portfolio", "portfolio_cvs"]
S_MAP = ["uniform", "de_zelinka", "sus", "portfolio_no_sus", "portfolio"]
CONFIGS = [
    ("se11", S_FLOW, 3000), ("le1", S_FLOW, 8000), ("chua_I", S_FLOW, 1000), ("jafari2026_lc", S_FLOW, 3000),
    ("huynh2019", S_MAP, 3000), ("jiang_nfe", S_MAP, 3000), ("jiang_nff", S_MAP, 3000), ("jiang_nfg", S_MAP, 3000),
    ("jiang_sfa", S_MAP, 3000), ("jiang_sfb", S_MAP, 3000),
    ("chua_II", S_FLOW, 6000),
]


def done(system):
    p = ROOT / "results" / f"e20_search_{system}{TAG}.json"
    if not p.exists():
        return set()
    return {k for k in json.loads(p.read_text(encoding="utf-8")) if k != "_meta"}


def main(R=50):
    log = ROOT / "results" / "stage3_log.txt"
    for system, strategies, budget in CONFIGS:
        todo = [s for s in strategies if s not in done(system)]
        if not todo:
            continue
        t0 = time.time()
        cmd = [PY, str(ROOT / "experiments" / "e20_search_benchmark.py"), system, "--strategies", ",".join(todo),
               "--R", str(R), "--budget", str(budget), "--tag", TAG]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        with log.open("a", encoding="utf-8") as f:
            f.write(f"=== {system} {todo} budget={budget} R={R} ({time.time() - t0:.0f} s) rc={res.returncode}\n")
            f.write(res.stdout[-3000:])
            if res.returncode != 0:
                f.write(res.stderr[-3000:])
        print(system, "done", round(time.time() - t0), "s rc", res.returncode, flush=True)


if __name__ == "__main__":
    main()
