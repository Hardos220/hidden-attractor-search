"""Portfolio with a stagger-and-step arm (2026-10-29): checks the corollary of Proposition 1 of the article.

Strategies (src/hsearch/search/strategies.py):
  portfolio_sas        splitting + active boundary + 10 stagger-and-step chains + uniform share 0.2;
  portfolio_sas_nosus  active boundary + 10 stagger-and-step chains + uniform share 0.2 (stagger replaces splitting);
  stagger_multi        the 10-chain stagger-and-step arm on its own (the autonomous arm of the corollary).
Problems and budgets are those of the article (e20 runs, n_init = 50, batch = 20), R = 50 runs each.
Each (problem, strategy) pair is run by e20_search_benchmark.py in its own process with its own tag, so the outputs
results/e20_search_<problem>_<tag>.json never collide; NUMBA_NUM_THREADS = 1 per process.

    python experiments/e67_portfolio_sas.py --workers 6
    python experiments/e67_portfolio_sas.py --report-only

Writes results/e67_portfolio_sas.{json,md}: N_0.9 with 95% bootstrap intervals, G = N0 / N_0.9 against the fixed
uniform reference of the article, the bound of Proposition 1 for the portfolio, and the D2 scaling exponents.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
R = 50
N_BOOT = 2000
C_BATCH = 50          # largest round of portfolio_sas (10 splitting + 20 boundary + 10 stagger + 10 uniform)

# problem, budget, uniform N0 of the article (as in e66_stagger_summary.py)
PROBLEMS = [
    ("chua_I", 1000, 66), ("chua_II", 6000, 818), ("se11", 3000, 317), ("le1", 8000, 1617),
    ("jafari2026_lc", 3000, 219), ("huynh2019", 3000, 221), ("jafari2026_chaos", 40000, 5265),
    ("emb_se3_m3_p0.001", 12000, 1932),
    ("edge_se3_m3_p0.01", 2000, 277), ("edge_se3_m3_p0.001", 10000, 1932),
    ("edge_se3_m3_p0.0001", 60000, 17416), ("edge_se3_m3_p1e-05", 400000, 189765),
]
STRATS = [("portfolio_sas", "_psas"), ("portfolio_sas_nosus", "_psasn"), ("stagger_multi", "_smulti")]
D2 = [("edge_se3_m3_p0.01", 1e-2), ("edge_se3_m3_p0.001", 1e-3), ("edge_se3_m3_p0.0001", 1e-4), ("edge_se3_m3_p1e-05", 1e-5)]


def out_file(name, tag):
    return RES / f"e20_search_{name}{tag}.json"


def run_one(job):
    name, budget, strat, tag = job
    f = out_file(name, tag)
    if f.exists() and strat in json.loads(f.read_text(encoding="utf-8")):
        return f"{name} {strat}: already done"
    env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    cmd = [sys.executable, str(ROOT / "experiments" / "e20_search_benchmark.py"), name, "--strategies", strat,
           "--R", str(R), "--budget", str(budget), "--tag", tag]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (p.stdout or "").strip().splitlines()[-1:] + (p.stderr or "").strip().splitlines()[-3:]
    return f"{name} {strat}: exit {p.returncode}; " + " | ".join(tail)


def q09(a):
    a = np.asarray(a, float)
    return float(np.quantile(a, 0.9)) if np.mean(np.isfinite(a)) >= 0.9 else math.inf


def stats(found, n0, rng):
    a = np.array([math.inf if x is None else float(x) for x in found], float)
    n9 = q09(a)
    boot = [q09(rng.choice(a, len(a))) for _ in range(N_BOOT)]
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    g = n0 / n9 if math.isfinite(n9) else 0.0
    return {"N09": n9, "N09_lo": lo, "N09_hi": hi, "G": g, "G_lo": n0 / hi if math.isfinite(hi) else 0.0,
            "G_hi": n0 / lo if math.isfinite(lo) else 0.0, "success": float(np.mean(np.isfinite(a))), "R": len(a)}


def load_found(name, strat, tag):
    f = out_file(name, tag)
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    return d.get(strat, {}).get("found_at")


def slope(ns, ps):
    x = np.log(1.0 / np.asarray(ps)); y = np.log(np.asarray(ns))
    return float(np.polyfit(x, y, 1)[0])


def report():
    rng = np.random.default_rng(67)
    e66 = json.loads((RES / "e66_stagger_summary.json").read_text(encoding="utf-8")) if (RES / "e66_stagger_summary.json").exists() else {}
    out = {"problems": {}}
    lines = ["# e67: portfolio with a stagger-and-step arm", "",
             "G = N0/N_0.9 against the uniform reference of the article; 95% bootstrap intervals; R = 50.",
             "Bound (Prop. 1): N_0.9(portfolio_sas) <= min{(N0+c)/0.2, (N_0.9(stagger_multi)+c)/omega}, c = 50,",
             "omega = 0.2 with the boundary arm active (0.4 without it; the smaller share is used).", "",
             "| problem | N0 | portfolio_sas | portfolio_sas_nosus | stagger_multi | bound for portfolio_sas | holds |",
             "|---|---|---|---|---|---|---|"]
    for name, budget, n0 in PROBLEMS:
        row = {}
        for strat, tag in STRATS:
            fa = load_found(name, strat, tag)
            if fa:
                row[strat] = stats(fa, n0, rng)
        out["problems"][name] = row
        if not row:
            continue
        sm = row.get("stagger_multi", {}).get("N09", math.inf)
        bound = min((n0 + C_BATCH) / 0.2, (sm + C_BATCH) / 0.2)
        ps = row.get("portfolio_sas", {}).get("N09", math.inf)
        row["bound_N09"] = bound
        row["holds"] = bool(ps <= bound)

        def cell(s):
            r = row.get(s)
            if not r:
                return "—"
            return f"{r['G']:.2f} [{r['G_lo']:.2f}, {r['G_hi']:.2f}] ({r['success']:.0%})"
        lines.append(f"| {name} | {n0} | {cell('portfolio_sas')} | {cell('portfolio_sas_nosus')} | {cell('stagger_multi')} "
                     f"| {bound:.0f} (N_0.9 {ps:.0f}) | {'yes' if row['holds'] else 'NO'} |")
    sl = {}
    for strat, _ in STRATS:
        ns = [out["problems"].get(n, {}).get(strat, {}).get("N09", math.inf) for n, _ in D2]
        if all(math.isfinite(v) for v in ns):
            sl[strat] = slope(ns, [p for _, p in D2])
    out["d2_exponents"] = sl
    lines += ["", "D2 scaling exponents s (N_0.9 ~ p^-s, p = 1e-2 ... 1e-5): "
              + ", ".join(f"{k} {v:.2f}" for k, v in sl.items()) + " (article: portfolio 0.45, single-chain stagger 0.32)"]
    if e66:
        out["e66"] = e66
    (RES / "e67_portfolio_sas.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (RES / "e67_portfolio_sas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("RMQ_CPU", "4")))
    a = ap.parse_args()
    if not a.report_only:
        jobs = [(n, b, s, t) for n, b, _ in PROBLEMS for s, t in STRATS]
        # long jobs first
        jobs.sort(key=lambda j: -j[1])
        done = 0
        prog = os.environ.get("RMQ_PROGRESS_FILE")
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for msg in ex.map(run_one, jobs):
                done += 1
                print(f"[{done}/{len(jobs)}] {msg}", flush=True)
                if prog:
                    Path(prog).write_text(f"{done}/{len(jobs)} e67 portfolio+stagger", encoding="utf-8")
    report()


if __name__ == "__main__":
    main()
