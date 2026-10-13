"""Стоимость одной оценки начального условия у разных способов счёта (2026-10-13), часть на Python.

У равномерной выборки номер первой точки, попавшей в цель, не зависит от того, как считается траектория.
Поэтому время перебора T_0.9 = N_0.9 × (средняя стоимость оценки), и способы оценки сравниваются на общем
наборе начальных условий: полный счёт (run_batch), двухэтапная оценка (screened_batch) — здесь, на одном потоке;
рекуррентный метод Attractors.jl — julia/recurrence_cost.jl на тех же точках.

    python experiments/e59_eval_cost.py export      # results/e59/ics_<задача>.csv — общие начальные условия
    rmq run --cpu 1 --ram 2 --time 30m -- python experiments/e59_eval_cost.py cost   # стоимости на одном потоке
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

OUT = ROOT / "results" / "e59"
PROBLEMS = {
    "chua_II": ({"system": "chua_II"}, None, 2000),
    "coupled_chua_II_eps0.1": ({"coupled": "chua_II", "k": 2, "eps": 0.1, "coord": 0, "ha": ["HA_a", "HA_b"],
                                "products": True, "alleq": True},
                               ["HA_aa", "HA_ab", "HA_ba", "HA_bb", "P_HA_a-E2", "P_E1-HA_a", "P_E2-HA_b"], 1000),
    "se11": ({"system": "se11"}, None, 4000),
    "le1": ({"system": "le1"}, None, 4000),
}


def export():
    import e50_batch_runner as runner
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, (var, targets, n) in PROBLEMS.items():
        s, refs, tidx, kidx, _ = runner.get_problem(var, targets)
        ics = s.sample_box(n, np.random.default_rng(2026))
        np.savetxt(OUT / f"ics_{name}.csv", ics, delimiter=",", fmt="%.17g")
        meta[name] = {"dim": s.dim, "box_lo": s.box_lo.tolist(), "box_hi": s.box_hi.tolist(), "dt": s.dt,
                      "n_tr": s.n_tr, "n_obs": s.n_obs, "esc_r": s.esc_r, "params": s.params.tolist(),
                      "refs": [(r.name, r.kind) for r in refs], "targets": [refs[k].name for k in tidx], "n": n}
        print(name, meta[name]["targets"], flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def cost():
    import numba
    numba.set_num_threads(1)
    import e50_batch_runner as runner
    from hsearch.identify import match
    from hsearch.integrate import run_batch
    from hsearch.search.harness import screened_batch, steps_of
    out = {}
    for name, (var, targets, n) in PROBLEMS.items():
        s, refs, tidx, kidx, _ = runner.get_problem(var, targets)
        ics = np.loadtxt(OUT / f"ics_{name}.csv", delimiter=",", ndmin=2)
        run_batch(s, ics[:4]); screened_batch(s, ics[:4], refs, kidx)          # компиляция
        t0 = time.perf_counter()
        F, X = run_batch(s, ics)
        t_full = time.perf_counter() - t0
        lab_full, _ = match(s, F, X, refs)
        t0 = time.perf_counter()
        F2, X2, lab2, st2, stop = screened_batch(s, ics, refs, kidx)
        t_scr = time.perf_counter() - t0
        hit = np.isin(lab_full, tidx)
        out[name] = {"n": int(len(ics)), "threads": 1,
                     "full": {"ms_per_ic": 1e3 * t_full / len(ics), "steps_per_ic": float(steps_of(F, s.n_tr, s.n_obs).mean())},
                     "screened": {"ms_per_ic": 1e3 * t_scr / len(ics), "steps_per_ic": float(st2.mean()),
                                  "stopped_frac": float(stop.mean()), "labels_agree": float((lab_full == lab2).mean())},
                     "p_target_sample": float(hit.mean()), "n_hits": int(hit.sum()),
                     "fractions": {r.name: float((lab_full == k).mean()) for k, r in enumerate(refs)},
                     "escape": float((lab_full == -1).mean()), "unknown": float((lab_full == -2).mean())}
        print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
        (OUT / "cost_python.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"export": export, "cost": cost}[sys.argv[1]]()
