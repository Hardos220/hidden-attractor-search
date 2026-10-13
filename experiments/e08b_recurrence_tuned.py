"""Базовая линия «сетка + рекуррентность» (аналог Attractors.jl) с настройкой под каждую систему.

Отличия от e08: начальные условия обрабатываются по одному (точная стоимость в IC),
сетка грубее (40 ячеек на измерение), порог рекуррентностей 500 шагов, бюджет шагов на IC
удвоен. Для каждой системы R повторов; стоимость — число IC и суммарное число шагов до первого
обнаружения целевого аттрактора. Выход: results/e08b_recurrence.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.baselines.recurrence import RecurrenceFinder  # noqa: E402
from hsearch.identify import build_reference_set, match  # noqa: E402
from hsearch.integrate import run_batch  # noqa: E402

SYSTEMS = ["chua_I", "wangchen1", "wei_ne2", "wangchen2", "se3", "se11", "le1", "rabinovich_100", "ne4"]
GRID = {"default": 40, "ne4": 40, "rabinovich_100": 40}
REC_STEPS = {"default": 500}


def main(names=None, R=8):
    names = names or SYSTEMS
    out_path = ROOT / "results" / "e08b_recurrence.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    e02 = json.loads((ROOT / "results" / "e02_uniform.json").read_text(encoding="utf-8"))
    for name in names:
        rng = np.random.default_rng(8)
        s = systems.get(name)
        refs, *_ = build_reference_set(s, rng, n_sample=3000, verbose=False)
        targets = [h for h, v in e02.get(name, {}).get("hidden", {}).items() if v["p"] > 0]
        tidx = [k for k, r in enumerate(refs) if r.name in targets or (r.kind in ("chaotic", "periodic") and r.name in s.hidden)]
        if not tidx:
            tidx = [k for k, r in enumerate(refs) if r.kind in ("chaotic", "periodic")]
        p_u = max([e02.get(name, {}).get("fractions", {}).get(refs[k].name, 0.0) for k in tidx] or [1e-9])
        N09 = int(np.ceil(np.log(0.1) / np.log1p(-p_u))) if p_u > 0 else 10 ** 6
        max_ic = int(min(20000, max(50, 20 * N09)))
        runs = []
        t0 = time.time()
        for r in range(R):
            rr = np.random.default_rng(200 + r)
            finder = RecurrenceFinder(s, GRID.get(name, GRID["default"]),
                                      consecutive_recurrences=REC_STEPS.get(name, REC_STEPS["default"]),
                                      attractor_locate_steps=2000, consecutive_hits=10, consecutive_lost=200,
                                      max_steps=2 * (s.n_tr + s.n_obs))
            found = None
            n_ic = 0
            n_checked = 0
            while n_ic < max_ic:
                labels, steps = finder.find(s.sample_box(1, rr))
                n_ic += 1
                if finder.n_att > n_checked:
                    seeds = finder.attractor_seeds()
                    sd = np.array([seeds[k] for k in range(n_checked + 1, finder.n_att + 1)])
                    fs, fn = run_batch(s, sd, n_tr=0)
                    lab, z = match(s, fs, fn, refs)
                    n_checked = finder.n_att
                    if np.any(np.isin(lab, tidx)):
                        found = n_ic
                        break
            runs.append({"n_ic": found, "steps": int(finder.total_steps), "n_attractors": int(finder.n_att),
                         "found": found is not None})
        ok = [u for u in runs if u["found"]]
        rec = {"targets": [refs[k].name for k in tidx], "p_uniform": p_u, "N_0.9_uniform": N09,
               "grid_n": GRID.get(name, GRID["default"]), "runs": runs,
               "success_rate": float(np.mean([u["found"] for u in runs])),
               "median_n_ic": float(np.median([u["n_ic"] for u in ok])) if ok else None,
               "median_steps": float(np.median([u["steps"] for u in ok])) if ok else None,
               "steps_per_uniform_traj": int(s.n_tr + s.n_obs), "time_s": round(time.time() - t0, 1)}
        out[name] = rec
        print(f"{name}: успех {rec['success_rate']:.2f}, медиана IC {rec['median_n_ic']}, медиана шагов {rec['median_steps']} "
              f"(перебор: N_0.9={N09}, шагов/траекторию {rec['steps_per_uniform_traj']}) [{rec['time_s']} с]")
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
