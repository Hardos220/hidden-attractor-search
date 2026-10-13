"""Скан параметра системы эталонного набора: где бассейн скрытого аттрактора HA естественно сжимается (2026-10-13).

Обобщение e56 на любую систему с параметрами: продолжение HA по параметру (интегрирование из состояния на
аттракторе при предыдущем значении; потерян, если траектория ушла или пришла на равновесие), равновесия —
методом Ньютона от равновесий при предыдущем значении, эталоны калибруются заново, доля бассейна HA — по
равномерной выборке с двухэтапной оценкой. Ядра не перекомпилируются: параметры передаются аргументом.

    rmq submit --cpu 8 --ram 2 --time 30m -- python experiments/e57_param_scan.py wangchen2 0 0.006 0.1 0.002 100000 _up
пишет results/e57_scan_<система>_p<индекс><суффикс>.json
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from hsearch import systems  # noqa: E402
from hsearch.identify import LAB_UNKNOWN, calibrate_reference, fp_references_from_equilibria, match  # noqa: E402
from hsearch.integrate import F_ESC, F_LAM, run_batch  # noqa: E402
from hsearch.search.harness import screened_batch  # noqa: E402
from hsearch.systems.coupled import _max_re, _newton_all  # noqa: E402


def main():
    import numba
    numba.set_num_threads(min(int(os.environ.get("RMQ_CPU", "8")), numba.config.NUMBA_NUM_THREADS))
    name, ip, a0, a1, da, n_sample = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), \
        float(sys.argv[5]), int(sys.argv[6])
    tag = sys.argv[7] if len(sys.argv) > 7 else ""
    base = systems.get(name)
    step = abs(da) if a1 > a0 else -abs(da)
    vals = np.round(np.arange(a0, a1 + step / 2, step), 8)
    states = {nm: np.asarray(v, float) for nm, v in base.known.items()}
    eqs = [np.asarray(e, float) for e in base.equilibria]
    out = {"system": name, "param_index": ip, "base_params": base.params.tolist(), "points": []}
    path = ROOT / "results" / f"e57_scan_{name}_p{ip}{tag}.json"
    pf = os.environ.get("RMQ_PROGRESS_FILE")
    rng = np.random.default_rng(57)
    for k, v in enumerate(vals):
        t0 = time.time()
        s = copy.copy(base)
        s.params = base.params.copy()
        s.params[ip] = v
        if eqs:
            eqs = _newton_all(s.rhs, s.params, eqs, s.dim)
        s.equilibria = eqs
        new_states, lost = {}, False
        for nm, st in states.items():
            tr = s.traj(st, n=s.n_tr)
            x = tr[-1]
            if not np.all(np.isfinite(x)):
                if nm == "HA":
                    lost = True
                continue
            new_states[nm] = x
        rec = {"value": float(v), "eq": [{"x": e.tolist(), "max_re": _max_re(s, e)} for e in eqs]}
        if not lost and "HA" in new_states:
            F, X = run_batch(s, new_states["HA"][None, :], n_tr=0, n_obs=s.n_obs)
            chk = fp_references_from_equilibria(s)
            lb, _ = match(s, F, X, chk) if chk else (np.array([-2]), None)
            lost = F[0, F_ESC] > 0.5 or (lb[0] >= 0)
            rec["lam"] = float(F[0, F_LAM])
        if lost or "HA" not in new_states:
            rec["HA"] = "lost"
            out["points"].append(rec)
            print(f"{name} p[{ip}]={v}: HA потерян", flush=True)
            break
        states = new_states
        refs = fp_references_from_equilibria(s)
        for nm, st in states.items():
            r = calibrate_reference(s, nm, st, rng, from_state=True)
            if r.kind != "escape":
                refs.append(r)
        names = [r.name for r in refs]
        ics = s.sample_box(n_sample, rng)
        Fs, Xs, lab, steps, stop = screened_batch(s, ics, refs, list(range(len(refs))))
        cnt = {nm: int((lab == i).sum()) for i, nm in enumerate(names)}
        cnt["unknown"] = int((lab == LAB_UNKNOWN).sum())
        cnt["escape"] = int((lab == -1).sum())
        rec.update({"HA": "ok", "kinds": {r.name: r.kind for r in refs}, "counts": cnt, "n": n_sample,
                    "p_HA": cnt.get("HA", 0) / n_sample, "states": {nm: st.tolist() for nm, st in states.items()},
                    "wall_s": round(time.time() - t0, 1)})
        out["points"].append(rec)
        print(f"{name} p[{ip}]={v}: lam={rec['lam']:.3f} p_HA={rec['p_HA']:.2e} ({cnt.get('HA', 0)}) "
              f"counts={cnt} {rec['wall_s']} с", flush=True)
        if pf:
            try:
                Path(pf).write_text(f"{k + 1}/{len(vals)} p={v}", encoding="utf-8")
            except OSError:
                pass
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
