"""Скан параметра цепи Чуа II: где бассейн скрытых хаотических аттракторов HA_a, HA_b естественно сжимается
(2026-10-13). Нужна редкая скрытая цель в реальной системе без структуры произведения.

Продолжение по параметру: на каждом новом значении параметр меняется на шаг, из состояния на HA_a при
предыдущем значении интегрируется переходный участок (T = 1000), проверяется, что траектория хаотическая и не
пришла на цикл или равновесие. HA_b — зеркальный образ HA_a (симметрия x -> -x). Затем эталоны (равновесия,
LC, HA_a, HA_b) калибруются заново, и доля бассейна цели p = доля HA_a + HA_b оценивается равномерной выборкой с
двухэтапной оценкой. Ядра интегратора не перекомпилируются: параметры передаются им аргументом.

    rmq submit --cpu 16 --ram 3 --time 40m -- python experiments/e56_chua_param_scan.py alpha 8.4 9.4 0.05
пишет results/e56_chua_scan_<параметр>.json
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
from hsearch.systems.flows3d import chua_equilibria  # noqa: E402

PIDX = {"alpha": 0, "beta": 1, "gamma": 2, "m0": 3, "m1": 4}
N_SAMPLE = 20000


def variant(base, params):
    s = copy.copy(base)
    s.params = np.asarray(params, float)
    s.equilibria = chua_equilibria(s.params)
    return s


def refs_for(s, states, rng):
    refs = fp_references_from_equilibria(s)
    for nm, st in states.items():
        r = calibrate_reference(s, nm, st, rng, from_state=True)
        refs.append(r)
    return refs


def main():
    import numba
    numba.set_num_threads(min(int(os.environ.get("RMQ_CPU", "16")), numba.config.NUMBA_NUM_THREADS))
    par, a0, a1, da = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    tag = sys.argv[5] if len(sys.argv) > 5 else ""          # суффикс файла результата (несколько сканов одного параметра)
    base = systems.get("chua_II")
    i_p = PIDX[par]
    vals = np.round(np.arange(a0, a1 + 1e-9 if a1 > a0 else a1 - 1e-9, da if a1 > a0 else -abs(da)), 6)
    st_a = np.asarray(base.known["HA_a"], float)
    st_lc = np.asarray(base.known["LC"], float)
    out = {"param": par, "base_params": base.params.tolist(), "points": []}
    pf = os.environ.get("RMQ_PROGRESS_FILE")
    rng = np.random.default_rng(56)
    for k, v in enumerate(vals):
        t0 = time.time()
        p = base.params.copy()
        p[i_p] = v
        s = variant(base, p)
        # продолжение HA_a и LC
        tr = s.traj(st_a, n=s.n_tr)
        x_a = tr[-1]
        F, X = run_batch(s, x_a[None, :], n_tr=0, n_obs=s.n_obs)
        rec = {"value": float(v), "eq": [e.tolist() for e in s.equilibria]}
        trl = s.traj(st_lc, n=s.n_tr)
        st_lc_new = trl[-1].copy()
        # аттрактор потерян, если траектория ушла, пришла на равновесие или на большой цикл LC; хаотический
        # и периодический (после обратного каскада удвоений) режимы отслеживаются одинаково
        lost = not np.all(np.isfinite(x_a)) or F[0, F_ESC] > 0.5
        if not lost:
            ref_lc = calibrate_reference(s, "LC", st_lc_new, rng, n_rep=8, from_state=True)
            chk = fp_references_from_equilibria(s) + [ref_lc]
            lb, _ = match(s, F, X, chk)
            lost = lb[0] >= 0
        if lost:
            rec["HA"] = "lost"
            rec["lam"] = float(F[0, F_LAM])
            out["points"].append(rec)
            print(f"{par}={v}: HA потерян (lam={F[0, F_LAM]:.4f}, esc={F[0, F_ESC]})", flush=True)
            break
        st_a = X[0].copy()
        st_lc = st_lc_new
        states = {"LC": st_lc, "HA_a": st_a, "HA_b": -st_a}
        refs = refs_for(s, states, rng)
        names = [r.name for r in refs]
        kinds = {r.name: r.kind for r in refs}
        ics = s.sample_box(N_SAMPLE, rng)
        Fs, Xs, lab, steps, stop = screened_batch(s, ics, refs, list(range(len(refs))))
        cnt = {nm: int((lab == i).sum()) for i, nm in enumerate(names)}
        cnt["unknown"] = int((lab == LAB_UNKNOWN).sum())
        cnt["escape"] = int((lab == -1).sum())
        n_ha = cnt.get("HA_a", 0) + cnt.get("HA_b", 0)
        rec.update({"HA": "ok", "lam": float(F[0, F_LAM]), "state_HA_a": st_a.tolist(), "state_LC": st_lc.tolist(),
                    "kinds": kinds, "counts": cnt, "n": N_SAMPLE, "p_HA": n_ha / N_SAMPLE,
                    "steps_per_eval": float(steps.mean()), "wall_s": round(time.time() - t0, 1)})
        out["points"].append(rec)
        print(f"{par}={v}: lam={rec['lam']:.3f} p_HA={rec['p_HA']:.2e} ({n_ha}) counts={cnt} kinds={kinds} "
              f"{rec['wall_s']} с", flush=True)
        if pf:
            try:
                Path(pf).write_text(f"{k + 1}/{len(vals)} {par}={v}", encoding="utf-8")
            except OSError:
                pass
        (ROOT / "results" / f"e56_chua_scan_{par}{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                    encoding="utf-8")
    (ROOT / "results" / f"e56_chua_scan_{par}{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
