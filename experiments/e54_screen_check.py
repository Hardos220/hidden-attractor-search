"""Проверка двухэтапной оценки (harness.screened_batch) против полного счёта на равномерных выборках.

Для каждой задачи: доля остановленных на этапе 1 траекторий, совпадение меток с полным счётом, точное
совпадение отпечатков продолженных траекторий, сохранность всех попаданий в цель и экономия шагов и
времени интегрирования. Запуск только через rmq:

    rmq run --cpu 8 --ram 4 --time 20m -- python experiments/e54_screen_check.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import e50_batch_runner as runner  # noqa: E402
from hsearch.identify import match  # noqa: E402
from hsearch.integrate import run_batch  # noqa: E402
from hsearch.search.harness import screened_batch, steps_of  # noqa: E402

PROBLEMS = [({"system": "chua_I"}, 2000), ({"system": "chua_II"}, 1500), ({"system": "se11"}, 4000),
            ({"system": "le1"}, 4000), ({"system": "jafari2026_lc"}, 3000), ({"system": "jafari2026_chaos"}, 3000),
            ({"system": "edge_se3_m3_p0.0001"}, 4000), ({"coupled": "se11", "k": 2, "eps": 0.01}, 4000)]


def main():
    import numba
    th = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    numba.set_num_threads(min(th, numba.config.NUMBA_NUM_THREADS))
    out = {}
    for var, n in PROBLEMS:
        s, refs, tidx, kidx, _ = runner.get_problem(var, None)
        rng = np.random.default_rng(11)
        ics = s.sample_box(n, rng)
        run_batch(s, ics[:4]); screened_batch(s, ics[:4], refs, kidx)          # компиляция
        t0 = time.perf_counter()
        F, X = run_batch(s, ics)
        t_full = time.perf_counter() - t0
        lab, _ = match(s, F, X, refs)
        st_full = steps_of(F, s.n_tr, s.n_obs)
        t0 = time.perf_counter()
        F2, X2, lab2, st2, stop = screened_batch(s, ics, refs, kidx)
        t_scr = time.perf_counter() - t0
        cont = ~stop & ~(F2[:, 0] > 0.5)
        same_cont = bool(np.array_equal(np.nan_to_num(F[cont], nan=7e300), np.nan_to_num(F2[cont], nan=7e300))
                         and np.array_equal(X[cont], X2[cont]))
        esc = F[:, 0] > 0.5
        same_esc = bool(np.array_equal(F[esc], F2[esc]))
        hit = np.isin(lab, tidx)
        hit2 = np.isin(lab2, tidx)
        rec = {"n": n, "frac_stopped": float(stop.mean()), "label_agree": float((lab == lab2).mean()),
               "label_agree_stopped": float((lab[stop] == lab2[stop]).mean()) if stop.any() else None,
               "cont_bit_exact": same_cont, "esc_bit_exact": same_esc,
               "hits_full": int(hit.sum()), "hits_screened": int(hit2.sum()), "hits_lost": int((hit & ~hit2).sum()),
               "steps_ratio": float(st2.sum() / st_full.sum()), "time_ratio": t_scr / t_full,
               "t_full_s": round(t_full, 3), "t_screened_s": round(t_scr, 3),
               "disagree": [(int(a), int(b)) for a, b in zip(lab[lab != lab2][:10], lab2[lab != lab2][:10])],
               "refs": [r.name for r in refs], "targets": [refs[k].name for k in tidx]}
        out[s.name] = rec
        print(s.name, json.dumps(rec, ensure_ascii=False), flush=True)
    p = ROOT / "results" / "e54_screen_check.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", p)


if __name__ == "__main__":
    main()
