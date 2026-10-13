"""Длинная проверка времени жизни для хрупких случаев: контроль Лоренца (4, 29, 2), Рабинович (6.8, -0.5, 0.99),
хаос Jafari 2026 у границы ограниченности. 8 копий, окна T_obs, до T_max = 1e5 (или уход/смена режима).
Выход: results/e13b_long_lifetime.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.identify import calibrate_reference, match  # noqa: E402
from hsearch.integrate import F_ESC, F_FP, F_LAM, nfeat, run_batch  # noqa: E402

CASES = {"lorenz_transient": "TC", "rabinovich_68": "HA", "jafari2026_chaos": "HA"}


def main(T_max=1e5, n_copies=8):
    out = {}
    for name, key in CASES.items():
        rng = np.random.default_rng(14)
        s = systems.get(name)
        ref = calibrate_reference(s, key, s.known[key], rng)
        x = ref.example + 1e-3 * (1 + np.abs(ref.example)) * rng.normal(size=(n_copies, s.dim))
        n_win = int(T_max / s.T_obs)
        t_end = np.full(n_copies, np.nan)
        end_state = ["alive"] * n_copies
        t0 = time.time()
        for w in range(n_win):
            feats, finals = run_batch(s, x, n_tr=0, n_obs=s.n_obs)
            labels, z = match(s, feats, finals, [ref])
            for j in range(n_copies):
                if not np.isnan(t_end[j]):
                    continue
                if feats[j, F_ESC] > 0.5:
                    t_end[j] = (w + 1) * s.T_obs; end_state[j] = "escape"
                elif feats[j, F_FP] > 0.5 or feats[j, F_LAM] < -0.002:
                    t_end[j] = (w + 1) * s.T_obs; end_state[j] = "decay"
                elif labels[j] != 0:
                    t_end[j] = (w + 1) * s.T_obs; end_state[j] = "other"
            x = np.where(np.isnan(t_end)[:, None], finals, x)
            if not np.isnan(t_end).any():
                break
        rec = {"T_max": T_max, "n_copies": n_copies, "t_end": [None if np.isnan(v) else float(v) for v in t_end],
               "end_state": end_state, "n_alive": int(np.isnan(t_end).sum()), "time_s": round(time.time() - t0, 1)}
        out[name] = rec
        te = [v for v in t_end if not np.isnan(v)]
        print(f"{name}/{key}: живых при T={T_max:.0f}: {rec['n_alive']}/{n_copies}; исходы {end_state}; "
              f"времена ухода {np.round(te, 0).tolist()} ({rec['time_s']} с)")
    (ROOT / "results" / "e13b_long_lifetime.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
