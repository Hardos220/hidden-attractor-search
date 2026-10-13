"""Полный скан силы связи двух цепей Чуа II: продолжение всех режимов по eps (2026-10-13).

Старт — задача при eps0 (по умолчанию 0.1): равновесия (все, методом Ньютона), эталоны и их состояния-примеры
(runner.get_problem: произведения, обнаруженные режимы — в том числе новый скрытый аттрактор «цикл + хаос»).
Дальше eps меняется шагами; на каждом шаге:
  1. равновесия — методом Ньютона от предыдущих;
  2. каждый отслеживаемый режим продолжается: из его прежнего состояния интегрируется переходный участок, затем
     состояние калибруется как эталон (вид, показатель Ляпунова); что бы ни получилось (тот же режим, цикл вместо
     хаоса, другой известный режим), это записывается — режим «потерян» только при уходе на бесконечность или
     совпадении с равновесием или с другим отслеживаемым режимом (тогда ветви сливаются);
  3. равномерная выборка n_sample точек с двухэтапной оценкой: доли всех режимов, неопознанные кластеризуются
     (identify.discover) и становятся новыми отслеживаемыми режимами;
  4. для каждого осциллирующего режима — что делает каждая цепь (ближайший режим одной цепи), синхронизация;
  5. скрытость: доли точек из окрестностей всех равновесий (радиусы 1e-3 и 1e-2 L, по N_HID на равновесие плюс
     неустойчивые направления), попавших в бассейн каждого режима.
Ядра не перекомпилируются (eps — последний параметр системы).

    rmq submit --cpu 8 --ram 3 --time 3h -- python experiments/e58_coupled_chua_eps_scan.py 0.1 0.0 0.005 10000 _down
    rmq submit --cpu 8 --ram 3 --time 3h -- python experiments/e58_coupled_chua_eps_scan.py 0.1 0.6 0.01 10000 _up
пишет results/e58_eps_scan<суффикс>.json
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import e50_batch_runner as runner  # noqa: E402
from e55_coupled_chua_census import sync_stats, unit_identity  # noqa: E402
from hsearch.identify import (LAB_ESCAPE, LAB_UNKNOWN, _duplicate_of, calibrate_reference, discover,  # noqa: E402
                              fp_references_from_equilibria, match, scale)
from hsearch.integrate import F_ESC, F_LAM, run_batch  # noqa: E402
from hsearch.search.harness import screened_batch  # noqa: E402
from hsearch.systems.coupled import _max_re, _newton_all  # noqa: E402

N_HID = 60
EPS_INDEX = 5          # params = [alpha, beta, gamma, m0, m1, eps]


def progress(k, total, note):
    pf = os.environ.get("RMQ_PROGRESS_FILE")
    if pf:
        try:
            Path(pf).write_text(f"{k}/{total} {note}", encoding="utf-8")
        except OSError:
            pass
    print(f"[{k}/{total}] {note}", flush=True)


def hiddenness(s, refs, rng, L):
    out = {}
    eqs = [np.asarray(e, float) for e in s.equilibria]
    for eps_rel in (1e-3, 1e-2):
        e_h = eps_rel * L
        pts = []
        for e in eqs:
            v = rng.normal(size=(N_HID, s.dim))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            rr = e_h * rng.random(N_HID) ** (1.0 / s.dim)
            pts.append(e + v * rr[:, None])
            J = np.zeros((s.dim, s.dim)); f0 = np.zeros(s.dim); f1 = np.zeros(s.dim)
            s.rhs(e, s.params, f0)
            for i in range(s.dim):
                xp = e.copy(); xp[i] += 1e-7
                s.rhs(xp, s.params, f1)
                J[:, i] = (f1 - f0) / 1e-7
            w, V = np.linalg.eig(J)
            for kk in range(len(w)):
                if np.real(w[kk]) > 1e-6:
                    u = np.real(V[:, kk]); u /= np.linalg.norm(u) + 1e-12
                    pts.append(np.vstack([e + e_h * u, e - e_h * u]))
        ics = np.vstack(pts)
        F, X = run_batch(s, ics)
        lab, _ = match(s, F, X, refs)
        out[str(eps_rel)] = {"n": int(len(ics)), "frac": {r.name: float((lab == k).mean()) for k, r in enumerate(refs)},
                             "unknown": float((lab == LAB_UNKNOWN).mean())}
    return out


def main():
    import numba
    numba.set_num_threads(min(int(os.environ.get("RMQ_CPU", "8")), numba.config.NUMBA_NUM_THREADS))
    eps0, eps1, de, n_sample = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    tag = sys.argv[5] if len(sys.argv) > 5 else ""
    step = abs(de) if eps1 > eps0 else -abs(de)
    vals = [round(v, 6) for v in np.arange(eps0, eps1 + step / 2, step)]
    var = {"coupled": "chua_II", "k": 2, "eps": eps0, "coord": 0, "ha": ["HA_a", "HA_b"], "products": True, "alleq": True}
    base, refs0, tidx, kidx, _ = runner.get_problem(var, None)
    base_refs = runner.get_problem({"system": "chua_II"}, None)[1]
    L = scale(base)
    # отслеживаемые режимы: все осциллирующие эталоны стартовой задачи с состоянием-примером
    tracked = {r.name: np.asarray(r.example, float) for r in refs0
               if r.kind in ("periodic", "chaotic") and r.example is not None}
    eqs = [np.asarray(e, float) for e in base.equilibria]
    rng = np.random.default_rng(58)
    path = ROOT / "results" / f"e58_eps_scan{tag}.json"
    out = {"eps0": eps0, "n_sample": n_sample, "tracked_start": sorted(tracked), "points": []}
    for k, eps in enumerate(vals):
        t0 = time.time()
        s = copy.copy(base)
        s.params = base.params.copy()
        s.params[EPS_INDEX] = eps
        eqs = _newton_all(s.rhs, s.params, eqs, s.dim)
        s.equilibria = eqs
        refs = fp_references_from_equilibria(s)
        n_fp = len(refs)
        regimes = {}
        new_states = {}
        # 2. продолжение отслеживаемых режимов
        for nm, st in list(tracked.items()):
            tr = s.traj(st, n=s.n_tr)
            x = tr[-1]
            if not np.all(np.isfinite(x)):
                regimes[nm] = {"status": "escaped"}
                continue
            r = calibrate_reference(s, nm, x, rng, from_state=True)
            if r.kind == "escape":
                regimes[nm] = {"status": "escaped"}
                continue
            dup = _duplicate_of(s, r, refs)
            if dup is not None:
                regimes[nm] = {"status": "merged", "into": refs[dup].name}
                continue
            refs.append(r)
            new_states[nm] = x.copy()
            regimes[nm] = {"status": "ok", "kind": r.kind, "lam": None if r.mu is None else round(float(r.mu[F_LAM]), 4)}
        tracked = new_states
        # 3. равномерная выборка, обнаружение неизвестных
        ics = s.sample_box(n_sample, rng)
        F, X, lab, steps, stop = screened_batch(s, ics, refs, list(range(len(refs))))
        unk = lab == LAB_UNKNOWN
        found_new = []
        if unk.sum() >= 5:
            cand = [r for r in discover(s, F, X, unk, rng, max_new=6) if _duplicate_of(s, r, refs) is None]
            for r in cand:
                r.name = f"new{eps:g}_{len(found_new)}"
                refs.append(r)
                tracked[r.name] = np.asarray(r.example, float)
                regimes[r.name] = {"status": "discovered", "kind": r.kind,
                                   "lam": None if r.mu is None else round(float(r.mu[F_LAM]), 4),
                                   "cluster_size": r.meta.get("cluster_size")}
                found_new.append(r.name)
            if cand:
                lab_u, _ = match(s, F[unk], X[unk], refs)
                lab[unk] = lab_u
        frac = {r.name: float((lab == i).mean()) for i, r in enumerate(refs)}
        frac["unknown"] = float((lab == LAB_UNKNOWN).mean())
        frac["escape"] = float((lab == LAB_ESCAPE).mean())
        # 4. характеристика осциллирующих режимов
        for r in refs[n_fp:]:
            g = regimes.setdefault(r.name, {"status": "ok", "kind": r.kind,
                                            "lam": None if r.mu is None else round(float(r.mu[F_LAM]), 4)})
            g["frac"] = frac[r.name]
            g["n_hits"] = int(round(frac[r.name] * n_sample))
            if r.mu is not None:
                g["units"] = unit_identity(r, base_refs)
                g["sync"] = sync_stats(s, r, n=20000)
        # 5. скрытость
        hid = hiddenness(s, refs, rng, L)
        for r in refs[n_fp:]:
            regimes[r.name]["h_eps"] = {e_: hid[e_]["frac"].get(r.name, 0.0) for e_ in hid}
            regimes[r.name]["hidden"] = all(hid[e_]["frac"].get(r.name, 0.0) == 0.0 for e_ in hid)
        rec = {"eps": eps, "equilibria": [{"x": [round(float(q), 4) for q in e], "max_re": round(_max_re(s, e), 4)} for e in eqs],
               "n_sample": n_sample, "fractions": frac, "regimes": regimes, "discovered": found_new,
               "hid_unknown": {e_: hid[e_]["unknown"] for e_ in hid}, "steps_per_eval": float(steps.mean()),
               "states": {nm: st.tolist() for nm, st in tracked.items()}, "wall_s": round(time.time() - t0, 1)}
        out["points"].append(rec)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        summ = "; ".join(f"{nm}:{g['status'][:3]}/{g.get('kind', '-')[:4]}/λ={g.get('lam')}/p={g.get('frac', 0):.1e}"
                         for nm, g in regimes.items() if g["status"] in ("ok", "discovered"))
        progress(k + 1, len(vals), f"eps={eps:g} unk={frac['unknown']:.1e} {summ} ({rec['wall_s']} с)")


if __name__ == "__main__":
    main()
