"""Перепись аттракторов двух связанных цепей Чуа II и поиск новых скрытых аттракторов (2026-10-13).

Система: две цепи Чуа с параметрами II (у одной цепи пять аттракторов: цикл LC с долей 0.99, два скрытых
хаотических HA_a и HA_b, два устойчивых равновесия E1, E2) со связью среднего поля по x силы eps.
Известными считаются все произведения аттракторов и устойчивых равновесий отдельных цепей (products), а
равновесия связанной системы ищутся все, включая несинхронные (alleq).

Шаги для каждого eps:
1. эталоны: равновесия, произведения, обнаружение по стартовой выборке (e20.build_problem);
2. перепись: равномерные выборки по n_round точек с двухэтапной оценкой; неопознанные траектории
   кластеризуются (identify.discover), кластеры калибруются как новые эталоны new<k>; доли всех режимов;
3. новые режимы: что делает каждая цепь (ближайший режим одной цепи по статистикам окна), синхронизация
   (корреляция x1 и x2), показатель Ляпунова; проверка, что это аттрактор, а не долгий переходный процесс
   (identify.verify_candidate: 20 окон наблюдения подряд и повтор с половинным шагом);
4. скрытость: доли начальных условий из eps_h-окрестностей всех равновесий (и вдоль их неустойчивых
   направлений), попавших в бассейн каждого режима, eps_h = 1e-3, 1e-2, 1e-1 L. Скрытый режим: ноль при
   eps_h = 1e-3 и 1e-2 L.

Только через rmq:  rmq submit --cpu 16 --ram 4 --time 40m -- python experiments/e55_coupled_chua_census.py 0.01 0.05
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
from hsearch.identify import LAB_ESCAPE, LAB_UNKNOWN, _duplicate_of, discover, match, scale, verify_candidate  # noqa: E402
from hsearch.integrate import F_LAM, NHEAD, run_batch  # noqa: E402
from hsearch.search.harness import screened_batch  # noqa: E402
from hsearch.systems.coupled import _max_re  # noqa: E402

N_ROUND = 10000
N_ROUNDS = 4
N_HID = 200
N_LONG = 400            # неопознанных траекторий, продолжаемых ещё на LONG_HOR горизонтов
LONG_HOR = 10


def progress(done, total, note):
    import os
    pf = os.environ.get("RMQ_PROGRESS_FILE")
    if pf:
        try:
            Path(pf).write_text(f"{done}/{total} {note}", encoding="utf-8")
        except OSError:
            pass
    print(f"[{done}/{total}] {note}", flush=True)


def unit_identity(ref, base_refs, d0=3, k=2):
    """Для каждой цепи — ближайший режим одной цепи по (среднему, СКО) окна наблюдения."""
    out = []
    d = d0 * k
    for u in range(k):
        m = ref.mu[NHEAD + u * d0: NHEAD + (u + 1) * d0]
        sd = ref.mu[NHEAD + d + u * d0: NHEAD + d + (u + 1) * d0]
        best = None
        for b in base_refs:
            if b.kind == "fp":
                if b.point is None:
                    continue
                bm, bs = np.asarray(b.point, float), np.zeros(d0)
            elif b.mu is not None:
                bm, bs = b.mu[NHEAD:NHEAD + d0], b.mu[NHEAD + d0:NHEAD + 2 * d0]
            else:
                continue
            dist = float(np.linalg.norm(np.concatenate([m - bm, sd - bs])) / (1.0 + np.linalg.norm(np.concatenate([bm, bs]))))
            if best is None or dist < best[1]:
                best = (b.name, dist)
        out.append({"nearest": best[0], "rel_dist": round(best[1], 3), "std": [round(float(v), 3) for v in sd]})
    return out


def sync_stats(s, ref, n=40000):
    if ref.example is None:
        return None
    tr = s.traj(np.asarray(ref.example, float), n=n)
    tr = tr[n // 2:]
    if not np.all(np.isfinite(tr)):
        return None
    x1, x2 = tr[:, 0], tr[:, 3]
    c = float(np.corrcoef(x1, x2)[0, 1]) if x1.std() > 1e-9 and x2.std() > 1e-9 else None
    return {"corr_x1x2": None if c is None else round(c, 3),
            "mean_abs_dx_over_L": round(float(np.mean(np.abs(x1 - x2)) / scale(s)), 4)}


def census(eps, total_steps, step0):
    var = {"coupled": "chua_II", "k": 2, "eps": eps, "coord": 0, "ha": ["HA_a", "HA_b"], "products": True, "alleq": True}
    t0 = time.time()
    s, refs, tidx, kidx, _ = runner.get_problem(var, None)
    refs = list(refs)
    n_base = len(refs)
    L = scale(s)
    progress(step0 + 1, total_steps, f"eps={eps}: эталоны построены ({len(refs)}), {time.time() - t0:.0f} с")
    rng = np.random.default_rng(55)
    Fs, Xs, labs = [], [], []
    for rd in range(N_ROUNDS):
        ics = s.sample_box(N_ROUND, rng)
        F, X, lab, steps, stop = screened_batch(s, ics, refs, list(range(len(refs))))
        Fs.append(F); Xs.append(X); labs.append(lab)
        F_all, X_all = np.vstack(Fs), np.vstack(Xs)
        lab_all = np.concatenate(labs)
        unk = lab_all == LAB_UNKNOWN
        if unk.sum() >= 5:
            new = [r for r in discover(s, F_all, X_all, unk, rng, max_new=10) if _duplicate_of(s, r, refs) is None]
            for r in new:
                r.name = f"new{sum(1 for q in refs if q.name.startswith('new'))}"
                refs.append(r)
            if new:
                # переопознание неизвестных с новыми эталонами (у неизвестных отпечатки полного счёта)
                lab_unk, _ = match(s, F_all[unk], X_all[unk], refs)
                lab_all[unk] = lab_unk
                labs = [lab_all]
                Fs, Xs = [F_all], [X_all]
        progress(step0 + 2 + rd, total_steps, f"eps={eps}: перепись, раунд {rd + 1}/{N_ROUNDS}, эталонов {len(refs)}, "
                                             f"неопознанных {(np.concatenate(labs) == LAB_UNKNOWN).sum()}")
    lab_all = np.concatenate(labs)
    F_all, X_all = np.vstack(Fs), np.vstack(Xs)
    # долгое продолжение неопознанных: долгий переходный процесс (например, медленная синхронизация) или аттрактор
    unk_rows = np.where(lab_all == LAB_UNKNOWN)[0]
    long_rec = {"n_unknown": int(len(unk_rows))}
    if len(unk_rows):
        sel = rng.choice(unk_rows, size=min(N_LONG, len(unk_rows)), replace=False)
        Xl = X_all[sel].copy()
        alive = np.ones(len(sel), bool)
        settled = {}
        for h in range(LONG_HOR):
            Fh, Xh = run_batch(s, np.ascontiguousarray(Xl[alive]), n_tr=s.n_tr, n_obs=s.n_obs)
            lh, _ = match(s, Fh, Xh, refs)
            idx_alive = np.where(alive)[0]
            Xl[idx_alive] = Xh
            for j, lb in zip(idx_alive, lh):
                if lb != LAB_UNKNOWN:
                    settled.setdefault(refs[lb].name if lb >= 0 else "escape", []).append(h + 1)
                    alive[j] = False
            if not alive.any():
                break
        long_rec.update({"n_followed": int(len(sel)), "settled": {k: {"n": len(v), "median_horizons": float(np.median(v))}
                                                                   for k, v in settled.items()},
                         "still_unknown": int(alive.sum())})
        if alive.sum() >= 5:
            Fh, Xh = run_batch(s, np.ascontiguousarray(Xl[alive]), n_tr=0, n_obs=s.n_obs)
            cand = [r for r in discover(s, Fh, Xh, np.ones(int(alive.sum()), bool), rng, max_new=5)
                    if _duplicate_of(s, r, refs) is None]
            for r in cand:
                r.name = f"new{sum(1 for q in refs if q.name.startswith('new'))}"
                refs.append(r)
            long_rec["new_from_long"] = [r.name for r in cand]
    progress(step0 + 1 + N_ROUNDS, total_steps, f"eps={eps}: продолжение неопознанных: {long_rec}")
    n = len(lab_all)
    frac = {"escape": float((lab_all == LAB_ESCAPE).mean()), "unknown": float((lab_all == LAB_UNKNOWN).mean())}
    for k, r in enumerate(refs):
        frac[r.name] = float((lab_all == k).mean())
    # скрытость: окрестности всех равновесий
    eqs = [np.asarray(e, float) for e in s.equilibria]
    hid = {}
    for eps_rel in (1e-3, 1e-2, 1e-1):
        e_h = eps_rel * L
        pts = []
        for e in eqs:
            v = rng.normal(size=(N_HID, s.dim))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            rr = e_h * rng.random(N_HID) ** (1.0 / s.dim)
            pts.append(e + v * rr[:, None])
            # неустойчивые направления (якобиан конечными разностями)
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
        hid[str(eps_rel)] = {"n": int(len(ics)), "frac": {r.name: float((lab == k).mean()) for k, r in enumerate(refs)},
                             "unknown": float((lab == LAB_UNKNOWN).mean()), "escape": float((lab == LAB_ESCAPE).mean())}
    progress(step0 + 2 + N_ROUNDS, total_steps, f"eps={eps}: проверка скрытости готова")
    # новые режимы: характеристика и проверка
    base_refs = runner.get_problem({"system": "chua_II"}, None)[1]
    regimes = []
    for k, r in enumerate(refs):
        rec = {"name": r.name, "kind": r.kind, "frac": frac.get(r.name, 0.0), "n_hits": int((lab_all == k).sum()),
               "lam": None if r.mu is None else round(float(r.mu[F_LAM]), 4),
               "is_new": k >= n_base or not (r.name.startswith(("P_", "HA_", "E")) or r.name == "LC"),
               "hidden_1e-3": hid["0.001"]["frac"].get(r.name, 0.0) == 0.0,
               "hidden_1e-2": hid["0.01"]["frac"].get(r.name, 0.0) == 0.0,
               "h_eps": {e_: hid[e_]["frac"].get(r.name, 0.0) for e_ in hid}}
        if r.kind in ("periodic", "chaotic") and r.mu is not None:
            rec["units"] = unit_identity(r, base_refs)
            rec["sync"] = sync_stats(s, r)
            # «рождённый связью»: хотя бы одна цепь ведёт себя не как режим отдельной цепи
            rec["coupling_induced"] = max(u["rel_dist"] for u in rec["units"]) > 0.1
            if (rec["is_new"] or rec["coupling_induced"]) and r.example is not None:
                v = verify_candidate(s, r.example, r, np.random.default_rng(7))
                rec["verify"] = {"survived": v["survived"], "lifetime_T": v["lifetime_T"], "halfstep_ok": v["halfstep_ok"]}
        elif r.kind == "fp" and r.point is not None:
            rec["point"] = [round(float(q), 4) for q in r.point]
            rec["max_re"] = round(_max_re(s, np.asarray(r.point, float)), 4)
        regimes.append(rec)
    out = {"eps": eps, "system": s.name, "n_census": int(n), "L": L, "equilibria": [
        {"x": [round(float(q), 4) for q in e], "max_re": round(_max_re(s, e), 4)} for e in eqs],
        "fractions": frac, "long_followup": long_rec, "hiddenness": hid, "regimes": regimes,
        "wall_s": round(time.time() - t0, 1)}
    return out


def main():
    import numba
    epss = [float(a) for a in sys.argv[1:]] or [0.01]
    th = int(__import__("os").environ.get("RMQ_CPU", "16"))
    numba.set_num_threads(min(th, numba.config.NUMBA_NUM_THREADS))
    total = len(epss) * (N_ROUNDS + 3)
    res = {}
    for i, eps in enumerate(epss):
        res[str(eps)] = census(eps, total, i * (N_ROUNDS + 3))
        p = ROOT / "results" / "e55_coupled_chua_census_v2.json"
        old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        old.update(res)
        p.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
        r = res[str(eps)]
        print(f"=== eps={eps}: {r['n_census']} точек, {r['wall_s']} с")
        for g in sorted(r["regimes"], key=lambda g: -g["frac"]):
            print(f"  {g['name']:16s} {g['kind']:9s} p={g['frac']:.2e} lam={g['lam']} new={g['is_new']} ind={g.get('coupling_induced')} "
                  f"hidden={g['hidden_1e-2']} units={[u['nearest'] for u in g.get('units', [])]} sync={g.get('sync')} "
                  f"verify={g.get('verify')}")
    progress(total, total, "готово")


if __name__ == "__main__":
    main()
