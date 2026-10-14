"""Длинные проверки времени жизни (2026-10-13): аттрактор или долгий переходный процесс.

(a) Режим «цикл + хаос» двух связанных цепей Чуа II (eps = 0.1; эталоны P_LC-HA_a / P_HA_a-LC задачи
    coupled_chua_II_k2_eps0.1_c0_prod_eq, как в e55): 16 копий состояния на режиме (относительное возмущение 1e-3, как в
    протоколе), RK4 библиотеки с dt = 0.0025 до T = 1e5 окнами наблюдения T_obs = 100 (40000 шагов): на каждом окне —
    сопоставление со всеми эталонами переписи (identify.match) и старший показатель Ляпунова; выживание и статистика
    λ по блокам 1000 единиц времени. Если расчёт до 1e5 занял меньше ~20 мин — 4 копии продолжаются до T = 1e6.
    Независимый интегратор: 8 копий, scipy DOP853 (rtol 1e-11, atol 1e-12) до T_dop (по умолчанию 2e4) отрезками;
    каждые 1000 единиц времени состояние DOP853 проверяется окном RK4 (полный отпечаток, λ, сопоставление), в конце —
    статистики последнего окна самой DOP853-траектории (z-оценка относительно эталона).
(b) Хаос Jafari 2026 (a3 = -3.548, система jafari2026_chaos из e13c): 32 копии с возмущением 1e-6, RK4 с шагом системы
    (0.005) до T = 1e4 окнами T_obs = 200: время ухода каждой копии (по шагу ухода), кривая выживания, распределение
    конечновременных показателей Ляпунова по окнам; 8 копий DOP853 до 1e4 с событием ухода |x| > R_esc.

    rmq submit --cpu 4 --ram 2 --time 1h --name e62 -- python experiments/e62_lifetime_checks.py
пишет results/e62_lifetime_checks.json, results/e62_lifetime_checks.md, results/e62_lifetime_checks.npz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

VAR = {"coupled": "chua_II", "k": 2, "eps": 0.1, "coord": 0, "ha": ["HA_a", "HA_b"], "products": True, "alleq": True}
REGIME = ("P_LC-HA_a", "P_HA_a-LC")
OUT = ROOT / "results" / "e62_lifetime_checks"


def progress(done, total, note):
    pf = os.environ.get("RMQ_PROGRESS_FILE")
    if pf:
        try:
            Path(pf).write_text(f"{done}/{total} {note}", encoding="utf-8")
        except OSError:
            pass
    print(f"[{done}/{total}] {note}", flush=True)


# ----------------------------------------------------------------------------------------------------- RK4 окнами
def rk4_windows(s, x0, refs, ok_idx, n_windows, log_every=100, note="", t_offset=0.0):
    """Интегрирует копии x0 окнами наблюдения (n_obs шагов, без переходного участка); на каждом окне — отпечаток,
    λ, метка по эталонам refs. Возвращает lam (n, W), lab (n, W), z (n, W), t_esc (n), состояния в конце."""
    from hsearch.identify import match
    from hsearch.integrate import F_ESC, F_FP, F_LAM, F_TESC, run_batch
    n = len(x0)
    x = np.array(x0, float)
    lam = np.full((n, n_windows), np.nan)
    lab = np.full((n, n_windows), -9, dtype=np.int64)
    zz = np.full((n, n_windows), np.nan)
    fp = np.zeros((n, n_windows), bool)
    t_esc = np.full(n, np.nan)
    alive = np.ones(n, bool)
    t0 = time.time()
    for w in range(n_windows):
        idx = np.where(alive)[0]
        if not len(idx):
            break
        feats, finals = run_batch(s, np.ascontiguousarray(x[idx]), n_tr=0, n_obs=s.n_obs)
        labels, z = match(s, feats, finals, refs)
        for j, i in enumerate(idx):
            if feats[j, F_ESC] > 0.5:
                t_esc[i] = t_offset + w * s.T_obs + feats[j, F_TESC] * s.dt
                alive[i] = False
                lab[i, w] = -1
                continue
            lam[i, w] = feats[j, F_LAM]
            lab[i, w] = labels[j]
            zz[i, w] = z[j]
            fp[i, w] = feats[j, F_FP] > 0.5
            x[i] = finals[j]
        if log_every and (w + 1) % log_every == 0:
            on = np.isin(lab[:, w], ok_idx).sum()
            print(f"  {note} окно {w + 1}/{n_windows}: на режиме {on}/{n}, ушло {int((~alive).sum())}, "
                  f"λ среднее {np.nanmean(lam[:, w]):.4f}, {time.time() - t0:.0f} с", flush=True)
    return {"lam": lam, "lab": lab, "z": zz, "fp": fp, "t_esc": t_esc, "x_end": x, "alive": alive, "wall_s": time.time() - t0}


def survival_table(res, ok_idx, T_obs, block_T, n_windows):
    """Таблица по блокам block_T: доля копий, не покидавших режим ни на одном окне (строго), доля на режиме в
    последнем окне блока, статистика λ по окнам блока."""
    lab, lam = res["lab"], res["lam"]
    n = lab.shape[0]
    per_block = int(round(block_T / T_obs))
    rows = []
    strict_alive = np.ones(n, bool)
    for b in range(n_windows // per_block):
        w0, w1 = b * per_block, (b + 1) * per_block
        blk = lab[:, w0:w1]
        strict_alive &= np.all(np.isin(blk, ok_idx), axis=1)
        cur = np.isin(lab[:, w1 - 1], ok_idx)
        lb = lam[:, w0:w1]
        lb_on = lb[np.isin(blk, ok_idx)]
        rows.append({"T": (b + 1) * block_T, "strict": int(strict_alive.sum()), "current": int(cur.sum()), "n": n,
                     "escaped": int(np.any(blk == -1, axis=1).sum()),
                     "lam_mean": None if not len(lb_on) else float(np.mean(lb_on)),
                     "lam_min": None if not len(lb_on) else float(np.min(lb_on)),
                     "lam_max": None if not len(lb_on) else float(np.max(lb_on)),
                     "lam_std": None if not len(lb_on) else float(np.std(lb_on))})
    return rows


def leave_events(res, refs, ok_idx, T_obs):
    """Для каждой копии: первое окно, где метка не из ok_idx (что это было), и доля окон на режиме."""
    lab = res["lab"]
    out = []
    for i in range(lab.shape[0]):
        row = lab[i]
        valid = row != -9
        on = np.isin(row[valid], ok_idx)
        bad = np.where(valid & ~np.isin(row, ok_idx))[0]
        first = None
        if len(bad):
            w = int(bad[0])
            lb = int(row[w])
            first = {"T": (w + 1) * T_obs, "label": "escape" if lb == -1 else ("unknown" if lb == -2 else refs[lb].name),
                     "z": None if np.isnan(res["z"][i, w]) else float(res["z"][i, w])}
        out.append({"frac_on_regime": float(on.mean()) if valid.any() else None, "n_off_windows": int(len(bad)),
                    "first_off": first, "t_esc": None if np.isnan(res["t_esc"][i]) else float(res["t_esc"][i]),
                    "lam_mean": float(np.nanmean(res["lam"][i])) if np.isfinite(res["lam"][i]).any() else None})
    return out


# ----------------------------------------------------------------------------------------------------- DOP853
def _dop_worker(args):
    """Один запуск DOP853: (kind, x0, T, seg, check_T, dt_sample, rtol, atol). Возвращает контрольные состояния
    каждые check_T, последнее окно (отсчёты с шагом dt_sample за последние T_obs), уход, nfev, время."""
    kind, x0, T, seg, check_T, dt_sample, T_obs, rtol, atol = args
    import numpy as np
    from scipy.integrate import solve_ivp
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "experiments"))
    if kind == "coupled":
        import e50_batch_runner as runner
        s = runner.build_variant(VAR)
    else:
        from hsearch import systems
        s = systems.get(kind)
    rhs, p, d, esc_r = s.rhs, s.params, s.dim, float(s.esc_r)

    def fun(t, y):
        out = np.empty(d)
        rhs(y, p, out)
        return out

    def esc_event(t, y):
        return esc_r - float(np.sqrt(np.dot(y, y)))
    esc_event.terminal = True
    esc_event.direction = -1

    t = 0.0
    y = np.array(x0, float)
    checks = []
    nfev = 0
    t0 = time.time()
    escaped_at = None
    n_seg = int(round(T / seg))
    last = None
    for k in range(n_seg):
        t1 = t + seg
        final_seg = k == n_seg - 1
        t_eval = None
        if final_seg:
            t_eval = t1 - T_obs + np.arange(0.0, T_obs + 1e-9, dt_sample)
            t_eval = t_eval[(t_eval >= t) & (t_eval <= t1)]
        sol = solve_ivp(fun, (t, t1), y, method="DOP853", rtol=rtol, atol=atol, t_eval=t_eval, events=esc_event)
        nfev += int(sol.nfev)
        if sol.status == 1 or not np.all(np.isfinite(sol.y[:, -1])):
            escaped_at = float(sol.t_events[0][0]) if (sol.t_events and len(sol.t_events[0])) else float(sol.t[-1])
            break
        y = sol.y[:, -1].copy()
        t = t1
        if final_seg:
            last = sol.y.T.copy()
        if abs((t / check_T) - round(t / check_T)) < 1e-9:
            checks.append((t, y.copy()))
    return {"checks": checks, "last_window": last, "escaped_at": escaped_at, "nfev": nfev, "wall_s": time.time() - t0, "y_end": y}


def dop853_batch(kind, x0s, T, seg, check_T, dt_sample, T_obs, workers, rtol=1e-11, atol=1e-12):
    import multiprocessing as mp
    tasks = [(kind, np.asarray(x, float), T, seg, check_T, dt_sample, T_obs, rtol, atol) for x in x0s]
    if workers <= 1:
        return [_dop_worker(a) for a in tasks]
    with mp.get_context("spawn").Pool(processes=min(workers, len(tasks))) as pool:
        return pool.map(_dop_worker, tasks)


def dop_report(s, refs, ok_idx, dres, T_obs, ref_for_window):
    """Проверка контрольных состояний DOP853 окнами RK4 и статистик последнего окна DOP853-траектории."""
    from hsearch.identify import features_from_trajectory, match
    from hsearch.integrate import F_ESC, F_FP, F_LAM, run_batch
    out = []
    for r in dres:
        rec = {"escaped_at": r["escaped_at"], "nfev": r["nfev"], "wall_s": round(r["wall_s"], 1), "checks": []}
        if r["checks"]:
            st = np.array([c[1] for c in r["checks"]])
            feats, finals = run_batch(s, st, n_tr=0, n_obs=s.n_obs)
            labels, z = match(s, feats, finals, refs)
            for (t, _), f, lb, zz in zip(r["checks"], feats, labels, z):
                rec["checks"].append({"T": t, "label": "escape" if f[F_ESC] > 0.5 else ("unknown" if lb == -2 else refs[lb].name),
                                      "on_regime": bool(lb in ok_idx), "lam": float(f[F_LAM]), "z": float(zz)})
        if r["last_window"] is not None:
            tr = r["last_window"]
            f = features_from_trajectory(tr, s.dt)
            # ворота по λ: статистики окна без показателя, λ берётся у эталона (проверяются средние/СКО/размах/частота)
            f[F_LAM] = ref_for_window.mu[F_LAM]
            lb, zz = match(s, f[None, :], tr[-1][None, :], refs)
            rec["last_window_dop"] = {"label": "unknown" if lb[0] == -2 else ("escape" if lb[0] == -1 else refs[lb[0]].name),
                                      "on_regime": bool(lb[0] in ok_idx), "z": float(zz[0]),
                                      "rmax": float(np.linalg.norm(tr, axis=1).max())}
        out.append(rec)
    return out


# ----------------------------------------------------------------------------------------------------- (a)
def part_a(args, th):
    import e50_batch_runner as runner
    from hsearch.identify import scale
    t0 = time.time()
    s, refs, tidx, kidx, _ = runner.get_problem(VAR, None)
    refs = list(refs)
    ok_idx = [k for k, r in enumerate(refs) if r.name in REGIME]
    k0 = [k for k, r in enumerate(refs) if r.name == REGIME[0]][0]
    state = np.asarray(refs[k0].example, float)
    print(f"(a) {s.name}: эталоны {[r.name for r in refs]}, режим {REGIME} -> {ok_idx}, λ эталона "
          f"{refs[k0].lam:.4f}, построено за {time.time() - t0:.0f} с", flush=True)
    rng = np.random.default_rng(62)
    n_cop = 16
    x0 = state + 1e-3 * (1.0 + np.abs(state)) * rng.normal(size=(n_cop, s.dim))
    T1 = args.T_a
    W1 = int(round(T1 / s.T_obs))
    rec = {"system": s.name, "regime": list(REGIME), "regime_idx": ok_idx, "refs": [r.name for r in refs],
           "lam_template": float(refs[k0].lam), "state": state.tolist(), "noise": 1e-3, "dt": float(s.dt), "T_obs": float(s.T_obs),
           "n_copies": n_cop, "T": T1}
    progress(1, 6, f"(a) RK4 16 копий до T={T1:g}")
    res = rk4_windows(s, x0, refs, ok_idx, W1, log_every=100, note="(a) 1e5")
    rec["rk4"] = {"wall_s": round(res["wall_s"], 1), "survival": survival_table(res, ok_idx, s.T_obs, 1000.0, W1),
                  "copies": leave_events(res, refs, ok_idx, s.T_obs),
                  "lam_all_mean": float(np.nanmean(res["lam"])), "lam_all_std": float(np.nanstd(res["lam"])),
                  "lam_all_min": float(np.nanmin(res["lam"])), "lam_all_max": float(np.nanmax(res["lam"])),
                  "n_strict_alive_end": int(np.all(np.isin(res["lab"], ok_idx), axis=1).sum()),
                  "n_current_on_end": int(np.isin(res["lab"][:, -1], ok_idx).sum())}
    npz = {"a_lam_1e5": res["lam"], "a_lab_1e5": res["lab"], "a_z_1e5": res["z"]}
    print(f"(a) до {T1:g}: строго на режиме {rec['rk4']['n_strict_alive_end']}/{n_cop}, в последнем окне "
          f"{rec['rk4']['n_current_on_end']}/{n_cop}, λ = {rec['rk4']['lam_all_mean']:.4f} ± {rec['rk4']['lam_all_std']:.4f}, "
          f"{res['wall_s']:.0f} с", flush=True)
    # длинное продолжение
    rec["rk4_long"] = None
    if res["wall_s"] < 20 * 60 and args.T_long > T1:
        progress(2, 6, f"(a) RK4 4 копии до T={args.T_long:g}")
        cand = [i for i in range(n_cop) if np.all(np.isin(res["lab"][i], ok_idx))] or list(range(n_cop))
        pick = cand[:4]
        W2 = int(round((args.T_long - T1) / s.T_obs))
        res2 = rk4_windows(s, res["x_end"][pick], refs, ok_idx, W2, log_every=1000, note="(a) 1e6", t_offset=T1)
        lab_full = np.hstack([res["lab"][pick], res2["lab"]])
        lam_full = np.hstack([res["lam"][pick], res2["lam"]])
        full = {"lab": lab_full, "lam": lam_full, "z": np.hstack([res["z"][pick], res2["z"]]), "t_esc": res2["t_esc"]}
        rec["rk4_long"] = {"copies_idx": pick, "T": args.T_long, "wall_s": round(res2["wall_s"], 1),
                           "survival": survival_table(full, ok_idx, s.T_obs, 1e4, W1 + W2) if args.T_long >= 1e5 else None,
                           "survival_1e5_blocks": survival_table(full, ok_idx, s.T_obs, 1e5, W1 + W2),
                           "copies": leave_events(full, refs, ok_idx, s.T_obs),
                           "n_strict_alive_end": int(np.all(np.isin(lab_full, ok_idx), axis=1).sum()),
                           "n_current_on_end": int(np.isin(lab_full[:, -1], ok_idx).sum()),
                           "lam_all_mean": float(np.nanmean(lam_full)), "lam_all_std": float(np.nanstd(lam_full)),
                           "lam_all_min": float(np.nanmin(lam_full)), "lam_all_max": float(np.nanmax(lam_full))}
        npz.update({"a_lam_long": lam_full, "a_lab_long": lab_full})
        print(f"(a) до {args.T_long:g}: строго {rec['rk4_long']['n_strict_alive_end']}/4, в последнем окне "
              f"{rec['rk4_long']['n_current_on_end']}/4, {res2['wall_s']:.0f} с", flush=True)
    else:
        print(f"(a) продолжение до {args.T_long:g} пропущено (расчёт до {T1:g} занял {res['wall_s'] / 60:.1f} мин)", flush=True)
    # DOP853
    progress(3, 6, f"(a) DOP853 8 копий до T={args.T_dop:g}")
    t1 = time.time()
    dres = dop853_batch("coupled", x0[:8], args.T_dop, seg=100.0, check_T=1000.0, dt_sample=s.dt, T_obs=s.T_obs, workers=th)
    rep = dop_report(s, refs, ok_idx, dres, s.T_obs, refs[k0])
    rec["dop853"] = {"T": args.T_dop, "rtol": 1e-11, "atol": 1e-12, "n_copies": 8, "wall_s": round(time.time() - t1, 1),
                     "copies": rep,
                     "n_on_regime_final_rk4window": int(sum(1 for r in rep if r["checks"] and r["checks"][-1]["on_regime"])),
                     "n_on_regime_last_window_dop": int(sum(1 for r in rep if r.get("last_window_dop", {}).get("on_regime"))),
                     "n_escaped": int(sum(1 for r in rep if r["escaped_at"] is not None)),
                     "n_checks_off": int(sum(1 for r in rep for c in r["checks"] if not c["on_regime"]))}
    print(f"(a) DOP853: на режиме (окно RK4 от конечного состояния) {rec['dop853']['n_on_regime_final_rk4window']}/8, "
          f"по статистикам последнего окна DOP853 {rec['dop853']['n_on_regime_last_window_dop']}/8, "
          f"контрольных точек вне режима {rec['dop853']['n_checks_off']}, {time.time() - t1:.0f} с", flush=True)
    return rec, npz


# ----------------------------------------------------------------------------------------------------- (b)
def part_b(args, th):
    from hsearch import systems
    from hsearch.identify import calibrate_reference
    s = systems.get("jafari2026_chaos")
    rng = np.random.default_rng(14)
    ref = calibrate_reference(s, "HA", s.known["HA"], rng)
    refs = [ref]
    state = np.asarray(ref.example, float)
    n_cop = 32
    x0 = state + 1e-6 * (1.0 + np.abs(state)) * rng.normal(size=(n_cop, s.dim))
    W = int(round(args.T_b / s.T_obs))
    progress(4, 6, f"(b) Jafari RK4 32 копии до T={args.T_b:g}")
    res = rk4_windows(s, x0, refs, [0], W, log_every=10, note="(b)")
    lam, lab, t_esc = res["lam"], res["lab"], res["t_esc"]
    grid = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    grid = [g for g in grid if g <= args.T_b]
    surv = [{"T": g, "alive": int(np.sum(~(t_esc <= g))), "n": n_cop} for g in grid]
    fin = lam[np.isfinite(lam)]
    rec = {"system": s.name, "a3": float(s.params[2]), "dt": float(s.dt), "T_obs": float(s.T_obs), "n_copies": n_cop, "noise": 1e-6,
           "T": args.T_b, "lam_template": float(ref.lam), "escape_frac_cal": float(ref.escape_frac), "state": state.tolist(),
           "rk4": {"wall_s": round(res["wall_s"], 1), "t_esc": [None if np.isnan(v) else float(v) for v in t_esc],
                   "n_alive_end": int(np.isnan(t_esc).sum()), "survival": surv,
                   "survival_blocks": survival_table(res, [0], s.T_obs, 1000.0, W),
                   "copies": leave_events(res, refs, [0], s.T_obs),
                   "ftle": {"n_windows": int(len(fin)), "mean": float(fin.mean()), "std": float(fin.std()), "min": float(fin.min()),
                            "max": float(fin.max()), "q05": float(np.quantile(fin, 0.05)), "q25": float(np.quantile(fin, 0.25)),
                            "median": float(np.median(fin)), "q75": float(np.quantile(fin, 0.75)), "q95": float(np.quantile(fin, 0.95)),
                            "frac_negative": float((fin < 0).mean()), "frac_below_0.005": float((fin < 0.005).mean())},
                   "per_copy_lam_mean": [float(np.nanmean(lam[i])) if np.isfinite(lam[i]).any() else None for i in range(n_cop)],
                   "label_counts": {str(k): int((lab == k).sum()) for k in np.unique(lab)}}}
    te = t_esc[~np.isnan(t_esc)]
    if len(te):
        rec["rk4"]["t_esc_sorted"] = sorted(float(v) for v in te)
        rec["rk4"]["t_esc_median"] = float(np.median(te))
        rec["rk4"]["t_esc_decades"] = {f"{lo:g}-{hi:g}": int(((te >= lo) & (te < hi)).sum())
                                       for lo, hi in ((0, 100), (100, 1000), (1000, 10000), (10000, 1e9))}
    print(f"(b) RK4: живых при T={args.T_b:g}: {rec['rk4']['n_alive_end']}/{n_cop}; времена ухода "
          f"{np.round(np.sort(te), 0).tolist() if len(te) else '—'}; λ по окнам {rec['rk4']['ftle']['mean']:.4f} ± "
          f"{rec['rk4']['ftle']['std']:.4f} [{rec['rk4']['ftle']['min']:.4f}, {rec['rk4']['ftle']['max']:.4f}]", flush=True)
    progress(5, 6, f"(b) Jafari DOP853 8 копий до T={args.T_b:g}")
    t1 = time.time()
    dres = dop853_batch("jafari2026_chaos", x0[:8], args.T_b, seg=200.0, check_T=1000.0, dt_sample=s.dt, T_obs=s.T_obs, workers=th)
    rep = dop_report(s, refs, [0], dres, s.T_obs, ref)
    rec["dop853"] = {"T": args.T_b, "rtol": 1e-11, "atol": 1e-12, "n_copies": 8, "wall_s": round(time.time() - t1, 1), "copies": rep,
                     "n_escaped": int(sum(1 for r in rep if r["escaped_at"] is not None)),
                     "t_esc": [r["escaped_at"] for r in rep],
                     "n_on_regime_final_rk4window": int(sum(1 for r in rep if r["checks"] and r["checks"][-1]["on_regime"])),
                     "n_on_regime_last_window_dop": int(sum(1 for r in rep if r.get("last_window_dop", {}).get("on_regime")))}
    print(f"(b) DOP853: ушло {rec['dop853']['n_escaped']}/8 (времена {rec['dop853']['t_esc']}), на режиме в конце "
          f"{rec['dop853']['n_on_regime_final_rk4window']}/8, {time.time() - t1:.0f} с", flush=True)
    return rec, {"b_lam": lam, "b_lab": lab, "b_t_esc": t_esc}


# ----------------------------------------------------------------------------------------------------- отчёт
def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def write_md(out):
    L = ["# e62: длинные проверки времени жизни", ""]
    a = out.get("a")
    if a:
        L += [f"## (a) Режим «цикл + хаос» двух связанных цепей Чуа II (eps = 0.1), эталоны {a['regime']}",
              "",
              f"Система {a['system']}, dt = {a['dt']}, окно T_obs = {a['T_obs']:g}; {a['n_copies']} копий состояния на режиме с "
              f"относительным возмущением {a['noise']:g}; λ эталона {a['lam_template']:.4f}. На каждом окне — сопоставление со всеми "
              f"эталонами переписи ({len(a['refs'])}): «строго» — ни одно окно до T не выпало из режима; «в окне» — метка последнего окна "
              "блока. λ — старший показатель Ляпунова по окнам блока (все копии на режиме).", "",
              "| T | строго на режиме | в окне | ушло | λ среднее | λ min | λ max | λ СКО |", "|---|---|---|---|---|---|---|---|"]
        rows = a["rk4"]["survival"]
        step = max(1, len(rows) // 20)
        for r in rows[::step] + ([rows[-1]] if (len(rows) - 1) % step else []):
            L.append(f"| {r['T']:g} | {r['strict']}/{r['n']} | {r['current']}/{r['n']} | {r['escaped']} | {fmt(r['lam_mean'])} | "
                     f"{fmt(r['lam_min'])} | {fmt(r['lam_max'])} | {fmt(r['lam_std'])} |")
        L += ["", f"Итог RK4 до T = {a['T']:g}: строго на режиме {a['rk4']['n_strict_alive_end']}/{a['n_copies']}, в последнем окне "
                  f"{a['rk4']['n_current_on_end']}/{a['n_copies']}; λ по всем окнам {a['rk4']['lam_all_mean']:.4f} ± {a['rk4']['lam_all_std']:.4f} "
                  f"[{a['rk4']['lam_all_min']:.4f}, {a['rk4']['lam_all_max']:.4f}]; {a['rk4']['wall_s']:.0f} с.", ""]
        offs = [c for c in a["rk4"]["copies"] if c["n_off_windows"]]
        if offs:
            L.append("Окна вне режима (копия: число окон, первое — T, метка, z): " + "; ".join(
                f"{i}: {c['n_off_windows']}, {c['first_off']['T']:g}, {c['first_off']['label']}, z={fmt(c['first_off']['z'], 2)}"
                for i, c in enumerate(a["rk4"]["copies"]) if c["n_off_windows"]))
            L.append("")
        if a.get("rk4_long"):
            g = a["rk4_long"]
            L += [f"### Продолжение 4 копий до T = {g['T']:g}", "", "| T | строго | в окне | λ среднее | λ min | λ max |", "|---|---|---|---|---|---|"]
            for r in g["survival_1e5_blocks"]:
                L.append(f"| {r['T']:g} | {r['strict']}/4 | {r['current']}/4 | {fmt(r['lam_mean'])} | {fmt(r['lam_min'])} | {fmt(r['lam_max'])} |")
            L += ["", f"Итог: строго {g['n_strict_alive_end']}/4, в последнем окне {g['n_current_on_end']}/4; λ {g['lam_all_mean']:.4f} ± "
                      f"{g['lam_all_std']:.4f} [{g['lam_all_min']:.4f}, {g['lam_all_max']:.4f}]; {g['wall_s']:.0f} с.", ""]
            offs = [(i, c) for i, c in enumerate(g["copies"]) if c["n_off_windows"]]
            if offs:
                L.append("Окна вне режима: " + "; ".join(f"копия {g['copies_idx'][i]}: {c['n_off_windows']}, первое T={c['first_off']['T']:g} "
                                                        f"({c['first_off']['label']}, z={fmt(c['first_off']['z'], 2)})" for i, c in offs))
                L.append("")
        d = a["dop853"]
        L += [f"### Независимый интегратор DOP853 (rtol {d['rtol']:g}, atol {d['atol']:g}), {d['n_copies']} копий до T = {d['T']:g}", "",
              "| копия | ушла | контрольных точек вне режима | λ (окно RK4 от конечного состояния) | метка в конце | z | z статистик последнего окна DOP853 | nfev | с |",
              "|---|---|---|---|---|---|---|---|---|"]
        for i, c in enumerate(d["copies"]):
            last = c["checks"][-1] if c["checks"] else None
            lw = c.get("last_window_dop")
            L.append(f"| {i} | {'' if c['escaped_at'] is None else f'{c['escaped_at']:.0f}'} | {sum(1 for q in c['checks'] if not q['on_regime'])}/{len(c['checks'])} | "
                     f"{fmt(last['lam']) if last else '—'} | {last['label'] if last else '—'} | {fmt(last['z'], 2) if last else '—'} | "
                     f"{(lw['label'] + ', z=' + fmt(lw['z'], 2)) if lw else '—'} | {c['nfev']} | {c['wall_s']:.0f} |")
        L += ["", f"Итог DOP853: на режиме в конце {d['n_on_regime_final_rk4window']}/{d['n_copies']} (окно RK4 от конечного состояния) и "
                  f"{d['n_on_regime_last_window_dop']}/{d['n_copies']} (статистики последнего окна самой DOP853-траектории); ушло {d['n_escaped']}; "
                  f"контрольных точек вне режима {d['n_checks_off']}; {d['wall_s']:.0f} с.", "",
              "**Вердикт (a).** " + a.get("verdict", "(заполняется по результатам)"), ""]
    b = out.get("b")
    if b:
        r = b["rk4"]
        L += [f"## (b) Хаос Jafari 2026, a3 = {b['a3']}", "",
              f"Система {b['system']}, dt = {b['dt']}, окно T_obs = {b['T_obs']:g}; {b['n_copies']} копий с относительным возмущением {b['noise']:g}; "
              f"λ эталона {b['lam_template']:.4f}, доля ушедших копий при калибровке (возмущение 1e-3) {b['escape_frac_cal']:.2f}.", "",
              "| T | живых (не ушло) | строго на эталоне | в окне | λ среднее | λ min | λ max |", "|---|---|---|---|---|---|---|"]
        sb = {q["T"]: q for q in r["survival_blocks"]}
        for q in r["survival"]:
            blk = sb.get(q["T"]) or (r["survival_blocks"][-1] if r["survival_blocks"] and q["T"] >= r["survival_blocks"][-1]["T"] else None)
            L.append(f"| {q['T']:g} | {q['alive']}/{q['n']} | {blk['strict'] if blk else '—'}/{q['n']} | {blk['current'] if blk else '—'}/{q['n']} | "
                     f"{fmt(blk['lam_mean']) if blk else '—'} | {fmt(blk['lam_min']) if blk else '—'} | {fmt(blk['lam_max']) if blk else '—'} |")
        f = r["ftle"]
        L += ["", f"Времена ухода (RK4): {r.get('t_esc_sorted', [])}; по декадам: {r.get('t_esc_decades', {})}; живых при T = {b['T']:g}: {r['n_alive_end']}/{b['n_copies']}.",
              f"Конечновременные λ по окнам (n = {f['n_windows']}): среднее {f['mean']:.4f}, СКО {f['std']:.4f}, min {f['min']:.4f}, max {f['max']:.4f}, "
              f"квантили 5/25/50/75/95 %: {f['q05']:.4f} / {f['q25']:.4f} / {f['median']:.4f} / {f['q75']:.4f} / {f['q95']:.4f}; "
              f"доля окон с λ < 0: {f['frac_negative']:.3f}, с λ < 0.005: {f['frac_below_0.005']:.3f}.", ""]
        d = b["dop853"]
        L += [f"DOP853 ({d['n_copies']} копий до T = {d['T']:g}): ушло {d['n_escaped']}, времена ухода {d['t_esc']}; на эталоне в конце "
              f"{d['n_on_regime_final_rk4window']}/{d['n_copies']} (окно RK4 от конечного состояния), {d['n_on_regime_last_window_dop']}/{d['n_copies']} "
              f"(статистики последнего окна DOP853); {d['wall_s']:.0f} с.", "",
              "| копия | ушла при T | контрольных точек вне эталона | λ в конце | z в конце |", "|---|---|---|---|---|"]
        for i, c in enumerate(d["copies"]):
            last = c["checks"][-1] if c["checks"] else None
            L.append(f"| {i} | {'' if c['escaped_at'] is None else f'{c['escaped_at']:.0f}'} | {sum(1 for q in c['checks'] if not q['on_regime'])}/{len(c['checks'])} | "
                     f"{fmt(last['lam']) if last else '—'} | {fmt(last['z'], 2) if last else '—'} |")
        L += ["", "**Вердикт (b).** " + b.get("verdict", "(заполняется по результатам)"), ""]
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="a,b")
    ap.add_argument("--T-a", type=float, default=1e5)
    ap.add_argument("--T-long", type=float, default=1e6)
    ap.add_argument("--T-dop", type=float, default=2e4)
    ap.add_argument("--T-b", type=float, default=1e4)
    ap.add_argument("--tag", default="", help="суффикс файлов результата (для проб)")
    args = ap.parse_args()
    global OUT
    if args.tag:
        OUT = ROOT / "results" / f"e62_lifetime_checks{args.tag}"
    import numba
    th = int(os.environ.get("RMQ_CPU", "4") or 4)
    numba.set_num_threads(max(1, min(th, numba.config.NUMBA_NUM_THREADS)))
    out = json.loads(OUT.with_suffix(".json").read_text(encoding="utf-8")) if OUT.with_suffix(".json").exists() else {}
    npz = dict(np.load(OUT.with_suffix(".npz"))) if OUT.with_suffix(".npz").exists() else {}
    t0 = time.time()
    for part in args.parts.split(","):
        rec, arrs = (part_a if part == "a" else part_b)(args, th)
        out[part] = rec
        npz.update(arrs)
        np.savez_compressed(OUT.with_suffix(".npz"), **npz)
        OUT.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        write_md(out)
    out["wall_s"] = round(time.time() - t0, 1)
    OUT.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(out)
    progress(6, 6, "готово")


if __name__ == "__main__":
    main()
