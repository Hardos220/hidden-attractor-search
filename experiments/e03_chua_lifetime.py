"""Протокол верификации на «скрытом аттракторе» Чуа (набор I): переходный хаос или аттрактор?

1. Кривая выживания: N копий состояния на хаотическом множестве (шум 1e-3),
   интегрирование окнами T=100 до T_max; в каждом окне — класс (хаос / периодика /
   затухание / уход) по показателю Ляпунова и флагам.
2. Зависимость от шага: dt = 0.01 и dt = 0.0025 (RK4).
3. Контроль интегратором высокой точности (DOP853, rtol 1e-11): периодичность по
   разбросу последовательных максимумов x.
4. Чувствительность долей к радиусу ухода esc_r (100 против 1000) в шаре r=3.
Выход: results/e03_chua_lifetime.json, results/e03_chua_survival.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.integrate import F_ESC, F_FP, F_LAM, nfeat, run_batch  # noqa: E402
from hsearch.identify import LAM_DECAY  # noqa: E402

LAM_CHAOS = 0.02


def classify_window(f):
    if f[F_ESC] > 0.5:
        return "escape"
    if f[F_FP] > 0.5 or f[F_LAM] < LAM_DECAY:
        return "decay"
    if f[F_LAM] > LAM_CHAOS:
        return "chaos"
    return "periodic"


def survival(s, state, n_copies, dt, T_win, T_max, rng, noise=1e-3):
    d = s.dim
    x = state + noise * (1.0 + np.abs(state)) * rng.normal(size=(n_copies, d))
    n_win = int(round(T_max / T_win))
    n_obs = int(round(T_win / dt))
    status = np.array(["chaos"] * n_copies, dtype=object)
    t_end = np.full(n_copies, np.nan)      # время, когда копия перестала быть хаотической
    end_state = np.array(["chaos"] * n_copies, dtype=object)
    lam_hist = np.zeros((n_copies, n_win))
    alive = np.ones(n_copies, dtype=bool)
    for w in range(n_win):
        feats = np.zeros((n_copies, nfeat(d))); finals = np.zeros((n_copies, d))
        s.batch(x, s.params, dt, 0, n_obs, s.esc_r, s.renorm_every, feats, finals)
        lam_hist[:, w] = feats[:, F_LAM]
        for j in range(n_copies):
            if not alive[j]:
                continue
            c = classify_window(feats[j])
            if c != "chaos":
                # подтверждение: периодика/затухание должна держаться два окна подряд (защита от шума λ)
                status[j] = c
                if w > 0 and classify_window_prev[j] == c or c == "escape":
                    alive[j] = False
                    t_end[j] = (w + 1) * T_win
                    end_state[j] = c
            else:
                status[j] = "chaos"
        classify_window_prev = [classify_window(feats[j]) for j in range(n_copies)]
        x = finals
        x[~alive] = state  # неживые копии «паркуем» (их результат зафиксирован)
    surv = [(w + 1) * T_win for w in range(n_win)]
    frac_alive = [float(np.mean(np.isnan(t_end) | (t_end > t))) for t in surv]
    return {"T": surv, "alive": frac_alive, "t_end": [None if np.isnan(v) else float(v) for v in t_end],
            "end_state": list(end_state), "lam_hist_mean_alive": [float(np.nanmean(lam_hist[np.isnan(t_end), w])) if np.isnan(t_end).any() else None for w in range(n_win)]}


def dop853_check(s, state, T_max=4000.0, T_win=100.0):
    """Высокоточное интегрирование: по каждому окну считаем число различных максимумов x."""
    from scipy.integrate import solve_ivp
    p = s.params
    a, b, g, m0, m1 = p

    def rhs(t, u):
        x, y, z = u
        fx = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1.0) - abs(x - 1.0))
        return [a * (y - x - fx), x - y + z, -(b * y + g * z)]

    out = []
    u = np.asarray(state, float)
    t = 0.0
    while t < T_max:
        sol = solve_ivp(rhs, (t, t + T_win), u, method="DOP853", rtol=1e-11, atol=1e-13,
                        max_step=0.05, dense_output=False, t_eval=np.linspace(t, t + T_win, 20001))
        xs = sol.y[0]
        dx = np.diff(xs)
        pk = np.where((dx[:-1] > 0) & (dx[1:] <= 0))[0] + 1
        peaks = xs[pk]
        n_distinct = len(np.unique(np.round(peaks, 3)))
        out.append({"t": t + T_win, "n_peaks": int(len(peaks)), "n_distinct_peaks": int(n_distinct),
                    "peak_std": float(np.std(peaks)) if len(peaks) else 0.0,
                    "rmax": float(np.max(np.linalg.norm(sol.y, axis=0)))})
        u = sol.y[:, -1]
        t += T_win
        if np.linalg.norm(u) > s.esc_r:
            out.append({"t": t, "escape": True})
            break
    return out


def main():
    rng = np.random.default_rng(3)
    s = systems.get("chua_I")
    res = {"system": "chua_I"}
    # состояние на хаотическом множестве: короткий переходный участок от известного IC
    feats, finals = run_batch(s, s.known["HA_a"][None, :], n_tr=20000, n_obs=10000)
    state = finals[0]
    res["state"] = state.tolist()
    res["lam_state_window"] = float(feats[0, F_LAM])
    print("состояние на множестве:", np.round(state, 4), "λ окна:", round(float(feats[0, F_LAM]), 4))
    for dt, n_copies, T_max in [(0.01, 256, 20000.0), (0.0025, 96, 10000.0)]:
        t0 = time.time()
        r = survival(s, state, n_copies, dt, 100.0, T_max, rng)
        r["dt"] = dt; r["n_copies"] = n_copies; r["time_s"] = round(time.time() - t0, 1)
        ends = {k: int(sum(1 for e in r["end_state"] if e == k)) for k in ("chaos", "periodic", "decay", "escape")}
        r["end_counts"] = ends
        res[f"survival_dt{dt}"] = r
        tt = np.array([v for v in r["t_end"] if v is not None])
        print(f"dt={dt}: копий {n_copies}, исходы {ends}, медиана времени ухода с множества: "
              f"{np.median(tt) if len(tt) else float('nan'):.0f}, доля живых при T_max: {r['alive'][-1]:.3f}  ({r['time_s']} с)")
    t0 = time.time()
    res["dop853"] = dop853_check(s, state)
    print(f"DOP853: {len(res['dop853'])} окон, {time.time() - t0:.1f} с; последние окна:",
          res["dop853"][-3:])
    # чувствительность к радиусу ухода
    from hsearch.identify import LAB_ESCAPE, calibrate_reference, fp_references_from_equilibria, match
    refs = fp_references_from_equilibria(s, names=["F0", "Fp", "Fm"])
    refs.append(calibrate_reference(s, "HA_a", s.known["HA_a"], rng))
    refs.append(calibrate_reference(s, "HA_b", s.known["HA_b"], rng))
    ics = s.sample_ball(10000, rng, radius=3.0)
    sens = {}
    for esc_r in (100.0, 1000.0):
        feats = np.zeros((len(ics), nfeat(3))); finals = np.zeros((len(ics), 3))
        s.batch(ics, s.params, s.dt, s.n_tr, s.n_obs, esc_r, s.renorm_every, feats, finals)
        labels, z = match(s, feats, finals, refs)
        sens[f"esc_r{int(esc_r)}"] = {"inf": float((labels == LAB_ESCAPE).mean()),
                                      **{r.name: float((labels == k).mean()) for k, r in enumerate(refs)},
                                      "unknown": float((labels == -2).mean())}
        print(f"esc_r={esc_r}: {sens[f'esc_r{int(esc_r)}']}")
    res["esc_sensitivity_r3"] = sens
    (ROOT / "results" / "e03_chua_lifetime.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    for key in ("survival_dt0.01", "survival_dt0.0025"):
        r = res[key]
        ax.plot(r["T"], r["alive"], label=f"RK4, dt={r['dt']}, N={r['n_copies']}")
    ax.set_xlabel("t"); ax.set_ylabel("доля копий, остающихся хаотическими"); ax.set_ylim(0, 1.02)
    ax.set_title("Чуа, набор I: кривая выживания на «скрытом» хаотическом множестве")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "results" / "e03_chua_survival.png")
    print("сохранено results/e03_chua_lifetime.json, results/e03_chua_survival.png")


if __name__ == "__main__":
    main()
