"""Стоимость классических методов на системах уровня A (в сравнении с перебором из e02).

Для каждой системы и её скрытой цели (цели берутся из results/e02_uniform.json):
  * PP  — перпетуальные точки: каждая интегрируется один раз; успех, если хоть одна ведёт к цели;
  * CVS — выборка на нульклинах x_i' = 0 (по n точек на каждой): доля p_i попаданий в цель,
          N_0.9 по лучшей нульклине (процедура I Godara и др.);
  * CC  — выборка на соединяющих кривых: доля попаданий, N_0.9;
  * REC — сетка + рекуррентность (аналог Attractors.jl): случайные IC обрабатываются
          последовательно до первого обнаружения цели; стоимость — число IC и число шагов;
          R повторов, медиана.
Выход: results/e08_classic.json, results/e08_classic.md
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.baselines.geometric import connecting_curve_samples, nullcline_samples, perpetual_points  # noqa: E402
from hsearch.baselines.recurrence import RecurrenceFinder  # noqa: E402
from hsearch.identify import LAB_UNKNOWN, build_reference_set, match  # noqa: E402
from hsearch.integrate import run_batch  # noqa: E402

SYSTEMS = ["chua_I", "ne4", "wangchen1", "wei_ne2", "wangchen2", "se3", "se11", "le1", "rabinovich_100"]
GRID_N = {"chua_I": 120, "default": 100}


def n_delta(p, delta=0.1):
    return float("inf") if p <= 0 else float(np.ceil(np.log(delta) / np.log1p(-p)))


def hits(system, refs, target_idx, ics):
    if len(ics) == 0:
        return 0, 0
    feats, finals = run_batch(system, ics)
    labels, z = match(system, feats, finals, refs)
    return int(np.sum(np.isin(labels, target_idx))), len(ics)


def main(names=None):
    names = names or SYSTEMS
    out_path = ROOT / "results" / "e08_classic.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    e02 = json.loads((ROOT / "results" / "e02_uniform.json").read_text(encoding="utf-8"))
    for name in names:
        rng = np.random.default_rng(8)
        s = systems.get(name)
        print(f"=== {name}")
        refs, ics0, feats0, finals0, labels0, hist = build_reference_set(s, rng, n_sample=5000, verbose=False)
        targets = [h for h in e02.get(name, {}).get("hidden", {}) if e02[name]["hidden"][h]["p"] > 0]
        tidx = [k for k, r in enumerate(refs) if r.name in targets or (r.kind in ("chaotic", "periodic") and r.name in s.hidden)]
        if not tidx:
            tidx = [k for k, r in enumerate(refs) if r.kind in ("chaotic", "periodic")]
        rec = {"targets": [refs[k].name for k in tidx], "p_uniform": {refs[k].name: e02.get(name, {}).get("fractions", {}).get(refs[k].name) for k in tidx}}
        # --- перпетуальные точки
        t0 = time.time()
        pp = perpetual_points(s, rng, n_starts=300)
        h, n = hits(s, refs, tidx, pp)
        rec["PP"] = {"n_points": int(n), "hits": int(h), "points": np.round(pp, 4).tolist(), "time_s": round(time.time() - t0, 1)}
        print(f"  PP: {n} точек, попаданий {h}")
        # --- нульклины
        t0 = time.time()
        cvs = {}
        for i in range(s.dim):
            pts = nullcline_samples(s, rng, i, n=1000)
            h, n = hits(s, refs, tidx, pts)
            p = h / n if n else 0.0
            cvs[f"x{i}"] = {"n": int(n), "hits": int(h), "p": p, "N_0.9": n_delta(p)}
        rec["CVS"] = cvs
        rec["CVS"]["time_s"] = round(time.time() - t0, 1)
        print("  CVS:", {k: (v["p"] if isinstance(v, dict) else v) for k, v in cvs.items()})
        # --- соединяющие кривые
        t0 = time.time()
        cc = connecting_curve_samples(s, rng, n=400, n_starts=3000)
        h, n = hits(s, refs, tidx, cc)
        p = h / n if n else 0.0
        rec["CC"] = {"n": int(n), "hits": int(h), "p": p, "N_0.9": n_delta(p), "time_s": round(time.time() - t0, 1)}
        print(f"  CC: {n} точек, p={p:.4f}")
        # --- сетка + рекуррентность
        t0 = time.time()
        R = 8
        costs = []
        for r in range(R):
            rr = np.random.default_rng(100 + r)
            finder = RecurrenceFinder(s, GRID_N.get(name, GRID_N["default"]))
            found_ic = None
            n_ic = 0
            max_ic = int(min(20000, 20 * max(n_delta(max(rec["p_uniform"].values() or [1e-9]) or 1e-9), 50)))
            while n_ic < max_ic:
                batch_ics = s.sample_box(50, rr)
                labels, steps = finder.find(batch_ics)
                n_ic += len(batch_ics)
                seeds = finder.attractor_seeds()
                if seeds:
                    sd = np.array([seeds[k] for k in sorted(seeds)])
                    fs, fn = run_batch(s, sd, n_tr=0)
                    lab, z = match(s, fs, fn, refs)
                    if np.any(np.isin(lab, tidx)):
                        # число IC до первого попадания: первая метка, равная найденному аттрактору
                        kfound = [k for k, l in zip(sorted(seeds), lab) if l in tidx][0]
                        found_ic = n_ic
                        break
            costs.append({"n_ic": found_ic, "steps": finder.total_steps, "n_attractors": finder.n_att,
                          "found": found_ic is not None})
        rec["REC"] = {"runs": costs, "grid_n": GRID_N.get(name, GRID_N["default"]),
                      "median_n_ic": float(np.median([c["n_ic"] for c in costs if c["found"]])) if any(c["found"] for c in costs) else None,
                      "median_steps": float(np.median([c["steps"] for c in costs if c["found"]])) if any(c["found"] for c in costs) else None,
                      "success_rate": float(np.mean([c["found"] for c in costs])), "time_s": round(time.time() - t0, 1)}
        print(f"  REC: успех {rec['REC']['success_rate']:.2f}, медиана IC {rec['REC']['median_n_ic']}, медиана шагов {rec['REC']['median_steps']}")
        out[name] = rec
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["| Система | цель | p (перебор) | N_0.9 перебор | PP: точек/попаданий | CVS: лучшая p / N_0.9 | CC: p / N_0.9 | REC: медиана IC |",
             "|---|---|---|---|---|---|---|---|"]
    for name, rec in out.items():
        for t in rec["targets"]:
            pu = rec["p_uniform"].get(t) or 0.0
            best = max((v for k, v in rec["CVS"].items() if isinstance(v, dict)), key=lambda v: v["p"])
            lines.append(f"| {name} | {t} | {pu:.4g} | {n_delta(pu):.0f} | {rec['PP']['n_points']}/{rec['PP']['hits']} | "
                         f"{best['p']:.3f} / {best['N_0.9']:.0f} | {rec['CC']['p']:.3f} / {rec['CC']['N_0.9']:.0f} | {rec['REC']['median_n_ic']} |")
    (ROOT / "results" / "e08_classic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
