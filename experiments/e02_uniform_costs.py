"""Базовая стоимость равномерного перебора для систем уровня A.

Для каждой системы: построение набора эталонов (известные IC + обнаружение),
доли бассейнов p в области B по выборке n_sample, стоимость N_0.9 = ln(10)/(-ln(1-p)).
Выход: results/e02_uniform.json, results/e02_uniform.md
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.identify import build_reference_set, label_fractions  # noqa: E402

SYSTEMS = {
    "chua_I": 50000, "chua_II": 50000, "ne4": 50000, "wangchen1": 50000, "wei_ne2": 50000,
    "wangchen2": 50000, "se3": 50000, "se11": 50000, "le1": 50000, "rabinovich_100": 20000,
}


def n_delta(p, delta=0.1):
    if p <= 0:
        return float("inf")
    return float(np.ceil(np.log(delta) / np.log1p(-p)))


def main(names=None):
    names = names or list(SYSTEMS)
    out_path = ROOT / "results" / "e02_uniform.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for name in names:
        rng = np.random.default_rng(2)
        s = systems.get(name)
        n = SYSTEMS.get(name, 20000)
        print(f"=== {name}: n={n}, box={s.box_lo.tolist()}..{s.box_hi.tolist()}")
        t0 = time.time()
        refs, ics, feats, finals, labels, hist = build_reference_set(s, rng, n_sample=n)
        frac = label_fractions(labels, refs)
        rec = {"n_sample": n, "box_lo": s.box_lo.tolist(), "box_hi": s.box_hi.tolist(),
               "dt": s.dt, "T_tr": s.T_tr, "T_obs": s.T_obs, "time_s": round(time.time() - t0, 1),
               "refs": [{"name": r.name, "kind": r.kind, "lam": round(float(r.lam), 4),
                         "escape_frac": round(float(r.escape_frac), 3),
                         "point": None if r.point is None else [round(float(v), 4) for v in r.point],
                         "example": None if r.example is None else [round(float(v), 5) for v in r.example]}
                        for r in refs],
               "fractions": {k: round(v, 6) for k, v in frac.items()},
               "hidden": {}, "history": hist}
        for h in s.hidden:
            p = frac.get(h, 0.0)
            rec["hidden"][h] = {"p": p, "N_0.9": n_delta(p), "N_0.5": n_delta(p, 0.5)}
        for r in refs:
            if r.name.startswith("new") and r.kind in ("periodic", "chaotic"):
                p = frac.get(r.name, 0.0)
                rec["hidden"][r.name] = {"p": p, "N_0.9": n_delta(p), "N_0.5": n_delta(p, 0.5), "discovered": True}
        out[name] = rec
        print("  эталоны:", [(r.name, r.kind, round(r.lam, 3)) for r in refs])
        print("  доли:", {k: round(v, 4) for k, v in frac.items()})
        print("  цели:", rec["hidden"], f"  ({rec['time_s']} с)")
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # таблица
    lines = ["| Система | n | p(∞) | p(unknown) | Цель | p | N_0.9 |", "|---|---|---|---|---|---|---|"]
    for name, rec in out.items():
        f = rec["fractions"]
        first = True
        for h, v in rec["hidden"].items():
            lines.append(f"| {name if first else ''} | {rec['n_sample'] if first else ''} | "
                         f"{f['escape']:.3f} | {f['unknown']:.4f} | {h} | {v['p']:.4g} | {v['N_0.9']:.0f} |")
            first = False
        if first:
            lines.append(f"| {name} | {rec['n_sample']} | {f['escape']:.3f} | {f['unknown']:.4f} | — | — | — |")
    (ROOT / "results" / "e02_uniform.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
