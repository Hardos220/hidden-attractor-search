"""Мера скрытости h_eps: доля начальных условий из eps-окрестностей равновесий (и вдоль их
неустойчивых направлений), попадающих в бассейн каждого аттрактора.

Для каждой системы с равновесиями и eps in (1e-3, 1e-2, 1e-1) * L: n точек в шаре радиуса eps
вокруг каждого равновесия + точки e ± eps v по неустойчивым собственным направлениям якобиана.
Скрытый аттрактор: h_eps = 0 при малых eps. Выход: results/e11_hiddenness.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.baselines.geometric import _jac  # noqa: E402
from hsearch.identify import LAB_ESCAPE, LAB_UNKNOWN, build_reference_set, match, scale  # noqa: E402
from hsearch.integrate import run_batch  # noqa: E402

SYSTEMS = ["chua_I", "chua_II", "wangchen2", "se3", "se11", "rabinovich_100", "lorenz_transient"]


def main(names=None, n=400):
    names = names or SYSTEMS
    out = {}
    for name in names:
        rng = np.random.default_rng(11)
        s = systems.get(name)
        L = scale(s)
        refs, *_ = build_reference_set(s, rng, n_sample=5000, verbose=False)
        # полный список равновесий: заданные + найденные как неподвижные эталоны
        eqs = [np.asarray(e, float) for e in s.equilibria]
        for r in refs:
            if r.kind == "fp" and r.point is not None and all(np.linalg.norm(r.point - e) > 1e-3 * L for e in eqs):
                eqs.append(r.point.copy())
        J = _jac(s)
        rec = {"L": L, "refs": [(r.name, r.kind) for r in refs], "equilibria": [e.tolist() for e in eqs], "eps": {}}
        print(f"=== {name}: равновесий {len(eqs)}, эталоны {[r.name for r in refs]}")
        for eps_rel in (1e-3, 1e-2, 1e-1):
            eps = eps_rel * L
            rows = []
            for e in eqs:
                v = rng.normal(size=(n, s.dim)); v /= np.linalg.norm(v, axis=1, keepdims=True)
                r = eps * rng.random(n) ** (1.0 / s.dim)
                pts = e + v * r[:, None]
                # неустойчивые направления
                w, V = np.linalg.eig(J(e))
                unst = [np.real(V[:, k]) for k in range(len(w)) if np.real(w[k]) > 1e-6]
                for u in unst:
                    u = u / (np.linalg.norm(u) + 1e-12)
                    pts = np.vstack([pts, e + eps * u, e - eps * u])
                rows.append(pts)
            ics = np.vstack(rows)
            feats, finals = run_batch(s, ics)
            labels, z = match(s, feats, finals, refs)
            frac = {"escape": float((labels == LAB_ESCAPE).mean()), "unknown": float((labels == LAB_UNKNOWN).mean())}
            for k, r in enumerate(refs):
                frac[r.name] = float((labels == k).mean())
            rec["eps"][str(eps_rel)] = {"eps": eps, "n": int(len(ics)), "fractions": frac}
            print(f"  eps = {eps_rel} L: " + ", ".join(f"{k}={v:.3f}" for k, v in frac.items() if v > 0))
        out[name] = rec
        (ROOT / "results" / "e11_hiddenness.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
