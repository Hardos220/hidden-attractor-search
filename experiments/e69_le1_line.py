"""LE1: how hidden is the attractor with respect to the line of equilibria x = y = 0? (2026-11-15)

The line consists of equilibria (0, 0, z); transversally they are stable for z < 0 and unstable for z > 0 (eigenvalues 0
and the roots of l^2 - z l + 1 = 0). Trajectories are started near points (0, 0, z0), z0 in [-3, 4] (140 values, z0 = 0
excluded), at radii 1e-8 and 1e-4 in 24 random directions, integrated by RK4 (step 0.005) for T = 600 and classified:
escape (|x| > 100), on the hidden attractor (within 0.05 of a reference sample of the attractor in at least 90 of the last
100 checks, every 0.2 time units), back to the line (distance to the line < 1e-3), other.

The reference sample is computed here from the known point of the basin (0, 0.5, 0.5) of the registry
(src/hsearch/systems/flows3d.py): transient 500 time units, then 2e4 points every 0.05 time units.

    python experiments/e69_le1_line.py
writes results/e69_le1_line.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e69_le1_line.json"
A, B = 15.0, 1.0


def f(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    return np.stack([y, -x + y * z, -x - A * x * y - B * x * z], axis=1)


def rk4(X, dt):
    k1 = f(X)
    k2 = f(X + 0.5 * dt * k1)
    k3 = f(X + 0.5 * dt * k2)
    k4 = f(X + dt * k3)
    return X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def reference(dt=0.005):
    X = np.array([[0.0, 0.5, 0.5]])
    for _ in range(int(500 / dt)):
        X = rk4(X, dt)
    pts = []
    for k in range(20000 * 10):
        X = rk4(X, dt)
        if k % 10 == 0:
            pts.append(X[0].copy())
    P = np.array(pts)
    assert np.abs(P).max() < 100 and np.hypot(P[:, 0], P[:, 1]).max() > 0.1, "reference orbit did not settle on the attractor"
    return P


def main():
    att = reference()
    tree = cKDTree(att)
    rng = np.random.default_rng(5)
    z0 = np.round(np.linspace(-3, 4, 141), 4)
    z0 = z0[z0 != 0]
    radii, nd = [1e-8, 1e-4], 24
    S, Z, Rr = [], [], []
    for r in radii:
        for zz in z0:
            d = rng.normal(size=(nd, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            S.append(np.array([0.0, 0.0, zz]) + r * d)
            Z += [zz] * nd
            Rr += [r] * nd
    X, Z, Rr = np.vstack(S), np.array(Z), np.array(Rr)
    alive = np.ones(len(X), bool)
    hit = np.zeros(len(X))
    dt, n = 0.005, 120000
    for k in range(n):
        X[alive] = rk4(X[alive], dt)
        alive &= (np.abs(X) < 100).all(axis=1)
        if k >= n - 4000 and k % 40 == 0:
            d, _ = tree.query(np.where(alive[:, None], X, 1e3))
            hit += (d < 0.05) & alive
    on = alive & (hit >= 90)
    line = alive & ~on & (np.hypot(X[:, 0], X[:, 1]) < 1e-3)
    other = alive & ~on & ~line
    out = {"z0_range": [float(z0.min()), float(z0.max())], "n_z0": int(len(z0)), "radii": radii, "directions": nd,
           "step": dt, "T": n * dt, "reference": "orbit from (0, 0.5, 0.5), transient 500, 2e4 points"}
    for r in radii:
        m = Rr == r
        share = {float(zz): float(np.mean(on[m & (Z == zz)])) for zz in z0}
        zh = [zz for zz, s in share.items() if s > 0]
        out[f"r={r:g}"] = {
            "starts": int(m.sum()), "attractor": int((on & m).sum()), "line": int((line & m).sum()),
            "escape": int((~alive & m).sum()), "other": int((other & m).sum()),
            "attractor_from_z0_lt_0": int((on & m & (Z < 0)).sum()),
            "z0_with_hits": [min(zh), max(zh)] if zh else None, "n_z0_with_hits": len(zh),
            "max_share_over_z0": max(share.values()),
            "mean_share_z0_0.3_1.0": float(np.mean([s for zz, s in share.items() if 0.3 < zz <= 1.0])),
            "share_by_z0": share}
        print(f"r={r:g}: " + json.dumps({k: v for k, v in out[f'r={r:g}'].items() if k != 'share_by_z0'}), flush=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
