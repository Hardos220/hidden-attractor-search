"""Классические геометрические эвристики локализации как генераторы стартовых точек.

* perpetual_points(system)   — перпетуальные точки: J(x) F(x) = 0, F(x) != 0 (Prasad 2015).
* nullcline_samples(system)  — точки на поверхностях критической скорости x_i' = 0 (Godara et al. 2018).
* connecting_curve_samples(system) — точки на соединяющих кривых a ∧ v = 0 (Gilmore et al. 2010;
                                     Guan & Xie 2021): минимизация |a ∧ v|^2 из случайных стартов.
Все функции возвращают массивы точек (k, d) внутри области B системы.
"""
import numpy as np
from scipy.optimize import brentq, least_squares, root


def _F(system):
    out = np.empty(system.dim)

    def f(x):
        system.rhs(np.asarray(x, float), system.params, out)
        return out.copy()
    return f


def _jac(system, h=1e-6):
    f = _F(system)
    d = system.dim

    def J(x):
        x = np.asarray(x, float)
        M = np.empty((d, d))
        for i in range(d):
            e = np.zeros(d); e[i] = h * (1.0 + abs(x[i]))
            M[:, i] = (f(x + e) - f(x - e)) / (2 * e[i])
        return M
    return J


def _dedupe(points, tol=1e-3):
    keep = []
    for p in points:
        if all(np.linalg.norm(p - q) > tol * (1 + np.linalg.norm(q)) for q in keep):
            keep.append(p)
    return np.array(keep) if keep else np.zeros((0, len(points[0]) if len(points) else 0))


def perpetual_points(system, rng, n_starts=400, eq_tol=1e-6):
    """Перпетуальные точки: решения a(x) = J(x) F(x) = 0 с F(x) != 0, найденные из случайных стартов в B."""
    f = _F(system); J = _jac(system)

    def acc(x):
        return J(x) @ f(x)

    sols = []
    for _ in range(n_starts):
        x0 = system.sample_box(1, rng)[0]
        try:
            r = root(acc, x0, method="hybr", tol=1e-10)
        except Exception:
            continue
        if not r.success:
            continue
        x = r.x
        if np.linalg.norm(acc(x)) > 1e-7 * (1 + np.linalg.norm(f(x))):
            continue
        if np.linalg.norm(f(x)) < eq_tol:
            continue          # это равновесие, не перпетуальная точка
        if np.any(x < system.box_lo - 1e-9) or np.any(x > system.box_hi + 1e-9):
            continue
        sols.append(x)
    return _dedupe(sols)


def nullcline_samples(system, rng, i, n=2000, tries=20):
    """Случайные точки на нульклине x_i' = 0: по остальным координатам — равномерно в B,
    по координате i — корень F_i на отрезке [lo_i, hi_i] (если есть смена знака)."""
    f = _F(system)
    d = system.dim
    pts = []
    lo, hi = system.box_lo, system.box_hi
    attempts = 0
    while len(pts) < n and attempts < n * tries:
        attempts += 1
        x = system.sample_box(1, rng)[0]

        def g(t):
            y = x.copy(); y[i] = t
            return f(y)[i]
        grid = np.linspace(lo[i], hi[i], 33)
        vals = np.array([g(t) for t in grid])
        s = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        if len(s) == 0:
            continue
        k = s[rng.integers(len(s))]
        try:
            t = brentq(g, grid[k], grid[k + 1], xtol=1e-10)
        except ValueError:
            continue
        y = x.copy(); y[i] = t
        pts.append(y)
    return np.array(pts) if pts else np.zeros((0, d))


def connecting_curve_samples(system, rng, n=500, n_starts=4000, tol=1e-8):
    """Точки на соединяющих кривых: минимизация |a ∧ v|^2, a = J F, v = F, из случайных стартов в B.
    В 3D a ∧ v — векторное произведение; в общем случае — все 2x2 миноры (компоненты бивектора)."""
    f = _F(system); J = _jac(system)
    d = system.dim

    def wedge(x):
        v = f(x); a = J(x) @ v
        comps = [a[i] * v[j] - a[j] * v[i] for i in range(d) for j in range(i + 1, d)]
        return np.array(comps) / (1.0 + np.linalg.norm(v) ** 2)

    pts = []
    for _ in range(n_starts):
        x0 = system.sample_box(1, rng)[0]
        try:
            r = least_squares(wedge, x0, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
        except Exception:
            continue
        x = r.x
        if np.linalg.norm(wedge(x)) > tol:
            continue
        if np.linalg.norm(f(x)) < 1e-8:
            continue          # равновесие
        if np.any(x < system.box_lo) or np.any(x > system.box_hi):
            continue
        pts.append(x)
        if len(pts) >= n:
            break
    return np.array(pts) if pts else np.zeros((0, d))
