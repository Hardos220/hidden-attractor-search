"""Резервуарный вычислитель (ESN) в замкнутом режиме как автономное отображение в R^N (этап 3).

Обучение: Lorenz (sigma, rho, beta) = (10, 28, 8/3), шаг выборки h; резервуар
    r_{k+1} = tanh(W r_k + W_in u_k + b),   W_out = ridge(r -> u_{k+1}).
Замкнутый цикл: u_k := W_out r_k  =>  r_{k+1} = tanh(A r_k + b),  A = W + W_in W_out.
Отображение регистрируется как System kind='map' размерности N; параметры p = [A (N*N), b (N)].
Область B — куб [-1, 1]^N (образ tanh) либо окрестность обучающих состояний.
"""
import numpy as np
from numba import njit

from . import System, register
from .flows3d import lorenz_rhs
from ..integrate import make_flow_kernels


def lorenz_data(n_steps, h=0.02, dt=0.002, x0=(1.0, 1.0, 1.0), transient=2000, params=(10.0, 28.0, 8.0 / 3.0)):
    """Траектория классической системы Лоренца с шагом выборки h (RK4 с шагом dt)."""
    _, traj = make_flow_kernels(lorenz_rhs, 3)
    p = np.array(params, float)
    sub = int(round(h / dt))
    n_total = (n_steps + transient) * sub
    full = traj(np.asarray(x0, float), p, dt, n_total, 1e6)
    return full[transient * sub::sub][:n_steps]


def train_esn(u, n_res=300, spectral_radius=1.1, input_scale=0.5, density=0.05, leak=1.0, ridge=1e-6,
              washout=500, seed=0):
    """Обучение ESN на многомерном ряде u (T, m); возвращает словарь с матрицами и статистиками."""
    rng = np.random.default_rng(seed)
    T, m = u.shape
    W = rng.normal(size=(n_res, n_res)) * (rng.random((n_res, n_res)) < density)
    ev = np.max(np.abs(np.linalg.eigvals(W)))
    W *= spectral_radius / ev
    W_in = rng.uniform(-input_scale, input_scale, size=(n_res, m))
    b = rng.uniform(-0.2, 0.2, size=n_res)
    scale = u.std(axis=0)
    un = u / scale
    R = np.zeros((T, n_res))
    r = np.zeros(n_res)
    for k in range(T):
        r = (1 - leak) * r + leak * np.tanh(W @ r + W_in @ un[k] + b)
        R[k] = r
    X = R[washout:-1]
    Y = un[washout + 1:]
    W_out = np.linalg.solve(X.T @ X + ridge * np.eye(n_res), X.T @ Y).T   # (m, n_res)
    train_err = np.sqrt(np.mean((X @ W_out.T - Y) ** 2))
    A = W + W_in @ W_out
    return {"W": W, "W_in": W_in, "W_out": W_out, "b": b, "A": A, "scale": scale, "train_rmse": float(train_err),
            "R_train": R[washout:], "u_norm": un, "n_res": n_res}


@njit(cache=False)
def esn_map(x, p, out):
    n = x.shape[0]
    # p = [A (n*n), b (n)]
    for i in range(n):
        acc = p[n * n + i]
        base = i * n
        for j in range(n):
            acc += p[base + j] * x[j]
        out[i] = np.tanh(acc)


def make_esn_system(name="esn_lorenz", n_res=300, seed=0, n_train=20000, box="cube", **esn_kw):
    u = lorenz_data(n_train)
    esn = train_esn(u, n_res=n_res, seed=seed, **esn_kw)
    A = np.ascontiguousarray(esn["A"])
    params = np.concatenate([A.ravel(), esn["b"]])
    R = esn["R_train"]
    if box == "cube":
        lo, hi = -np.ones(n_res), np.ones(n_res)
    else:
        lo, hi = R.min(axis=0) - 0.1, R.max(axis=0) + 0.1
    s = System(
        # dt = h: одна итерация замкнутого резервуара соответствует шагу выборки h = 0.02
        name=name, kind="map", dim=n_res, rhs=esn_map, params=params, box_lo=lo, box_hi=hi,
        dt=0.02, n_tr=4000, n_obs=2000, esc_r=1e6, renorm_every=1,
        equilibria=[],
        known={"LORENZ": R[-1].copy()},
        hidden=[],
        refs=["JaegerHaas2004", "Pathak2017Chaos", "LuHuntOtt2018"],
        notes=f"ESN N={n_res}, обучен на Лоренце (h=0.02), замкнутый цикл; RMSE обучения {esn['train_rmse']:.3g}.",
        level="E",
    )
    s.esn = esn
    return register(s)
