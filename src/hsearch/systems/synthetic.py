"""Уровень D: синтетическое семейство с аналитически известной мерой бассейна.

Конструкция «вложение с воротами». Базовая трёхмерная система dx/dt = F(x) со скрытым
аттрактором HA и тривиальным аттрактором (равновесием) e вкладывается в R^{3+m}:

    dx/dt = F(x) - g(w) (x - e),      dw/dt = -kappa w,
    g(w) = H * sigma((|w| - rho) / eps),   sigma(u) = 1 / (1 + exp(-u)).

Пока |w(t)| > rho «ворота» g включены и сильно тянут x к e; w затухает экспоненциально,
так что ворота выключаются в момент t* = ln(|w0| / rho) / kappa. При больших H траектория
с |w0| > rho успевает попасть в бассейн e, поэтому бассейн HA в области
B = B_x × [-1, 1]^m есть приблизительно W_HA × {|w0| <= rho}:

    p(rho, m) ≈ p_W · V_m(rho) / 2^m,   V_m(rho) = pi^{m/2} rho^m / Gamma(m/2 + 1).

Равновесия полной системы — равновесия базовой × {w = 0} (там g = 0), поэтому скрытость
HA сохраняется. Истинная мера измеряется перебором при больших p и экстраполируется.
"""
import math

import numpy as np
from numba import njit

from . import System, get, register


def analytic_p(p_base, m, rho):
    """Аналитическая оценка доли бассейна HA в B = B_x × [-1, 1]^m."""
    vol_ball = math.pi ** (m / 2.0) * rho ** m / math.gamma(m / 2.0 + 1.0)
    return p_base * vol_ball / 2.0 ** m


def rho_for_p(p_target, p_base, m):
    """Радиус ворот rho, дающий целевую долю p_target при базовой доле p_base."""
    vol_needed = p_target / p_base * 2.0 ** m
    return (vol_needed * math.gamma(m / 2.0 + 1.0) / math.pi ** (m / 2.0)) ** (1.0 / m)


def make_edge_embedded(base_name, m, rho, kappa=1.0, name=None, w_half=1.0):
    """Семейство D2 «краевое состояние»: развязанное вложение
        dx/dt = F(x),   dw/dt = kappa * w * (|w|^2 / rho^2 - 1).
    Сфера |w| = rho отталкивающая: при |w0| < rho координаты w затухают к нулю, и x следует
    базовой системе; при |w0| > rho происходит взрыв |w| за конечное время (уход). Бассейн
    скрытого аттрактора — ровно W_HA × {|w0| < rho}: p = p_W V_m(rho) / 2^m (точно).
    Время ухода при |w0| = rho (1 + delta) растёт как ln(1/(2 delta)) / (2 kappa): у границы
    бассейна переходный процесс нетипично долог — это и есть градиентный сигнал канала B.
    Равновесия — равновесия базы × ({0} ∪ сфера); их окрестности не пересекают бассейн HA,
    если равновесие базы устойчиво (se3), так что скрытость сохраняется.
    """
    base = get(base_name)
    d0 = base.dim
    d = d0 + m
    base_rhs = base.rhs
    n_base_p = len(base.params)

    @njit(cache=False)
    def rhs(x, p, out):
        base_rhs(x[:d0], p[:n_base_p], out[:d0])
        kappa_ = p[n_base_p]; rho_ = p[n_base_p + 1]
        r2 = 0.0
        for i in range(d0, d):
            r2 += x[i] * x[i]
        g = kappa_ * (r2 / (rho_ * rho_) - 1.0)
        for i in range(d0, d):
            out[i] = g * x[i]

    params = np.concatenate([base.params, [kappa, rho]])
    lo = np.concatenate([base.box_lo, -w_half * np.ones(m)])
    hi = np.concatenate([base.box_hi, w_half * np.ones(m)])
    known = {k: np.concatenate([v, np.zeros(m)]) for k, v in base.known.items()}
    eqs = [np.concatenate([np.asarray(e, float), np.zeros(m)]) for e in base.equilibria]
    nm = name or f"edge_{base_name}_m{m}_rho{rho:g}"
    s = System(
        name=nm, kind="flow", dim=d, rhs=rhs, params=params, box_lo=lo, box_hi=hi,
        dt=base.dt, n_tr=base.n_tr, n_obs=base.n_obs, esc_r=base.esc_r,
        renorm_every=base.renorm_every, equilibria=eqs, known=known, hidden=list(base.hidden),
        refs=list(base.refs), notes=f"Краевое вложение {base_name} в R^{d}: m={m}, rho={rho}, kappa={kappa}.",
        level="D2",
    )
    s.meta = {"base": base_name, "m": m, "rho": rho, "kappa": kappa, "family": "edge"}
    return register(s)


def make_embedded(base_name, m, rho, H=50.0, eps=None, kappa=0.5, name=None, w_half=1.0):
    base = get(base_name)
    if base.kind != "flow":
        raise ValueError("базовая система должна быть потоком")
    if eps is None:
        # Хвост сигмоиды смещает эффективный радиус: delta ~ (eps/rho) ln(H eps / (kappa rho)).
        # При eps = rho/400 аргумент логарифма < 1 и смещение пренебрежимо; ширина переключения
        # по времени eps/(kappa rho) = 1/200 — порядка одного шага RK4 (однократная локальная ошибка).
        eps = rho / 400.0
    d0 = base.dim
    d = d0 + m
    base_rhs = base.rhs
    # точка e — первое равновесие базовой системы (тривиальный аттрактор), иначе 0
    e_pt = np.asarray(base.equilibria[0], float) if base.equilibria else np.zeros(d0)
    n_base_p = len(base.params)

    @njit(cache=False)
    def rhs(x, p, out):
        # p = [params базовой..., H, eps, rho, kappa, e_0..e_{d0-1}]
        base_rhs(x[:d0], p[:n_base_p], out[:d0])
        H_ = p[n_base_p]; eps_ = p[n_base_p + 1]; rho_ = p[n_base_p + 2]; kappa_ = p[n_base_p + 3]
        wn = 0.0
        for i in range(d0, d):
            wn += x[i] * x[i]
        wn = math.sqrt(wn)
        u = (wn - rho_) / eps_
        if u > 40.0:
            g = H_
        elif u < -40.0:
            g = 0.0
        else:
            g = H_ / (1.0 + math.exp(-u))
        for i in range(d0):
            out[i] -= g * (x[i] - p[n_base_p + 4 + i])
        for i in range(d0, d):
            out[i] = -kappa_ * x[i]

    params = np.concatenate([base.params, [H, eps, rho, kappa], e_pt])
    lo = np.concatenate([base.box_lo, -w_half * np.ones(m)])
    hi = np.concatenate([base.box_hi, w_half * np.ones(m)])
    known = {k: np.concatenate([v, np.zeros(m)]) for k, v in base.known.items()}
    eqs = [np.concatenate([np.asarray(e, float), np.zeros(m)]) for e in base.equilibria]
    nm = name or f"emb_{base_name}_m{m}_rho{rho:g}"
    # ворота выключаются за время ln(sqrt(m)/rho)/kappa; добавляем запас к переходному участку
    extra = int(math.ceil((math.log(math.sqrt(m) * w_half / rho) / kappa + 5.0) / base.dt))
    s = System(
        name=nm, kind="flow", dim=d, rhs=rhs, params=params, box_lo=lo, box_hi=hi,
        dt=base.dt, n_tr=base.n_tr + extra, n_obs=base.n_obs, esc_r=base.esc_r,
        renorm_every=base.renorm_every, equilibria=eqs, known=known, hidden=list(base.hidden),
        refs=list(base.refs), notes=f"Вложение {base_name} в R^{d}: m={m}, rho={rho}, H={H}, eps={eps}, kappa={kappa}.",
        level="D",
    )
    s.meta = {"base": base_name, "m": m, "rho": rho, "H": H, "eps": eps, "kappa": kappa}
    return register(s)
