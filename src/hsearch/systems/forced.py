"""Периодически возбуждаемые осцилляторы как стробоскопические отображения (2026-10-13).

Осциллятор ван дер Поля–Дуффинга с внешним возбуждением (Chudzik, Perlikowski, Stefański, Kapitaniak 2011;
Brezetskyi, Dudkowski, Kapitaniak 2015):
    x'' - alpha (1 - x^2) x' + x^3 = F sin(omega t),   alpha = 0.2, F = 1.0,
вблизи главного резонанса (omega ~ 0.93–0.98) мультистабилен: сосуществуют хаотический аттрактор с доминирующим
бассейном, квазипериодический и несколько периодических с очень малыми бассейнами — «редкие» и, по определению
через сечение Пуанкаре, скрытые. При omega = 0.955 (Chudzik 2011, табл. 1, область B = [-2, 2] x [-2, 3]):
хаос 0.9227, квазипериодический 0.0519, периодические 21T, 35T, 21T — 0.0087, 0.0086, 0.0081.

Система задаётся как отображение за период T = 2 pi / omega (сечение в фазе t = 0 mod T): один шаг отображения —
n_sub шагов РК4 по времени. Единица стоимости — один период. Равновесий у отображения в смысле потока нет; для
меры скрытости используются неподвижные точки отображения (периодические орбиты периода T), которые ищутся
методом Ньютона (experiments/e60_vdpd_census.py) и записываются в meta.
"""
import math

import numpy as np
from numba import njit

from . import System, register


@njit(cache=False)
def vdpd_map(x, p, out):
    alpha, F, omega = p[0], p[1], p[2]
    n_sub = int(p[3])
    T = 2.0 * math.pi / omega
    h = T / n_sub
    u = x[0]
    v = x[1]
    t = 0.0
    for _ in range(n_sub):
        # k1
        a1 = v
        b1 = alpha * (1.0 - u * u) * v - u * u * u + F * math.sin(omega * t)
        # k2
        u2 = u + 0.5 * h * a1
        v2 = v + 0.5 * h * b1
        a2 = v2
        b2 = alpha * (1.0 - u2 * u2) * v2 - u2 * u2 * u2 + F * math.sin(omega * (t + 0.5 * h))
        # k3
        u3 = u + 0.5 * h * a2
        v3 = v + 0.5 * h * b2
        a3 = v3
        b3 = alpha * (1.0 - u3 * u3) * v3 - u3 * u3 * u3 + F * math.sin(omega * (t + 0.5 * h))
        # k4
        u4 = u + h * a3
        v4 = v + h * b3
        a4 = v4
        b4 = alpha * (1.0 - u4 * u4) * v4 - u4 * u4 * u4 + F * math.sin(omega * (t + h))
        u += h / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        v += h / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
        t += h
    out[0] = u
    out[1] = v


def make_vdpd(omega, name=None, n_sub=200, n_tr=1500, n_obs=1000, alpha=0.2, F=1.0):
    s = System(
        name=name or f"vdpd_w{omega:g}", kind="map", dim=2, rhs=vdpd_map,
        params=np.array([alpha, F, omega, float(n_sub)]),
        box_lo=[-2.0, -2.0], box_hi=[2.0, 3.0],
        dt=1.0, n_tr=n_tr, n_obs=n_obs, esc_r=50.0, renorm_every=1,
        equilibria=[], known={}, hidden=[],
        refs=["Chudzik2011Rare", "Brezetskyi2015RareHidden"],
        notes=f"Вынужденный осциллятор ван дер Поля–Дуффинга, alpha={alpha}, F={F}, omega={omega}; "
              f"стробоскопическое отображение, {n_sub} шагов РК4 на период.",
        level="E",
    )
    s.meta = {"family": "forced", "omega": float(omega), "alpha": float(alpha), "F": float(F), "n_sub": int(n_sub)}
    return register(s)


for _w in (0.955, 0.958, 0.962):
    make_vdpd(_w)
