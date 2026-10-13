"""Системы размерности 4–6 (уровень C)."""
import numpy as np
from numba import njit

from . import System, register


@njit(cache=False)
def li_sprott_4d_rhs(x, p, out):
    a, b = p[0], p[1]
    out[0] = x[1] - x[0]
    out[1] = -x[0] * x[2] + x[3]
    out[2] = x[0] * x[1] - a
    out[3] = -b * x[1]


# Li & Sprott (2014): a = 6, b = 0.1 — тор (IC (1,-1,1,-1)) сосуществует с симметричной парой
# странных аттракторов (IC (0, ±4, 0, ∓5)); равновесий нет — все аттракторы скрыты;
# границы бассейнов фрактальны (рис. 11 статьи).
register(System(
    name="li_sprott_4d_a6", kind="flow", dim=4, rhs=li_sprott_4d_rhs, params=np.array([6.0, 0.1]),
    box_lo=[-15.0, -15.0, -15.0, -15.0], box_hi=[15.0, 15.0, 15.0, 15.0],
    dt=0.005, n_tr=100000, n_obs=40000, esc_r=500.0, renorm_every=20,
    equilibria=[],
    known={"T": np.array([1.0, -1.0, 1.0, -1.0]),
           "SA_a": np.array([0.0, 4.0, 0.0, -5.0]), "SA_b": np.array([0.0, -4.0, 0.0, 5.0])},
    hidden=["T", "SA_a", "SA_b"],
    refs=["LiSprott2014"],
    notes="4D упрощённая система Лоренца без равновесий; тор + два странных аттрактора (ЛП 0.252, 0, -0.005, -1.247).",
    level="C",
))

# a = 7, b = 0.1 — предельный цикл (IC (1,-1,1,-1)) + пара странных аттракторов (IC (0, ±7.8, 0, ∓5)).
register(System(
    name="li_sprott_4d_a7", kind="flow", dim=4, rhs=li_sprott_4d_rhs, params=np.array([7.0, 0.1]),
    box_lo=[-15.0, -15.0, -15.0, -15.0], box_hi=[15.0, 15.0, 15.0, 15.0],
    dt=0.005, n_tr=100000, n_obs=40000, esc_r=500.0, renorm_every=20,
    equilibria=[],
    known={"LC": np.array([1.0, -1.0, 1.0, -1.0]),
           "SA_a": np.array([0.0, 7.8, 0.0, -5.0]), "SA_b": np.array([0.0, -7.8, 0.0, 5.0])},
    hidden=["LC", "SA_a", "SA_b"],
    refs=["LiSprott2014"],
    notes="4D упрощённая система Лоренца: предельный цикл + два странных аттрактора (ЛП 0.227, 0, -0.010, -1.217).",
    level="C",
))

@njit(cache=False)
def jahanshahi_rhs(x, p, out):
    a, b, c, e, f, g = p[0], p[1], p[2], p[3], p[4], p[5]
    out[0] = x[1]
    out[1] = x[2]
    out[2] = x[3]
    out[3] = -a * x[3] + b * x[0] * x[0] - c * x[1] * x[1] + e * x[0] * x[1] + f * x[0] * x[2] + g


# Jahanshahi et al. (Entropy 21:156, 2019): 4D гиперджерк без равновесий (b, g > 0),
# ЛП (0.185, 0, -0.195, -1.034) при g = 1.15; начальное условие в статье не приведено — находится выборкой.
register(System(
    name="jahanshahi2019_4d", kind="flow", dim=4, rhs=jahanshahi_rhs,
    params=np.array([1.05, 0.7, 0.19, 1.37, 1.79, 1.15]),
    box_lo=[-10.0, -10.0, -10.0, -10.0], box_hi=[10.0, 10.0, 10.0, 10.0],
    dt=0.005, n_tr=100000, n_obs=40000, esc_r=500.0, renorm_every=20,
    equilibria=[],
    known={"HA": np.array([-3.17442, 4.76556, -4.784, -27.60316])},   # найдено обнаружением в e02
    hidden=["HA"],
    refs=["Jahanshahi2019"],
    notes="4D гиперджерк без равновесий; скрытый хаотический аттрактор (λ ≈ 0.19), p ≈ 0.51 в [-10, 10]^4.",
    level="C",
))

# a = 2.6, b = 0.44 — гиперхаос (IC (2, 4, 0, 0)), ЛП (0.0704, ...).
register(System(
    name="li_sprott_4d_hyper", kind="flow", dim=4, rhs=li_sprott_4d_rhs, params=np.array([2.6, 0.44]),
    box_lo=[-15.0, -15.0, -15.0, -15.0], box_hi=[15.0, 15.0, 15.0, 15.0],
    dt=0.005, n_tr=100000, n_obs=40000, esc_r=500.0, renorm_every=20,
    equilibria=[],
    known={"HC": np.array([2.0, 4.0, 0.0, 0.0])},
    hidden=["HC"],
    refs=["LiSprott2014"],
    notes="4D упрощённая система Лоренца: гиперхаотический скрытый аттрактор.",
    level="C",
))
