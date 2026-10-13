"""Двумерные отображения (уровень B).

Семейство Цзяна и др. (2016) в записи Ouannas et al. (2018, Entropy 20:720, ур. (1)):
    x_{n+1} = y_n,
    y_{n+1} = x_n + a1 x_n^2 + a2 y_n^2 - a3 x_n y_n - a4.
Три набора параметров без неподвижных точек со скрытыми хаотическими аттракторами
(Ouannas et al. 2018, рис. 1): (0.2, 0.71, 0.91, 1.14), IC (0.93, -0.44);
(0.51, 1, 1.51, 0.74), IC (-0.81, 0.51); (0.6, 1, 1.6, 0.72), IC (-0.26, 0.18).
Jiang et al. отмечают, что бассейны некоторых скрытых аттракторов крошечные.
"""
import numpy as np
from numba import njit

from . import System, register


@njit(cache=False)
def jiang_map(x, p, out):
    a1, a2, a3, a4 = p[0], p[1], p[2], p[3]
    out[0] = x[1]
    out[1] = x[0] + a1 * x[0] * x[0] + a2 * x[1] * x[1] - a3 * x[0] * x[1] - a4


for _name, _p, _ic in [("jiang_a", (0.2, 0.71, 0.91, 1.14), (0.93, -0.44)),
                       ("jiang_b", (0.51, 1.0, 1.51, 0.74), (-0.81, 0.51)),
                       ("jiang_c", (0.6, 1.0, 1.6, 0.72), (-0.26, 0.18))]:
    register(System(
        name=_name, kind="map", dim=2, rhs=jiang_map, params=np.array(_p),
        box_lo=[-2.0, -2.0], box_hi=[2.0, 2.0],
        dt=1.0, n_tr=20000, n_obs=5000, esc_r=100.0, renorm_every=1,
        equilibria=[],
        known={"HA": np.array(_ic)},
        hidden=["HA"],
        refs=["Jiang2016Maps", "Ouannas2018"],
        notes=f"Отображение Цзяна без неподвижных точек, параметры {_p}.",
        level="B",
    ))


@njit(cache=False)
def henon_map(x, p, out):
    a, b = p[0], p[1]
    out[0] = 1.0 - a * x[0] * x[0] + x[1]
    out[1] = b * x[0]


register(System(
    name="henon", kind="map", dim=2, rhs=henon_map, params=np.array([1.4, 0.3]),
    box_lo=[-2.0, -2.0], box_hi=[2.0, 2.0],
    dt=1.0, n_tr=5000, n_obs=2000, esc_r=50.0, renorm_every=1,
    equilibria=[],
    known={"SA": np.array([0.0, 0.0])},
    hidden=[],
    refs=[],
    notes="Контрольное отображение Хенона (самовозбуждающийся аттрактор, уход на бесконечность).",
    level="B",
))


@njit(cache=False)
def jiang_general_map(x, p, out):
    """Общее семейство Jiang et al. (2016), ур. (1):
    x' = y,  y' = a1 x + a2 y + a3 x^2 + a4 y^2 + a5 x y + a6."""
    a1, a2, a3, a4, a5, a6 = p[0], p[1], p[2], p[3], p[4], p[5]
    out[0] = x[1]
    out[1] = a1 * x[0] + a2 * x[1] + a3 * x[0] * x[0] + a4 * x[1] * x[1] + a5 * x[0] * x[1] + a6


# Табл. 1 и 3 Jiang et al. (2016): случаи с крошечными бассейнами хаоса (NF I e, f, g, i) и
# с единственной устойчивой неподвижной точкой (SF I a–d). Коэффициенты (a1..a6), IC хаоса,
# неподвижная точка (для SF).
_JIANG_CASES = {
    "jiang_nfe": ((1.0, 0.0, 0.0, -0.3, 0.3, 2.98), (3.02, -2.78), None, 8.0),
    "jiang_nff": ((1.0, 0.0, 0.0, 0.38, -0.38, -1.6), (-0.07, 0.71), None, 5.0),
    "jiang_nfg": ((1.0, 0.0, 0.0, 0.12, -0.12, -3.62), (1.03, 2.68), None, 8.0),
    "jiang_nfi": ((1.0, 0.0, 0.0, -0.3, 0.3, 1.09), (1.33, 0.12), None, 5.0),
    "jiang_sfa": ((-0.33, 0.17, -0.48, 0.47, 0.01, -0.9), (0.32, -1.85), (-0.7759, -0.7759), 5.0),
    "jiang_sfb": ((-0.84, 0.0, 0.0, 0.15, -0.15, -5.85), (4.61, -6.99), (-3.1793, -3.1793), 12.0),
    "jiang_sfc": ((-0.99, 0.0, 0.0, 0.23, -0.23, -4.24), (3.15, -4.82), (-2.1307, -2.1307), 10.0),
    "jiang_sfd": ((-1.29, 2.0, -0.35, -0.85, 1.2, 0.17), (-1.44, -0.23), (0.5862, 0.5862), 5.0),
}
for _name, (_p, _ic, _fp, _half) in _JIANG_CASES.items():
    register(System(
        name=_name, kind="map", dim=2, rhs=jiang_general_map, params=np.array(_p),
        box_lo=[-_half, -_half], box_hi=[_half, _half],
        dt=1.0, n_tr=20000, n_obs=5000, esc_r=1e3, renorm_every=1,
        equilibria=[] if _fp is None else [np.array(_fp)],
        known={"HA": np.array(_ic)} if _fp is None else {"FP": np.array(_fp), "HA": np.array(_ic)},
        hidden=["HA"],
        refs=["Jiang2016Maps"],
        notes=f"Jiang et al. (2016), случай {_name[-3:].upper()}: коэффициенты {_p}.",
        level="B",
    ))


@njit(cache=False)
def huynh_map(x, p, out):
    a, b, c, d = p[0], p[1], p[2], p[3]
    out[0] = x[0] + x[1]
    out[1] = x[1] - a * abs(x[1]) - x[0] * x[1] + b * x[0] * x[0] - c * x[1] * x[1] + d


register(System(
    name="huynh2019", kind="map", dim=2, rhs=huynh_map, params=np.array([0.01, 0.1, 2.0, 0.1]),
    box_lo=[-3.0, -3.0], box_hi=[3.0, 3.0],
    dt=1.0, n_tr=20000, n_obs=5000, esc_r=100.0, renorm_every=1,
    equilibria=[],
    known={"HA": np.array([1.5, 0.5])},
    hidden=["HA"],
    refs=["Huynh2019"],
    notes="Отображение Хуиня и др. (Entropy 21:279, 2019) без неподвижных точек: x' = x + y, y' = y − a|y| − xy + bx² − cy² + d.",
    level="B",
))


@njit(cache=False)
def wangding_map(x, p, out):
    k, c, d, e, f = p[0], p[1], p[2], p[3], p[4]
    u = (k * c + 1.0) * x[0] + k * (d - 1.0) * x[1] + e
    v = c * x[0] + d * x[1] + f
    out[0] = u - np.floor(u)
    out[1] = v - np.floor(v)


register(System(
    name="wangding", kind="map", dim=2, rhs=wangding_map, params=np.array([2.0, 1.0, 2.0, 0.1, 0.2]),
    box_lo=[0.0, 0.0], box_hi=[1.0, 1.0],
    dt=1.0, n_tr=5000, n_obs=2000, esc_r=50.0, renorm_every=1,
    equilibria=[],
    known={"HA": np.array([0.7, 0.6])},
    hidden=["HA"],
    refs=["WangDing2018Map", "GuanXie2025"],
    notes="Отображение Ванга–Дина на торе без неподвижных точек (уточнить условия на коэффициенты).",
    level="B",
))
