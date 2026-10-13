"""Трёхмерные потоки эталонного набора (уровень A) и отрицательные контроли.

Источники: docs/04_etalonnye_sistemy_draft.md (Guan & Xie 2025; Leonov et al. 2011;
Jafari & Sprott 2013; Wang & Chen 2012; Molaie et al. 2013; Wei 2011;
Sprott & Munmuangsaen 2018).
"""
import numpy as np
from numba import njit

from . import System, register


# ---------------------------------------------------------------- Чуа
@njit(cache=False)
def chua_rhs(x, p, out):
    a, b, g, m0, m1 = p[0], p[1], p[2], p[3], p[4]
    fx = m1 * x[0] + 0.5 * (m0 - m1) * (abs(x[0] + 1.0) - abs(x[0] - 1.0))
    out[0] = a * (x[1] - x[0] - fx)
    out[1] = x[0] - x[1] + x[2]
    out[2] = -(b * x[1] + g * x[2])


def chua_equilibria(p):
    """Равновесия кусочно-линейной системы Чуа: 0 и две симметричные точки во внешних зонах."""
    a, b, g, m0, m1 = p
    eqs = [np.zeros(3)]
    # внешняя зона |x| > 1: f(x) = m1 x + (m0 - m1) sign(x)
    # y = -g z / b  (из третьего уравнения), x = y - z, y - x - f(x) = 0
    # => подставляем z = x (1 + b/g)^{-1}... решаем линейно: пусть z = t, y = -g t / b, x = y - z = -t (g/b + 1)
    # условие: y - x - (m1 x + (m0 - m1) s) = 0, s = sign(x)
    for s in (1.0, -1.0):
        # x = -t (g/b + 1); y = -g t / b; подставляем
        # -g t/b + t (g/b + 1) - m1 (-t (g/b + 1)) - (m0 - m1) s = 0
        # t [ -g/b + g/b + 1 + m1 (g/b + 1) ] = (m0 - m1) s
        coef = 1.0 + m1 * (g / b + 1.0)
        t = (m0 - m1) * s / coef
        x = -t * (g / b + 1.0)
        y = -g * t / b
        z = t
        if abs(x) > 1.0:
            eqs.append(np.array([x, y, z]))
    return eqs


_chua_I_p = np.array([8.4562218418, 12.0732335925, 0.0051631393, -0.1767573476, -1.1467573476])
register(System(
    name="chua_I", kind="flow", dim=3, rhs=chua_rhs, params=_chua_I_p,
    box_lo=[-10.0, -10.0, -15.0], box_hi=[10.0, 10.0, 15.0],
    # dt=0.01 для кусочно-линейной системы даёт численный «переходный хаос» (e03); dt=0.0025 сходится
    dt=0.0025, n_tr=400000, n_obs=40000, esc_r=100.0, renorm_every=40,
    equilibria=chua_equilibria(_chua_I_p),
    known={"F0": np.zeros(3),
           "HA_a": np.array([0.1698, 1.2067, 3.9548]),
           "HA_b": np.array([-0.1698, -1.2067, -3.9548])},
    hidden=["HA_a", "HA_b"],
    refs=["Leonov2011PLA", "GuanXie2025", "Wagemakers2025Zoo"],
    notes="Набор параметров I: устойчивый фокус-узел в нуле, два седло-фокуса, два симметричных скрытых хаотических аттрактора.",
))

_chua_II_p = np.array([8.4, 12.0, -0.005, -1.2, -0.05])
register(System(
    name="chua_II", kind="flow", dim=3, rhs=chua_rhs, params=_chua_II_p,
    box_lo=[-10.0, -10.0, -15.0], box_hi=[10.0, 10.0, 15.0],
    dt=0.0025, n_tr=400000, n_obs=40000, esc_r=100.0, renorm_every=40,
    equilibria=chua_equilibria(_chua_II_p),
    # состояния на аттракторах найдены обнаружением в e02 (выборка 5·10^4 в боксе Зелинки)
    known={"LC": np.array([-7.28686, -2.77261, 36.02167]),
           "HA_a": np.array([1.52759, 0.28302, -1.81919]),
           "HA_b": np.array([-1.30415, 0.02917, 2.38749])},
    hidden=["LC", "HA_a", "HA_b"],
    refs=["GuanXie2025", "Kuznetsov2023Chua"],
    notes="Набор параметров II: пять сосуществующих аттракторов (два к устойчивым F±, два скрытых хаотических HA_a/HA_b с долями ~0.0014, скрытый предельный цикл LC с долей 0.99).",
))


# ---------------------------------------------------------------- системы без равновесий
@njit(cache=False)
def ne4_rhs(x, p, out):
    a = p[0]
    out[0] = -0.1 * x[1] + a
    out[1] = x[0] + x[2]
    out[2] = x[0] * x[2] - 3.0 * x[1]


register(System(
    name="ne4", kind="flow", dim=3, rhs=ne4_rhs, params=np.array([1.0]),
    # аттрактор велик: x in [-25, 3.3], z in [-52, 178] (проверено в e02); бокс — удвоенный экстент
    box_lo=[-40.0, -30.0, -80.0], box_hi=[20.0, 50.0, 220.0],
    # скорости порядка |z| ~ 200 требуют мелкого шага; медленная перемежаемость (ЛП 0.02) — длинного окна
    dt=0.001, n_tr=500000, n_obs=500000, esc_r=2000.0, renorm_every=100,
    equilibria=[],
    known={"HA": np.array([-8.2, 0.0, -5.0])},
    hidden=["HA"],
    refs=["JafariSprottGolpayegani2013", "GuanXie2025", "Godara2018CVS"],
    notes="NE4 Джафари–Спротта: без равновесий; перпетуальная точка (0, 10/31, 0) уходит на бесконечность.",
))


@njit(cache=False)
def wangchen1_rhs(x, p, out):
    a = p[0]
    out[0] = x[1]
    out[1] = x[2]
    out[2] = -x[1] + 3.0 * x[1] * x[1] - x[0] * x[0] - x[0] * x[2] + a


register(System(
    name="wangchen1", kind="flow", dim=3, rhs=wangchen1_rhs, params=np.array([-0.05]),
    box_lo=[-10.0, -10.0, -10.0], box_hi=[10.0, 10.0, 10.0],
    dt=0.01, n_tr=30000, n_obs=10000, esc_r=100.0, renorm_every=20,
    equilibria=[],
    known={"HA": np.array([1.0, 0.0, 1.0])},
    hidden=["HA"],
    refs=["GuanXie2025"],
    notes="Джерк-система Ванга–Чена #1 без равновесий; PP P1 ведёт к аттрактору, P2 — на бесконечность.",
))


@njit(cache=False)
def wei_rhs(x, p, out):
    a, c, d = p[0], p[1], p[2]
    out[0] = -x[1]
    out[1] = c * x[0] + x[2]
    out[2] = a * x[1] * x[1] + x[0] * x[2] - d


register(System(
    name="wei_ne2", kind="flow", dim=3, rhs=wei_rhs, params=np.array([2.0, 1.0, 0.35]),
    box_lo=[-10.0, -10.0, -10.0], box_hi=[10.0, 10.0, 10.0],
    dt=0.01, n_tr=30000, n_obs=10000, esc_r=100.0, renorm_every=20,
    equilibria=[],
    known={"HA": np.array([-1.6, 0.82, 1.9])},
    hidden=["HA"],
    refs=["Wei2011", "GuanXie2025"],
    notes="Система Вэя (модифицированная Спротт D, NE2) без равновесий.",
))


# ---------------------------------------------------------------- единственное устойчивое равновесие
@njit(cache=False)
def wangchen2_rhs(x, p, out):
    a = p[0]
    out[0] = x[1] * x[2] + a
    out[1] = x[0] * x[0] - x[1]
    out[2] = 1.0 - 4.0 * x[0]


register(System(
    name="wangchen2", kind="flow", dim=3, rhs=wangchen2_rhs, params=np.array([0.006]),
    box_lo=[-10.0, -10.0, -10.0], box_hi=[10.0, 10.0, 10.0],
    dt=0.01, n_tr=30000, n_obs=10000, esc_r=100.0, renorm_every=20,
    equilibria=[np.array([0.25, 0.0625, -0.096])],
    known={"FP": np.array([0.25, 0.0625, -0.096]), "HA": np.array([0.0, 0.0, -2.0])},
    hidden=["HA"],
    refs=["WangChen2012", "GuanXie2025"],
    notes="Единственное устойчивое равновесие (0.25, 0.0625, -0.096).",
))


@njit(cache=False)
def se3_rhs(x, p, out):
    out[0] = x[1]
    out[1] = x[2]
    out[2] = -3.4 * x[0] - x[1] - 4.0 * x[2] + x[1] * x[1] + x[0] * x[1]


register(System(
    name="se3", kind="flow", dim=3, rhs=se3_rhs, params=np.array([0.0]),
    box_lo=[-10.0, -10.0, -10.0], box_hi=[10.0, 10.0, 10.0],
    dt=0.01, n_tr=30000, n_obs=10000, esc_r=100.0, renorm_every=20,
    equilibria=[np.zeros(3)],
    known={"FP": np.zeros(3), "HA": np.array([-2.0, 0.0, 2.4])},
    hidden=["HA"],
    refs=["Molaie2013", "GuanXie2025"],
    notes="SE3: устойчивое равновесие в нуле, перпетуальных точек нет.",
))


@njit(cache=False)
def se11_rhs(x, p, out):
    out[0] = x[1]
    out[1] = -x[0] + x[1] * x[2]
    out[2] = -x[1] - 12.0 * x[2] + x[0] * x[0] + 9.0 * x[0] * x[2] - 1.0


register(System(
    name="se11", kind="flow", dim=3, rhs=se11_rhs, params=np.array([0.0]),
    box_lo=[-10.0, -10.0, -10.0], box_hi=[10.0, 10.0, 10.0],
    dt=0.01, n_tr=30000, n_obs=10000, esc_r=100.0, renorm_every=20,
    equilibria=[np.array([0.0, 0.0, -1.0 / 12.0])],
    known={"FP": np.array([0.0, 0.0, -1.0 / 12.0]), "HA": np.array([-2.0, 0.0, 0.1])},
    hidden=["HA"],
    refs=["Molaie2013", "GuanXie2025"],
    notes="SE11: устойчивое равновесие (0, 0, -1/12).",
))


# ---------------------------------------------------------------- линия равновесий
@njit(cache=False)
def le1_rhs(x, p, out):
    a, b = p[0], p[1]
    out[0] = x[1]
    out[1] = -x[0] + x[1] * x[2]
    out[2] = -x[0] - a * x[0] * x[1] - b * x[0] * x[2]


register(System(
    name="le1", kind="flow", dim=3, rhs=le1_rhs, params=np.array([15.0, 1.0]),
    box_lo=[-10.0, -10.0, -10.0], box_hi=[10.0, 10.0, 10.0],
    dt=0.01, n_tr=30000, n_obs=10000, esc_r=100.0, renorm_every=20,
    equilibria=[],
    eq_project=lambda x: np.array([0.0, 0.0, x[2]]),
    known={"HA": np.array([0.0, 0.5, 0.5])},
    hidden=["HA"],
    refs=["JafariSprott2013", "GuanXie2025"],
    notes="LE1: линия равновесий (0,0,z), устойчивых при z<0; PP ведёт к устойчивому равновесию (0,0,-0.7778).",
))


# ---------------------------------------------------------------- без неподвижных и перпетуальных точек (Jafari et al. 2026)
@njit(cache=False)
def jafari2026_rhs(x, p, out):
    a1, a2, a3, a4, a5 = p[0], p[1], p[2], p[3], p[4]
    out[0] = x[1]
    out[1] = x[2]
    out[2] = a1 * x[2] + a2 * x[0] * x[1] + a3 * x[1] * x[2] + a4 * x[2] * x[0] + a5


register(System(
    name="jafari2026_lc", kind="flow", dim=3, rhs=jafari2026_rhs, params=np.array([6.0, -1.0, -1.0, -1.0, -1.0]),
    box_lo=[-10.0, -10.0, -10.0], box_hi=[10.0, 10.0, 10.0],
    # медленные уходы (дрейф x -> +inf) отсекаются порогом 60: цикл лежит в |x| < 12
    dt=0.01, n_tr=300000, n_obs=20000, esc_r=60.0, renorm_every=10,
    equilibria=[],
    known={"LC": np.array([5.7, 1.3, 1.9])},
    hidden=["LC"],
    refs=["Jafari2026NoPP"],
    notes="ż = 6z − xy − yz − zx − 1: нет ни неподвижных, ни перпетуальных точек; устойчивый предельный цикл (ЛП 0, −0.014, −0.014).",
))

register(System(
    name="jafari2026_chaos", kind="flow", dim=3, rhs=jafari2026_rhs, params=np.array([6.0, -1.0, -3.548, -1.0, -1.0]),
    box_lo=[-10.0, -10.0, -10.0], box_hi=[10.0, 10.0, 10.0],
    dt=0.005, n_tr=200000, n_obs=40000, esc_r=60.0, renorm_every=20,
    equilibria=[],
    known={"HA": np.array([6.744486820710031, 0.4214444979978726, -1.7846332513720344])},   # состояние из скана e12
    hidden=["HA"],
    refs=["Jafari2026NoPP"],
    notes="a3 = -3.548 — внутри окна хаоса [-3.5468, -3.5492] (скан e12: λ ≈ 0.05); уход на бесконечность при a3 <= -3.5494.",
))


# ---------------------------------------------------------------- Рабинович (обобщённая система Лоренца)
@njit(cache=False)
def rabinovich_rhs(x, p, out):
    r, a, b = p[0], p[1], p[2]
    sigma = -a * r
    out[0] = -sigma * (x[0] - x[1]) - a * x[1] * x[2]
    out[1] = r * x[0] - x[1] - x[0] * x[2]
    out[2] = -b * x[2] + x[0] * x[1]


register(System(
    name="rabinovich_100", kind="flow", dim=3, rhs=rabinovich_rhs,
    params=np.array([100.0, -0.009965, 0.077454]),
    box_lo=[-60.0, -60.0, -150.0], box_hi=[60.0, 60.0, 150.0],
    dt=0.002, n_tr=100000, n_obs=50000, esc_r=1000.0, renorm_every=20,
    equilibria=[np.zeros(3)],
    known={"F0": np.zeros(3), "HA": np.array([-0.2385, 49.1403, -101.4613])},
    hidden=["HA"],
    refs=["Kuznetsov2018Rabinovich", "GuanXie2025"],
    notes="(r, a, b) = (100, -0.009965, 0.077454): скрытый аттрактор найден продолжением; PP (-0.2385, 49.1403, -101.4613) ведёт к нему. Ненулевые равновесия добавляются численно.",
))

register(System(
    name="rabinovich_68", kind="flow", dim=3, rhs=rabinovich_rhs,
    params=np.array([6.8, -0.5, 0.99]),
    box_lo=[-10.0, -10.0, -10.0], box_hi=[10.0, 10.0, 15.0],
    dt=0.005, n_tr=60000, n_obs=20000, esc_r=200.0, renorm_every=20,
    equilibria=[np.zeros(3)],
    known={"F0": np.zeros(3), "HA": np.array([-0.1629, -0.2154, 1.9553])},
    hidden=["HA"],
    refs=["Kuznetsov2018Rabinovich", "GuanXie2025", "SprottMunmuangsaen2018Comment"],
    notes="(r, a, b) = (6.8, -0.5, 0.99): «скрытый аттрактор» по Кузнецову и др.; по Sprott & Munmuangsaen — переходное множество с τ ~ 5e7. Тест протокола верификации.",
))


# ---------------------------------------------------------------- Лоренц: переходный хаос (отрицательный контроль)
@njit(cache=False)
def lorenz_rhs(x, p, out):
    a, r, b = p[0], p[1], p[2]
    out[0] = a * (x[1] - x[0])
    out[1] = -x[0] * x[2] + r * x[0] - x[1]
    out[2] = x[0] * x[1] - b * x[2]


_lz = np.sqrt(2.0 * 28.0)
register(System(
    name="lorenz_transient", kind="flow", dim=3, rhs=lorenz_rhs, params=np.array([4.0, 29.0, 2.0]),
    box_lo=[-20.0, -20.0, 0.0], box_hi=[20.0, 20.0, 50.0],
    dt=0.005, n_tr=60000, n_obs=20000, esc_r=500.0, renorm_every=20,
    equilibria=[np.zeros(3), np.array([_lz, _lz, 28.0]), np.array([-_lz, -_lz, 28.0])],
    known={"Fp": np.array([_lz, _lz, 28.0]), "Fm": np.array([-_lz, -_lz, 28.0]), "TC": np.array([5.0, 5.0, 5.0])},
    hidden=[],
    refs=["Munmuangsaen2018", "SprottMunmuangsaen2018Comment"],
    notes="(a, r, b) = (4, 29, 2): заявленный скрытый аттрактор — переходный хаос с τ ~ 1e4. Отрицательный контроль протокола.",
    level="ctrl",
))
