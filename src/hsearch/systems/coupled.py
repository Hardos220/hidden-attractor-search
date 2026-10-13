"""Реальная многомерная система для проверки зависимости выигрыша от размерности (батч B4 e50).

k копий трёхмерной системы со скрытым аттрактором (по умолчанию SE11) со связью среднего поля
по координате coord:

    dx_u/dt = F(x_u) + eps * (mean_v x_v[coord] - x_u[coord]) e_coord,   u = 1..k.

Цель — режим «все k блоков на скрытом аттракторе» (HA^k). При eps = 0 блоки независимы, бассейн
цели — прямое произведение бассейнов, и p = p_1^k точно (p_1 — доля бассейна HA базы в её области).
При малом eps > 0 система по-настоящему связанная, и p измеряется перебором.

Равновесие одно — синхронное равновесие базы; оно устойчиво, поэтому HA^k скрыт. Эталон цели
калибруется с рассинхронизированного состояния: блоки ставятся в точки одной траектории базы на
аттракторе, разнесённые по времени.
"""
import numpy as np
from numba import njit

from . import System, get, register


def _max_re(base, x, h=1e-6):
    """Наибольшая действительная часть собственных чисел якобиана базы в точке x (конечные разности)."""
    d = base.dim
    J = np.zeros((d, d))
    fp_, fm_ = np.zeros(d), np.zeros(d)
    for i in range(d):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        base.rhs(xp, base.params, fp_)
        base.rhs(xm, base.params, fm_)
        J[:, i] = (fp_ - fm_) / (2 * h)
    return float(np.linalg.eigvals(J).real.max())


def _newton_all(rhs, params, guesses, d, tol=1e-11, max_it=60, h=1e-7):
    """Равновесия методом Ньютона (якобиан конечными разностями) из начальных приближений; без повторов."""
    out = []
    f0, f1 = np.zeros(d), np.zeros(d)
    for g in guesses:
        x = np.asarray(g, float).copy()
        ok = False
        for _ in range(max_it):
            rhs(x, params, f0)
            if np.linalg.norm(f0) < tol:
                ok = True
                break
            J = np.zeros((d, d))
            for i in range(d):
                xp = x.copy()
                xp[i] += h
                rhs(xp, params, f1)
                J[:, i] = (f1 - f0) / h
            try:
                x = x - np.linalg.solve(J, f0)
            except np.linalg.LinAlgError:
                break
        if ok and all(np.linalg.norm(x - e) > 1e-6 * (1.0 + np.linalg.norm(e)) for e in out):
            out.append(x)
    return out


def make_coupled(base_name="se11", k=2, eps=0.0, coord=0, name=None, lag_steps=1500, ha_names=None, products=False,
                 all_equilibria=False):
    """ha_names — имена скрытых аттракторов базы, из которых составляется цель (по умолчанию один "HA", как в
    батчах B4 и B8). Если задан список (например, ["HA_a", "HA_b"] для Чуа II), цель — все сочетания
    «каждый блок на одном из них» (эталоны HA_aa, HA_ab, …), а прочие известные состояния базы переносятся
    произведением того же вида (блоки в разнесённых по времени точках траектории базы).

    products=True (2026-10-12): известными считаются все сочетания аттракторов блоков, кроме целевых, — каждый
    блок на одном из известных состояний базы или в одном из её устойчивых равновесий (эталоны P_<имена>).
    Аттракторы отдельного блока известны, поэтому известны и их произведения; без этого смешанные состояния
    («часть блоков на скрытом аттракторе, часть в равновесии») при поиске остаются неопознанными, и двухэтапная
    оценка не может остановить траектории, пришедшие на них.

    all_equilibria=True (2026-10-13): список равновесий — не только синхронные (все блоки в одном равновесии базы),
    но и все остальные: метод Ньютона из произведений равновесий базы (для слабой связи они лишь немного сдвигаются).
    Нужен для проверки скрытости режимов связанной системы, которая требует всех равновесий."""
    base = get(base_name)
    if base.kind != "flow":
        raise ValueError("базовая система должна быть потоком")
    d0 = base.dim
    d = d0 * k
    base_rhs = base.rhs
    n_bp = len(base.params)
    kk = int(k)
    c0 = int(coord)

    @njit(cache=False)
    def rhs(x, p, out):
        for u in range(kk):
            base_rhs(x[u * d0:(u + 1) * d0], p[:n_bp], out[u * d0:(u + 1) * d0])
        e = p[n_bp]
        if e != 0.0:
            m = 0.0
            for u in range(kk):
                m += x[u * d0 + c0]
            m /= kk
            for u in range(kk):
                out[u * d0 + c0] += e * (m - x[u * d0 + c0])

    params = np.concatenate([base.params, [float(eps)]])
    lo = np.tile(base.box_lo, kk)
    hi = np.tile(base.box_hi, kk)
    if ha_names is None:
        # состояние на HA^k: точки одной траектории базы на аттракторе, разнесённые на lag_steps шагов
        tr = base.traj(base.known["HA"], n=base.n_tr + kk * lag_steps + 1)
        states = [tr[base.n_tr + u * lag_steps] for u in range(kk)]
        known = {"HA": np.concatenate(states)}
        if "FP" in base.known:
            known["FP"] = np.tile(np.asarray(base.known["FP"], float), kk)
        hidden = ["HA"]
    else:
        import itertools
        trs = {nm: base.traj(base.known[nm], n=base.n_tr + kk * lag_steps + 1) for nm in base.known}
        known, hidden = {}, []
        for combo in itertools.product(list(ha_names), repeat=kk):
            tag = "HA_" + "".join(c.split("_")[-1] for c in combo)
            known[tag] = np.concatenate([trs[c][base.n_tr + u * lag_steps] for u, c in enumerate(combo)])
            hidden.append(tag)
        for nm in base.known:
            if nm not in ha_names:
                known[nm] = np.concatenate([trs[nm][base.n_tr + u * lag_steps] for u in range(kk)])
    if products:
        import itertools
        tnames = list(ha_names) if ha_names is not None else ["HA"]
        trs = {nm: base.traj(base.known[nm], n=base.n_tr + kk * lag_steps + 1) for nm in base.known}
        unit = {nm: [trs[nm][base.n_tr + u * lag_steps] for u in range(kk)] for nm in base.known}
        L0 = float(np.max(base.box_hi - base.box_lo)) / 2.0
        for i, e in enumerate(base.equilibria):
            e = np.asarray(e, float)
            if _max_re(base, e) >= 0.0:
                continue
            if any(np.linalg.norm(v[0] - e) < 1e-3 * L0 for v in unit.values()):
                continue                                   # равновесие уже есть среди известных состояний базы
            unit[f"E{i}"] = [e] * kk
        for combo in itertools.product(list(unit), repeat=kk):
            if all(c in tnames for c in combo) or len(set(combo)) == 1:
                continue                                   # целевые сочетания и одинаковые блоки (уже есть)
            known["P_" + "-".join(combo)] = np.concatenate([unit[c][u] for u, c in enumerate(combo)])
    eqs = [np.tile(np.asarray(e_, float), kk) for e_ in base.equilibria]
    if all_equilibria:
        import itertools
        guesses = [np.concatenate([np.asarray(base.equilibria[i], float) for i in combo])
                   for combo in itertools.product(range(len(base.equilibria)), repeat=kk)]
        eqs = _newton_all(rhs, params, guesses, d)
    nm = name or f"coupled_{base_name}_k{kk}_eps{eps:g}"
    s = System(
        name=nm, kind="flow", dim=d, rhs=rhs, params=params, box_lo=lo, box_hi=hi,
        dt=base.dt, n_tr=base.n_tr, n_obs=base.n_obs, esc_r=base.esc_r * np.sqrt(kk),
        renorm_every=base.renorm_every, equilibria=eqs, known=known, hidden=hidden,
        refs=list(base.refs), notes=f"{kk} копий {base_name}, связь среднего поля eps={eps} по координате {coord}.",
        level="D3",
    )
    s.meta = {"base": base_name, "k": kk, "eps": float(eps), "coord": c0, "family": "coupled", "products": bool(products)}
    if ha_names is not None:
        s.meta["ha_names"] = list(ha_names)
    return register(s)
