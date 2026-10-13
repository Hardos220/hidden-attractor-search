"""Поиск аттракторов по рекуррентностям на сетке — реализация идеи Attractors.jl
(Datseris & Wagemakers 2022) на Python/numba для сравнения стоимости.

Конечный автомат над потоком: ячейки сетки получают метки
  0            — не посещалась;
  VISIT (-2)   — посещена текущей траекторией (временно);
  2k           — ячейка аттрактора k (k >= 1);
  2k+1         — ячейка бассейна аттрактора k;
  ESC (-1)     — ячейка, с которой траектория ушла из сетки.
Правила:
  * consecutive_recurrences шагов подряд в ячейках VISIT -> новый аттрактор,
    далее attractor_locate_steps шагов помечают ячейки аттрактора;
  * consecutive_hits шагов подряд в ячейках аттрактора/бассейна k -> траектория принадлежит k;
  * consecutive_lost шагов подряд вне сетки (или |x| > horizon) -> уход (ESC);
  * max_steps — предохранитель; неразрешённые траектории получают метку UNDECIDED (-3).
Стоимость учитывается в шагах интегрирования.
"""
import numpy as np
from numba import njit

VISIT = -2
ESC = -1
UNDECIDED = -3


def make_recurrence_runner(system):
    """Возвращает функцию run(ics, cells, state) -> (labels, steps, seeds, n_att_new)."""
    rhs = system.rhs
    d = system.dim
    is_map = system.kind == "map"

    @njit(cache=False)
    def _step(x, p, dt, k1, k2, k3, k4, tmp):
        if is_map:
            rhs(x, p, tmp)
            for i in range(d):
                x[i] = tmp[i]
            return
        rhs(x, p, k1)
        for i in range(d):
            tmp[i] = x[i] + 0.5 * dt * k1[i]
        rhs(tmp, p, k2)
        for i in range(d):
            tmp[i] = x[i] + 0.5 * dt * k2[i]
        rhs(tmp, p, k3)
        for i in range(d):
            tmp[i] = x[i] + dt * k3[i]
        rhs(tmp, p, k4)
        for i in range(d):
            x[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    @njit(cache=False)
    def _cell(x, lo, hi, n, strides):
        idx = 0
        for i in range(d):
            if x[i] < lo[i] or x[i] >= hi[i]:
                return -1
            c = int((x[i] - lo[i]) / (hi[i] - lo[i]) * n[i])
            if c >= n[i]:
                c = n[i] - 1
            idx += c * strides[i]
        return idx

    @njit(cache=False)
    def run(ics, p, dt, lo, hi, n, strides, cells, n_att_start, horizon,
            consecutive_recurrences, attractor_locate_steps, consecutive_hits, consecutive_lost, max_steps,
            labels, steps, seeds):
        n_att = n_att_start
        x = np.empty(d)
        k1 = np.empty(d); k2 = np.empty(d); k3 = np.empty(d); k4 = np.empty(d); tmp = np.empty(d)
        visited = np.empty(max_steps + attractor_locate_steps + 1, dtype=np.int64)
        h2 = horizon * horizon
        for j in range(ics.shape[0]):
            for i in range(d):
                x[i] = ics[j, i]
            nvis = 0
            rec = 0
            hit = 0
            hit_label = -999
            lost = 0
            label = UNDECIDED
            nsteps = 0
            while nsteps < max_steps:
                _step(x, p, dt, k1, k2, k3, k4, tmp)
                nsteps += 1
                r2 = 0.0
                for i in range(d):
                    r2 += x[i] * x[i]
                if r2 > h2 or r2 != r2:
                    label = ESC
                    break
                c = _cell(x, lo, hi, n, strides)
                if c < 0:
                    lost += 1
                    if lost >= consecutive_lost:
                        label = ESC
                        break
                    continue
                lost = 0
                lab = cells[c]
                if lab == 0:
                    cells[c] = VISIT
                    visited[nvis] = c
                    nvis += 1
                    rec = 0
                    hit = 0
                elif lab == VISIT:
                    rec += 1
                    hit = 0
                    if rec >= consecutive_recurrences:
                        n_att += 1
                        k = n_att
                        for i in range(d):
                            seeds[k, i] = x[i]
                        for t in range(attractor_locate_steps):
                            _step(x, p, dt, k1, k2, k3, k4, tmp)
                            nsteps += 1
                            c2 = _cell(x, lo, hi, n, strides)
                            if c2 >= 0:
                                cells[c2] = 2 * k
                        for v in range(nvis):
                            if cells[visited[v]] == VISIT:
                                cells[visited[v]] = 2 * k + 1
                        nvis = 0
                        label = k
                        break
                else:
                    # ячейка существующего аттрактора/бассейна или ухода
                    m = lab // 2 if lab > 0 else ESC
                    if m == hit_label:
                        hit += 1
                    else:
                        hit_label = m
                        hit = 1
                    rec = 0
                    if hit >= consecutive_hits:
                        label = m
                        break
            # финализация временных меток
            if label == UNDECIDED:
                for v in range(nvis):
                    if cells[visited[v]] == VISIT:
                        cells[visited[v]] = 0
            elif label == ESC:
                for v in range(nvis):
                    if cells[visited[v]] == VISIT:
                        cells[visited[v]] = ESC
            elif nvis > 0:
                for v in range(nvis):
                    if cells[visited[v]] == VISIT:
                        cells[visited[v]] = 2 * label + 1
            labels[j] = label
            steps[j] = nsteps
        return n_att

    return run


class RecurrenceFinder:
    """Обёртка с состоянием сетки. find(ics) обрабатывает начальные условия последовательно."""

    def __init__(self, system, grid_n, grid_lo=None, grid_hi=None, horizon=None,
                 consecutive_recurrences=2000, attractor_locate_steps=4000, consecutive_hits=10,
                 consecutive_lost=200, max_steps=None, max_attractors=64):
        self.s = system
        d = system.dim
        self.lo = np.asarray(system.box_lo if grid_lo is None else grid_lo, float)
        self.hi = np.asarray(system.box_hi if grid_hi is None else grid_hi, float)
        self.n = np.asarray(grid_n if np.ndim(grid_n) else [grid_n] * d, dtype=np.int64)
        strides = np.ones(d, dtype=np.int64)
        for i in range(d - 1, 0, -1):
            strides[i - 1] = strides[i] * self.n[i]
        self.strides = strides
        self.cells = np.zeros(int(np.prod(self.n)), dtype=np.int64)
        self.horizon = float(system.esc_r if horizon is None else horizon)
        self.cfg = dict(consecutive_recurrences=int(consecutive_recurrences),
                        attractor_locate_steps=int(attractor_locate_steps),
                        consecutive_hits=int(consecutive_hits), consecutive_lost=int(consecutive_lost),
                        max_steps=int(max_steps if max_steps is not None else system.n_tr + system.n_obs))
        self.seeds = np.zeros((max_attractors + 1, d))
        self.n_att = 0
        self._run = make_recurrence_runner(system)
        self.total_steps = 0

    def find(self, ics):
        ics = np.ascontiguousarray(ics, dtype=np.float64)
        labels = np.zeros(len(ics), dtype=np.int64)
        steps = np.zeros(len(ics), dtype=np.int64)
        self.n_att = self._run(ics, self.s.params, self.s.dt, self.lo, self.hi, self.n, self.strides,
                               self.cells, self.n_att, self.horizon,
                               self.cfg["consecutive_recurrences"], self.cfg["attractor_locate_steps"],
                               self.cfg["consecutive_hits"], self.cfg["consecutive_lost"], self.cfg["max_steps"],
                               labels, steps, self.seeds)
        self.total_steps += int(steps.sum())
        return labels, steps

    def attractor_seeds(self):
        return {k: self.seeds[k].copy() for k in range(1, self.n_att + 1)}
