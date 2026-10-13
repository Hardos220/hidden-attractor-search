"""Бисекция к границам бассейнов — базовая стратегия для сравнения с портфелем (канал B без расщепления).

Идея та же, что у отслеживания краевого состояния (Skufca, Yorke, Eckhardt 2006) и у «зажатых»
траекторий Nusse–Yorke (1989): граница между бассейнами находится делением отрезка пополам, а у
границы переходный процесс длинный. Цепь берёт точку a с длинным переходным процессом (из верхней
доли top_frac по координате transient_length) и ближайшую к ней точку b другого класса (класс —
метка эталона; ушедшие траектории — отдельный класс), затем n_bisect раз оценивает середину отрезка
[a, b] и заменяет ею тот конец, чей класс середина унаследовала. Если середина принадлежит третьему
классу, деление продолжается между a и серединой. После n_bisect делений цепь начинается заново.
Так оценки сосредоточиваются у границ известных бассейнов, к которым может примыкать бассейн цели.

Доля eps равномерных точек в каждом раунде (по умолчанию 0) делает стратегию сопоставимой с
портфелем по гарантии худшего случая. Опорная выборка — первые n_init равномерных точек, общие
с перебором (общие случайные числа).
"""
import numpy as np

from ..integrate import F_ESC
from .harness import Strategy
from .sus import transient_length


class EdgeBisection(Strategy):
    name = "edge_bisect"

    def __init__(self, problem, batch=20, n_init=50, n_chains=10, n_bisect=12, top_frac=0.2, eps=0.0,
                 max_pool=20000, seed_ics=None, **kw):
        super().__init__(problem, batch=batch)
        self.n_init = int(n_init)
        self.n_chains = int(n_chains)
        self.n_bisect = int(n_bisect)
        self.top_frac = float(top_frac)
        self.eps = float(eps)
        self.max_pool = int(max_pool)
        self.seed_ics = seed_ics
        self.X = None
        self.y = None
        self.xi = None
        self.chains = []
        self._kind = None            # что было предложено последним: init / mid / uniform
        self._n_mid = 0

    def _cls(self, feats, labels):
        return np.where(feats[:, F_ESC] > 0.5, -1, labels).astype(np.int64)

    def _norm(self, X):
        return (X - self.lo) / (self.hi - self.lo)

    def _pool(self):
        """Индексы точек, среди которых ищутся пары (последние max_pool точек — ограничение стоимости)."""
        n = len(self.X)
        return np.arange(max(0, n - self.max_pool), n)

    def _new_chain(self):
        idx = self._pool()
        y, xi = self.y[idx], self.xi[idx]
        k = max(1, int(self.top_frac * len(idx)))
        top = idx[np.argpartition(-xi, k - 1)[:k]] if k < len(idx) else idx
        ia = top[self.rng.integers(len(top))]
        other = idx[y != self.y[ia]]
        if len(other) == 0:
            return None
        d = np.linalg.norm(self._norm(self.X[other]) - self._norm(self.X[ia])[None, :], axis=1)
        ib = other[int(np.argmin(d))]
        return {"a": self.X[ia].copy(), "b": self.X[ib].copy(), "ya": int(self.y[ia]), "yb": int(self.y[ib]), "steps": 0}

    def propose(self):
        if self.X is None:
            self._kind = "init"
            if self.seed_ics is not None:
                ics = np.asarray(self.seed_ics, float)[: self.n_init]
                if len(ics) < self.n_init:
                    ics = np.vstack([ics, self.s.sample_box(self.n_init - len(ics), self.rng)])
                return ics
            return self.s.sample_box(self.n_init, self.rng)
        if len(np.unique(self.y)) < 2:
            self._kind = "uniform"
            return self.s.sample_box(self.batch, self.rng)
        while len(self.chains) < self.n_chains:
            c = self._new_chain()
            if c is None:
                break
            self.chains.append(c)
        mids = np.array([0.5 * (c["a"] + c["b"]) for c in self.chains]) if self.chains else np.zeros((0, self.d))
        self._n_mid = len(mids)
        n_u = int(np.ceil(self.eps / max(1e-9, 1.0 - self.eps) * len(mids))) if len(mids) else self.batch
        self._kind = "mid"
        if n_u:
            return np.vstack([mids, self.s.sample_box(n_u, self.rng)]) if len(mids) else self.s.sample_box(n_u, self.rng)
        return mids

    def observe(self, ics, feats, finals, labels):
        y = self._cls(feats, labels)
        xi = transient_length(self.s, feats)
        if self._kind == "mid" and self._n_mid:
            n = min(self._n_mid, len(ics))          # пачка могла быть усечена в конце бюджета
            for c, m, ym in zip(self.chains[:n], ics[:n], y[:n]):
                if ym == c["ya"]:
                    c["a"] = m.copy()
                else:
                    c["b"] = m.copy()                # класс b или третий класс: граница между a и серединой
                    c["yb"] = int(ym)
                c["steps"] += 1
            self.chains = [c for c in self.chains if c["steps"] < self.n_bisect]
        if self.X is None:
            self.X, self.y, self.xi = ics.copy(), y, xi
        else:
            self.X = np.vstack([self.X, ics])
            self.y = np.concatenate([self.y, y])
            self.xi = np.concatenate([self.xi, xi])


class BisectArm(EdgeBisection):
    """Рукав бисекции для портфеля: пары «длинный переходный процесс — ближайшая точка другого класса»
    строятся по всем точкам, оценённым портфелем (последние max_pool), а цепи обновляются только своими
    серединами. Точки копятся кусками, без склейки на каждой пачке."""
    name = "bisect_arm"

    def __init__(self, problem, batch=20, n_chains=10, n_bisect=12, top_frac=0.2, max_pool=20000, **kw):
        super().__init__(problem, batch=batch, n_chains=n_chains, n_bisect=n_bisect, top_frac=top_frac,
                         eps=0.0, max_pool=max_pool)
        self._Xc, self._yc, self._xic = [], [], []
        self._n = 0
        self.classes = set()
        self._prop = []              # цепи, предложившие середины в последней пачке (по порядку)

    def add_data(self, ics, feats, labels):
        y = self._cls(feats, labels)
        self._Xc.append(np.array(ics, dtype=float))
        self._yc.append(y)
        self._xic.append(transient_length(self.s, feats))
        self._n += len(ics)
        self.classes.update(np.unique(y).tolist())
        while len(self._Xc) > 1 and self._n - len(self._Xc[0]) >= self.max_pool:
            self._n -= len(self._Xc[0])
            self._Xc.pop(0); self._yc.pop(0); self._xic.pop(0)

    def _materialize(self):
        X, y, xi = np.vstack(self._Xc), np.concatenate(self._yc), np.concatenate(self._xic)
        self.X, self.y, self.xi = X[-self.max_pool:], y[-self.max_pool:], xi[-self.max_pool:]

    def propose_mids(self, n_max):
        """До n_max середин отрезков; пусто, пока в данных меньше двух классов."""
        self._prop = []
        if n_max <= 0 or len(self.classes) < 2:
            return np.zeros((0, self.d))
        if len(self.chains) < min(self.n_chains, n_max):
            self._materialize()
            while len(self.chains) < min(self.n_chains, n_max):
                c = self._new_chain()
                if c is None:
                    break
                self.chains.append(c)
        self._prop = self.chains[:n_max]
        return np.array([0.5 * (c["a"] + c["b"]) for c in self._prop]) if self._prop else np.zeros((0, self.d))

    def observe_mids(self, ics, feats, labels):
        y = self._cls(feats, labels)
        for c, m, ym in zip(self._prop, ics, y):
            if ym == c["ya"]:
                c["a"] = m.copy()
            else:
                c["b"] = m.copy()
                c["yb"] = int(ym)
            c["steps"] += 1
        self.chains = [c for c in self.chains if c["steps"] < self.n_bisect]
