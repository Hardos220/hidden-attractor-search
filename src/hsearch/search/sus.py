"""Многоуровневое расщепление в пространстве начальных условий (subset simulation; Au & Beck 2001)
с координатой реакции «редкость переходного процесса».

Для статической задачи «начальное условие -> отпечаток траектории» статическим аналогом
адаптивного многоуровневого расщепления (Cérou & Guyader 2007) является subset simulation:
уровни задаются адаптивно квантилем координаты реакции xi, выборка над уровнем пополняется
условными марковскими цепями (случайное блуждание, принимаемое только при xi >= уровня;
при равномерной мере на B допустимость сводится к этому условию).

Координата реакции. Отпечаток переходного процесса z(x0) = (флаг ухода, ln t_esc / ln N,
среднее и СКО раннего окна в масштабе L). Редкость xi(x0) — среднее расстояние от z(x0) до
k ближайших соседей в опорной выборке Z_ref (первые n0 равномерных траекторий), по
стандартизованным координатам. Опорная выборка фиксируется при (пере)запуске, поэтому xi
стационарна в пределах одного запуска цепей, как требует теория расщепления.

Физическая мотивация: вблизи границы бассейна траектория долго следует устойчивому
многообразию инвариантного множества на границе (краевого состояния), поэтому её переходный
процесс нетипичен (длинное время ухода, задержка вблизи седла). Бассейны скрытых аттракторов
окружены такими границами, и подъём по редкости ведёт к ним.
"""
import numpy as np

from ..integrate import F_ESC, F_TESC, feature_slices
from .harness import Strategy


def reflect(y, lo, hi):
    """Зеркальное отражение точки в параллелепипед [lo, hi] (сохраняет равномерную меру)."""
    span = hi - lo
    t = np.mod(y - lo, 2 * span)
    t = np.where(t > span, 2 * span - t, t)
    return lo + t


def transient_descriptor(system, feats):
    """Матрица описаний переходного процесса (n, 2d + 2)."""
    d = system.dim
    sl = feature_slices(d)
    L = float(np.max(system.box_hi - system.box_lo)) / 2.0
    n_total = float(system.n_tr + system.n_obs)
    esc = (feats[:, F_ESC] > 0.5).astype(float)
    lt = np.where(esc > 0.5, np.log(np.maximum(feats[:, F_TESC], 1.0)) / np.log(n_total), 1.0)
    return np.column_stack([esc, lt, feats[:, sl["emean"]] / L, feats[:, sl["estd"]] / L])


def transient_length(system, feats):
    """Одномерная координата «длительность переходного процесса» (лексикографическая):
    ушедшие траектории — ln t_esc / ln N ∈ [0, 1]; ограниченные — 1 + D, где D — расстояние
    (в масштабе L) между статистиками раннего окна (среднее, СКО) и асимптотического окна.
    Траектория, быстро вышедшая на свой режим, имеет D ≈ 0; задержавшаяся у краевого
    состояния на границе бассейна — большое D (или большое t_esc)."""
    d = system.dim
    sl = feature_slices(d)
    L = float(np.max(system.box_hi - system.box_lo)) / 2.0
    n_total = float(system.n_tr + system.n_obs)
    esc = feats[:, F_ESC] > 0.5
    lt = np.log(np.maximum(feats[:, F_TESC], 1.0)) / np.log(n_total)
    dm = feats[:, sl["emean"]] - feats[:, sl["mean"]]
    ds = feats[:, sl["estd"]] - feats[:, sl["std"]]
    D = np.sqrt((dm * dm).sum(axis=1) + (ds * ds).sum(axis=1)) / L
    return np.where(esc, lt, 1.0 + D)


class Rarity:
    """Редкость относительно фиксированной опорной выборки."""

    def __init__(self, Z_ref, k=5):
        self.k = int(k)
        self.mu = Z_ref.mean(axis=0)
        self.sd = Z_ref.std(axis=0)
        self.sd = np.where(self.sd < 1e-3, 1e-3, self.sd)
        self.Z = (Z_ref - self.mu) / self.sd
        from sklearn.neighbors import NearestNeighbors
        self.nn = NearestNeighbors(n_neighbors=min(self.k + 1, len(Z_ref))).fit(self.Z)

    def score(self, Z, self_included=False):
        Zs = (Z - self.mu) / self.sd
        k = min(self.k + (1 if self_included else 0), len(self.Z))
        dist, _ = self.nn.kneighbors(Zs, n_neighbors=k)
        if self_included:
            dist = dist[:, 1:]
        return dist.mean(axis=1)


class SubsetSim(Strategy):
    """Subset simulation по координате реакции «редкость» (или «время ухода», score='tesc')."""
    name = "sus"

    def __init__(self, problem, batch=20, n0=100, p0=0.2, sigma0=0.1, max_levels=10, stall_levels=2,
                 score="rarity", k=5, target_acc=0.4, seed_ics=None, rel_stall=0.02, sigma_min=0.005, **kw):
        self.rel_stall = float(rel_stall)
        self.sigma_min = float(sigma_min)
        super().__init__(problem, batch=batch)
        self.n0 = int(n0)
        self.p0 = float(p0)
        self.n_chains = max(1, int(round(self.n0 * self.p0)))
        self.steps_per_level = int(round(1.0 / self.p0))
        self.sigma0 = float(sigma0)
        self.max_levels = int(max_levels)
        self.stall_levels = int(stall_levels)
        self.score_kind = score
        self.k = int(k)
        self.target_acc = float(target_acc)
        self.seed_ics = seed_ics
        self.n_restarts = 0
        self.level_log = []
        self._reset()

    def _reset(self):
        self.phase = "init"
        self.rarity = None
        self.level = -np.inf
        self.n_level = 0
        self.stall = 0
        self.sigma = self.sigma0
        self.chain_x = None
        self.chain_xi = None
        self.lvl_x = []
        self.lvl_xi = []
        self.step = 0
        self.acc = []

    # --- координата реакции
    def _xi(self, feats, self_included=False):
        if self.score_kind == "tesc":
            n_total = float(self.s.n_tr + self.s.n_obs)
            esc = feats[:, F_ESC] > 0.5
            return np.where(esc, np.log(np.maximum(feats[:, F_TESC], 1.0)) / np.log(n_total), 2.0)
        if self.score_kind == "trans":
            return transient_length(self.s, feats)
        Z = transient_descriptor(self.s, feats)
        return self.rarity.score(Z, self_included=self_included)

    def _proposal(self, x):
        y = x + self.sigma * (self.hi - self.lo) * self.rng.normal(size=x.shape)
        return reflect(y, self.lo, self.hi)

    def propose(self):
        if self.phase == "init":
            if self.seed_ics is not None and self.n_restarts == 0:
                ics = np.asarray(self.seed_ics, float)[: self.n0]
                if len(ics) < self.n0:
                    ics = np.vstack([ics, self.s.sample_box(self.n0 - len(ics), self.rng)])
                return ics
            return self.s.sample_box(self.n0, self.rng)
        return np.array([self._proposal(x) for x in self.chain_x])

    def _start_level(self, X, XI):
        order = np.argsort(-XI)
        top = order[: self.n_chains]
        new_level = float(XI[order[self.n_chains - 1]]) if len(order) >= self.n_chains else float(XI.min())
        # застой: относительный прирост уровня мал или шаг блуждания схлопнулся
        if np.isfinite(self.level) and (new_level - self.level <= self.rel_stall * max(abs(self.level), 1e-9)
                                        or self.sigma < self.sigma_min):
            self.stall += 1
        else:
            self.stall = 0
        self.level = new_level
        self.chain_x = X[top].copy()
        self.chain_xi = XI[top].copy()
        self.lvl_x = [self.chain_x.copy()]
        self.lvl_xi = [self.chain_xi.copy()]
        self.step = 0
        self.acc = []
        self.n_level += 1
        self.level_log.append((self.n_restarts, self.n_level, self.level, self.sigma))

    def observe(self, ics, feats, finals, labels):
        if self.phase == "init":
            if self.score_kind == "rarity":
                self.rarity = Rarity(transient_descriptor(self.s, feats), k=self.k)
                XI = self._xi(feats, self_included=True)
            else:
                XI = self._xi(feats)
            self.phase = "level"
            self._start_level(ics.copy(), XI)
            return
        if len(ics) != len(self.chain_x):
            # последняя пачка, усечённая концом бюджета (run_search): запуск на ней заканчивается,
            # цепочки обновлять незачем (как в Portfolio.observe, который учит рукав только на полных пачках)
            return
        xi = self._xi(feats)
        accept = xi >= self.level
        self.acc.append(float(accept.mean()))
        self.chain_x[accept] = ics[accept]
        self.chain_xi[accept] = xi[accept]
        self.lvl_x.append(self.chain_x.copy())
        self.lvl_xi.append(self.chain_xi.copy())
        self.step += 1
        if self.step >= self.steps_per_level - 1:
            # адаптация шага по доле принятия за уровень
            a = float(np.mean(self.acc)) if self.acc else 0.0
            if a < 0.5 * self.target_acc:
                self.sigma = max(self.sigma * 0.5, 1e-7)
            elif a < self.target_acc:
                self.sigma = max(self.sigma * 0.8, 1e-7)
            elif a > 1.5 * self.target_acc:
                self.sigma = min(self.sigma * 1.5, 0.5)
            X = np.vstack(self.lvl_x)
            XI = np.concatenate(self.lvl_xi)
            if self.n_level >= self.max_levels or self.stall >= self.stall_levels:
                self.n_restarts += 1
                self._reset()
                return
            self._start_level(X, XI)


class SubsetSimCoord(SubsetSim):
    """SubsetSim с ходами двух видов (диагностика Р3, 2026-10-10): изотропный гауссов шаг, как в SubsetSim,
    или независимая перевыборка случайного подмножества координат (каждая с вероятностью 1/d, хотя бы одна)
    из равномерного распределения в области. Оба хода симметричны относительно равномерной меры, так что цепь
    по-прежнему выбирает равномерно на множестве уровня. Перевыборка исследует координаты без градиента и не
    требует уменьшать шаг, который нужен для разрешения тонких координат."""
    name = "sus_coord"

    def __init__(self, problem, redraw_frac=0.5, **kw):
        super().__init__(problem, **kw)
        self.redraw_frac = float(redraw_frac)

    def _proposal(self, x):
        if self.rng.random() < self.redraw_frac:
            m = self.rng.random(self.d) < 1.0 / self.d
            if not m.any():
                m[self.rng.integers(self.d)] = True
            y = x.copy()
            y[m] = self.lo[m] + (self.hi[m] - self.lo[m]) * self.rng.random(int(m.sum()))
            return y
        return super()._proposal(x)
