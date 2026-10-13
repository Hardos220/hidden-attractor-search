"""Стратегии поиска.

* Uniform        — равномерный перебор (точка отсчёта).
* AMS            — адаптивное многоуровневое расщепление по начальным условиям: координата реакции
                   xi = (класс, значение): ушедшие траектории ранжируются по времени ухода, ограниченные —
                   по новизне отпечатка относительно известных режимов; худшая доля популяции
                   замещается мутациями выживших с условием превышения уровня.
* NoveltyME      — MAP-Elites по дескриптору поведения (класс ухода / показатель Ляпунова / размер /
                   положение раннего окна); элита ячейки — наиболее новая; потомки — мутации элит.
* ActiveBoundary — активное обучение границ бассейнов: kNN-классификатор на начальных условиях,
                   запросы с максимальной энтропией предсказания (плюс доля случайных).
* DEZelinka      — дифференциальная эволюция с функцией стоимости Зелинки (2016): стоимость 0, если
                   максимум нормы на последних 25 % окна лежит в заданном диапазоне «диаметра»;
                   требует знания размера цели (оракул).
Все стратегии предлагают по batch начальных условий за вызов.
"""
import numpy as np

from ..identify import LAB_ESCAPE, LAB_UNKNOWN
from ..integrate import F_ESC, F_LAM, F_RMAX, F_TESC, feature_slices
from .harness import Strategy


class Uniform(Strategy):
    name = "uniform"

    def propose(self):
        return self.s.sample_box(self.batch, self.rng)


# ----------------------------------------------------------------------------
class AMS(Strategy):
    """Адаптивное многоуровневое расщепление по начальным условиям."""
    name = "ams"

    def __init__(self, problem, batch=20, n_pop=100, kill_frac=0.2, sigma0=0.05, sigma_min=1e-4,
                 max_retry=3, seed_ics=None, **kw):
        super().__init__(problem, batch=batch)
        self.n_pop = int(n_pop)
        self.k = max(1, int(kill_frac * n_pop))
        self.sigma = float(sigma0)
        self.sigma_min = float(sigma_min)
        self.max_retry = int(max_retry)
        self.pop = None            # (n_pop, d)
        self.xi = None             # (n_pop,) координата реакции
        self.level = -np.inf
        self.pending = None        # индексы замещаемых слотов для текущей пачки
        self.retry = 0
        self.seed_ics = seed_ics
        self.acc_hist = []

    def score(self, feats, labels):
        """xi: ушедшие -> время ухода (в долях длины интегрирования), ограниченные -> 1 + новизна."""
        n_total = float(self.s.n_tr + self.s.n_obs)
        xi = np.where(feats[:, F_ESC] > 0.5, feats[:, F_TESC] / n_total, 1.0 + self.novelty(feats, labels))
        return xi

    def propose(self):
        if self.pop is None:
            n0 = self.n_pop
            ics = self.s.sample_box(n0, self.rng) if self.seed_ics is None else np.array(self.seed_ics)[:n0]
            if len(ics) < n0:
                ics = np.vstack([ics, self.s.sample_box(n0 - len(ics), self.rng)])
            self.phase = "init"
            return ics
        # итерация: убить k худших, предложить k мутаций выживших
        order = np.argsort(self.xi)
        self.pending = order[: self.k]
        self.level = float(self.xi[order[self.k]])
        survivors = order[self.k:]
        parents = self.rng.choice(survivors, size=self.k)
        cand = np.array([self.mutate(self.pop[i], self.sigma) for i in parents])
        self.phase = "iter"
        return cand

    def observe(self, ics, feats, finals, labels):
        xi = self.score(feats, labels)
        if self.phase == "init":
            self.pop = ics.copy()
            self.xi = xi
            return
        accepted = xi >= self.level
        # принятые замещают убитых; непринятые тоже замещают (чтобы не терять бюджет),
        # но с меньшим приоритетом: их xi остаётся ниже уровня и они будут убиты первыми
        self.pop[self.pending] = ics
        self.xi[self.pending] = xi
        acc = float(accepted.mean())
        self.acc_hist.append(acc)
        # адаптация шага мутации: цель — доля принятия около 0.5
        if acc < 0.3:
            self.sigma = max(self.sigma * 0.7, self.sigma_min)
        elif acc > 0.7:
            self.sigma = min(self.sigma * 1.3, 0.5)


# ----------------------------------------------------------------------------
class NoveltyME(Strategy):
    """MAP-Elites: архив элит по дискретизированному дескриптору поведения."""
    name = "novelty_me"

    def __init__(self, problem, batch=20, n_init=100, sigma0=0.05, bins=12, seed_ics=None, **kw):
        super().__init__(problem, batch=batch)
        self.n_init = int(n_init)
        self.sigma = float(sigma0)
        self.bins = int(bins)
        self.archive = {}          # cell -> (ic, fitness)
        self.seed_ics = seed_ics
        self.initialized = False
        self.n_total = float(self.s.n_tr + self.s.n_obs)
        self.L = float(np.max(self.hi - self.lo)) / 2.0

    def descriptor(self, feats):
        """Дескриптор: (класс, b1, b2) — для ушедших: (0, бин log t_esc, бин нормы раннего окна);
        для ограниченных: (1, бин λ, бин log rmax)."""
        sl = feature_slices(self.d)
        esc = feats[:, F_ESC] > 0.5
        cells = []
        for i in range(len(feats)):
            if esc[i]:
                b1 = int(np.clip(np.log10(max(feats[i, F_TESC], 1.0)) / np.log10(self.n_total) * self.bins, 0, self.bins - 1))
                er = np.linalg.norm(feats[i, sl["emean"]]) / (self.L + 1e-9)
                b2 = int(np.clip(er * self.bins, 0, self.bins - 1))
                cells.append((0, b1, b2))
            else:
                lam = feats[i, F_LAM]
                b1 = int(np.clip((lam + 0.05) / 0.3 * self.bins, 0, self.bins - 1))
                b2 = int(np.clip(np.log10(max(feats[i, F_RMAX], 1e-3) / self.L + 1e-9) + 3, 0, self.bins - 1))
                cells.append((1, b1, b2))
        return cells

    def propose(self):
        if not self.initialized:
            ics = self.s.sample_box(self.n_init, self.rng) if self.seed_ics is None else np.array(self.seed_ics)[: self.n_init]
            if len(ics) < self.n_init:
                ics = np.vstack([ics, self.s.sample_box(self.n_init - len(ics), self.rng)])
            return ics
        keys = list(self.archive)
        cand = []
        for _ in range(self.batch):
            if keys and self.rng.random() < 0.9:
                ic, _ = self.archive[keys[self.rng.integers(len(keys))]]
                cand.append(self.mutate(ic, self.sigma))
            else:
                cand.append(self.s.sample_box(1, self.rng)[0])
        return np.array(cand)

    def observe(self, ics, feats, finals, labels):
        self.initialized = True
        nov = self.novelty(feats, labels)
        cells = self.descriptor(feats)
        # приспособленность: новизна для ограниченных, время ухода для ушедших (поздний уход лучше)
        fit = np.where(feats[:, F_ESC] > 0.5, feats[:, F_TESC] / self.n_total, 1.0 + nov)
        n_new = 0
        for ic, c, f in zip(ics, cells, fit):
            if c not in self.archive or f > self.archive[c][1]:
                if c not in self.archive:
                    n_new += 1
                self.archive[c] = (ic.copy(), float(f))
        # адаптация шага: если новых ячеек мало — уменьшаем шаг (локальный поиск), иначе увеличиваем
        if n_new == 0:
            self.sigma = max(self.sigma * 0.8, 1e-4)
        else:
            self.sigma = min(self.sigma * 1.2, 0.5)


# ----------------------------------------------------------------------------
class ActiveBoundary(Strategy):
    """Активное обучение границ бассейнов: запросы в точках максимальной неопределённости kNN.

    Пул кандидатов: доля local_frac — гауссовы возмущения обучающих точек, у которых среди
    k соседей есть точка другого класса (масштаб — половина расстояния до ближайшего соседа
    другого класса); остальное — равномерные кандидаты. Классификатор переобучается, когда
    обучающая выборка выросла на refit_growth (экономия при больших бюджетах).
    """
    name = "active_boundary"

    def __init__(self, problem, batch=20, n_init=100, k=10, n_cand=2000, explore=0.2, local_frac=0.5,
                 refit_growth=0.1, seed_ics=None, cand_sampler=None, knn_algo="auto", max_train=None, **kw):
        super().__init__(problem, batch=batch)
        # max_train (2026-10-13): классификатор обучается только на последних max_train точках; без ограничения его
        # стоимость растёт с длиной запуска (на трёх копиях Чуа II при 1e5 оценок — до половины времени)
        self.max_train = int(max_train) if max_train else None
        # knn_algo="fast" (2026-10-12): при размерности >= 6 или обучающей выборке <= 2000 точек классификатор ищет
        # соседей прямым перебором (BLAS), а не деревом: в 3-10 раз быстрее при тех же соседях (с точностью до
        # округления расстояний); "auto" — выбор sklearn, как в прежних расчётах
        self.knn_algo = knn_algo
        self.n_init = int(n_init)
        self.k = int(k)
        self.n_cand = int(n_cand)
        self.explore = float(explore)
        self.local_frac = float(local_frac)
        self.refit_growth = float(refit_growth)
        self.seed_ics = seed_ics
        self.cand_sampler = cand_sampler      # функция (n, rng) -> кандидаты (например, нульклины)
        # обучающая выборка копится кусками и склеивается только при переобучении (раз на refit_growth
        # роста): склейка на каждой пачке и np.unique по всем меткам делали запуск квадратичным по n
        self._Xc, self._yc = [], []
        self._n = 0
        self.classes = set()
        self._X_fit = None                    # выборка на момент последнего переобучения
        self.knn = None
        self.n_fit = 0
        self.edge_idx = None
        self.edge_scale = None

    def _collapse(self):
        if len(self._Xc) > 1:
            self._Xc = [np.vstack(self._Xc)]
            self._yc = [np.concatenate(self._yc)]

    @property
    def X(self):
        if not self._Xc:
            return None
        self._collapse()
        return self._Xc[0]

    @property
    def y(self):
        if not self._yc:
            return None
        self._collapse()
        return self._yc[0]

    def _label_class(self, feats, labels):
        return np.where(feats[:, F_ESC] > 0.5, -1, labels)

    def _norm(self, X):
        return (X - self.lo) / (self.hi - self.lo)

    def _refit(self):
        from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
        X_all, y_all = self.X, self.y
        if self.max_train and len(X_all) > self.max_train:
            X_all, y_all = X_all[-self.max_train:], y_all[-self.max_train:]
        Xs = self._norm(X_all)
        k = min(self.k, len(Xs))
        algo = "brute" if self.knn_algo == "fast" and (self.d >= 6 or len(Xs) <= 2000) else "auto"
        self.knn = KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm=algo).fit(Xs, y_all)
        nn = NearestNeighbors(n_neighbors=k).fit(Xs)
        dist, idx = nn.kneighbors(Xs)
        other = y_all[idx] != y_all[:, None]
        has = other.any(axis=1)
        self.edge_idx = np.where(has)[0]
        if len(self.edge_idx):
            d_other = np.where(other[has], dist[has], np.inf).min(axis=1)
            self.edge_scale = 0.5 * d_other
        self._X_fit = X_all
        self.n_fit = self._n                  # всего точек на момент переобучения (при ограничении — не длина выборки)

    def _candidates(self, n):
        n_loc = int(self.local_frac * n) if (self.edge_idx is not None and len(self.edge_idx)) else 0
        parts = []
        if n_loc:
            pick = self.rng.integers(0, len(self.edge_idx), size=n_loc)
            base = self._X_fit[self.edge_idx[pick]]       # индексы edge_idx — из последнего переобучения
            sc = self.edge_scale[pick][:, None] * (self.hi - self.lo)
            parts.append(np.clip(base + sc * self.rng.normal(size=base.shape), self.lo, self.hi))
        n_rest = n - n_loc
        if self.cand_sampler is not None:
            half = n_rest // 2
            c = self.cand_sampler(half, self.rng)
            if len(c):
                parts.append(c)
                n_rest -= len(c)
        parts.append(self.s.sample_box(n_rest, self.rng))
        return np.vstack(parts)

    def propose(self, n=None):
        n = self.batch if n is None else int(n)
        if self._n == 0:
            if self.seed_ics is not None:
                ics = np.asarray(self.seed_ics, float)[: self.n_init]
                if len(ics) < self.n_init:
                    ics = np.vstack([ics, self.s.sample_box(self.n_init - len(ics), self.rng)])
                return ics
            return self.s.sample_box(self.n_init, self.rng)
        if len(self.classes) < 2:
            return self.s.sample_box(n, self.rng)
        if self.knn is None or self._n >= (1 + self.refit_growth) * self.n_fit:
            self._refit()
        cand = self._candidates(self.n_cand)
        proba = self.knn.predict_proba(self._norm(cand))
        ent = -np.sum(np.where(proba > 0, proba * np.log(proba + 1e-12), 0.0), axis=1)
        n_exp = int(round(self.explore * n))
        n_act = n - n_exp
        pick = cand[np.argsort(-ent)[:n_act]]
        if n_exp > 0:
            pick = np.vstack([pick, self.s.sample_box(n_exp, self.rng)])
        return pick

    def observe(self, ics, feats, finals, labels):
        y = self._label_class(feats, labels)
        self._Xc.append(np.array(ics, dtype=float))
        self._yc.append(np.asarray(y))
        self._n += len(ics)
        self.classes.update(np.unique(y).tolist())


# ----------------------------------------------------------------------------
class Portfolio(Strategy):
    """Портфель: доля eps равномерных точек (гарантия G >= eps в худшем случае), рукав активного
    обучения границ (обучается на всех оценённых точках) и рукав расщепления по редкости.
    Порядок точек в пачке перемешивается, чтобы ни один рукав не получал преимущества в учёте
    момента первого попадания."""
    name = "portfolio"

    def __init__(self, problem, batch=20, eps=0.2, use_boundary=True, use_sus=True, n_init=100,
                 sus_kw=None, ab_kw=None, uniform_sampler=None, seed_ics=None, sus_cls=None, **kw):
        super().__init__(problem, batch=batch)
        from .sus import SubsetSim
        self.eps = float(eps)
        self.arms = []
        if use_sus:
            self.sus = (sus_cls or SubsetSim)(problem, batch=batch, n0=n_init, seed_ics=seed_ics, **(sus_kw or {}))
            self.arms.append(self.sus)
        else:
            self.sus = None
        if use_boundary:
            self.ab = ActiveBoundary(problem, batch=batch, n_init=n_init, explore=0.0, seed_ics=seed_ics, **(ab_kw or {}))
            self.arms.append(self.ab)
        else:
            self.ab = None
        self.uniform_sampler = uniform_sampler
        self.n_init = int(n_init)
        self.seed_ics = seed_ics
        self.phase = "init"
        self._perm = None

    def propose(self):
        if self.phase == "init":
            # общая опорная выборка для обоих рукавов (без перемешивания: общие случайные числа с перебором)
            if self.seed_ics is not None:
                X = np.asarray(self.seed_ics, float)[: self.n_init]
                if len(X) < self.n_init:
                    X = np.vstack([X, self.s.sample_box(self.n_init - len(X), self.rng)])
            else:
                X = self.s.sample_box(self.n_init, self.rng)
            self._perm = np.arange(len(X))
            self._owners = np.array(["init"] * len(X))
            return X
        parts, owners = [], []
        if self.sus is not None:
            p = self.sus.propose()
            parts.append(p); owners += ["sus"] * len(p)
        # рукав границ работает, только когда в данных есть граница (не меньше двух классов);
        # без рукава расщепления он работает всегда (как самостоятельное активное обучение)
        ab_active = self.ab is not None and (self.sus is None or len(self.ab.classes) >= 2)
        if ab_active:
            p = self.ab.propose(self.batch)
            parts.append(p); owners += ["ab"] * len(p)
        n_arm = sum(len(p) for p in parts)
        n_u = int(np.ceil(self.eps / max(1e-9, 1 - self.eps) * n_arm)) if parts else self.batch
        if n_u:
            if self.uniform_sampler is not None and self.rng.random() < 0.5:
                u = self.uniform_sampler(n_u, self.rng)
                if len(u) < n_u:
                    u = np.vstack([u, self.s.sample_box(n_u - len(u), self.rng)]) if len(u) else self.s.sample_box(n_u, self.rng)
            else:
                u = self.s.sample_box(n_u, self.rng)
            parts.append(u); owners += ["u"] * n_u
        X = np.vstack(parts)
        self._owners = np.array(owners)
        self._perm = self.rng.permutation(len(X))
        return X[self._perm]

    def observe(self, ics, feats, finals, labels):
        if self.phase == "init":
            self.phase = "run"
            if self.sus is not None:
                self.sus.observe(ics, feats, finals, labels)     # опорная выборка и первый уровень
            if self.ab is not None:
                self.ab.observe(ics, feats, finals, labels)
            return
        n = len(ics)
        orig = self._perm[:n]                 # исходные позиции оценённых точек
        order = np.argsort(orig)
        ics, feats, finals, labels = ics[order], feats[order], finals[order], labels[order]
        own = self._owners[orig[order]]
        complete = n == len(self._perm)       # пачка усечена только в конце бюджета
        if self.sus is not None and complete:
            m = own == "sus"
            if m.any():
                self.sus.observe(ics[m], feats[m], finals[m], labels[m])
        if self.ab is not None:
            # рукав границ учится на всех точках
            self.ab.observe(ics, feats, finals, labels)


# ----------------------------------------------------------------------------
class DEZelinka(Strategy):
    """Дифференциальная эволюция DE/rand/1/bin по начальным условиям с функцией стоимости Зелинки.
    diam = (d_lo, d_hi): если max|x| на последней четверти окна наблюдения лежит в [d_lo, d_hi],
    стоимость 0, иначе стоимость = max|x| (для ушедших — 1e6)."""
    name = "de_zelinka"

    def __init__(self, problem, batch=20, n_pop=20, F=0.9, CR=0.3, diam=(3.0, 11.0), **kw):
        super().__init__(problem, batch=batch)
        self.n_pop = int(n_pop)
        self.F = float(F)
        self.CR = float(CR)
        self.diam = diam
        self.pop = None
        self.cost = None
        self.trial = None

    def _cost(self, feats):
        rmax = feats[:, F_RMAX]
        esc = feats[:, F_ESC] > 0.5
        c = np.where((rmax >= self.diam[0]) & (rmax <= self.diam[1]), 0.0, rmax)
        c[esc] = 1e6
        return c

    def propose(self):
        if self.pop is None:
            return self.s.sample_box(self.n_pop, self.rng)
        n = self.n_pop
        trial = np.empty_like(self.pop)
        for i in range(n):
            r1, r2, r3 = self.rng.choice([j for j in range(n) if j != i], size=3, replace=False)
            mutant = self.pop[r3] + self.F * (self.pop[r1] - self.pop[r2])
            jrand = self.rng.integers(self.d)
            mask = (self.rng.random(self.d) < self.CR)
            mask[jrand] = True
            trial[i] = np.where(mask, mutant, self.pop[i])
        self.trial = np.clip(trial, self.lo, self.hi)
        return self.trial

    def observe(self, ics, feats, finals, labels):
        c = self._cost(feats)
        if self.pop is None:
            self.pop = ics.copy()
            self.cost = c
            return
        better = c <= self.cost
        self.pop[better] = ics[better]
        self.cost[better] = c[better]


def _sus(problem, **kw):
    from .sus import SubsetSim
    return SubsetSim(problem, n0=kw.pop("n_init", 100), **{k: v for k, v in kw.items() if k in ("batch", "seed_ics", "score", "p0", "sigma0", "max_levels", "stall_levels", "k")})


STRATEGIES = {
    "uniform": Uniform, "ams": AMS, "novelty_me": NoveltyME, "active_boundary": ActiveBoundary,
    "de_zelinka": DEZelinka,
    # второе поколение
    "sus_rarity": lambda pr, **kw: _sus(pr, score="rarity", **kw),
    "sus_tesc": lambda pr, **kw: _sus(pr, score="tesc", **kw),
    "sus": lambda pr, **kw: _sus(pr, score="trans", **kw),
    "portfolio": lambda pr, **kw: Portfolio(pr, sus_kw={"score": "trans"}, **kw),
    "portfolio_no_sus": lambda pr, **kw: Portfolio(pr, use_sus=False, **kw),
    "portfolio_no_ab": lambda pr, **kw: Portfolio(pr, use_boundary=False, sus_kw={"score": "trans"}, **kw),
    "portfolio_eps0": lambda pr, **kw: Portfolio(pr, eps=0.0, sus_kw={"score": "trans"}, **kw),
    # робастность к гиперпараметрам (этап 3)
    "portfolio_eps01": lambda pr, **kw: Portfolio(pr, eps=0.1, sus_kw={"score": "trans"}, **kw),
    "portfolio_eps03": lambda pr, **kw: Portfolio(pr, eps=0.3, sus_kw={"score": "trans"}, **kw),
    "portfolio_p01": lambda pr, **kw: Portfolio(pr, sus_kw={"score": "trans", "p0": 0.1}, **kw),
    "portfolio_p03": lambda pr, **kw: Portfolio(pr, sus_kw={"score": "trans", "p0": 0.3}, **kw),
    "portfolio_sig03": lambda pr, **kw: Portfolio(pr, sus_kw={"score": "trans", "sigma0": 0.3}, **kw),
    "portfolio_n100": lambda pr, **kw: Portfolio(pr, sus_kw={"score": "trans"}, **{**kw, "n_init": 100}),
}


def _edge(problem, eps=0.0, **kw):
    from .edge import EdgeBisection
    return EdgeBisection(problem, eps=eps, **{k: v for k, v in kw.items() if k in ("batch", "n_init", "seed_ics")})


# бисекция к границам бассейнов (базовая стратегия, батч B6 e50): без равномерной доли и с 20 %
STRATEGIES["edge_bisect"] = lambda pr, **kw: _edge(pr, eps=0.0, **kw)
STRATEGIES["edge_bisect_u20"] = lambda pr, **kw: _edge(pr, eps=0.2, **kw)


# диагностика предела разрешения расщепления (Р3, 2026-10-10): уровень с шагом цепочек sigma < sigma_min
# засчитывается как застой, два застоя — перезапуск каскада; по умолчанию sigma_min = 0.005 ширины области
def _sus_trans(problem, sigma_min, **kw):
    from .sus import SubsetSim
    return SubsetSim(problem, n0=kw.pop("n_init", 100), score="trans", sigma_min=sigma_min,
                     **{k: v for k, v in kw.items() if k in ("batch", "seed_ics")})


STRATEGIES["sus_smin"] = lambda pr, **kw: _sus_trans(pr, 5e-4, **kw)
STRATEGIES["sus_nostall"] = lambda pr, **kw: _sus_trans(pr, 0.0, **kw)
STRATEGIES["portfolio_smin"] = lambda pr, **kw: Portfolio(pr, sus_kw={"score": "trans", "sigma_min": 5e-4}, **kw)


# диагностика (Р3): расщепление с перевыборкой подмножеств координат и портфель с таким рукавом
def _sus_coord(problem, **kw):
    from .sus import SubsetSimCoord
    return SubsetSimCoord(problem, n0=kw.pop("n_init", 100), score="trans",
                          **{k: v for k, v in kw.items() if k in ("batch", "seed_ics")})


def _portfolio_coord(problem, **kw):
    from .sus import SubsetSimCoord
    return Portfolio(problem, sus_kw={"score": "trans"}, sus_cls=SubsetSimCoord, **kw)


STRATEGIES["sus_coord"] = _sus_coord
STRATEGIES["portfolio_coord"] = _portfolio_coord


# ----------------------------------------------------------------------------
class PortfolioPlus(Portfolio):
    """Портфель с четвёртым рукавом — бисекцией к границам бассейнов (2026-10-12).

    Фиксированный вариант: в каждой пачке рукав расщепления предлагает свои n_chains точек, рукав границ —
    ab_n, бисекция — bis_n середин отрезков, равномерная доля — eps от суммы.
    Адаптивный вариант (adaptive=True) меняет ab_n и bis_n по наблюдаемому состоянию каскада расщепления:
    после перезапуска каскада (застой уровней или их предел — признак отсутствия градуированного сигнала)
    в течение stall_window пачек бюджет отдаётся рукавам границ (ab_n_stall, bis_n_stall), а когда уровни
    растут без перезапусков — расщеплению (ab_n_prog, bis_n_prog). Каждый рукав сохраняет не меньше
    четверти неравномерной части пачки, а равномерная доля не меняется, поэтому гарантии утверждения о
    портфеле сохраняются с долями w_j >= (1 - eps) / 4.
    """
    name = "portfolio_plus"

    def __init__(self, problem, adaptive=False, bis_n=None, ab_n_stall=20, bis_n_stall=10, ab_n_prog=5, bis_n_prog=5,
                 stall_window=10, **kw):
        super().__init__(problem, **kw)
        from .edge import BisectArm
        # по умолчанию половина пачки рукава границ (10 при batch = 20, как в B11 и B12): ширина пачки
        # портфеля масштабируется одним параметром batch (вместе с n_init — числом цепей расщепления)
        self.bis_n = int(bis_n) if bis_n is not None else max(1, self.batch // 2)
        self.bis = BisectArm(problem, batch=self.batch, n_chains=max(10, self.bis_n))
        self.adaptive = bool(adaptive)
        self.ab_n_stall, self.bis_n_stall = int(ab_n_stall), int(bis_n_stall)
        self.ab_n_prog, self.bis_n_prog = int(ab_n_prog), int(bis_n_prog)
        self.stall_window = int(stall_window)
        self._restarts_seen = 0
        self._since_restart = 10 ** 9
        self.mode_counts = {"stall": 0, "progress": 0, "neutral": 0}

    def _counts(self):
        ab_n, bis_n = self.batch, self.bis_n
        if not self.adaptive or self.sus is None:
            return ab_n, bis_n
        if self.sus.n_restarts != self._restarts_seen:
            self._restarts_seen = self.sus.n_restarts
            self._since_restart = 0
        else:
            self._since_restart += 1
        if self._since_restart < self.stall_window:
            self.mode_counts["stall"] += 1
            return self.ab_n_stall, self.bis_n_stall
        if self.sus.n_level >= 2:
            self.mode_counts["progress"] += 1
            return self.ab_n_prog, self.bis_n_prog
        self.mode_counts["neutral"] += 1
        return ab_n, bis_n

    def propose(self):
        if self.phase == "init":
            return super().propose()
        parts, owners = [], []
        if self.sus is not None:
            p = self.sus.propose()
            parts.append(p); owners += ["sus"] * len(p)
        ab_n, bis_n = self._counts()
        if self.ab is not None and (self.sus is None or len(self.ab.classes) >= 2):
            p = self.ab.propose(ab_n)
            parts.append(p); owners += ["ab"] * len(p)
        m = self.bis.propose_mids(bis_n)
        if len(m):
            parts.append(m); owners += ["bis"] * len(m)
        n_arm = sum(len(p) for p in parts)
        n_u = int(np.ceil(self.eps / max(1e-9, 1 - self.eps) * n_arm)) if parts else self.batch
        if n_u:
            if self.uniform_sampler is not None and self.rng.random() < 0.5:
                u = self.uniform_sampler(n_u, self.rng)
                if len(u) < n_u:
                    u = np.vstack([u, self.s.sample_box(n_u - len(u), self.rng)]) if len(u) else self.s.sample_box(n_u, self.rng)
            else:
                u = self.s.sample_box(n_u, self.rng)
            parts.append(u); owners += ["u"] * n_u
        X = np.vstack(parts)
        self._owners = np.array(owners)
        self._perm = self.rng.permutation(len(X))
        return X[self._perm]

    def observe(self, ics, feats, finals, labels):
        if self.phase == "init":
            super().observe(ics, feats, finals, labels)
            self.bis.add_data(ics, feats, labels)
            return
        n = len(ics)
        orig = self._perm[:n]
        order = np.argsort(orig)
        ics, feats, finals, labels = ics[order], feats[order], finals[order], labels[order]
        own = self._owners[orig[order]]
        complete = n == len(self._perm)
        if self.sus is not None and complete:
            m = own == "sus"
            if m.any():
                self.sus.observe(ics[m], feats[m], finals[m], labels[m])
        if self.ab is not None:
            self.ab.observe(ics, feats, finals, labels)
        self.bis.add_data(ics, feats, labels)
        if complete:
            m = own == "bis"
            if m.any():
                self.bis.observe_mids(ics[m], feats[m], labels[m])


STRATEGIES["portfolio_bis"] = lambda pr, **kw: PortfolioPlus(pr, adaptive=False, sus_kw={"score": "trans"}, **kw)
STRATEGIES["portfolio_ad"] = lambda pr, **kw: PortfolioPlus(pr, adaptive=True, sus_kw={"score": "trans"}, **kw)


def _portfolio_fast(problem, max_train=None, **kw):
    """Портфель с быстрым kNN рукава границ (knn_algo="fast"); в остальном — portfolio. max_train — обучающая
    выборка рукава границ ограничена последними max_train точками (portfolio_cap)."""
    ab = dict(kw.pop("ab_kw", None) or {})
    ab["knn_algo"] = "fast"
    if max_train:
        ab["max_train"] = int(max_train)
    return Portfolio(problem, sus_kw={"score": "trans"}, ab_kw=ab, **kw)


STRATEGIES["portfolio_fast"] = _portfolio_fast
STRATEGIES["portfolio_cap"] = lambda pr, **kw: _portfolio_fast(pr, max_train=20000, **kw)
