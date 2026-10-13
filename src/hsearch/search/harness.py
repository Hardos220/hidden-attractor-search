"""Каркас измерения стоимости поиска.

Задача (Problem): система, набор эталонов (все известные режимы, включая цель), индексы целевых
эталонов. Стратегия (Strategy) предлагает начальные условия пачками и получает отпечатки и метки;
она не знает, какая метка целевая. Каркас считает бюджет (число интегрирований) и момент первого
попадания в цель. Для отображений единица бюджета — одна орбита.

Семантика «известного»: стратегии получают список индексов известных (нецелевых) эталонов,
чтобы вычислять новизну как расстояние до известного; попадание в любой нецелевой и не известный
режим считается «открытием» и тоже фиксируется (поле discoveries).
"""
import time
from dataclasses import dataclass, field

import numpy as np

from ..identify import LAB_ESCAPE, LAB_UNKNOWN, match, stat_index
from ..integrate import F_ESC, F_FP, F_LAM, F_RMAX, F_TESC, feature_slices, nfeat, run_batch


def steps_of(feats, n_tr, n_obs):
    """Число шагов интегратора на траекторию: шаг переходного участка — один, шаг окна наблюдения — два
    (основная и сопутствующая траектории); ушедшая траектория обрывается на шаге ухода F_TESC."""
    esc = feats[:, F_ESC] > 0.5
    t = feats[:, F_TESC]
    return np.where(esc, np.where(t <= n_tr, t, n_tr + 2.0 * (t - n_tr)), float(n_tr + 2 * n_obs))


SCREEN_DEFAULT = {"n1": None, "m1": None, "strict": 0.5, "amp_tol": 0.02, "asym": "window"}
SCREEN_AUTO_MIN = 200                     # после стольких оценок правило auto решает, выгодна ли двухэтапная оценка
SCREEN_WINDOW = 10                        # решение — по последним SCREEN_WINDOW пачкам с двухэтапной оценкой
SCREEN_PROBE = 10                         # когда она выключена, каждая SCREEN_PROBE-я пачка — проба


def screened_batch(system, ics, refs, known_idx, tol=6.0, n1=None, m1=None, strict=0.5, amp_tol=0.02, timing=None,
                   asym="window"):
    """Двухэтапная оценка (досрочная остановка траекторий, уже пришедших на известный режим).

    Этап 1: переходный участок n1 шагов (не короче раннего окна min(n_tr, n_obs)) и короткое окно m1
    (по умолчанию n_obs / 4). Траектория останавливается, если на этом окне она уверенно опознана как
    известный нецелевой режим: равновесие — сошлась (признак F_FP) или затухает с амплитудой меньше
    amp_tol·L; цикл или хаотический аттрактор — z-оценка меньше strict·tol. Ушедшие траектории тоже
    окончательны. Остальные (цель, неизвестное, неуверенное) продолжаются с конечного состояния этапа 1
    до полной длины n_tr + n_obs; их отпечатки совпадают с отпечатками полного счёта бит в бит (та же
    основная траектория, раннее окно — с этапа 1, окно наблюдения — то же), так что подтверждение цели
    идёт по полному протоколу. Возвращает (feats, finals, labels, steps, stopped); в timing (dict) копятся
    времена интегрирования "int" и сопоставления "match", а также "saved" — шаги, сэкономленные
    остановкой (полная длина минус этап 1), и "extra" — лишние шаги этапа 1 у остальных траекторий
    (на них правило auto в Problem решает, оставлять ли двухэтапную оценку).

    asym — асимптотические статистики (среднее и СКО окна наблюдения) остановленных траекторий: "window" — по
    короткому окну этапа 1; "ref" — статистики опознанного режима: у равновесия — его точка и нулевой разброс (как
    у сошедшейся траектории полного счёта), у цикла или хаотического аттрактора — средние по калибровке эталона.
    От них зависит координата расщепления transient_length (расстояние раннего окна до асимптотического)."""
    tm = timing if timing is not None else {}
    for key in ("int", "match", "saved", "extra"):
        tm.setdefault(key, 0.0)
    s = system
    d = s.dim
    n_tr, n_obs = int(s.n_tr), int(s.n_obs)
    n_early = min(n_tr, n_obs)
    n1 = max(n_early, int(n1) if n1 else n_early)
    m1 = int(m1) if m1 else max(1, n_obs // 4)
    n = len(ics)
    if s.batch_x is None or n1 + m1 >= n_tr:
        t0 = time.perf_counter()
        feats, finals = run_batch(s, ics)
        t1 = time.perf_counter()
        labels, _ = match(s, feats, finals, refs, tol=tol)
        tm["int"] += t1 - t0; tm["match"] += time.perf_counter() - t1
        return feats, finals, labels, steps_of(feats, n_tr, n_obs), np.zeros(n, bool)
    from ..identify import scale as sys_scale
    L = sys_scale(s)
    F1 = np.zeros((n, nfeat(d)))
    X1 = np.zeros((n, d))
    t0 = time.perf_counter()
    s.batch_x(ics, s.params, s.dt, n1, m1, n_early, 0, s.esc_r, s.renorm_every, F1, X1)
    t1 = time.perf_counter()
    lab1, z1 = match(s, F1, X1, refs, tol=tol)
    tm["int"] += t1 - t0; tm["match"] += time.perf_counter() - t1
    esc1 = F1[:, F_ESC] > 0.5
    known = np.isin(lab1, np.asarray(known_idx, dtype=np.int64)) & ~esc1
    kind = np.array([refs[k].kind if k >= 0 else "" for k in lab1])
    sl = feature_slices(d)
    amp = F1[:, sl["std"]].max(axis=1)
    fp_ok = (F1[:, F_FP] > 0.5) | (amp < amp_tol * L)
    osc_ok = z1 < strict * tol
    stop = known & np.where(kind == "fp", fp_ok, osc_ok)
    cont = ~esc1 & ~stop
    if asym == "ref" and stop.any():
        for i in np.where(stop)[0]:
            r = refs[lab1[i]]
            if r.kind == "fp":
                c = r.point if r.point is not None else (r.project(X1[i]) if r.project is not None else X1[i])
                F1[i, sl["mean"]] = c
                F1[i, sl["std"]] = 0.0
            else:
                F1[i, sl["mean"]] = r.mu[sl["mean"]]
                F1[i, sl["std"]] = r.mu[sl["std"]]
    feats, finals, labels = F1, X1, lab1.copy()
    steps = steps_of(F1, n1, m1)
    full_len = float(n_tr + 2 * n_obs)
    tm["saved"] += float(stop.sum()) * (full_len - (n1 + 2.0 * m1))
    tm["extra"] += float((steps[esc1] - steps_of(F1[esc1], n_tr, n_obs)).sum())
    if cont.any():
        idx = np.where(cont)[0]
        F2 = np.zeros((len(idx), nfeat(d)))
        X2 = np.zeros((len(idx), d))
        t_off = n1 + m1
        n_tr2 = n_tr - t_off
        t0 = time.perf_counter()
        s.batch_x(np.ascontiguousarray(X1[idx]), s.params, s.dt, n_tr2, n_obs, 0, t_off, s.esc_r, s.renorm_every, F2, X2)
        tm["int"] += time.perf_counter() - t0
        for key in ("emean", "estd", "emin", "emax"):
            F2[:, sl[key]] = F1[idx][:, sl[key]]
        feats[idx] = F2
        finals[idx] = X2
        t1 = time.perf_counter()
        labels[idx], _ = match(s, F2, X2, refs, tol=tol)
        tm["match"] += time.perf_counter() - t1
        t = F2[:, F_TESC]
        e2 = F2[:, F_ESC] > 0.5
        steps2 = np.where(e2, np.where(t - t_off <= n_tr2, t - t_off, n_tr2 + 2.0 * (t - n_tr)), float(n_tr2 + 2 * n_obs))
        steps[idx] = n1 + 2.0 * m1 + steps2
        tm["extra"] += float((steps[idx] - steps_of(F2, n_tr, n_obs)).sum())
    return feats, finals, labels, steps, stop


@dataclass
class Problem:
    system: object
    refs: list
    target_idx: list
    known_idx: list                       # индексы эталонов, считающихся известными до поиска
    rng: np.random.Generator
    tol: float = 6.0
    n_eval: int = 0
    screen: object = None                 # None — полный счёт; True или dict — двухэтапная оценка (screened_batch);
                                          # "auto" — двухэтапная, пока она выгодна (см. evaluate)
    n_steps: float = 0.0                  # шаги интегратора (steps_of) — машинно-независимая стоимость
    n_stopped: int = 0                    # траектории, остановленные на этапе 1 двухэтапной оценки
    t_int: float = 0.0                    # время интегрирования, с
    t_match: float = 0.0                  # время сопоставления с эталонами, с
    screen_log: dict = field(default_factory=dict)

    def _screen_on(self):
        """Правило auto: первые SCREEN_AUTO_MIN оценок — двухэтапно; дальше двухэтапная оценка остаётся, пока на
        последних SCREEN_WINDOW пачках с ней сэкономленные шаги больше лишних. Когда она не окупается (известных
        режимов, на которые траектории быстро приходят, нет), каждая SCREEN_PROBE-я пачка всё же считается
        двухэтапно — проба: поиск может прийти в область, где остановка выгодна (например, к смешанным состояниям
        связанных блоков). Правило одинаково для всех стратегий, включая перебор; решения зависят только от шагов,
        не от времени, поэтому воспроизводимы."""
        if not self.screen:
            return False
        if self.screen != "auto":
            return True
        lg = self.screen_log
        lg["batch"] = lg.get("batch", 0) + 1
        hist = lg.setdefault("hist", [])
        if self.n_eval < SCREEN_AUTO_MIN or not hist:
            return True
        win = hist[-SCREEN_WINDOW:]
        if sum(h[0] for h in win) > sum(h[1] for h in win):
            return True
        if lg.get("off_at") is None:
            lg["off_at"] = self.n_eval
        lg["n_off"] = lg.get("n_off", 0) + 1
        return lg["batch"] % SCREEN_PROBE == 0

    def evaluate(self, ics):
        ics = np.ascontiguousarray(ics, dtype=np.float64)
        t0 = time.perf_counter()
        if self._screen_on():
            kw = dict(SCREEN_DEFAULT, **(self.screen if isinstance(self.screen, dict) else {}))
            tm = {}
            feats, finals, labels, steps, stop = screened_batch(self.system, ics, self.refs, self.known_idx,
                                                               tol=self.tol, timing=tm, **kw)
            self.t_int += tm["int"]
            self.t_match += tm["match"]
            self.n_stopped += int(stop.sum())
            for key in ("saved", "extra"):
                self.screen_log[key] = self.screen_log.get(key, 0.0) + tm[key]
            self.screen_log.setdefault("hist", []).append((tm["saved"], tm["extra"]))
        else:
            feats, finals = run_batch(self.system, ics)
            t1 = time.perf_counter()
            labels, z = match(self.system, feats, finals, self.refs, tol=self.tol)
            self.t_int += t1 - t0
            self.t_match += time.perf_counter() - t1
            steps = steps_of(feats, self.system.n_tr, self.system.n_obs)
        self.n_eval += len(ics)
        self.n_steps += float(steps.sum())
        return feats, finals, labels

    def known_feature_bank(self):
        """Матрица отпечатков известных осциллирующих эталонов (для новизны)."""
        si = stat_index(self.system.dim)
        rows = [self.refs[k].mu[si] for k in self.known_idx
                if self.refs[k].kind in ("periodic", "chaotic") and self.refs[k].mu is not None]
        return np.array(rows) if rows else np.zeros((0, len(si)))


class Strategy:
    name = "base"

    def __init__(self, problem, batch=20, **kw):
        self.p = problem
        self.s = problem.system
        self.rng = problem.rng
        self.batch = int(batch)
        self.d = self.s.dim
        self.lo = self.s.box_lo
        self.hi = self.s.box_hi
        self.L = float(np.max(self.hi - self.lo)) / 2.0

    def propose(self):
        raise NotImplementedError

    def observe(self, ics, feats, finals, labels):
        pass

    # --- общие утилиты
    def mutate(self, x, sigma_rel):
        y = x + sigma_rel * (self.hi - self.lo) * self.rng.normal(size=x.shape)
        return np.clip(y, self.lo, self.hi)

    def novelty(self, feats, labels):
        """Новизна отпечатков относительно банка известных режимов: нормированное расстояние до
        ближайшего известного эталона; для ушедших — 0 (их асимптотика не определена)."""
        bank = self.p.known_feature_bank()
        si = stat_index(self.d)
        F = feats[:, si]
        nov = np.zeros(len(feats))
        if len(bank) == 0:
            nov[feats[:, F_ESC] < 0.5] = 1.0
            return nov
        scale = np.abs(bank).mean(axis=0) + 1e-6
        D = np.linalg.norm((F[:, None, :] - bank[None, :, :]) / scale[None, None, :], axis=2)
        nov = D.min(axis=1)
        nov[feats[:, F_ESC] > 0.5] = 0.0
        return nov


def run_search(problem, strategy, budget, stop_on_target=True):
    """Запуск одной стратегии до исчерпания бюджета или первого попадания в цель."""
    found_at = None
    discoveries = []
    history = []
    while problem.n_eval < budget:
        ics = strategy.propose()
        ics = ics[: max(1, min(len(ics), budget - problem.n_eval))]
        feats, finals, labels = problem.evaluate(ics)
        strategy.observe(ics, feats, finals, labels)
        for j, lab in enumerate(labels):
            n_here = problem.n_eval - len(ics) + j + 1
            if lab in problem.target_idx and found_at is None:
                found_at = n_here
            if lab == LAB_UNKNOWN or (lab >= 0 and lab not in problem.known_idx and lab not in problem.target_idx):
                discoveries.append((n_here, int(lab)))
        history.append((problem.n_eval, int(np.sum(np.isin(labels, problem.target_idx)))))
        if found_at is not None and stop_on_target:
            break
    return {"found_at": found_at, "n_eval": problem.n_eval, "discoveries": discoveries[:20], "history": history}


def pn_curve(found_list, budget, grid=None):
    """Оценка P(N) = доля запусков с found_at <= N по сетке N."""
    grid = np.unique(np.round(np.logspace(0, np.log10(budget), 60)).astype(int)) if grid is None else grid
    fa = np.array([np.inf if f is None else f for f in found_list], float)
    return grid, np.array([(fa <= n).mean() for n in grid])


def n_delta_from_runs(found_list, delta=0.1, n_boot=2000, rng=None):
    """N_delta — квантиль уровня 1-delta распределения found_at (inf, если не достигнут);
    бутстрап-интервал по запускам."""
    rng = np.random.default_rng(0) if rng is None else rng
    fa = np.array([np.inf if f is None else f for f in found_list], float)
    q = float(np.quantile(fa, 1 - delta))
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(fa, size=len(fa), replace=True)
        boots.append(np.quantile(samp, 1 - delta))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return q, float(lo), float(hi)
