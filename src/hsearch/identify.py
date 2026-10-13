"""Протокол идентификации аттракторов.

1. Калибровка эталона: с известного начального условия траектория выводится на
   аттрактор, затем n_rep возмущённых копий дают распределение отпечатков
   (mu, sigma) — «коробку допусков» эталона.
2. Сопоставление: траектория относится к эталону, если максимальный
   нормированный уклон признаков z = max_i |f_i - mu_i| / (sigma_i + floor_i)
   меньше tol; иначе она — кандидат в новые аттракторы.
3. Обнаружение: кандидаты кластеризуются (DBSCAN) и из медоидов калибруются
   новые эталоны; цикл повторяется, пока остаются крупные кластеры.
4. Верификация кандидата: время жизни при продолжении интегрирования,
   повторное интегрирование с половинным шагом, сходимость показателя Ляпунова
   по окнам.
"""
from dataclasses import dataclass, field

import numpy as np

from .integrate import (F_CONTR, F_ESC, F_FP, F_LAM, F_PEAKS, F_RMAX, F_SPEED, NHEAD,
                        feature_slices, nfeat, run_batch)

LAB_ESCAPE = -1
LAB_UNKNOWN = -2

LAM_DECAY = -2e-3        # явно отрицательный показатель (затухание), вместе с сжатием амплитуды
CONTR_DECAY = 0.9        # коэффициент сжатия амплитуды (вторая половина окна / первая) для затухания
LAM_CHAOS = 0.02         # медианный показатель эталона выше порога -> эталон хаотический
LAM_CHAOS_GATE = 0.005   # траектория может соответствовать хаотическому эталону, если её показатель выше
TOL_CONV = 1e-3          # допуск сошедшейся неподвижной точки (в долях масштаба системы L)
TOL_DECAY = 0.05         # допуск для центра затухающей спирали (в долях L)
TOL_DUP = 1e-2           # допуск совпадения равновесий (в долях L)


def scale(system):
    """Характерный масштаб системы: половина наибольшего ребра области B."""
    return float(np.max(system.box_hi - system.box_lo)) / 2.0


def stat_index(d):
    """Индексы признаков, участвующих в сопоставлении."""
    return np.array([F_LAM, F_SPEED, F_RMAX, F_PEAKS] + list(range(NHEAD, NHEAD + 4 * d)))


@dataclass
class Reference:
    name: str
    kind: str                       # 'fp' | 'periodic' | 'chaotic' | 'escape'
    mu: np.ndarray = None
    sigma: np.ndarray = None
    point: np.ndarray = None        # для 'fp' — координаты равновесия
    lam: float = 0.0
    n_cal: int = 0
    escape_frac: float = 0.0        # доля возмущённых копий, ушедших на бесконечность
    example: np.ndarray = None      # состояние на аттракторе (для повторного выхода)
    meta: dict = field(default_factory=dict)
    project: object = None          # для континуума равновесий: функция x -> ближайшая точка множества

    def fp_distance(self, probes):
        """Расстояния от точек probes (k, d) до равновесия (точки или множества)."""
        if self.project is not None:
            proj = np.array([self.project(q) for q in probes])
            return np.linalg.norm(probes - proj, axis=1)
        return np.linalg.norm(probes - self.point[None, :], axis=1)


def _kind_from_feats(system, feats_row):
    if feats_row[F_ESC] > 0.5:
        return "escape"
    if feats_row[F_FP] > 0.5:
        return "fp"
    return "chaotic" if feats_row[F_LAM] > LAM_CHAOS else "periodic"


def calibrate_reference(system, name, ic, rng, n_rep=64, noise=1e-3, from_state=False):
    """Строит эталон по начальному условию ic (или по состоянию на аттракторе при from_state=True)."""
    ic = np.asarray(ic, float)
    d = system.dim
    if from_state:
        state = ic.copy()
    else:
        feats0, finals0 = run_batch(system, ic[None, :])
        if feats0[0, F_ESC] > 0.5:
            return Reference(name=name, kind="escape", n_cal=1)
        state = finals0[0]
        # неподвижная точка: сошлась или медленно затухает к известному равновесию
        fp_point = _fixed_point_of(system, feats0[0], finals0[0])
        if fp_point is not None:
            return Reference(name=name, kind="fp", point=fp_point, mu=feats0[0].copy(),
                             sigma=np.zeros(nfeat(d)), lam=float(feats0[0, F_LAM]), n_cal=1,
                             example=finals0[0].copy())
    # копии берутся из случайных точек длинного отрезка траектории на аттракторе
    # (рандомизация фазы — иначе у длинных циклов разброс статистик окна занижен)
    traj = system.traj(state, n=4 * system.n_obs)
    traj = traj[~np.isnan(traj).any(axis=1)]
    picks = traj[rng.integers(0, len(traj), size=n_rep)] if len(traj) > 10 else np.repeat(state[None, :], n_rep, axis=0)
    ics = picks + noise * (1.0 + np.abs(picks)) * rng.normal(size=(n_rep, d))
    feats, finals = run_batch(system, ics)
    esc = feats[:, F_ESC] > 0.5
    keep = ~esc
    ref = Reference(name=name, kind="escape", n_cal=int(keep.sum()), escape_frac=float(esc.mean()),
                    example=state.copy())
    if keep.sum() == 0:
        return ref
    f = feats[keep]
    kinds = [_kind_from_feats(system, row) for row in f]
    vals, counts = np.unique(kinds, return_counts=True)
    ref.kind = str(vals[np.argmax(counts)])
    ref.mu = f.mean(axis=0)
    ref.sigma = f.std(axis=0)
    ref.lam = float(np.median(f[:, F_LAM]))
    if ref.kind == "fp":
        ref.point = finals[keep].mean(axis=0)
    return ref


def _is_decaying(system, feats_row):
    """Затухающий переходный процесс: явно отрицательный показатель И сжатие амплитуды."""
    return feats_row[F_LAM] < LAM_DECAY and feats_row[F_CONTR] < CONTR_DECAY


def _fixed_point_of(system, feats_row, final):
    """Равновесие (точка), к которому сошлась (или затухает) траектория, иначе None.

    Для затухающих спиралей сравнивается не конечная точка, а среднее по окну
    (центр спирали), с мягким допуском TOL_DECAY * L.
    """
    if feats_row[F_ESC] > 0.5:
        return None
    converged = feats_row[F_FP] > 0.5
    decaying = _is_decaying(system, feats_row)
    if not (converged or decaying):
        return None
    d = system.dim
    L = scale(system)
    probe = final if converged else feats_row[feature_slices(d)["mean"]]
    tol = (TOL_CONV if converged else TOL_DECAY) * L
    cands = []
    for e in system.equilibria:
        e = np.asarray(e, float)
        cands.append((np.linalg.norm(probe - e), e))
    if getattr(system, "eq_project", None) is not None:
        e = np.asarray(system.eq_project(probe), float)
        cands.append((np.linalg.norm(probe - e), e))
    if cands:
        dist, e = min(cands, key=lambda c: c[0])
        if dist < tol:
            return e.copy()
    if converged:
        return final.copy()   # сошлась к неизвестному ранее равновесию
    return None


def fp_references_from_equilibria(system, names=None):
    """Эталоны-равновесия напрямую из списка равновесий системы (без интегрирования)."""
    refs = []
    for i, e in enumerate(system.equilibria):
        nm = f"E{i}" if names is None else names[i]
        refs.append(Reference(name=nm, kind="fp", point=np.asarray(e, float), n_cal=0))
    if getattr(system, "eq_project", None) is not None:
        refs.append(Reference(name="Eset", kind="fp", project=system.eq_project, n_cal=0))
    return refs


def match(system, feats, finals, refs, tol=6.0, floor_rel=0.02):
    """Возвращает метки: индекс эталона, LAB_ESCAPE или LAB_UNKNOWN, и массив z-оценок."""
    n = feats.shape[0]
    d = system.dim
    L = scale(system)
    labels = np.full(n, LAB_UNKNOWN, dtype=np.int64)
    zbest = np.full(n, np.inf)
    esc = feats[:, F_ESC] > 0.5
    labels[esc] = LAB_ESCAPE
    converged = (feats[:, F_FP] > 0.5) & ~esc
    sl = feature_slices(d)
    decaying = ~esc & ~converged & (feats[:, F_LAM] < LAM_DECAY) & (feats[:, F_CONTR] < CONTR_DECAY)
    isfp = converged | decaying
    # неподвижные точки — по расстоянию до известных равновесий
    # (сошедшиеся — конечная точка с жёстким допуском, затухающие — центр спирали с мягким)
    fp_refs = [(k, r) for k, r in enumerate(refs) if r.kind == "fp" and (r.point is not None or r.project is not None)]
    if isfp.any() and fp_refs:
        idx = np.array([k for k, _ in fp_refs])
        rows = np.where(isfp)[0]
        probe = np.where(converged[rows][:, None], finals[rows], feats[rows][:, sl["mean"]])
        dist = np.column_stack([r.fp_distance(probe) for _, r in fp_refs])
        j = dist.argmin(axis=1)
        dmin = dist[np.arange(len(rows)), j]
        tol_row = np.where(converged[rows], TOL_CONV, TOL_DECAY) * L
        ok = dmin < tol_row
        lab = np.where(ok, idx[j], LAB_UNKNOWN)
        labels[rows] = lab
        zbest[rows] = dmin
        # затухающие, но далёкие от известных равновесий, остаются кандидатами в осцилляторы
        isfp = isfp.copy()
        isfp[rows[~ok & decaying[rows]]] = False
    # осциллирующие режимы — по коробке допусков
    osc = ~esc & ~isfp
    osc_refs = [(k, r) for k, r in enumerate(refs) if r.kind in ("periodic", "chaotic") and r.mu is not None]
    if osc.any() and osc_refs:
        si = stat_index(d)
        F = feats[osc][:, si]
        lam = feats[osc][:, F_LAM]
        Z = np.empty((F.shape[0], len(osc_refs)))
        for c, (k, r) in enumerate(osc_refs):
            mu = r.mu[si]
            sig = r.sigma[si] + floor_rel * (1.0 + np.abs(mu))
            Z[:, c] = np.max(np.abs(F - mu) / sig, axis=1)
            # класс по показателю Ляпунова обязан быть совместим: хаотический эталон требует
            # явно положительного показателя, периодический — не хаотического
            kind_ok = (lam > LAM_CHAOS_GATE) if r.kind == "chaotic" else (lam < LAM_CHAOS)
            Z[~kind_ok, c] = np.inf
        c = Z.argmin(axis=1)
        zmin = Z[np.arange(F.shape[0]), c]
        lab = np.where(zmin < tol, np.array([k for k, _ in osc_refs])[c], LAB_UNKNOWN)
        labels[np.where(osc)[0]] = lab
        zbest[np.where(osc)[0]] = zmin
    return labels, zbest


def discover(system, feats, finals, unknown_mask, rng, eps=0.6, min_samples=5, max_new=10, n_rep=48):
    """Кластеризует неизвестные траектории и калибрует новые эталоны по медоидам кластеров."""
    from sklearn.cluster import DBSCAN
    idx = np.where(unknown_mask)[0]
    if len(idx) == 0:
        return []
    d = system.dim
    si = stat_index(d)
    X = feats[idx][:, si]
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 0.02 * (1.0 + np.abs(mu))
    Xs = (X - mu) / sd
    lab = DBSCAN(eps=eps * np.sqrt(Xs.shape[1]) / 3.0, min_samples=min_samples).fit_predict(Xs)
    new_refs = []
    clusters = [c for c in np.unique(lab) if c >= 0]
    clusters.sort(key=lambda c: -(lab == c).sum())
    for c in clusters[:max_new]:
        members = np.where(lab == c)[0]
        centre = Xs[members].mean(axis=0)
        medoid = members[np.argmin(np.linalg.norm(Xs[members] - centre, axis=1))]
        row = idx[medoid]
        state = finals[row]
        ref = calibrate_reference(system, f"new{len(new_refs)}", state, rng, n_rep=n_rep, from_state=True)
        ref.meta = {"cluster_size": int(len(members)), "seed_row": int(row)}
        if ref.kind != "escape":
            new_refs.append(ref)
    return new_refs


def build_reference_set(system, rng, n_sample=20000, max_rounds=4, tol=6.0, verbose=True, sample="box"):
    """Полный цикл: эталоны из известных IC и равновесий, выборка, обнаружение неизвестных, повтор."""
    refs = fp_references_from_equilibria(system)
    for name, ic in system.known.items():
        r = calibrate_reference(system, name, ic, rng)
        if r.kind == "fp":
            # заменить ближайшее равновесие-эталон калиброванным (если это та же точка)
            L = scale(system)
            refs = [q for q in refs if not (q.kind == "fp" and q.point is not None and
                                            np.linalg.norm(q.point - r.point) < TOL_DUP * L)]
        refs.append(r)
    ics = system.sample_box(n_sample, rng) if sample == "box" else sample(n_sample, rng)
    feats, finals = run_batch(system, ics)
    history = []
    prev_unknown = None
    for rnd in range(max_rounds):
        labels, z = match(system, feats, finals, refs, tol=tol)
        unk = labels == LAB_UNKNOWN
        history.append({"round": rnd, "unknown": int(unk.sum()), "n_refs": len(refs)})
        if verbose:
            print(f"  раунд {rnd}: эталонов {len(refs)}, неизвестных {unk.sum()} из {len(labels)}")
        if unk.sum() < 3 or (prev_unknown is not None and unk.sum() >= prev_unknown):
            break
        prev_unknown = int(unk.sum())
        new = discover(system, feats, finals, unk, rng)
        added = 0
        for r in new:
            if _duplicate_of(system, r, refs, tol=tol) is not None:
                continue
            r.name = f"new{sum(1 for q in refs if q.name.startswith('new'))}"
            refs.append(r)
            added += 1
        if added == 0:
            break
    labels, z = match(system, feats, finals, refs, tol=tol)
    return refs, ics, feats, finals, labels, history


def _duplicate_of(system, ref, refs, tol=6.0):
    """Индекс существующего эталона, которому соответствует новый (по его среднему отпечатку), иначе None."""
    if ref.kind == "fp":
        L = scale(system)
        for k, q in enumerate(refs):
            if q.kind == "fp" and (q.point is not None or q.project is not None) and \
                    q.fp_distance(ref.point[None, :])[0] < TOL_DUP * L:
                return k
        return None
    if ref.mu is None:
        return None
    feats = ref.mu[None, :].copy()
    finals = (ref.example if ref.example is not None else np.zeros(system.dim))[None, :]
    labels, z = match(system, feats, finals, refs, tol=tol)
    return int(labels[0]) if labels[0] >= 0 else None


def label_fractions(labels, refs):
    """Доли меток: словарь имя -> доля."""
    n = len(labels)
    out = {"escape": float((labels == LAB_ESCAPE).mean()), "unknown": float((labels == LAB_UNKNOWN).mean())}
    for k, r in enumerate(refs):
        out[r.name] = float((labels == k).mean())
    return out


# ----------------------------------------------------------------------------
# Верификация кандидата
# ----------------------------------------------------------------------------

def features_from_trajectory(traj, dt):
    """Статистики, совпадающие по смыслу с ядром (без ляпуновского показателя)."""
    d = traj.shape[1]
    x = traj
    v = np.diff(x, axis=0) / dt
    f = np.zeros(nfeat(d))
    f[F_SPEED] = np.linalg.norm(v, axis=1).mean()
    f[F_RMAX] = np.linalg.norm(x, axis=1).max()
    dx0 = np.diff(x[:, 0])
    f[F_PEAKS] = np.sum((dx0[:-1] > 0) & (dx0[1:] <= 0)) / (dt * (len(x) - 1))
    sl = feature_slices(d)
    f[sl["mean"]] = x.mean(axis=0)
    f[sl["std"]] = x.std(axis=0)
    f[sl["min"]] = x.min(axis=0)
    f[sl["max"]] = x.max(axis=0)
    return f


def verify_candidate(system, state, ref, rng, life_factor=20, n_windows=4, tol=6.0):
    """Проверяет, что состояние state действительно лежит на аттракторе ref.

    Возвращает словарь: survived (bool), lifetime_T (время, пока отпечаток оставался в допуске),
    halfstep_ok (повторное интегрирование с dt/2 даёт тот же эталон), lam_windows (список).
    """
    d = system.dim
    rep = {"name": ref.name}
    # 1) время жизни: окна наблюдения подряд
    x = np.asarray(state, float).copy()
    n_win = life_factor
    lam_w = []
    survived = True
    t_alive = 0.0
    for w in range(n_win):
        feats, finals = run_batch(system, x[None, :], n_tr=0, n_obs=system.n_obs)
        labels, z = match(system, feats, finals, [ref], tol=tol)
        if labels[0] != 0:
            survived = False
            break
        lam_w.append(float(feats[0, F_LAM]))
        t_alive += system.T_obs
        x = finals[0]
    rep["survived"] = survived
    rep["lifetime_T"] = t_alive
    rep["lam_windows"] = lam_w[:n_windows] + lam_w[-n_windows:] if len(lam_w) > 2 * n_windows else lam_w
    # 2) половинный шаг
    if system.kind == "flow":
        saved = system.dt
        feats_h = np.zeros((1, nfeat(d))); fin_h = np.zeros((1, d))
        system.batch(np.asarray(state, float)[None, :], system.params, saved / 2.0, 0, 2 * system.n_obs,
                     system.esc_r, system.renorm_every, feats_h, fin_h)
        labels_h, z_h = match(system, feats_h, fin_h, [ref], tol=tol)
        rep["halfstep_ok"] = bool(labels_h[0] == 0)
        rep["halfstep_z"] = float(z_h[0])
    else:
        rep["halfstep_ok"] = True
        rep["halfstep_z"] = 0.0
    return rep
