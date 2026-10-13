"""Батч-интегрирование и отпечатки траекторий (numba, параллельно по начальным условиям).

Отпечаток траектории (вектор длины nfeat(d) = 7 + 4 d):
  [0] F_ESC   — 1.0, если траектория ушла за радиус esc_r (или NaN);
  [1] F_FP    — 1.0, если траектория сошлась к неподвижной точке;
  [2] F_LAM   — оценка старшего ляпуновского показателя (метод двух траекторий);
  [3] F_SPEED — средняя скорость |F(x)| (для отображений — средний шаг |x_{k+1}-x_k|);
  [4] F_RMAX  — максимум |x| на окне наблюдения;
  [5] F_PEAKS — число локальных максимумов x_0 на единицу времени (частотный признак);
  [6] F_TESC  — для ушедших траекторий: номер шага ухода, иначе 0;
  [7] F_CONTR — коэффициент сжатия: СКО второй половины окна / СКО первой половины
                (суммарно по координатам); ~1 для стационарных режимов, < 1 для затухающих;
  [8 : 8+d]      — среднее по координатам на окне наблюдения;
  [8+d : 8+2d]   — стандартное отклонение;
  [8+2d : 8+3d]  — минимум;
  [8+3d : 8+4d]  — максимум;
  [8+4d : 8+8d]  — те же четыре статистики по «раннему окну»: первые n_early = min(n_tr, n_obs)
                   шагов переходного участка (описывают, куда траектория ушла в начале;
                   заполняются и для ушедших траекторий, если уход случился позже n_early).
"""
import math

import numpy as np
from numba import njit, prange

F_ESC, F_FP, F_LAM, F_SPEED, F_RMAX, F_PEAKS, F_TESC, F_CONTR = 0, 1, 2, 3, 4, 5, 6, 7
NHEAD = 8


def nfeat(d):
    return NHEAD + 8 * d


def feature_slices(d):
    return {
        "mean": slice(NHEAD, NHEAD + d),
        "std": slice(NHEAD + d, NHEAD + 2 * d),
        "min": slice(NHEAD + 2 * d, NHEAD + 3 * d),
        "max": slice(NHEAD + 3 * d, NHEAD + 4 * d),
        "emean": slice(NHEAD + 4 * d, NHEAD + 5 * d),
        "estd": slice(NHEAD + 5 * d, NHEAD + 6 * d),
        "emin": slice(NHEAD + 6 * d, NHEAD + 7 * d),
        "emax": slice(NHEAD + 7 * d, NHEAD + 8 * d),
    }


def feature_names(d):
    names = ["esc", "fp", "lam", "speed", "rmax", "peaks", "tesc", "contr"]
    for key in ("mean", "std", "min", "max", "emean", "estd", "emin", "emax"):
        names += [f"{key}{i}" for i in range(d)]
    return names


# ----------------------------------------------------------------------------
# Потоки
# ----------------------------------------------------------------------------

def make_flow_kernels(rhs, d, with_x=False):
    """Создаёт numba-ядра для потока dx/dt = rhs(x, p).

    rhs — функция @njit вида rhs(x, p, out), записывающая производную в out.
    Возвращает (batch, trajectory) или, при with_x=True, (batch, trajectory, batch_x):
      batch(ics, p, dt, n_tr, n_obs, esc_r, renorm_every, feats, finals)
      trajectory(x0, p, dt, n, esc_r) -> массив (n+1, d)
      batch_x(ics, p, dt, n_tr, n_obs, n_early, t_off, esc_r, renorm_every, feats, finals) — то же, что batch,
        но длина раннего окна n_early задаётся явно (batch берёт min(n_tr, n_obs)), а к номеру шага ухода
        прибавляется t_off. Нужно двухэтапной оценке (run_batch_screened): второй этап продолжает траекторию
        с шага t_off, и номер шага ухода остаётся номером шага исходной траектории.
    """
    nf = nfeat(d)

    @njit(cache=False)
    def _step(x, p, dt, k1, k2, k3, k4, tmp):
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
    def trajectory(x0, p, dt, n, esc_r):
        out = np.empty((n + 1, d))
        x = np.empty(d)
        k1 = np.empty(d); k2 = np.empty(d); k3 = np.empty(d); k4 = np.empty(d); tmp = np.empty(d)
        for i in range(d):
            x[i] = x0[i]
            out[0, i] = x[i]
        esc2 = esc_r * esc_r
        for s in range(n):
            _step(x, p, dt, k1, k2, k3, k4, tmp)
            r2 = 0.0
            for i in range(d):
                out[s + 1, i] = x[i]
                r2 += x[i] * x[i]
            if r2 > esc2 or r2 != r2:
                for t in range(s + 2, n + 1):
                    for i in range(d):
                        out[t, i] = np.nan
                break
        return out

    @njit(parallel=True, cache=False)
    def batch_x(ics, p, dt, n_tr, n_obs, n_early, t_off, esc_r, renorm_every, feats, finals):
        n = ics.shape[0]
        esc2 = esc_r * esc_r
        for j in prange(n):
            x = np.empty(d); y = np.empty(d)
            k1 = np.empty(d); k2 = np.empty(d); k3 = np.empty(d); k4 = np.empty(d); tmp = np.empty(d)
            mean = np.zeros(d); m2 = np.zeros(d); mn = np.empty(d); mx = np.empty(d)
            for i in range(d):
                x[i] = ics[j, i]
                mn[i] = 1e300
                mx[i] = -1e300
            for q in range(nf):
                feats[j, q] = 0.0
            escaped = False
            t_esc = 0
            # раннее окно: статистики первых n_early шагов переходного участка
            es = np.zeros(d); eq = np.zeros(d); emn = np.empty(d); emx = np.empty(d)
            for i in range(d):
                emn[i] = 1e300
                emx[i] = -1e300
            n_e = 0
            for s in range(n_tr):
                _step(x, p, dt, k1, k2, k3, k4, tmp)
                r2 = 0.0
                for i in range(d):
                    r2 += x[i] * x[i]
                if r2 > esc2 or r2 != r2:
                    escaped = True
                    t_esc = t_off + s + 1
                    break
                if s < n_early:
                    n_e += 1
                    for i in range(d):
                        es[i] += x[i]
                        eq[i] += x[i] * x[i]
                        if x[i] < emn[i]:
                            emn[i] = x[i]
                        if x[i] > emx[i]:
                            emx[i] = x[i]
            if n_e > 0:
                for i in range(d):
                    a = es[i] / n_e
                    feats[j, NHEAD + 4 * d + i] = a
                    feats[j, NHEAD + 5 * d + i] = math.sqrt(max(eq[i] / n_e - a * a, 0.0))
                    feats[j, NHEAD + 6 * d + i] = emn[i]
                    feats[j, NHEAD + 7 * d + i] = emx[i]
            if escaped:
                feats[j, F_ESC] = 1.0
                feats[j, F_TESC] = t_esc
                for i in range(d):
                    finals[j, i] = x[i]
                continue
            # сопутствующая траектория для оценки ляпуновского показателя
            scale = 0.0
            for i in range(d):
                if abs(x[i]) > scale:
                    scale = abs(x[i])
            d0 = 1e-7 * (1.0 + scale)
            for i in range(d):
                y[i] = x[i]
            y[0] += d0
            sum_log = 0.0
            n_renorm = 0
            speed_sum = 0.0
            rmax2 = 0.0
            peaks = 0
            prev0 = x[0]
            prev_d = 0.0
            h1s = np.zeros(d); h1q = np.zeros(d); h2s = np.zeros(d); h2q = np.zeros(d)
            half = n_obs // 2
            for s in range(n_obs):
                _step(x, p, dt, k1, k2, k3, k4, tmp)
                _step(y, p, dt, k1, k2, k3, k4, tmp)
                r2 = 0.0
                for i in range(d):
                    r2 += x[i] * x[i]
                if r2 > esc2 or r2 != r2:
                    escaped = True
                    t_esc = t_off + n_tr + s + 1
                    break
                if r2 > rmax2:
                    rmax2 = r2
                for i in range(d):
                    delta = x[i] - mean[i]
                    mean[i] += delta / (s + 1)
                    m2[i] += delta * (x[i] - mean[i])
                    if x[i] < mn[i]:
                        mn[i] = x[i]
                    if x[i] > mx[i]:
                        mx[i] = x[i]
                    if s < half:
                        h1s[i] += x[i]
                        h1q[i] += x[i] * x[i]
                    else:
                        h2s[i] += x[i]
                        h2q[i] += x[i] * x[i]
                rhs(x, p, k1)
                sp = 0.0
                for i in range(d):
                    sp += k1[i] * k1[i]
                speed_sum += math.sqrt(sp)
                cur_d = x[0] - prev0
                if prev_d > 0.0 and cur_d <= 0.0:
                    peaks += 1
                prev_d = cur_d
                prev0 = x[0]
                if (s + 1) % renorm_every == 0:
                    dist2 = 0.0
                    for i in range(d):
                        dist2 += (y[i] - x[i]) * (y[i] - x[i])
                    dist = math.sqrt(dist2)
                    if dist > 0.0:
                        sum_log += math.log(dist / d0)
                        n_renorm += 1
                        f = d0 / dist
                        for i in range(d):
                            y[i] = x[i] + (y[i] - x[i]) * f
            if escaped:
                feats[j, F_ESC] = 1.0
                feats[j, F_TESC] = t_esc
                for i in range(d):
                    finals[j, i] = x[i]
                continue
            t_obs = n_obs * dt
            if n_renorm > 0:
                feats[j, F_LAM] = sum_log / (n_renorm * renorm_every * dt)
            feats[j, F_SPEED] = speed_sum / n_obs
            feats[j, F_RMAX] = math.sqrt(rmax2)
            feats[j, F_PEAKS] = peaks / t_obs
            v1 = 0.0
            v2 = 0.0
            n1 = half
            n2 = n_obs - half
            for i in range(d):
                a1 = h1s[i] / n1
                a2 = h2s[i] / n2
                v1 += max(h1q[i] / n1 - a1 * a1, 0.0)
                v2 += max(h2q[i] / n2 - a2 * a2, 0.0)
            feats[j, F_CONTR] = math.sqrt(v2 / v1) if v1 > 0.0 else 1.0
            std_max = 0.0
            for i in range(d):
                sd = math.sqrt(m2[i] / n_obs)
                feats[j, NHEAD + i] = mean[i]
                feats[j, NHEAD + d + i] = sd
                feats[j, NHEAD + 2 * d + i] = mn[i]
                feats[j, NHEAD + 3 * d + i] = mx[i]
                if sd > std_max:
                    std_max = sd
            if std_max < 1e-6 * (1.0 + scale) and feats[j, F_SPEED] < 1e-5 * (1.0 + scale):
                feats[j, F_FP] = 1.0
            for i in range(d):
                finals[j, i] = x[i]

    @njit(cache=False)
    def batch(ics, p, dt, n_tr, n_obs, esc_r, renorm_every, feats, finals):
        n_early = n_tr if n_tr < n_obs else n_obs
        batch_x(ics, p, dt, n_tr, n_obs, n_early, 0, esc_r, renorm_every, feats, finals)

    if with_x:
        return batch, trajectory, batch_x
    return batch, trajectory


# ----------------------------------------------------------------------------
# Отображения
# ----------------------------------------------------------------------------

def make_map_kernels(fmap, d):
    """Создаёт numba-ядра для отображения x_{k+1} = fmap(x_k, p).

    fmap — функция @njit вида fmap(x, p, out).
    Сигнатуры совпадают с потоковыми; dt игнорируется (принимается 1.0),
    renorm_every — период перенормировки сопутствующей траектории.
    """
    nf = nfeat(d)

    @njit(cache=False)
    def trajectory(x0, p, dt, n, esc_r):
        out = np.empty((n + 1, d))
        x = np.empty(d); tmp = np.empty(d)
        for i in range(d):
            x[i] = x0[i]
            out[0, i] = x[i]
        esc2 = esc_r * esc_r
        for s in range(n):
            fmap(x, p, tmp)
            r2 = 0.0
            for i in range(d):
                x[i] = tmp[i]
                out[s + 1, i] = x[i]
                r2 += x[i] * x[i]
            if r2 > esc2 or r2 != r2:
                for t in range(s + 2, n + 1):
                    for i in range(d):
                        out[t, i] = np.nan
                break
        return out

    @njit(parallel=True, cache=False)
    def batch(ics, p, dt, n_tr, n_obs, esc_r, renorm_every, feats, finals):
        n = ics.shape[0]
        esc2 = esc_r * esc_r
        for j in prange(n):
            x = np.empty(d); y = np.empty(d); tmp = np.empty(d)
            mean = np.zeros(d); m2 = np.zeros(d); mn = np.empty(d); mx = np.empty(d)
            for i in range(d):
                x[i] = ics[j, i]
                mn[i] = 1e300
                mx[i] = -1e300
            for q in range(nf):
                feats[j, q] = 0.0
            escaped = False
            t_esc = 0
            n_early = n_tr if n_tr < n_obs else n_obs
            es = np.zeros(d); eq = np.zeros(d); emn = np.empty(d); emx = np.empty(d)
            for i in range(d):
                emn[i] = 1e300
                emx[i] = -1e300
            n_e = 0
            for s in range(n_tr):
                fmap(x, p, tmp)
                r2 = 0.0
                for i in range(d):
                    x[i] = tmp[i]
                    r2 += x[i] * x[i]
                if r2 > esc2 or r2 != r2:
                    escaped = True
                    t_esc = s + 1
                    break
                if s < n_early:
                    n_e += 1
                    for i in range(d):
                        es[i] += x[i]
                        eq[i] += x[i] * x[i]
                        if x[i] < emn[i]:
                            emn[i] = x[i]
                        if x[i] > emx[i]:
                            emx[i] = x[i]
            if n_e > 0:
                for i in range(d):
                    a = es[i] / n_e
                    feats[j, NHEAD + 4 * d + i] = a
                    feats[j, NHEAD + 5 * d + i] = math.sqrt(max(eq[i] / n_e - a * a, 0.0))
                    feats[j, NHEAD + 6 * d + i] = emn[i]
                    feats[j, NHEAD + 7 * d + i] = emx[i]
            if escaped:
                feats[j, F_ESC] = 1.0
                feats[j, F_TESC] = t_esc
                for i in range(d):
                    finals[j, i] = x[i]
                continue
            scale = 0.0
            for i in range(d):
                if abs(x[i]) > scale:
                    scale = abs(x[i])
            d0 = 1e-7 * (1.0 + scale)
            for i in range(d):
                y[i] = x[i]
            y[0] += d0
            sum_log = 0.0
            n_renorm = 0
            step_sum = 0.0
            rmax2 = 0.0
            h1s = np.zeros(d); h1q = np.zeros(d); h2s = np.zeros(d); h2q = np.zeros(d)
            half = n_obs // 2
            for s in range(n_obs):
                fmap(x, p, tmp)
                st = 0.0
                r2 = 0.0
                for i in range(d):
                    st += (tmp[i] - x[i]) * (tmp[i] - x[i])
                    x[i] = tmp[i]
                    r2 += x[i] * x[i]
                step_sum += math.sqrt(st)
                fmap(y, p, tmp)
                for i in range(d):
                    y[i] = tmp[i]
                if r2 > esc2 or r2 != r2:
                    escaped = True
                    t_esc = n_tr + s + 1
                    break
                if r2 > rmax2:
                    rmax2 = r2
                for i in range(d):
                    delta = x[i] - mean[i]
                    mean[i] += delta / (s + 1)
                    m2[i] += delta * (x[i] - mean[i])
                    if x[i] < mn[i]:
                        mn[i] = x[i]
                    if x[i] > mx[i]:
                        mx[i] = x[i]
                    if s < half:
                        h1s[i] += x[i]
                        h1q[i] += x[i] * x[i]
                    else:
                        h2s[i] += x[i]
                        h2q[i] += x[i] * x[i]
                if (s + 1) % renorm_every == 0:
                    dist2 = 0.0
                    for i in range(d):
                        dist2 += (y[i] - x[i]) * (y[i] - x[i])
                    dist = math.sqrt(dist2)
                    if dist > 0.0:
                        sum_log += math.log(dist / d0)
                        n_renorm += 1
                        f = d0 / dist
                        for i in range(d):
                            y[i] = x[i] + (y[i] - x[i]) * f
            if escaped:
                feats[j, F_ESC] = 1.0
                feats[j, F_TESC] = t_esc
                for i in range(d):
                    finals[j, i] = x[i]
                continue
            if n_renorm > 0:
                # dt — физическая длительность одной итерации (1 для обычных отображений,
                # шаг выборки h для резервуара), чтобы λ был в единицах на единицу времени
                feats[j, F_LAM] = sum_log / (n_renorm * renorm_every * dt)
            feats[j, F_SPEED] = step_sum / n_obs
            feats[j, F_RMAX] = math.sqrt(rmax2)
            feats[j, F_PEAKS] = 0.0
            v1 = 0.0
            v2 = 0.0
            n1 = half
            n2 = n_obs - half
            for i in range(d):
                a1 = h1s[i] / n1
                a2 = h2s[i] / n2
                v1 += max(h1q[i] / n1 - a1 * a1, 0.0)
                v2 += max(h2q[i] / n2 - a2 * a2, 0.0)
            feats[j, F_CONTR] = math.sqrt(v2 / v1) if v1 > 0.0 else 1.0
            std_max = 0.0
            for i in range(d):
                sd = math.sqrt(m2[i] / n_obs)
                feats[j, NHEAD + i] = mean[i]
                feats[j, NHEAD + d + i] = sd
                feats[j, NHEAD + 2 * d + i] = mn[i]
                feats[j, NHEAD + 3 * d + i] = mx[i]
                if sd > std_max:
                    std_max = sd
            if std_max < 1e-9 * (1.0 + scale) and feats[j, F_SPEED] < 1e-9 * (1.0 + scale):
                feats[j, F_FP] = 1.0
            for i in range(d):
                finals[j, i] = x[i]

    return batch, trajectory


def run_batch(system, ics, n_tr=None, n_obs=None):
    """Удобная обёртка: интегрирует набор начальных условий системы, возвращает (feats, finals)."""
    ics = np.ascontiguousarray(ics, dtype=np.float64)
    n = ics.shape[0]
    d = system.dim
    feats = np.zeros((n, nfeat(d)))
    finals = np.zeros((n, d))
    system.batch(ics, system.params, system.dt,
                 system.n_tr if n_tr is None else int(n_tr),
                 system.n_obs if n_obs is None else int(n_obs),
                 system.esc_r, system.renorm_every, feats, finals)
    return feats, finals
