"""Батч-интегрирование потоков на видеокарте (CuPy, FP64) — то же, что integrate.make_flow_kernels.batch.

Каждая нить ведёт одно начальное условие тем же РК4 и считает те же признаки отпечатка в том же порядке
операций. Сборка идёт без слияния умножения и сложения (--fmad=false), поэтому траектории и признаки
совпадают с numba-ядром бит в бит. Исключение — log в F_LAM: у CUDA и у CPU он может отличаться в последнем
бите, на метки это не влияет.

Windows сбрасывает ядро видеокарты, если оно работает дольше ~2 с (TDR). Поэтому интегрирование идёт
короткими запусками ядра: состояние каждой траектории (точка, сопутствующая траектория, накопленные
статистики, номер шага) хранится в памяти видеокарты между запусками. После каждого запуска
закончившиеся траектории выбрасываются из списка активных, и нити не простаивают.

Поддержаны правые части chua (chua_I, chua_II), se11, le1, jafari2026 и связанные копии
(hsearch.systems.coupled) любой из них.
"""
import time

import numpy as np

from .integrate import nfeat

# правые части в том же порядке операций, что @njit-функции в systems/flows3d.py
_RHS_C = {
    "chua": """
__device__ __forceinline__ void base_rhs(const double* x, const double* p, double* out) {
    const double a = p[0], b = p[1], g = p[2], m0 = p[3], m1 = p[4];
    const double fx = m1 * x[0] + 0.5 * (m0 - m1) * (fabs(x[0] + 1.0) - fabs(x[0] - 1.0));
    out[0] = a * (x[1] - x[0] - fx);
    out[1] = x[0] - x[1] + x[2];
    out[2] = -(b * x[1] + g * x[2]);
}""",
    "se11": """
__device__ __forceinline__ void base_rhs(const double* x, const double* p, double* out) {
    out[0] = x[1];
    out[1] = -x[0] + x[1] * x[2];
    out[2] = -x[1] - 12.0 * x[2] + x[0] * x[0] + 9.0 * x[0] * x[2] - 1.0;
}""",
    "le1": """
__device__ __forceinline__ void base_rhs(const double* x, const double* p, double* out) {
    const double a = p[0], b = p[1];
    out[0] = x[1];
    out[1] = -x[0] + x[1] * x[2];
    out[2] = -x[0] - a * x[0] * x[1] - b * x[0] * x[2];
}""",
    "jafari2026": """
__device__ __forceinline__ void base_rhs(const double* x, const double* p, double* out) {
    const double a1 = p[0], a2 = p[1], a3 = p[2], a4 = p[3], a5 = p[4];
    out[0] = x[1];
    out[1] = x[2];
    out[2] = a1 * x[2] + a2 * x[0] * x[1] + a3 * x[1] * x[2] + a4 * x[2] * x[0] + a5;
}""",
}

_SINGLE = """
__device__ __forceinline__ void rhs(const double* x, const double* p, double* out) { base_rhs(x, p, out); }
"""

# связанные копии: как rhs в systems/coupled.py (сначала правые части блоков, затем связь среднего поля)
_COUPLED = """
#define KC @K@
#define D0 @D0@
#define NBP @NBP@
#define C0 @C0@
__device__ __forceinline__ void rhs(const double* x, const double* p, double* out) {
    for (int u = 0; u < KC; u++) base_rhs(x + u * D0, p, out + u * D0);
    const double e = p[NBP];
    if (e != 0.0) {
        double m = 0.0;
        for (int u = 0; u < KC; u++) m += x[u * D0 + C0];
        m /= (double)KC;
        for (int u = 0; u < KC; u++) out[u * D0 + C0] += e * (m - x[u * D0 + C0]);
    }
}
"""

_KERNEL = r"""
#define D @D@
#define NF (8 + 8 * D)
#define NS (14 * D + 7)
#define NI 6
@RHS@

__device__ __forceinline__ void rk4(double* x, const double* p, const double dt,
                                    double* k1, double* k2, double* k3, double* k4, double* tmp) {
    rhs(x, p, k1);
    for (int i = 0; i < D; i++) tmp[i] = x[i] + 0.5 * dt * k1[i];
    rhs(tmp, p, k2);
    for (int i = 0; i < D; i++) tmp[i] = x[i] + 0.5 * dt * k2[i];
    rhs(tmp, p, k3);
    for (int i = 0; i < D; i++) tmp[i] = x[i] + dt * k3[i];
    rhs(tmp, p, k4);
    for (int i = 0; i < D; i++) x[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

// состояние траектории (double): x y mean m2 mn mx es eq emn emx h1s h1q h2s h2q (по D), затем
// sum_log speed_sum rmax2 prev0 prev_d d0 scale; целые: phase (0 переходный, 1 окно, 2 готово), s, n_e,
// n_renorm, peaks, t_esc
extern "C" __global__ void advance(const int n_act, const int* __restrict__ act, const double* __restrict__ p,
        const double dt, const long long n_tr, const long long n_obs, const double esc_r,
        const long long renorm_every, const long long max_steps,
        double* __restrict__ st, long long* __restrict__ sti, double* __restrict__ feats,
        double* __restrict__ finals)
{
    const int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_act) return;
    const long long j = act[t];
    long long* I = sti + j * NI;
    long long phase = I[0];
    if (phase == 2) return;
    double* S = st + j * NS;
    double* F = feats + j * NF;
    double x[D], y[D], mean[D], m2[D], mn[D], mx[D], es[D], eq[D], emn[D], emx[D], h1s[D], h1q[D], h2s[D], h2q[D];
    double k1[D], k2[D], k3[D], k4[D], tmp[D];
    for (int i = 0; i < D; i++) {
        x[i] = S[i]; y[i] = S[D + i]; mean[i] = S[2 * D + i]; m2[i] = S[3 * D + i];
        mn[i] = S[4 * D + i]; mx[i] = S[5 * D + i]; es[i] = S[6 * D + i]; eq[i] = S[7 * D + i];
        emn[i] = S[8 * D + i]; emx[i] = S[9 * D + i]; h1s[i] = S[10 * D + i]; h1q[i] = S[11 * D + i];
        h2s[i] = S[12 * D + i]; h2q[i] = S[13 * D + i];
    }
    double sum_log = S[14 * D], speed_sum = S[14 * D + 1], rmax2 = S[14 * D + 2], prev0 = S[14 * D + 3];
    double prev_d = S[14 * D + 4], d0 = S[14 * D + 5], scale = S[14 * D + 6];
    long long s = I[1], n_e = I[2], n_renorm = I[3], peaks = I[4], t_esc = I[5];
    const double esc2 = esc_r * esc_r;
    const long long n_early = n_tr < n_obs ? n_tr : n_obs;
    long long budget = max_steps;

    if (phase == 0) {
        bool escaped = false;
        while (s < n_tr && budget > 0) {
            rk4(x, p, dt, k1, k2, k3, k4, tmp);
            double r2 = 0.0;
            for (int i = 0; i < D; i++) r2 += x[i] * x[i];
            if (r2 > esc2 || r2 != r2) { escaped = true; t_esc = s + 1; s += 1; break; }
            if (s < n_early) {
                n_e += 1;
                for (int i = 0; i < D; i++) {
                    es[i] += x[i];
                    eq[i] += x[i] * x[i];
                    if (x[i] < emn[i]) emn[i] = x[i];
                    if (x[i] > emx[i]) emx[i] = x[i];
                }
            }
            s += 1;
            budget -= 1;
        }
        if (escaped || s >= n_tr) {
            if (n_e > 0) {
                for (int i = 0; i < D; i++) {
                    const double a = es[i] / (double)n_e;
                    const double v = eq[i] / (double)n_e - a * a;
                    F[8 + 4 * D + i] = a;
                    F[8 + 5 * D + i] = sqrt((0.0 > v) ? 0.0 : v);
                    F[8 + 6 * D + i] = emn[i];
                    F[8 + 7 * D + i] = emx[i];
                }
            }
            if (escaped) {
                F[0] = 1.0;
                F[6] = (double)t_esc;
                for (int i = 0; i < D; i++) finals[j * D + i] = x[i];
                phase = 2;
            } else {
                scale = 0.0;
                for (int i = 0; i < D; i++) if (fabs(x[i]) > scale) scale = fabs(x[i]);
                d0 = 1e-7 * (1.0 + scale);
                for (int i = 0; i < D; i++) y[i] = x[i];
                y[0] += d0;
                sum_log = 0.0; n_renorm = 0; speed_sum = 0.0; rmax2 = 0.0; peaks = 0;
                prev0 = x[0]; prev_d = 0.0;
                for (int i = 0; i < D; i++) { h1s[i] = 0.0; h1q[i] = 0.0; h2s[i] = 0.0; h2q[i] = 0.0; }
                phase = 1;
                s = 0;
            }
        }
    }
    if (phase == 1 && budget > 0) {
        const long long half = n_obs / 2;
        bool escaped = false;
        while (s < n_obs && budget > 0) {
            rk4(x, p, dt, k1, k2, k3, k4, tmp);
            rk4(y, p, dt, k1, k2, k3, k4, tmp);
            double r2 = 0.0;
            for (int i = 0; i < D; i++) r2 += x[i] * x[i];
            if (r2 > esc2 || r2 != r2) { escaped = true; t_esc = n_tr + s + 1; s += 1; break; }
            if (r2 > rmax2) rmax2 = r2;
            for (int i = 0; i < D; i++) {
                const double delta = x[i] - mean[i];
                mean[i] += delta / (double)(s + 1);
                m2[i] += delta * (x[i] - mean[i]);
                if (x[i] < mn[i]) mn[i] = x[i];
                if (x[i] > mx[i]) mx[i] = x[i];
                if (s < half) { h1s[i] += x[i]; h1q[i] += x[i] * x[i]; }
                else { h2s[i] += x[i]; h2q[i] += x[i] * x[i]; }
            }
            rhs(x, p, k1);
            double sp = 0.0;
            for (int i = 0; i < D; i++) sp += k1[i] * k1[i];
            speed_sum += sqrt(sp);
            const double cur_d = x[0] - prev0;
            if (prev_d > 0.0 && cur_d <= 0.0) peaks += 1;
            prev_d = cur_d;
            prev0 = x[0];
            if ((s + 1) % renorm_every == 0) {
                double dist2 = 0.0;
                for (int i = 0; i < D; i++) dist2 += (y[i] - x[i]) * (y[i] - x[i]);
                const double dist = sqrt(dist2);
                if (dist > 0.0) {
                    sum_log += log(dist / d0);
                    n_renorm += 1;
                    const double f = d0 / dist;
                    for (int i = 0; i < D; i++) y[i] = x[i] + (y[i] - x[i]) * f;
                }
            }
            s += 1;
            budget -= 1;
        }
        if (escaped) {
            F[0] = 1.0;
            F[6] = (double)t_esc;
            for (int i = 0; i < D; i++) finals[j * D + i] = x[i];
            phase = 2;
        } else if (s >= n_obs) {
            const double t_obs = (double)n_obs * dt;
            if (n_renorm > 0) F[2] = sum_log / ((double)(n_renorm * renorm_every) * dt);
            F[3] = speed_sum / (double)n_obs;
            F[4] = sqrt(rmax2);
            F[5] = (double)peaks / t_obs;
            double v1 = 0.0, v2 = 0.0;
            const long long n1 = half, n2 = n_obs - half;
            for (int i = 0; i < D; i++) {
                const double a1 = h1s[i] / (double)n1;
                const double a2 = h2s[i] / (double)n2;
                const double t1 = h1q[i] / (double)n1 - a1 * a1;
                const double t2 = h2q[i] / (double)n2 - a2 * a2;
                v1 += (0.0 > t1) ? 0.0 : t1;
                v2 += (0.0 > t2) ? 0.0 : t2;
            }
            F[7] = v1 > 0.0 ? sqrt(v2 / v1) : 1.0;
            double std_max = 0.0;
            for (int i = 0; i < D; i++) {
                const double sd = sqrt(m2[i] / (double)n_obs);
                F[8 + i] = mean[i];
                F[8 + D + i] = sd;
                F[8 + 2 * D + i] = mn[i];
                F[8 + 3 * D + i] = mx[i];
                if (sd > std_max) std_max = sd;
            }
            if (std_max < 1e-6 * (1.0 + scale) && F[3] < 1e-5 * (1.0 + scale)) F[1] = 1.0;
            for (int i = 0; i < D; i++) finals[j * D + i] = x[i];
            phase = 2;
        }
    }
    for (int i = 0; i < D; i++) {
        S[i] = x[i]; S[D + i] = y[i]; S[2 * D + i] = mean[i]; S[3 * D + i] = m2[i];
        S[4 * D + i] = mn[i]; S[5 * D + i] = mx[i]; S[6 * D + i] = es[i]; S[7 * D + i] = eq[i];
        S[8 * D + i] = emn[i]; S[9 * D + i] = emx[i]; S[10 * D + i] = h1s[i]; S[11 * D + i] = h1q[i];
        S[12 * D + i] = h2s[i]; S[13 * D + i] = h2q[i];
    }
    S[14 * D] = sum_log; S[14 * D + 1] = speed_sum; S[14 * D + 2] = rmax2; S[14 * D + 3] = prev0;
    S[14 * D + 4] = prev_d; S[14 * D + 5] = d0; S[14 * D + 6] = scale;
    I[0] = phase; I[1] = s; I[2] = n_e; I[3] = n_renorm; I[4] = peaks; I[5] = t_esc;
}
"""


def _base_key(rhs):
    from .systems import flows3d
    table = {flows3d.chua_rhs: "chua", flows3d.se11_rhs: "se11", flows3d.le1_rhs: "le1",
             flows3d.jafari2026_rhs: "jafari2026"}
    return table.get(rhs)


def rhs_source(system):
    """CUDA-код правой части системы; ValueError, если её нет в поддержанных."""
    meta = getattr(system, "meta", None) or {}
    if meta.get("family") == "coupled":
        from .systems import get
        base = get(meta["base"])
        key = _base_key(base.rhs)
        if key is None:
            raise ValueError(f"нет CUDA-версии правой части базы {meta['base']}")
        return _RHS_C[key] + (_COUPLED.replace("@K@", str(int(meta["k"]))).replace("@D0@", str(base.dim))
                              .replace("@NBP@", str(len(base.params))).replace("@C0@", str(int(meta["coord"]))))
    key = _base_key(system.rhs)
    if key is None:
        raise ValueError(f"нет CUDA-версии правой части системы {system.name}")
    return _RHS_C[key] + _SINGLE


def supported(system):
    try:
        rhs_source(system)
        return system.kind == "flow"
    except ValueError:
        return False


class GpuFlow:
    """run_batch(ics) -> (feats, finals) на видеокарте, как hsearch.integrate.run_batch."""

    def __init__(self, system, block=128, launch_s=0.15):
        import cupy as cp
        self.cp = cp
        self.s = system
        self.d = system.dim
        self.block = int(block)
        self.launch_s = float(launch_s)      # целевая длительность одного запуска ядра (TDR Windows — 2 с)
        src = _KERNEL.replace("@D@", str(self.d)).replace("@RHS@", rhs_source(system))   # в коде есть %
        self.mod = cp.RawModule(code=src, options=("--fmad=false", "-std=c++17"))
        self.kern = self.mod.get_function("advance")
        self.p = cp.asarray(np.asarray(system.params, dtype=np.float64))
        self.steps = 50                      # шагов на запуск, подстраивается под launch_s
        self.gpu_time = 0.0

    def run_batch(self, ics, n_tr=None, n_obs=None, stop_check=None):
        """stop_check() -> True прерывает интегрирование между запусками ядра (мягкая остановка rmq):
        бросается исключение, которое stop_check вернул бы вызывающему — см. e51_gpu_uniform."""
        cp = self.cp
        s, d = self.s, self.d
        n_tr = s.n_tr if n_tr is None else int(n_tr)
        n_obs = s.n_obs if n_obs is None else int(n_obs)
        ics = np.ascontiguousarray(ics, dtype=np.float64)
        n = len(ics)
        ns, nf = 14 * d + 7, nfeat(d)
        st = cp.zeros((n, ns), dtype=cp.float64)
        st[:, 0:d] = cp.asarray(ics)
        st[:, 4 * d:5 * d] = 1e300
        st[:, 5 * d:6 * d] = -1e300
        st[:, 8 * d:9 * d] = 1e300
        st[:, 9 * d:10 * d] = -1e300
        sti = cp.zeros((n, 6), dtype=cp.int64)
        feats = cp.zeros((n, nf), dtype=cp.float64)
        finals = cp.zeros((n, d), dtype=cp.float64)
        act = cp.arange(n, dtype=cp.int32)
        args_tail = (self.p, np.float64(s.dt), np.int64(n_tr), np.int64(n_obs), np.float64(s.esc_r),
                     np.int64(s.renorm_every))
        t0 = time.perf_counter()
        while len(act):
            if stop_check is not None:
                stop_check()
            ta = time.perf_counter()
            grid = ((len(act) + self.block - 1) // self.block,)
            self.kern(grid, (self.block,), (np.int32(len(act)), act) + args_tail +
                      (np.int64(self.steps), st, sti, feats, finals))
            cp.cuda.Device().synchronize()
            dt_launch = time.perf_counter() - ta
            # шагов на запуск — под целевую длительность; рост не больше чем вдвое за раз: короткий запуск
            # мог быть коротким оттого, что траектории в нём закончились, а не оттого, что шагов мало
            f = min(max(self.launch_s / max(dt_launch, 1e-4), 0.1), 2.0)
            self.steps = int(min(max(self.steps * f, 10), 2_000_000))
            act = act[sti[act, 0] != 2]
        self.gpu_time += time.perf_counter() - t0
        return feats.get(), finals.get()
