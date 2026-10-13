"""Реестр эталонных систем."""
from dataclasses import dataclass, field

import numpy as np

from ..integrate import make_flow_kernels, make_map_kernels


@dataclass
class System:
    name: str
    kind: str                 # 'flow' | 'map'
    dim: int
    rhs: object               # @njit rhs(x, p, out) или fmap(x, p, out)
    params: np.ndarray
    box_lo: np.ndarray        # область начальных условий B
    box_hi: np.ndarray
    dt: float
    n_tr: int                 # шагов переходного процесса
    n_obs: int                # шагов окна наблюдения
    esc_r: float              # радиус ухода на бесконечность
    renorm_every: int = 20
    equilibria: list = field(default_factory=list)     # список np.ndarray
    known: dict = field(default_factory=dict)          # имя -> IC известного аттрактора
    hidden: list = field(default_factory=list)         # имена скрытых (целевых) аттракторов
    refs: list = field(default_factory=list)
    notes: str = ""
    level: str = "A"
    eq_project: object = None       # для континуума равновесий: x -> ближайшая точка множества

    def __post_init__(self):
        self.params = np.asarray(self.params, dtype=np.float64)
        self.box_lo = np.asarray(self.box_lo, dtype=np.float64)
        self.box_hi = np.asarray(self.box_hi, dtype=np.float64)
        if self.kind == "flow":
            self.batch, self.trajectory, self.batch_x = make_flow_kernels(self.rhs, self.dim, with_x=True)
        else:
            self.batch, self.trajectory = make_map_kernels(self.rhs, self.dim)
            self.batch_x = None

    @property
    def T_tr(self):
        return self.n_tr * self.dt

    @property
    def T_obs(self):
        return self.n_obs * self.dt

    def sample_box(self, n, rng):
        return self.box_lo + (self.box_hi - self.box_lo) * rng.random((n, self.dim))

    def sample_ball(self, n, rng, center=None, radius=1.0):
        """Равномерная выборка в шаре (для воспроизведения табл. 1 Guan & Xie)."""
        c = np.zeros(self.dim) if center is None else np.asarray(center, float)
        v = rng.normal(size=(n, self.dim))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        r = radius * rng.random(n) ** (1.0 / self.dim)
        return c + v * r[:, None]

    def traj(self, x0, n=None, dt=None):
        n = self.n_tr + self.n_obs if n is None else int(n)
        return self.trajectory(np.asarray(x0, float), self.params,
                               self.dt if dt is None else dt, n, self.esc_r)


_REGISTRY = {}


def register(sys_obj):
    _REGISTRY[sys_obj.name] = sys_obj
    return sys_obj


def get(name):
    if not _REGISTRY:
        _load_all()
    return _REGISTRY[name]


def names(level=None):
    if not _REGISTRY:
        _load_all()
    return [k for k, v in _REGISTRY.items() if level is None or v.level == level]


def _load_all():
    from . import flows3d  # noqa: F401  (регистрация при импорте)
    from . import maps2d   # noqa: F401
    from . import highdim  # noqa: F401
    from . import forced   # noqa: F401
