"""SIAM (SIADS) versions of the article figures -> paper/siads/figures/*.pdf|png.

Reuses paper/chaos/make_figures_en.py (English strings, layout patches, strategy line styles, Arial with an 8 pt
minimum, Type 42 fonts) on the SIAM page: wide figures are drawn on a canvas of the text width W2 = 5.125 in and are
included with width=\\textwidth, column figures on W1 = 3.18 in and are included with width=0.62\\textwidth, so that
the labels print at 8 pt. The background is white. figures_siads.py holds the SIADS-only redraws, including the
figures of the separate scripts e53 (fig27, fig29), e54 (fig30_cost_gains) and e61 (fig31_new_attractor), which are
drawn from their saved results. fig_product_gain is drawn by experiments/e68_product_theory.py --figure.

Basin maps and large trajectory batches are cached in paper/siads/_build/cache, so only the first run integrates
(about ten minutes on 8 threads; run it through the queue):

    rmq run --cpu 8 --ram 4 --time 30m -- python paper/siads/make_figures_siads.py
    python paper/siads/make_figures_siads.py e40:fig_maps e54     # selected figures (keys below)

Keys: e10:fig_chua_transient; e26:fig_pn, fig_scaling_p, fig_scaling_n, fig_ablation, fig_signal; e40:fig_chua_I,
fig_chua_II, fig_gallery, fig_li_sprott, fig_maps, fig_jafari, fig_d2_mechanism, fig_sus_chains, fig_verification;
e53 (fig27, fig29), e54, e61, e68.
"""
import functools
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("make_figures_en", ROOT / "paper" / "chaos" / "make_figures_en.py")
mfe = importlib.util.module_from_spec(_spec)
sys.modules["make_figures_en"] = mfe
_spec.loader.exec_module(mfe)            # also puts src/, experiments/ and paper/chaos/ on sys.path
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from hsearch import viz  # noqa: E402
import figures_siads  # noqa: E402

W1, W2 = 3.18, 5.125          # 0.62\textwidth and \textwidth of siamart, inches
OUT = HERE / "figures"
BUILD = HERE / "_build"
CACHE = BUILD / "cache"

mfe.W1, mfe.W2 = W1, W2
mfe.OUT, mfe.OUTDIRS, mfe.BUILD, mfe.SIDE = OUT, {}, BUILD, BUILD / "side"
mfe.FIGSIZE = {
    "e10:fig_chua_transient": (W1, 4.5),
    "e26:fig_pn": (W2, 5.1), "e26:fig_scaling_p": (W2, 3.3), "e26:fig_scaling_n": (W1, 3.4),
    "e26:fig_ablation": (W2, 3.7), "e26:fig_signal": (W1, 2.9),
    "e40:fig_chua_I": (W2, 3.0), "e40:fig_chua_II": (W2, 2.85), "e40:fig_gallery": (W2, 7.0),
    "e40:fig_li_sprott": (W2, 2.55), "e40:fig_maps": (W2, 1.95), "e40:fig_jafari": (W1, 2.35),
    "e40:fig_d2_mechanism": (W2, 2.35), "e40:fig_sus_chains": (W1, 3.65),
    "e40:fig_verification": lambda w, h: (W2, 0.17 * (h - 1.8) / 0.28 + 1.25),
}
mfe.OVERRIDE.update(figures_siads.OVERRIDE)
mfe.PATCH = mfe.PATCH + figures_siads.PATCH
WANTED = {"e10": ["fig_chua_transient"],
          "e26": ["fig_pn", "fig_scaling_p", "fig_scaling_n", "fig_ablation", "fig_signal"],
          "e40": ["fig_chua_I", "fig_chua_II", "fig_gallery", "fig_li_sprott", "fig_maps", "fig_jafari",
                  "fig_d2_mechanism", "fig_sus_chains", "fig_verification"]}
SEPARATE = ("e53", "e54", "e61", "e68")
SIZES = []


def white_setup_en():
    """setup_en of the Chaos figures with a white background (SIAM print look)."""
    _setup_en()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


_setup_en = viz.setup_en
viz.setup_en = white_setup_en
_plain_plot = Axes.plot                  # before the strategy-style hook: the separate scripts keep their styles


def _extra(fig):
    """Axis labels of 3D axes, which the tight bounding box of matplotlib leaves out (the z label of a panel in the
    right column would be cropped)."""
    return [ax.zaxis.label for ax in fig.axes if hasattr(ax, "zaxis") and ax.zaxis.label.get_text()]


def _artists(fig):
    extra = _extra(fig)
    return fig.get_default_bbox_extra_artists() + extra if extra else None


def _tight(fig):
    return fig.get_tightbbox(fig.canvas.get_renderer(), bbox_extra_artists=_artists(fig))


def fit_width(fig, target):
    """Resize the canvas until the tight bounding box (legends outside the axes included) has the target width,
    so that LaTeX does not scale the 8 pt labels down. A figure more than 4% too wide is left as it is and reported:
    shrinking it would squeeze the axes, its layout has to be fixed instead."""
    pad = 2 * float(plt.rcParams["savefig.pad_inches"])
    for i in range(8):
        fig.canvas.draw()
        w = _tight(fig).width + pad
        if abs(w - target) <= 0.01:
            break
        if i == 0 and w > 1.04 * target:
            print(f"WARNING: tight width {w:.2f} in > {target} in; layout left unchanged", flush=True)
            break
        W, H = fig.get_size_inches()
        fig.set_size_inches(W - (w - target), H)
        if getattr(fig, "_en_tl", None) is not None:
            fig.tight_layout(**fig._en_tl)


def save(fig, path_without_ext):
    name = Path(str(path_without_ext)).name
    post = figures_siads.POST.get(name)
    if post is not None:
        post(fig)
    w0 = getattr(fig, "_en_w0", float(fig.get_size_inches()[0]))
    fit_width(fig, W1 if w0 <= W1 + 0.1 else W2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_extra_artists=_artists(fig))
    fig.savefig(OUT / f"{name}.png", bbox_extra_artists=_artists(fig))
    fig.canvas.draw()
    bb = _tight(fig)
    pad = 2 * float(plt.rcParams["savefig.pad_inches"])
    SIZES.append((name, round(bb.width + pad, 2), round(bb.height + pad, 2)))
    plt.close(fig)


def cached(fn, kind):
    """Cache of run_batch (batches of at least 5000 points) and basin_grid results, keyed by the system, its
    integration settings and the arguments, so that layout iterations do not integrate again."""
    @functools.wraps(fn)
    def wrapped(s, *args, **kw):
        h = hashlib.sha1(kind.encode())
        h.update(s.name.encode()); h.update(np.asarray(s.params, float).tobytes())
        h.update(repr((s.dt, s.n_tr, s.n_obs, s.esc_r, s.dim)).encode())
        if kind == "run_batch":
            ics = np.ascontiguousarray(args[0], dtype=np.float64)
            if len(ics) < 5000 or args[1:] or kw:
                return fn(s, *args, **kw)
            h.update(ics.tobytes())
        else:
            refs, rest = args[0], args[1:]
            h.update(repr([r.name for r in refs]).encode()); h.update(repr((rest, sorted(kw.items()))).encode())
        f = CACHE / f"{kind}_{s.name}_{h.hexdigest()[:16]}.npz"
        if f.exists():
            z = np.load(f)
            return (z["feats"], z["finals"]) if kind == "run_batch" else (z["a"], z["b"], z["img"])
        res = fn(s, *args, **kw)
        CACHE.mkdir(parents=True, exist_ok=True)
        if kind == "run_batch":
            np.savez_compressed(f, feats=res[0], finals=res[1])
        else:
            np.savez_compressed(f, a=res[0], b=res[1], img=res[2])
        return res
    return wrapped


def run(script, only):
    """make_figures_en.run with the cache of the trajectory batches and basin maps."""
    path = mfe.script_path(mfe.SCRIPTS[script])
    src = mfe.translate(path.read_text(encoding="utf-8"), script)
    ns = {"__name__": f"siads_{script}", "__file__": str(path), "_fs": mfe._fs}
    exec(compile(src, str(path), "exec"), ns)
    for nm in ("run_batch", "basin_grid"):
        if nm in ns:
            ns[nm] = cached(ns[nm], nm)

    def wrap(f, key, name):
        def wrapped(*a, **k):
            mfe.STATE["current"] = key
            try:
                return f(*a, **k)
            finally:
                mfe.STATE["current"] = None
        wrapped.__name__ = name
        return wrapped

    for name, f in list(ns.items()):
        if name.startswith("fig_") and callable(f):
            key = f"{script}:{name}"
            if key not in only:
                ns[name] = lambda *a, **k: None
                continue
            if key in mfe.OVERRIDE:
                f = functools.partial(mfe.OVERRIDE[key], ns)
            ns[name] = wrap(f, key, name)
    ns["main"]()
    for key, f in mfe.EXTRA.items():
        s, name = key.split(":")
        if s == script and key in only:
            wrap(functools.partial(f, ns), key, name)()


def separate(key):
    hooked = Axes.plot
    Axes.plot = _plain_plot
    try:
        viz.setup()                      # hooked: setup + white setup_en
        if key == "e53":
            figures_siads.fig_enlarged_and_smallp(W2, W1, save)
        elif key == "e54":
            figures_siads.fig_cost_gains(W1, save)
        elif key == "e61":
            figures_siads.fig_new_attractor(W2, save)
        elif key == "e68":
            subprocess.run([sys.executable, str(ROOT / "experiments" / "e68_product_theory.py"), "--figure"], check=True)
    finally:
        Axes.plot = hooked


def main(argv):
    sel = [a for a in argv if not a.startswith("--")]
    mfe.install_hooks()
    viz.save = save                      # after install_hooks: column and text-width fitting, output to OUT
    for script, names in WANTED.items():
        only = {f"{script}:{n}" for n in names}
        if sel:
            only = {k for k in only if k in sel}
        if only:
            print(f"== {script}: {sorted(only)}", flush=True)
            run(script, only)
    for key in SEPARATE:
        if not sel or key in sel:
            print(f"== {key}", flush=True)
            separate(key)
    for name, w, h in SIZES:
        print(f"{name:28s} {w:5.2f} x {h:5.2f} in")


if __name__ == "__main__":
    main(sys.argv[1:])
