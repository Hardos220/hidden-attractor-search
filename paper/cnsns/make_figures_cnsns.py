"""Figures of the manuscript -> paper/cnsns/figures/*.pdf|png.

The machinery of paper/figures/make_figures_base.py is used at the elsarticle text width (390 pt = 5.40 in;
column figures 0.62 of it), with the grid beneath the data on every figure (axes.axisbelow), and these redraws:
  fig37_d2_mechanism  -- (a) Delta from 1e-3 (no points below), larger orange points above a lighter law line;
  fig21_pn_levelAB    -- two columns and four rows, the legend in the eighth slot;
  fig_product_gain    -- drawn by experiments/e68_product_theory.py --figure --cnsns (axis to 1e6, legend lower right).
Large trajectory batches and basin maps are cached in paper/figures/_build/cache; the first run integrates them
(about ten minutes on 8 threads):

    python paper/cnsns/make_figures_cnsns.py
    python paper/cnsns/make_figures_cnsns.py e40:fig_d2_mechanism e26:fig_pn e68   # selected figures
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("make_figures_base",
                                               ROOT / "paper" / "figures" / "make_figures_base.py")
msf = importlib.util.module_from_spec(_spec)
sys.modules["make_figures_base"] = msf
_spec.loader.exec_module(msf)

import matplotlib.pyplot as plt  # noqa: E402
import figures_base  # noqa: E402

mfe = msf.mfe
W2 = 390 / 72.27                  # \textwidth of elsarticle preprint 12pt, inches
W1 = 0.62 * W2
msf.W1, msf.W2 = W1, W2
mfe.W1, mfe.W2 = W1, W2
msf.OUT = mfe.OUT = HERE / "figures"
for _k, _v in list(mfe.FIGSIZE.items()):       # canvases of the base widths -> elsarticle widths, heights kept
    if isinstance(_v, tuple):
        mfe.FIGSIZE[_k] = ({5.125: W2, 3.18: W1}.get(_v[0], _v[0]), _v[1])
mfe.FIGSIZE["e40:fig_d2_mechanism"] = (W2, 2.45)
mfe.FIGSIZE["e26:fig_pn"] = (W2, 6.9)


def _post_d2_mechanism(fig):
    """fig37 (a): the uniform sample was 2-pt light grey points, hard to see; 4-pt orange points, half transparent,
    distinct from the law and from the blue chains of (b), drawn above the law line, which is lighter. No uniform
    point has Delta < 1e-3, so the axis starts there."""
    ax = fig.axes[0]
    pts, law = ax.lines[0], ax.lines[1]
    pts.set_markersize(4.0)
    pts.set_color("#d9661f")
    pts.set_alpha(0.5)
    pts.set_zorder(3)
    law.set_color("0.55")
    law.set_linewidth(1.0)
    law.set_zorder(2)
    ax.set_xlim(1e-3, None)
    h, _ = ax.get_legend_handles_labels()
    ax.legend(h, ["uniform initial conditions", "exact law"], loc="lower left", markerscale=1.5)


figures_base.POST["fig37_d2_mechanism"] = _post_d2_mechanism


def _post_chua_II(fig):
    """fig31: the basin map (c) sat at the right edge of the canvas, so its tick labels and its legend were clipped;
    the two 3D panels are narrowed and moved left, (c) with its legend moved left."""
    a3, b3, basin = fig.axes[0], fig.axes[1], fig.axes[2]
    a3.set_position([0.0, 0.25, 0.3, 0.75])
    b3.set_position([0.28, 0.25, 0.3, 0.75])       # its z label would fall on the y label of (c)
    basin.set_position([0.77, 0.34, 0.19, 0.6])
    fig.legends[0].set_bbox_to_anchor((0.32, 0.0), transform=fig.transFigure)
    fig.legends[1].set_bbox_to_anchor((0.82, 0.0), transform=fig.transFigure)


figures_base.POST["fig31_chua_II_coexisting"] = _post_chua_II


def fig_pn(ns):
    """fig21 at two columns and four rows: larger panels, the legend in the empty eighth slot. x labels only under
    the bottom panel of each column, y labels only in the left column."""
    load, title, color, label = ns["load"], ns["TITLE"], ns["COLOR"], ns["LABEL"]
    sysl = [s for s in ns["LEVEL_AB"] if load(s)]
    show = ["uniform", "de_zelinka", "sus", "portfolio_no_sus", "portfolio"]
    fig, axes = plt.subplots(4, 2, figsize=(W2, 6.9), squeeze=False)
    flat = axes.ravel()
    for i, (ax, name) in enumerate(zip(flat, sysl)):
        d = load(name)
        for sname in show:
            if sname in d:
                r = d[sname]
                ax.step(r["pn_grid"], r["pn"], where="post", color=color[sname], label=label[sname],
                        lw=2.4 if sname == "portfolio" else 1.6)
        ax.axhline(0.9, color=ns["MUTED"], lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.03)
        tg = {"chua_I": "both chaotic", "chua_II": "both chaotic"}.get(name)
        ax.set_title(f"{title.get(name, name)}: {tg}" if tg else title.get(name, name))
        if i >= len(sysl) - 2:
            ax.set_xlabel("N, integrations")
        if i % 2 == 0:
            ax.set_ylabel("P(N)")
    for ax in flat[len(sysl):]:
        ax.axis("off")
    h, lab = flat[0].get_legend_handles_labels()
    flat[-1].legend(h, lab, loc="center", handlelength=2.8, frameon=False)
    fig.tight_layout(h_pad=0.6, w_pad=1.5)
    msf.save(fig, ns["FIG"] / "fig21_pn_levelAB")


mfe.OVERRIDE["e26:fig_pn"] = fig_pn

_save = msf.save


def save(fig, path_without_ext):
    """Grid beneath the data on every axes (the default puts it above scatter points and filled areas)."""
    for ax in fig.axes:
        ax.set_axisbelow(True)
    _save(fig, path_without_ext)


msf.save = save
msf.SEPARATE = ("e53", "e54", "e61")      # e68 is drawn in its CNSNS version below


def main(argv):
    sel = [a for a in argv if a != "e68"]
    if sel or not argv:
        msf.main(sel)
    if not argv or "e68" in argv:
        subprocess.run([sys.executable, str(ROOT / "experiments" / "e68_product_theory.py"), "--figure", "--cnsns"],
                       check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
