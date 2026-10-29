# hidden-attractor-search

Code, experiments, results, and manuscript sources for the article

> M. Elisov, *Finding rare hidden attractors: Information bounds and an adaptive search portfolio with a worst-case guarantee*, manuscript for *SIAM Journal on Applied Dynamical Systems* (2026).

Archived in Zenodo: [doi:10.5281/zenodo.23348470](https://doi.org/10.5281/zenodo.23348470) (all versions; v1.1.0 is the current one). Versions 1.0.x accompany an earlier version of the manuscript, *Searching for hidden attractors as a rare-event problem: An adaptive strategy portfolio with a worst-case guarantee* (`paper/chaos/`).

The search for a hidden attractor with a small basin is treated as the detection of a rare event in the space of initial conditions. The article proves an information bound (no algorithm can guarantee a gain over uniform sampling larger than about 2.6/μ(Q), where Q is the set around the target on which the observations are constant), shows that for k uncoupled units ideal multilevel splitting along the escape time attains this bound up to a constant, and proposes a portfolio of search strategies with a worst-case guarantee. The repository contains the benchmark of systems with hidden attractors, the identification and verification protocol, the search strategies (uniform sampling, subset simulation driven by transient duration, active learning of basin boundaries, their portfolio, a portfolio arm of stagger-and-step chains, bisection toward basin boundaries, a stagger-and-step chain, and Zelinka's differential evolution), every numerical result reported in the article, and the scripts that produce its tables and figures.

## Layout

| Path | Contents |
| --- | --- |
| `src/hsearch/` | Library. `integrate.py`: fixed-step RK4 kernels (numba) that compute trajectory fingerprints inside the integrator; `systems/`: registry of benchmark systems (3D flows, 2D maps, 4D systems, synthetic families, coupled copies); `identify.py`: templates, matching, and the verification checks; `search/`: strategies (`strategies.py`, including `PortfolioSAS` and `StaggerMulti`) and the measurement harness (`harness.py`, including the two-stage evaluation); `gpu.py`: uniform sampling on the GPU (optional, CuPy); `viz.py`: plotting helpers. |
| `src/tools/` | Bibliography verification against Crossref and Semantic Scholar (`verify_bib.py`), the Zenodo deposit script, and small summaries. |
| `experiments/` | Numbered scripts; each writes to `results/`. The map to the article is below. |
| `results/` | JSON outputs of every experiment; `e50/<batch>/<job>.jsonl` holds one line per search run of the long batches; `e53_snippets/` holds table rows of the earlier manuscript; `e59/` holds the comparison with Attractors.jl. |
| `julia/` | `recurrence_cost.jl`: cost per evaluation of `AttractorsViaRecurrences` (Attractors.jl 2.0) on the same initial conditions as the Python code; `chua_attractors.jl`: independent check of the Chua basin fractions; `setup.jl`: packages. |
| `data/` | `bibliography.json`: the verified bibliographic records (DOI or arXiv id, title similarity) of the references. |
| `paper/siads/` | The manuscript (SIAM `siamart` class): `main.tex` + `sections/*.tex`, the supplementary materials `supplement.tex` + `sm/*.tex`, the shared preamble `shared.tex`, `refs.bib`, the figures (`figures/`), `build.py` (compiles both documents and checks for undefined references and overfull lines), `make_figures_siads.py` with `figures_siads.py` (all figures at the SIAM text width), and the compiled `main.pdf` and `supplement.pdf`. |
| `paper/chaos/` | The earlier version of the manuscript (REVTeX 4.2): sources, figures, `build_chaos.py`, `make_figures_en.py` (its figure machinery is reused by `paper/siads/make_figures_siads.py`), and the compiled `manuscript.pdf`. |

## Setup

```
pip install -r requirements.txt
```

Python 3.11 or later; the article's results were produced with Python 3.14, numpy, numba, scipy, scikit-learn, and matplotlib. The experiment scripts add `src/` to `sys.path` themselves and are run from the repository root. The GPU path (`experiments/e51_gpu_uniform.py`) needs a CUDA build of CuPy. The Julia comparison needs Julia 1.11 or later with the packages listed in `julia/setup.jl` (DynamicalSystems 3.7 / Attractors 2.0.5 were used).

## Reproducing the results

All runs use fixed seeds. Results that the article reports are already in `results/`; every script skips what is already computed where that applies. Section numbers below refer to `paper/siads/main.pdf`; SM*n* are sections of `paper/siads/supplement.pdf`.

Benchmark and protocol (section 5, SM1–SM3):

```
python experiments/e01_chua_table1.py      # Chua I basin fractions and the z = 0 basin map
python experiments/e02_uniform_costs.py    # cost of uniform sampling for every target (also the nonescaping fractions of Corollary 2.2)
python experiments/e03_chua_lifetime.py    # numerical "transient chaos" of Chua I versus the step
python experiments/e04_chua_dt_convergence.py
python experiments/e07_unknown_residues.py # unmatched regimes, discovery of new templates
python experiments/e08_classic_baselines.py  # perpetual points, nullclines, connecting curves
python experiments/e09_synthetic_check.py  # basin measure of the synthetic families D1, D2
python experiments/e11_hiddenness.py       # hiddenness measure h_eps
python experiments/e12_jafari2026_scan.py  # the chaotic window of the Jafari 2026 system
python experiments/e13_verify_targets.py   # lifetime, dt/2, Lyapunov sign; e13b: long lifetimes; e13c: Jafari fragility
```

Search benchmark on levels A–D (sections 6.3–6.5, SM4, SM6, SM7):

```
python experiments/e20_search_benchmark.py chua_II --strategies uniform,de_zelinka,sus,portfolio_no_sus,portfolio,portfolio_cvs --R 100
# sus = splitting by transient duration; portfolio_no_sus = active learning with the 20% uniform share; portfolio_cvs = portfolio with nullclines
python experiments/e25_stage2_driver.py    # all level-A/B/D configurations, with resume
python experiments/e27_stage3_driver.py    # repeats with R = 50 and the Jiang maps
python experiments/e28_robustness_driver.py  # hyperparameters
python experiments/e22_signal_analysis.py  # channel diagnostics d(alpha) and AUC (SM7)
python experiments/e23_edge_check.py       # bisection toward basin boundaries
python experiments/e65_stagger.py          # stagger-and-step baseline on Chua II, SE11, LE1, D2 (p = 1e-4)
python experiments/e66_stagger_summary.py  # stagger-and-step against splitting and the portfolio (Table SM7; with batch B23)
python experiments/e67_portfolio_sas.py    # stagger-and-step chains as a portfolio arm, Corollary 3.2 (Table 5; 12 problems, R = 50)
```

Product targets (Proposition 4.2, Table 2, Figure 2): `python experiments/e68_product_theory.py` computes the nonescaping fraction q of SE11, the gain of ideal splitting and the information bound for k copies, and the base of the measured geometric growth from the batch outputs of `results/e50/B8`; `--figure` redraws Figure 2 only.

Long batches (coupled copies, enlarged regions, D2 down to p = 1e-7, cost in steps and time, the one-thread timing, the coupling-born attractor):

```
python experiments/e50_batch_runner.py list        # batches B0..B23 of experiments/e50_batches.py with time estimates
RMQ_TASK_ID=local RMQ_CPU=8 python experiments/e50_batch_runner.py run B13
python experiments/e50_batch_runner.py summarize B13
```

The runner was written for a local job queue (`rmq`); the two environment variables above let it run directly with the given number of threads. Uniform sampling of the enlarged-region and coupled-copy batches can be computed on the GPU with `experiments/e51_gpu_uniform.py` (bit-exact with the CPU kernels). Batch outputs are appended to `results/e50/<batch>/<job>.jsonl`, so an interrupted batch resumes.

Coupled Chua circuits, the coupling-born attractor, and the comparison with Attractors.jl (sections 6.2, 6.6, 6.7, SM8.3):

```
python experiments/e55_coupled_chua_census.py       # regimes, equilibria, hiddenness of the coupled circuits
python experiments/e58_coupled_chua_eps_scan.py     # continuation of the coupling-born attractor in eps
python experiments/e62_lifetime_checks.py           # lifetime of the coupling-born attractor (1e5-1e6, DOP853) and of the Jafari chaos
python experiments/e63_hiddenness_big.py            # hiddenness with 2e5 starts per radius (upper bounds on h_eps)
python experiments/e56_chua_param_scan.py; python experiments/e57_param_scan.py   # basin measure toward the crisis
python experiments/e59_eval_cost.py export          # common initial conditions -> results/e59/ics_*.csv
julia julia/recurrence_cost.jl                      # Attractors.jl cost per evaluation -> results/e59/cost_julia.json
python experiments/e59_eval_cost.py cost            # Python cost per evaluation
python experiments/e59_cost_table.py; python experiments/e59_paper_table.py
```

## Building the manuscript

```
python paper/siads/make_figures_siads.py   # all figures of paper/siads at the SIAM text width (8-pt labels)
python paper/siads/make_figures_siads.py e40:fig_maps e54   # selected figure functions only
python paper/siads/build.py                # supplement -> main -> supplement -> main (cross-references via xr-hyper)
```

The figure driver runs the plotting scripts of `experiments/` (e10, e26, e40, e53, e54, e61, e68) through the machinery of `paper/chaos/make_figures_en.py` with SIAM page sizes and the SIADS-only redraws of `paper/siads/figures_siads.py`; the first run integrates the trajectory batches and basin maps of the attractor figures (about ten minutes on 8 threads) and caches them in `paper/siads/_build/cache`. `build.py` compiles with Tectonic (set the `TECTONIC` environment variable to its path); the sources also compile with pdflatex and BibTeX in the same order. The SIAM class may be distributed only together with the whole SIAM macro package, so it is not included: download the `siamart` macros (version 251216) from SIAM (https://www.siam.org/journals/) and put `siamart251216.cls` and `siamplain.bst` next to `paper/siads/main.tex`.

The earlier manuscript (`paper/chaos/`):

```
python experiments/e54_cost_figure.py     # cost in steps and time (table rows and figure)
python experiments/e54_fill_tables.py     # inserts the rows between the TABCOST/TABTIME markers of sections/50_results.tex
python experiments/e53_paper_results.py   # enlarged regions and small-p figures
python experiments/e61_new_attractor_figure.py   # the coupling-born attractor
python paper/chaos/make_figures_en.py     # all other figures (runs experiments/e10, e26, e40 with English labels)
python paper/chaos/build_chaos.py         # refs.bib, manuscript.tex, manuscript.pdf
```

## License and citation

The code is released under the MIT License (see `LICENSE`). Please cite the article; `CITATION.cff` has the reference.
