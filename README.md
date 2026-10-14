# hidden-attractor-search

Code, experiments, results, and manuscript sources for the article

> M. Elisov, *Searching for hidden attractors as a rare-event problem: An adaptive strategy portfolio with a worst-case guarantee*, manuscript for *Chaos: An Interdisciplinary Journal of Nonlinear Science* (2026).

Archived in Zenodo: [doi:10.5281/zenodo.23348470](https://doi.org/10.5281/zenodo.23348470) (all versions; v1.0.2 is the current one).

The search for a hidden attractor with a small basin is treated as the detection of a rare event in the space of initial conditions. The repository contains the benchmark of systems with hidden attractors, the identification and verification protocol, the search strategies (uniform sampling, subset simulation driven by transient duration, active learning of basin boundaries, their portfolio, bisection toward basin boundaries, a stagger-and-step chain, and Zelinka's differential evolution), every numerical result reported in the article, and the scripts that produce its tables and figures.

## Layout

| Path | Contents |
| --- | --- |
| `src/hsearch/` | Library. `integrate.py`: fixed-step RK4 kernels (numba) that compute trajectory fingerprints inside the integrator; `systems/`: registry of benchmark systems (3D flows, 2D maps, 4D systems, synthetic families, coupled copies); `identify.py`: templates, matching, and the verification checks; `search/`: strategies and the measurement harness (`harness.py`, including the two-stage evaluation); `gpu.py`: uniform sampling on the GPU (optional, CuPy); `viz.py`: plotting helpers. |
| `src/tools/` | Bibliography verification against Crossref and Semantic Scholar (`verify_bib.py`) and small summaries. |
| `experiments/` | Numbered scripts; each writes to `results/`. The map to the article is below. |
| `results/` | JSON outputs of every experiment; `e50/<batch>/<job>.jsonl` holds one line per search run of the long batches; `e53_snippets/` holds the table rows inserted into the manuscript; `e59/` holds the comparison with Attractors.jl. |
| `julia/` | `recurrence_cost.jl`: cost per evaluation of `AttractorsViaRecurrences` (Attractors.jl 2.0) on the same initial conditions as the Python code; `chua_attractors.jl`: independent check of the Chua basin fractions; `setup.jl`: packages. |
| `data/` | `bibliography.json`: the verified bibliographic records (DOI or arXiv id, title similarity) from which `paper/chaos/refs.bib` is generated. |
| `paper/chaos/` | REVTeX 4.2 sources (`main.tex`, `sections/*.tex`), the figures (`figures/`), `build_chaos.py` (generates `refs.bib`, flattens the sources into `manuscript.tex`, compiles the PDF, and checks for undefined references and overfull lines), `make_figures_en.py` (English, journal-size figures), the compiled `manuscript.pdf`, and `alt_text.txt` (alternative text for the figures and tables). |

## Setup

```
pip install -r requirements.txt
```

Python 3.11 or later; the article's results were produced with Python 3.14, numpy, numba, scipy, scikit-learn, and matplotlib. The experiment scripts add `src/` to `sys.path` themselves and are run from the repository root. The GPU path (`experiments/e51_gpu_uniform.py`) needs a CUDA build of CuPy. The Julia comparison needs Julia 1.11 or later with the packages listed in `julia/setup.jl` (DynamicalSystems 3.7 / Attractors 2.0.5 were used).

## Reproducing the results

All runs use fixed seeds. Results that the article reports are already in `results/`; every script skips what is already computed where that applies.

Benchmark and protocol (Sec. II, Appendix A--C):

```
python experiments/e01_chua_table1.py      # Chua I basin fractions and the z = 0 basin map
python experiments/e02_uniform_costs.py    # cost of uniform sampling for every target
python experiments/e03_chua_lifetime.py    # numerical "transient chaos" of Chua I versus the step
python experiments/e04_chua_dt_convergence.py
python experiments/e07_unknown_residues.py # unmatched regimes, discovery of new templates
python experiments/e08_classic_baselines.py  # perpetual points, nullclines, connecting curves
python experiments/e09_synthetic_check.py  # basin measure of the synthetic families D1, D2
python experiments/e11_hiddenness.py       # hiddenness measure h_eps
python experiments/e12_jafari2026_scan.py  # the chaotic window of the Jafari 2026 system
python experiments/e13_verify_targets.py   # lifetime, dt/2, Lyapunov sign; e13b: long lifetimes; e13c: Jafari fragility
```

Search benchmark on levels A--D (Sec. V A--C, Appendix D, G):

```
python experiments/e20_search_benchmark.py chua_II --strategies uniform,de_zelinka,sus,portfolio_no_sus,portfolio,portfolio_cvs --R 100
# sus = splitting by transient duration; portfolio_no_sus = active learning with the 20% uniform share; portfolio_cvs = portfolio with nullclines
python experiments/e25_stage2_driver.py    # all level-A/B/D configurations, with resume
python experiments/e27_stage3_driver.py    # repeats with R = 50 and the Jiang maps
python experiments/e28_robustness_driver.py  # hyperparameters
python experiments/e22_signal_analysis.py  # channel diagnostics d(q)
python experiments/e23_edge_check.py       # bisection toward basin boundaries
python experiments/e65_stagger.py          # stagger-and-step baseline on Chua II, SE11, LE1, D2 (p = 1e-4)
python experiments/e66_stagger_summary.py  # stagger-and-step against splitting and the portfolio (Table XIII; with batch B23)
```

Long batches (enlarged regions, D2 down to p = 1e-7, coupled copies, cost in steps and time, the one-thread timing, the coupling-born attractor):

```
python experiments/e50_batch_runner.py list        # batches B0..B23 of experiments/e50_batches.py with time estimates
RMQ_TASK_ID=local RMQ_CPU=8 python experiments/e50_batch_runner.py run B13
python experiments/e50_batch_runner.py summarize B13
```

The runner was written for a local job queue (`rmq`); the two environment variables above let it run directly with the given number of threads. Uniform sampling of the enlarged-region and coupled-copy batches can be computed on the GPU with `experiments/e51_gpu_uniform.py` (bit-exact with the CPU kernels). Batch outputs are appended to `results/e50/<batch>/<job>.jsonl`, so an interrupted batch resumes.

Coupled Chua circuits, the coupling-born attractor, and the comparison with Attractors.jl (Sec. V D--F):

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

Tables and figures of the manuscript:

```
python experiments/e54_cost_figure.py     # Table VI rows and Fig. 8 (cost in steps and time)
python experiments/e59_paper_table.py     # Table VII rows (three evaluation schemes)
python experiments/e54_fill_tables.py     # inserts the rows between the TABCOST/TABTIME markers of sections/50_results.tex
python experiments/e53_paper_results.py   # enlarged regions and small-p figures (Appendix E, F)
python experiments/e61_new_attractor_figure.py   # Fig. 7, the coupling-born attractor
python paper/chaos/make_figures_en.py     # all other figures (runs experiments/e10, e26, e40 with English labels)
python paper/chaos/build_chaos.py         # refs.bib, manuscript.tex, manuscript.pdf
```

`build_chaos.py` compiles with Tectonic (set the `TECTONIC` environment variable to its path); `manuscript.tex` and `refs.bib` also compile with pdflatex and BibTeX.

## License and citation

The code is released under the MIT License (see `LICENSE`). Please cite the article; `CITATION.cff` has the reference.
