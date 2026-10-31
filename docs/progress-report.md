# Progress report (October 30, 2026)

Where the Grokfast project stands: what we ran, how, what we found, and what's still open.
Everything here is on the `salman` branch.

| Part | Status | Details |
|---|---|---|
| Paper reading and analysis | Done | `docs/reading-guide.md`, `docs/ideas.md`, team summary doc |
| Code set up and running | Done | Runs on Mac (MPS), Windows (CPU) and Colab (T4 GPU) |
| Replication (paper's Table 3) | **Done**, 3 seeds | `docs/results-replication.md` |
| H2: weight decay vs. the filter | **Done**, 75 runs | `docs/results-h2.md` |
| H1: momentum vs. slow gradients | **Running**, 13 of 42 runs | Preliminary numbers below |
| H3: induced grokking conditions (MNIST) | Not started | Open |

## How we ran the experiments

- **Task:** the paper's main task, x·y mod 97, with its 2-layer Transformer, 50/50 train/validation
  split, Adam, learning rate 1e-3 and batch size 512. Training is the paper's own `main.py`.
- **Compute:** Google Colab, T4 GPU, running 8 experiments at once (the model is tiny, so one run
  barely uses the GPU). A batch of 3 seeds of a slow baseline takes about an hour this way.
- **One notebook:** `notebooks/grokfast_experiments.ipynb` defines every experiment as a "grid"
  of runs, launches them in the background, tracks progress, and makes the tables and plots.
  Open it in Colab from GitHub; to add an experiment, add a grid function and a section.
- **Seeds:** every configuration is run with seeds 0, 1 and 2. The seed also changes the
  train/validation split, as in the paper's code.
- **Stopping rules** (added to `main.py`): a run stops once validation accuracy has stayed at
  or above 95% for 1,000 steps, or at the step budget. A run whose training accuracy is still
  below 50% at step 20,000 is stopped and counted as a failure (the model can't even fit the
  training set).
- **Two metrics per run:**
  - **First reach:** first step with validation accuracy ≥ 95% (what the paper reports).
  - **Stable:** first step after which validation accuracy *stays* ≥ 95% for 1,000 steps. We
    added this because many runs reach 95% and then crash for a while, sometimes to under 10%.
- **Results files:** every run's full training curve is saved in `results/<experiment>/`.

## 1. Replication

Same settings as the paper's Table 3: baseline Adam with no weight decay, Grokfast-MA (window 100,
λ 5, weight decay 0.01) and Grokfast-EMA (α 0.98, λ 2, weight decay 0.005).

| Method | Steps to 95% val (seeds 0, 1, 2) | Median | Paper |
|---|---|---|---|
| Baseline | 56,800 / 79,640 / 48,650 | 56,800 | 39,890 |
| Grokfast-MA | 790 / 900 / 1,050 | 900 | 790 |
| Grokfast-EMA | 910 / 1,350 / 920 | 920 | 910 |

- Grokfast's numbers reproduce exactly for seed 0, on three different machines.
- The baseline is slower than the paper's and varies a lot between seeds, so the speedup
  ratio depends heavily on the seed (roughly 46× to 100×).
- Grokfast-EMA crashes repeatedly after it first groks, then recovers. Grokfast-MA doesn't.

## 2. H2: is the speedup from weight decay or from the filter?

**Question.** The paper's 50× compares Grokfast *with* weight decay (0.005–0.01) against a
baseline *without* any. How much of the speedup does tuned weight decay give by itself?

**Design.** 75 runs: optimizer {Adam, AdamW} × weight decay {0, 0.001, 0.005, 0.01, 0.05, 0.1,
1.0} × filter {none, Grokfast-EMA} × 3 seeds. With Adam (the paper's setup) weight decay is
added to the gradient (L2 regularization); with AdamW it's applied separately.

**Results** (median over seeds, thousands of steps; "fail" = couldn't fit the training set):

| Weight decay | Adam, no filter | Adam + EMA | AdamW, no filter | AdamW + EMA |
|---|---|---|---|---|
| | *first reach / stable* | *first reach / stable* | *first reach / stable* | *first reach / stable* |
| 0 | 56.8 / 67.3 | 9.5 / 9.8 | (same as Adam) | (same as Adam) |
| 0.001 | **1.6** / 10.0 | 2.1 / 6.5 | 61.3 / 66.1 | 11.9 / 12.2 |
| 0.005 | 1.9 / 17.9 | **0.9** / 6.3 | 40.9 / 46.0 | 11.9 / 11.9 |
| 0.01 | fail | 1.4 / **4.8** | 38.6 / 38.7 | 8.3 / 8.3 |
| 0.05 | fail | fail | 15.2 / 15.9 | 12.7 / 13.3 |
| 0.1 | fail | fail | 9.4 / 10.3 | 9.1 / 9.1 |
| 1.0 | fail | fail | **1.1** / fail | 2.2 / 8.2 |

Grokfast-MA at the paper's settings: 0.9 / 0.9 (from the replication).

**Findings**

1. **Tuned weight decay alone gives almost all of the speed to first reach 95%.** With no filter,
   the best settings first reach 95% in 1.1k (AdamW, decay 1.0) or 1.6k steps (Adam, decay
   0.001), against 0.9k for the best Grokfast setting. Against a tuned baseline, the paper's
   roughly 60× advantage shrinks to about 1.2×.
2. **The filters make grokking stable.** Requiring 95% to hold for 1,000 steps, the best
   no-filter setting needs about 10k steps, Grokfast-EMA 4.8–6.3k, and Grokfast-MA 0.9k (it never
   crashed). AdamW at decay 1.0, the fastest to first reach 95%, only stabilized in 1 of 3 seeds.
3. **With Adam, the filter quietly weakens weight decay.** Grokfast multiplies the gradient by up
   to 3× *before* Adam adds the weight-decay term, so the same decay setting is relatively weaker
   with the filter on. Without the filter, Adam can't fit the data at decay 0.01; with it, 0.01 is
   one of the best settings. Part of what looks like a filter effect in the paper is really this.
4. **Without enough weight decay, the filter helps a lot** (56.8k → 9.5k steps at decay 0).
5. **At AdamW's best decay, the filter slows the first reach** (1.1k → 2.2k) but stabilizes it.

**What it means.** The paper's 50× mostly comes from comparing against an untuned baseline.
Grokfast's real advantage is that generalization sticks once it arrives, especially with MA.
Full write-up and plot: `docs/results-h2.md`.

## 3. H1: is it just more momentum, or the slow part of the gradient? (running)

**Design.** 42 runs with Adam at weight decay 0 and 0.005, 3 seeds:
- filters at the same strength (λ 2, α 0.98): none, Grokfast-EMA, a **high-pass** version that
  amplifies the *fast* part of the gradient instead, and a **random** term of the same size as
  Grokfast's (new filters in `grokfast.py`)
- more Adam momentum with no filter: β₁ ∈ {0.95, 0.98, 0.99} (default 0.9)

**Preliminary numbers, seed 0 only** (steps to first reach 95%, thousands; 13 of 42 runs done):

| | Weight decay 0 | Weight decay 0.005 |
|---|---|---|
| No filter (β₁ 0.9) | 56.8 | 8.5 |
| No filter, β₁ 0.95 / 0.98 / 0.99 | 43.6 / 40.3 / 27.1 | 8.8 / fail / fail |
| Grokfast-EMA | **8.1** | **0.9** |
| High-pass | not reached by 65k | 14.7, then crashed |
| Random term | not reached by 53k | 1.7 |

Early pattern, to be checked on seeds 1 and 2: more momentum helps somewhat (up to about 2×)
but doesn't come close to Grokfast; amplifying the fast part blocks learning; and without weight
decay only the slow (EMA) filter gives a big speedup, which supports the paper's mechanism. With
weight decay, even the random term nearly matches EMA, consistent with H2's point that anything
added to the gradient before Adam weakens the effective decay. Seed 1 so far follows the same
pattern. Expected to finish tonight.

## 4. H3: does it only work in artificial grokking setups? (not started)

The paper's MNIST, QM9 and IMDb experiments use tiny training sets and inflated initial weights
to force grokking. H3 tests MNIST across training-set sizes {1k, 10k, 50k} and initialization
scales {1, 3, 8}, with a held-out test set. Two things we found while reading the code:
- In the paper's MNIST setup, the baseline uses weight decay 0.01 and Grokfast uses 2.0, the same
  confound as H2. Include a no-filter run at 2.0.
- The README's Grokfast commands for MNIST, IMDb and QM9 never turn the filter on (the scripts
  default to no filter), so set `--filter ema` explicitly.

`main_mnist.py` still needs the stopping options that `main.py` has.

## What's open

- **H1:** finish the analysis and write-up once the runs are done; add an SGD-with-momentum
  baseline and an "equal update size" comparison.
- **H2 follow-ups:** add Grokfast-MA to the weight-decay grid (it was the most stable setting,
  but H2 only tested EMA); tune the learning rate for the baselines; log weight norms over time.
- **H3:** everything.
- **Report and presentation:** combine the write-ups into the final story.

## Changes to the paper's code

- `main.py`: stopping options (`--stop_threshold`, `--stop_patience`, `--abort_step`,
  `--abort_train_acc`), `--out_dir`, SGD with `--momentum`, and the new filters.
- `grokfast.py`: two control filters for H1, `gradfilter_ema_highpass` and
  `gradfilter_ema_random`. The original filters are unchanged.
- Everything else (experiment grids, running, analysis) is in the notebook.

## File map

| File | What it is |
|---|---|
| `notebooks/grokfast_experiments.ipynb` | Runs and analyzes every experiment |
| `docs/reading-guide.md` | How to read the paper, key numbers, terms |
| `docs/ideas.md` | Project ideas and early notes |
| `docs/ta-summary.html` / `.pdf` | One-page brief for the TA meeting |
| `docs/results-replication.md` | Replication write-up |
| `docs/results-h2.md` | H2 write-up |
| `docs/progress-report.md` | This file |
| `results/replication/`, `results/h2/` | Saved runs (training curves) |
