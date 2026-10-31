# Full replication of the paper's claims

Every experiment in the Grokfast paper, rerun with the official scripts on a Colab T4 from the
`salman` branch, 3 seeds each (seed 0, 1, 2). Runs are defined and analyzed in section 5 of
`notebooks/grokfast_experiments.ipynb` and saved in `results/replication/` and `results/paper_*/`.

**Status (October 31, 2026):** Table 3, the ablations and IMDb are complete. MNIST has 5 of 9
runs and QM9 8 of 9; the remaining runs are still training, and the cells marked "running"
will be filled in when they finish.

## Summary

| Claim in the paper | Reproduces? | Notes |
|---|---|---|
| Grokfast-MA/EMA grok ~50× faster on x·y mod 97 (Table 3) | **Yes, the Grokfast numbers** | Seed 0 matches exactly. Our baseline is slower and more variable than the paper's single run. |
| Window and λ each have a sweet spot at w = 100, λ = 5 (ablations) | **Yes** | Same ranking across all 3 seeds. |
| 2-stage beats 1-stage; the slow component alone fails (Figs. 4–5) | **Yes** | Slow-only never learned in any seed. |
| MNIST: 44,022 → 2,001 steps, 89.8% → 91.2% (Table 4) | **Numbers yes, cause no** | The no-filter run with the same weight decay also gets 2,001 steps. |
| QM9: lower validation loss (Table 5) | **Partly** | Different loss scale; seed-to-seed spread is larger than the effect. |
| IMDb: 0.84 → 0.90 accuracy, 0.517 → 0.412 loss (Table 6) | **Numbers roughly, cause no** | Weight decay alone gives the same gain. |

The recurring finding: on MNIST and IMDb, the paper's "Grokfast" command also changes weight decay
(0.01 → 2.0 on MNIST, 1.0 → 10.0 on IMDb). Running that weight decay *without* the filter gives
the same improvement. This is the same confound as H2 on the main task (`docs/results-h2.md`).

## 1. Main task: x·y mod 97 (Table 3)

Full write-up: `docs/results-replication.md`. Steps to first reach 95% validation accuracy:

| Method | Seeds 0 / 1 / 2 | Median | Paper |
|---|---|---|---|
| Baseline (no filter, weight decay 0) | 56,800 / 79,640 / 48,650 | 56,800 | 39,890 |
| Grokfast-MA (w 100, λ 5, weight decay 0.01) | 790 / 900 / 1,050 | 900 | 790 |
| Grokfast-EMA (α 0.98, λ 2, weight decay 0.005) | 910 / 1,350 / 920 | 920 | 910 |

## 2. Ablations: window size, λ, 2-stage, slow-only (Table 1, Figs. 4–5)

Grokfast-MA at weight decay 0, as in the paper's ablations. "2-stage" turns the filter on after
the first 500 steps (the official `--two_stage` flag). "Slow only" replaces the gradient with
λ × its moving average instead of adding the two (`--filter ma_slow`). Runs stop once 95% has
held for 1,000 steps, or are stopped at step 20,000 if training accuracy is still below 50%.

Steps to first reach 95% validation accuracy:

| Variant | Seeds 0 / 1 / 2 | Median | Paper |
|---|---|---|---|
| **1-stage, w 100, λ 5** | 3,100 / 2,630 / 2,100 | **2,630** | 2,940 |
| **2-stage, w 100, λ 5** | 1,840 / 1,620 / 1,670 | **1,670** | 1,920 |
| Slow only, 1-stage | never (val acc 2–6% at 20k) | – | "much slower and unstable" |
| Slow only, 2-stage | never (val acc 2–5% at 20k) | – | same |
| λ 1 | 23,800 / 22,070 / 21,650 | 22,070 | |
| λ 2 | 4,920 / 3,660 / 3,350 | 3,660 | |
| λ 10 | 8,240 / 7,390 / 8,210 | 8,210 | |
| w 2 | 36,540 / 21,100 / 22,020 | 22,020 | |
| w 5 | 17,790 / 12,290 / 12,520 | 12,520 | |
| w 10 | 12,570 / 9,450 / 14,850 | 12,570 | |
| w 20 | 4,470 / 3,310 / 2,570 | 3,310 | |
| w 50 | 3,490 / 3,240 / 2,870 | 3,240 | |
| w 200 | 3,640 / 3,580 / 2,390 | 3,580 | |

- **The ablations reproduce.** The 1-stage and 2-stage numbers land within seed noise of the
  paper. λ = 5 and w = 100 are the best settings in every seed, and the shape matches the
  paper's "each hyperparameter has a sweet spot" (λ = 10 is 3× slower than λ = 5; w ≤ 10 is
  5–8× slower).
- **Windows from 20 to 200 are nearly equivalent** (about 2.5–3.6k steps). Only very short
  windows hurt.
- **The slow component alone fails completely.** The paper's Figure 5 says it "cannot be used
  alone as a replacement"; in our runs it never even fit the training set in 20k steps.
- At weight decay 0, Grokfast-MA needs about 2.6k steps. With the paper's weight decay of 0.01
  (Table 3) it needs 0.9k. Weight decay matters here as well.

## 3. MNIST (Table 4)

The paper's setup: a 3-layer MLP trained on 1,000 MNIST images with weights scaled up 8× at
initialization. The paper reports steps "@ 95% Val. Acc.", but its baseline never gets above 90%;
the threshold that reproduces its numbers is **85%**. Two commands from the README:

- **base:** no filter, weight decay 0.01 (the README baseline)
- **readme:** the README's "Grokfast" command run as written. It sets α, λ and weight decay 2.0
  but never passes `--filter`, so the filter is off.
- **ema:** the same command plus `--filter ema`, which is what the paper describes.

| Run | Steps to first 85% (seeds 0 / 1 / 2) | Final val acc | Paper |
|---|---|---|---|
| base | 43,355 / 48,691 / running | 89.3% / 88.3% | 44,022 steps, 89.8% |
| readme (no filter, weight decay 2.0) | 2,001 / 1,334 / running | 90.6% / 90.6% | |
| ema (filter, weight decay 2.0) | 2,001 / running / running | 90.7% (best 91.6%) | 2,001 steps, 91.2% |

- **The paper's numbers reproduce**, including the 2,001-step Grokfast result.
- **The filter doesn't cause them.** Without the filter, weight decay 2.0 also reaches 85% at
  step 2,001 (seed 0) and ends at the same accuracy. The speedup comes from the weight decay.
- Caveat: these runs evaluate every ~667 steps (the script's own schedule), so steps are only
  resolved to that granularity; 2,001 is the third evaluation.

## 4. QM9 (Table 5)

A small graph network on 100 molecules from QM9, initial weights scaled 3×. Lowest validation loss:

| Run | Seed 0 | Seed 1 | Seed 2 | Paper |
|---|---|---|---|---|
| base (no filter, no weight decay) | 3.56 | 0.95 | 1.82 | 0.00659 |
| readme (no filter, weight decay 0.01) | 2.27 | 0.85 | 3.67 | |
| ema (filter, weight decay 0.01) | 1.77 | 0.71 | running | 0.00348 |

- **Different scale.** The published `main_qm9.py` trains on raw (unnormalized) targets, so our
  losses are about 100–1000× the paper's. Only comparisons within a seed are meaningful.
- **The filter seems to help beyond weight decay** in seeds 0 and 1 (16–22% lower loss
  than weight decay alone). This is the only task where the filter itself appears to add
  something.
- **But seeds disagree.** In seed 2, weight decay alone made things *worse*, and the spread
  between seeds is larger than any of the effects. We'll judge this once seed 2's filter run
  is in.

## 5. IMDb (Table 6)

A 2-layer LSTM sentiment classifier on 1,000 IMDb reviews, initial weights scaled 6×. The
validation set is small, so accuracy moves in steps of 0.02.

| Run | Best val acc (seeds 0 / 1 / 2) | Median | Lowest val loss (seeds 0 / 1 / 2) | Median |
|---|---|---|---|---|
| base (no filter, weight decay 1.0) | 0.86 / 0.90 / 0.90 | 0.90 | 0.569 / 0.461 / 0.415 | 0.461 |
| readme (no filter, weight decay 10) | 0.86 / 0.88 / 0.88 | 0.88 | 0.361 / 0.329 / 0.290 | **0.329** |
| ema (filter, weight decay 10) | 0.84 / 0.94 / 0.90 | 0.90 | 0.499 / 0.359 / 0.352 | 0.359 |
| *Paper* | *0.84 → 0.90* | | *0.517 → 0.412* | |

- **The loss improvement reproduces** (0.461 → 0.359 with Grokfast), but **weight decay alone
  does at least as well** (0.329). The filter adds nothing on top.
- **Accuracy doesn't separate the methods.** Our baseline already reaches 0.90, and the
  differences are within one or two validation examples.

## Caveats

- 3 seeds per configuration: enough to see the spread, not to pin down small effects.
- MNIST, QM9 and IMDb runs use the paper's own evaluation schedule and validation sets,
  which are small and coarse.
- QM9 losses aren't on the paper's scale (see above).
