# H1: status and raw results (paused, for handoff)

H1 asks whether Grokfast's speedup is just "more momentum", or whether it really comes from the
*slow* part of the gradient. It's the `h1` grid in `notebooks/grokfast_experiments.ipynb`
(section 7): Adam at weight decay 0 and 0.005, 3 seeds, with

- **filters at the same strength (λ 2, α 0.98):** none, Grokfast-EMA, a high-pass version that
  amplifies the *fast* part of the gradient instead (`ema_hp`), and a random direction with the
  same size as Grokfast's term (`ema_rand`)
- **more Adam momentum with no filter:** β₁ ∈ {0.95, 0.98, 0.99} (default 0.9)

**Status:** paused on 2026-10-30 after **26 of 42 runs** so the paper replication could run first.
Finished runs are in `results/h1/` (each has a `done_*.txt` marker). Eight runs were stopped part
way; their partial curves are also in `results/h1/` but they have no `done_` marker. Eight more
never started (all seed 2 at weight decay 0.005, and seed 2 at β₁ 0.99 / weight decay 0).

This file only reports numbers. Interpreting them and writing up H1 is up to whoever takes it over.

## Raw results

Steps until validation accuracy **first reaches 95%** / **stays ≥ 95% for 1,000 steps**, per seed.
"never" = not reached; "stopped at N" = paused at step N before reaching it; "fail" = training
accuracy below 50% at step 20,000, so the run was ended.

**Weight decay 0**

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| No filter, β₁ 0.9 (paper's baseline) | 56,800 / 67,290 | 79,640 / 82,630 | stopped at 46k (91%) |
| No filter, β₁ 0.95 | 43,570 / 45,410 | 40,510 / 41,180 | stopped at 25k |
| No filter, β₁ 0.98 | 40,340 / 42,650 | 43,750 / 45,710 | stopped at 21k |
| No filter, β₁ 0.99 | 27,140 / 40,070 | 23,330 / 25,400 | not run |
| Grokfast-EMA | 8,100 / 8,100 | 10,390 / 10,390 | 9,460 / 9,790 |
| High-pass | never in 100k (3% val) | stopped at 88k (2%) | stopped at 27k (2%) |
| Random term | never in 100k (89% val) | stopped at 69k (52%) | stopped at 16k (53%) |

**Weight decay 0.005**

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| No filter, β₁ 0.9 | 8,470 / 37,520 | 1,860 / 11,310 | not run |
| No filter, β₁ 0.95 | 8,830 / 8,920 | 2,630 / 6,780 | not run |
| No filter, β₁ 0.98 | fail | 48,760 / stopped at 49k | not run |
| No filter, β₁ 0.99 | fail | fail | not run |
| Grokfast-EMA (paper's setting) | 910 / 6,840 | 1,350 / 6,280 | not run |
| High-pass | 14,700 / never (crashed to 2%) | 6,490 / never | not run |
| Random term | 1,680 / 25,760 | 1,610 / 9,060 | not run |

Sanity check: the β₁ 0.9 no-filter and Grokfast-EMA rows repeat runs from the replication and
H2 with identical settings, and give identical numbers, so the setup is deterministic.

## Resuming H1

In a fresh Colab runtime, run notebook sections 1–4, copy the finished-run markers so they're
skipped, then launch:

```python
!mkdir -p {RESULTS}/h1 && cp {REPO_DIR}/results/h1/done_*.txt {RESULTS}/h1/
launch("h1", parallel=4)
```

This reruns the 16 unfinished runs. The slowest are the weight-decay-0 runs (up to 100,000
steps each). Possible additions before resuming: an SGD-with-momentum baseline (`main.py` supports
`--optimizer SGD --momentum 0.9`) and a comparison at matched update sizes.
