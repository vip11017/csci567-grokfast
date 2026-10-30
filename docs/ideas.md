# Grokfast Final Project — Working Ideas

CSCI 567 final project. Team: Vishal Patel, Salman Sattar, Tomohiko Uemura, Chinenye Okoro.

Paper: Lee et al., *Grokfast: Accelerated Grokking by Amplifying Slow Gradients*, arXiv:2405.20233.
Official code: https://github.com/ironjr/grokfast (filter lives in `grokfast.py`).

These notes collect ideas from the first review of the team's paper analysis. They are
proposals to discuss, not decisions.

## 0. Status of the repo (2026-10-30)

The repo is the official Grokfast code plus MPS (Mac) device support. Saved runs for the
modular-multiplication task, measured from the `.pt` files (val evaluated every 10 steps):

| Run | Steps to 95% val | Paper (Table 3) | Train ≥ 99% at |
|---|---|---|---|
| Baseline (`algorithmic_data_baseline/`) | 58,600 | 39,890 | 440 |
| Grokfast-MA (`algorithmic_data_grokfast_ma/`) | 790 | 790 | 840 |
| Grokfast-EMA (`algorithmic_data_grokfast/`) | 910 | 910 | 890 |

- MA and EMA match the paper exactly; the baseline is ~1.5× slower than reported. One
  seed only, so this is the first number to re-check across seeds: baseline variance
  directly changes the headline "50×" ratio.
- With Grokfast, validation crosses 95% at about the same time training crosses 99%, so
  the "delay" itself is gone, not just shortened.

Code details that matter for the hypotheses:
- **`main.py` defaults to `torch.optim.Adam`, not AdamW.** Passing `--weight_decay` to
  Adam adds coupled L2 regularization, which Adam's normalization weakens. So the paper's
  "weight decay 0.01" is L2-in-Adam, not decoupled decay. H2 should compare both
  (`--optimizer AdamW` already works through the CLI).
- The output label keeps only `int(lamb)`, so lambda 0.1 and 0.5 both become `_l0` and
  overwrite each other. Fix before any lambda sweep.
- There's no separate test split: the 50% held-out set is used both for tuning and for
  reporting. Split it before tuning anything.
- `fir` is listed as a `--filter` choice but not implemented.

## 1. What the method actually does

The whole method runs between `loss.backward()` and `optimizer.step()`:

```python
# Grokfast-EMA (per parameter)
ema = alpha * ema + (1 - alpha) * grad
grad = grad + lamb * ema

# Grokfast-MA: same, but ema = mean of the last `window_size` gradients
# (only applied once the window is full when warmup=True)
```

- This is a low-pass boost: the steady part of the gradient (frequency ≈ 0) gets gain
  about `1 + lamb` (3× for EMA with lamb=2), and fast-changing parts get about 1×.
- The filtered gradient then goes into Adam(W), which already keeps its own EMA of
  gradients (beta1 = 0.9) and divides by their running magnitude.

Paper's headline settings (modular multiplication, x·y mod 97, 50/50 split):

| Method | Filter params | Weight decay | Steps to 95% val |
|---|---|---|---|
| Baseline | — | 0 | 39,890 |
| Grokfast-MA | window 100, lamb 5 | 0.01 | 790 |
| Grokfast-EMA | alpha 0.98, lamb 2 | 0.005 | 910 |

**The headline comparison changes filtering and weight decay at the same time.**

## 2. Project structure (from the team analysis)

1. **Replication** of the modular-arithmetic baseline vs. MA vs. EMA results.
2. **H1, ordinary optimization:** the gain comes from better momentum or step size, not
   from slow components carrying special information.
3. **H2, regularization and weight scale:** weight decay and parameter-norm dynamics
   explain most of the gain.
4. **H3, induced grokking conditions:** the benefit shrinks with more data or standard
   initialization (MNIST: train sizes {1k, 10k, 50k} × init scales {1, 3, 8}).

## 3. Ideas to strengthen the project

### H1: a precise framing
- Grokfast-EMA (alpha 0.98) is effectively a second, longer-memory gradient average
  stacked on top of Adam's own (beta1 0.9).
- Natural controls: Adam with beta1 ∈ {0.95, 0.98, 0.99}, and heavy-ball SGD with tuned
  momentum.
- Adam's normalization cancels much of a uniform gradient scaling, so match **actual
  parameter-update norms**, not raw gradient norms (already in the team plan; keep it).

### New: test whether "slow" specifically matters
None of H1–H3 tests frequency specificity head-on. Two cheap controls would:
- **High-pass control:** `grad = grad + lamb * (grad - ema)` boosts the fast components
  instead. If the paper's mechanism is right, this should not help (or should hurt).
- **Norm-matched control:** add a perturbation with the same norm as `lamb * ema` per
  step but no time structure (e.g. a random direction, or a time-shuffled past gradient).

If Grokfast beats both, that's the strongest evidence for the paper's mechanism.

### H2: widen the weight-decay grid
- The planned grid {0, 0.001, 0.005, 0.01, 0.05} is narrow.
- Grokking literature (e.g. Nanda et al., progress measures for grokking) uses AdamW with
  weight decay around 1.0, which on its own sharply speeds up grokking.
- Add **0.1 and 1.0**. If a tuned no-filter baseline groks in a few thousand steps,
  much of the "50×" claim goes away.
- Log parameter norm and update norm every eval step for both arms.

### Measurement hygiene
- Evaluate often (e.g. every 10 steps) near the threshold; 790 vs 910 is a small gap
  and coarse evaluation blurs it.
- Report **steps to threshold** as the primary metric, **median ± IQR over ≥3 seeds**.
- Treat wall-clock time as secondary (hardware-dependent).
- Keep a test split that is never used for tuning; give every method the same tuning
  budget.
- Record runs that never reach the threshold (censored), don't drop them.
- MNIST: use an explicitly defined 85% threshold and say so (paper is inconsistent:
  Table 4 says 95%, Figure 9 marks 85%).

## 4. Compute and ordering

- The modular task is tiny, but a baseline run is ~40k steps; seeds × tuning × factorial
  designs multiply quickly.
- Suggested order:
  1. Replication (sanity check the harness).
  2. H2 (cheapest, most likely to change the conclusion).
  3. H1 + frequency-specificity controls.
  4. H3 on MNIST (fast per run).
- With four people, one owner per block works naturally.

## 5. Harness to build

A single PyTorch harness for the modular task:
- 2-layer decoder-only Transformer for x·y mod 97, 50/50 split, fixed seeds.
- Pluggable gradient filters: none, MA, EMA, high-pass, norm-matched random.
- Config-driven sweeps (optimizer, lr, betas, weight decay, filter params, seed).
- Logs: train/val loss and accuracy, parameter norm, update norm, steps and time to
  threshold; CSV/JSON per run plus a plotting script.
