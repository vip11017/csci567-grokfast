# Grokfast: Reading Guide

Lee, Kang, Kim & Lee, *Grokfast: Accelerated Grokking by Amplifying Slow Gradients*,
arXiv:2405.20233 (v2, June 2024). PDF: https://arxiv.org/pdf/2405.20233v2

## The paper in five sentences

1. **Grokking** means a model reaches ~100% training accuracy early, then reaches high
   validation accuracy only much later. In their baseline, the gap is ~97× more steps.
2. The authors treat each parameter's gradient over training steps as a time signal.
   They argue that its **fast-changing** part causes overfitting and its **slow-changing**
   part causes generalization.
3. **Grokfast** boosts the slow part. Before each optimizer step it adds a smoothed copy of
   recent gradients to the current gradient: `g ← g + λ · smooth(g)`. The smoothing is a
   100-step moving average (MA) or an exponential moving average (EMA).
4. On x·y mod 97 this reaches 95% validation accuracy in 790 steps (MA) or 910 (EMA),
   versus 39,890 for the baseline. That's the "50×" claim, but it uses weight decay too
   (see below).
5. It also helps on MNIST, QM9 molecules and IMDb, all set up to grok on purpose (tiny
   training sets, oversized initial weights).

## Read it in this order

| Order | Section | What to get out of it | Effort |
|---|---|---|---|
| 1 | Abstract + §1 Intro | The idea: gradients as a signal over time, with slow parts that generalize. Eq. (1)–(2) just set up notation. Look at Figure 2. | Read |
| 2 | §2.1 Filter design | Eq. (3): `ĝ = g + h * g`. In frequency terms (Eq. 5), the gain is `1 + H(ω)`, so low frequencies get boosted. Algorithms 1 and 2 are the whole method. | Read carefully |
| 3 | §2.2 Experiment + Figure 4 | Grid search over λ and window w. Best is λ=5, w=100. Both have a sweet spot. | Read |
| 4 | §2.3 Discussion (Q1–Q3) | **The most important section for our project.** Q1: slow-only gradients fail. Q2: starting the filter at step 500 is even faster. Q3: weight decay is needed for the 50×. | Read carefully |
| 5 | §3 EMA version | Same idea with an EMA, which needs much less memory. Eq. (7). | Skim |
| 6 | §4.1 + Figure 8 | EMA sweeps over α, λ and weight decay. Recommended ranges: λ ∈ [0.1, 5], α ∈ [0.8, 0.99]. | Read |
| 7 | §4.2–4.4 | MNIST, QM9, IMDb. Note how each one is made to grok. | Skim |
| 8 | §5.1 Grokfast vs. momentum | Their argument that this isn't just momentum. **Key for our H1.** | Read carefully |
| 9 | §5.2 + Table 2 | PCA trajectories: Grokfast ends up closer to the starting weights. | Skim |
| 10 | Appendix A | Proof that filtering gradients equals filtering updates, but only for *linear* optimizers (SGD, momentum). | Skim the statement |
| 11 | Appendix B, D | Exact setups (B) and the headline numbers table (D, Table 3). | Reference |
| — | §6 Related work, references, App. C, E | Background, code listing, more PCA plots. | Optional |

## Numbers to remember

| Setting | Steps to 95% val acc |
|---|---|
| Baseline (Adam, no weight decay) | 39,890 |
| Grokfast-MA, **no** weight decay | 2,940 (13.6×) |
| Grokfast-MA, filter only after step 500 | 1,920 (20.8×) |
| Grokfast-MA + weight decay 0.01 | **790 (50.5×)** |
| Grokfast-EMA + weight decay 0.005 | **910 (43.8×)** |

Other tasks: MNIST 44,022 → 2,001 steps (22×). QM9 min val loss 0.00659 → 0.00348.
IMDb best val acc 0.84 → 0.90.

## Things to notice while reading (these matter for our project)

- **The 50× is the filter plus weight decay.** On its own, the MA filter gives 13.6×
  (§2.2, Table 1). Weight decay adds another 3.7× (§2.3 Q3). The authors also say weight
  decay *without* the filter made training unstable (Figure 7). Our H2 tests exactly this.
- **They use Adam with weight decay, not AdamW**, on the algorithmic task (Appendix B.1).
  In PyTorch's Adam, weight decay is plain L2 added to the gradient, so it gets filtered
  and rescaled too. The other tasks use AdamW.
- **§5.1 is their answer to "isn't this just momentum?"** Their points: the smoothed
  gradient is *added to* the raw gradient rather than replacing it, and it's applied
  *before* the optimizer. Adam already keeps its own moving average, so this is a
  second, slower one on top. H1 asks whether a tuned optimizer gets the same result.
- **Appendix A's proof assumes a linear optimizer.** Adam divides by a running gradient
  magnitude, which isn't linear, so the theory doesn't cover the optimizer they actually
  used. The Adam results are empirical only.
- **Every task is set up to grok.** Tiny datasets (1,000 MNIST images, 100 molecules,
  1,000 reviews) and inflated initial weights (8× for MNIST, 3× for QM9). Our H3 asks
  whether the gain survives normal conditions.
- **No error bars on the speedups.** Each speedup is a single run. Only the parameter
  distances in Table 2 report five runs.
- **The MNIST threshold is inconsistent.** Table 4 says 95% val accuracy, but §4.2 says
  final accuracy is only ~91%, so 95% can't be the threshold used.

## Terms you'll meet

- **Low-pass filter (LPF):** keeps slow trends in a signal and removes fast wiggles. A
  moving average is the simplest one.
- **Transfer function H(ω):** how much a filter scales each frequency ω. Low ω means slow
  trends, high ω means fast changes.
- **Impulse response h(t):** the filter's weights over past steps. For MA it's flat over
  w steps. For EMA it decays like αᵗ.
- **EMA momentum α:** how long the EMA remembers. α=0.98 averages over roughly the last
  50 steps.
- **λ (amplifier gain):** how strongly the smoothed gradient is added back.
- **States A → B → C:** initialized → overfitted (≈ step 500) → generalized.
