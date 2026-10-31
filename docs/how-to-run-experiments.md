# How to run an experiment

Everything runs from one notebook, `notebooks/grokfast_experiments.ipynb`, on the `salman`
branch. Training is the paper's own code (`main.py` for modular arithmetic, `main_mnist.py` for
MNIST); the notebook decides what to run, runs it, and summarizes the results.

## 1. Open the notebook in Colab

Open
<https://colab.research.google.com/github/vip11017/csci567-grokfast/blob/salman/notebooks/grokfast_experiments.ipynb>,
then *Runtime → Change runtime type → T4 GPU*. The first time you run a cell, Colab warns that
the notebook isn't from Google; click **Run anyway**.

Run sections 1–4 (setup, grids, runner, result readers). *Run all* is safe: the cells that start
long GPU jobs are commented out.

## 2. Run an existing experiment

In that experiment's section, uncomment the `launch(...)` line and run it. It returns right
away; the runs continue in the background, 8 at a time.

```python
launch("h3", seeds=(0,))   # try one seed first
status("h3")               # rerun any time to check progress
```

If the notebook's kernel restarts, run sections 1–4 again and call `launch` again: finished runs
are skipped, and runs still in progress are picked up rather than started twice.

## 3. Add a new experiment

1. In section 2, write a grid function. It gets a seed and yields one tuple per run:
   `(name, label, args)` for `main.py`, or `(name, label, args, "main_mnist.py")` for MNIST.
   `name` must be unique; `label` is passed to the script and becomes part of the result file
   name (the script adds the filter and weight-decay settings to it). Keep `_s{seed}_` in the
   label so results group by seed.
2. Add it to `GRIDS`.
3. Add a section with a description, a commented `launch(...)` cell, `status(...)`, and a cell
   that loads and tabulates the results.

Example (Grokfast-MA at several weight decays):

```python
def ma_decay_grid(seed):
    for wd in (0, 0.001, 0.005, 0.01, 0.05):
        yield (f"mawd_s{seed}_wd{wd:g}", f"mawd_s{seed}",
               MA + ["--weight_decay", str(wd)])

GRIDS["ma_decay"] = ma_decay_grid
```

## 4. Read the results

`load_runs(folder)` returns every run in a folder with its training curves (`its`, `train_acc`,
`val_acc`, `train_loss`, `val_loss`) and two summary numbers:

- `first_95`: first step with validation accuracy ≥ 95%
- `held_95`: first step after which it stays ≥ 95% for 1,000 steps

Pass `threshold=` to use another level (H3 uses 0.85) and `prefix="mnist_"` for MNIST runs.
`by_group`, `cell` and `plot_val_acc` group runs by everything after the seed, take the median
over seeds, and plot curves.

## 5. Keep the results

The Colab runtime is wiped when it shuts down. Either set `USE_DRIVE = True` in section 1
before launching (results go straight to your Google Drive), or copy them at the end
(section 9). Then add the result files to `results/<experiment>/` in the repo and commit them
with a short write-up in `docs/`, like `docs/results-h2.md`.

## Useful script options

`main.py` (modular arithmetic) and `main_mnist.py` both support:

| Option | Meaning |
|---|---|
| `--stop_threshold 0.95 --stop_patience 1000` | stop once validation accuracy has stayed at or above 0.95 for 1,000 steps |
| `--abort_step 20000 --abort_train_acc 0.5` | give up if training accuracy is still below 50% at step 20,000 |
| `--out_dir <folder>` | where results and plots go |
| `--filter none / ma / ema` | Grokfast filter (`main.py` also has `ema_hp` and `ema_rand`, the H1 controls) |

`main_mnist.py` also has `--val_points N` (validation images taken from the training set, with
the test set scored once at the end) and `--log_every N`. `main.py` also has `--optimizer SGD
--momentum 0.9`.
