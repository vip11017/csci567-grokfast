"""Run a named grid of main.py experiments (x*y mod 97) with a fixed number in parallel.

    python experiments/sweep.py --grid h2 --out /content/results/h2 --parallel 8

Grids:
  replication  baseline, Grokfast-MA and Grokfast-EMA with the paper's settings.
  h2           weight decay vs. filter: optimizer {Adam (coupled L2), AdamW}
               x weight decay {0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0}
               x filter {none, Grokfast-EMA}. Skips AdamW at 0 (same as Adam) and
               Adam/none/0 (the replication baseline).
  h1           ordinary momentum vs. the "slow gradient" mechanism, with Adam at weight
               decay 0 and 0.005: filter {none, EMA, high-pass EMA, norm-matched random}
               at beta1 0.9, plus no filter at beta1 {0.95, 0.98, 0.99}.

Every run stops once val accuracy has held >= 95% for 1,000 steps, or at --budget steps.
Runs whose train accuracy is still below 50% at step 20,000 are aborted as failures
(strong coupled L2 decay keeps the model from fitting at all).

Runs are queued seed by seed, so a partial grid still covers every config for seed 0.
Finished runs leave a done_<name>.txt marker, and rerunning the command skips them.
A status.json in the output folder lists what's done, running and queued.
"""
import argparse
import json
import os
import subprocess
import sys
import time

COMMON = ["--stop_threshold", "0.95", "--stop_patience", "1000",
          "--abort_step", "20000", "--abort_train_acc", "0.5"]
MA = ["--filter", "ma", "--window_size", "100", "--lamb", "5.0"]
EMA = ["--alpha", "0.98", "--lamb", "2.0"]  # used with --filter ema / ema_hp / ema_rand


def replication(seed):
    yield f"rep_s{seed}_none", f"rep_s{seed}", []
    yield f"rep_s{seed}_ma", f"rep_s{seed}", MA + ["--weight_decay", "0.01"]
    yield f"rep_s{seed}_ema", f"rep_s{seed}", ["--filter", "ema"] + EMA + ["--weight_decay", "0.005"]


def h2(seed):
    for optimizer in ("Adam", "AdamW"):
        for filt in ("none", "ema"):  # long no-filter runs first, for better packing
            for wd in (0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0):
                if wd == 0 and (optimizer == "AdamW" or filt == "none"):
                    continue
                args = ["--optimizer", optimizer, "--weight_decay", str(wd)]
                if filt == "ema":
                    args += ["--filter", "ema"] + EMA
                yield f"h2_s{seed}_{optimizer.lower()}_{filt}_wd{wd:g}", f"h2_s{seed}_{optimizer.lower()}", args


def h1(seed):
    for wd in (0, 0.005):
        for filt in ("none", "ema", "ema_hp", "ema_rand"):
            args = ["--weight_decay", str(wd)]
            if filt != "none":
                args += ["--filter", filt] + EMA
            yield f"h1_s{seed}_b0.9_{filt}_wd{wd:g}", f"h1_s{seed}_b0.9", args
        for beta1 in (0.95, 0.98, 0.99):
            yield (f"h1_s{seed}_b{beta1}_none_wd{wd:g}", f"h1_s{seed}_b{beta1}",
                   ["--weight_decay", str(wd), "--beta1", str(beta1)])


GRIDS = {"replication": replication, "h2": h2, "h1": h1}


def build_runs(grid, seeds, budget, out):
    runs = []
    for seed in seeds:
        for name, label, args in GRIDS[grid](seed):
            cmd = [sys.executable, "main.py", "--label", label, "--seed", str(seed),
                   "--budget", str(budget), "--out_dir", out] + COMMON + args
            runs.append({"name": name, "cmd": cmd})
    return runs


def main(args):
    os.makedirs(args.out, exist_ok=True)
    runs = build_runs(args.grid, args.seeds, args.budget, args.out)
    done = lambda r: os.path.exists(os.path.join(args.out, f"done_{r['name']}.txt"))
    queue = [r for r in runs if not done(r)]
    running = {}
    print(f"{args.grid}: {len(runs)} runs in grid, {len(queue)} still to do", flush=True)

    last_status = 0
    while queue or running:
        while queue and len(running) < args.parallel:
            run = queue.pop(0)
            log = open(os.path.join(args.out, f"log_{run['name']}.txt"), "w")
            running[run["name"]] = (subprocess.Popen(run["cmd"], stdout=log, stderr=subprocess.STDOUT), log, time.time())
            print(f"start {run['name']}", flush=True)

        for name, (proc, log, started) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            del running[name]
            minutes = (time.time() - started) / 60
            with open(os.path.join(args.out, f"done_{name}.txt"), "w") as f:
                f.write(f"rc={rc} minutes={minutes:.1f}\n")
            print(f"done {name} rc={rc} after {minutes:.1f} min", flush=True)

        if time.time() - last_status > 60 or not (queue or running):
            status = {
                "grid": args.grid,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total": len(runs),
                "done": sum(done(r) for r in runs),
                "running": sorted(running),
                "queued": [r["name"] for r in queue],
            }
            with open(os.path.join(args.out, "status.json"), "w") as f:
                json.dump(status, f, indent=1)
            last_status = time.time()
        time.sleep(10)
    print("sweep finished", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=sorted(GRIDS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    main(parser.parse_args())
