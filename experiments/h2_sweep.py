"""H2 sweep: does weight decay alone close the gap, or does the filter still add value?

Grid on x*y mod 97 (main.py), seeds 0-2:
  optimizer     Adam (coupled L2, as in the paper's main.py) and AdamW (decoupled)
  weight decay  0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0
  filter        none, and Grokfast-EMA (alpha 0.98, lambda 2)

Skipped: AdamW at weight decay 0 (identical to Adam) and Adam/none/0 (the replication
baseline, already in results/replication). Runs stop once val accuracy has held >= 95%
for 1,000 steps, or at --budget steps. Runs whose train accuracy is still below 50% at
step 20,000 are aborted as failures (strong coupled L2 decay keeps the model from
fitting at all, and those runs would otherwise use the whole budget).

Runs are queued seed by seed, so a partial sweep still covers every config for seed 0.
Finished runs leave a done_<name>.txt marker, and restarting the script skips them.

Usage (Colab):
    python experiments/h2_sweep.py --out /content/drive/MyDrive/csci567-grokfast/h2 --parallel 8
"""
import argparse
import json
import os
import subprocess
import sys
import time

WEIGHT_DECAYS = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
EMA = ["--filter", "ema", "--alpha", "0.98", "--lamb", "2.0"]


def build_runs(seeds, budget, out):
    runs = []
    for seed in seeds:
        for optimizer in ("Adam", "AdamW"):
            for filt in ("none", "ema"):  # long no-filter runs first, for better packing
                for wd in WEIGHT_DECAYS:
                    if wd == 0 and (optimizer == "AdamW" or filt == "none"):
                        continue
                    label = f"h2_s{seed}_{optimizer.lower()}"
                    name = f"{label}_{filt}_wd{wd:g}"
                    cmd = [sys.executable, "main.py", "--label", label, "--seed", str(seed),
                           "--optimizer", optimizer, "--weight_decay", str(wd),
                           "--budget", str(budget), "--stop_threshold", "0.95",
                           "--stop_patience", "1000", "--abort_step", "20000",
                           "--abort_train_acc", "0.5", "--out_dir", out]
                    if filt == "ema":
                        cmd += EMA
                    runs.append({"name": name, "cmd": cmd})
    return runs


def main(args):
    os.makedirs(args.out, exist_ok=True)
    runs = build_runs(args.seeds, args.budget, args.out)
    done = lambda r: os.path.exists(os.path.join(args.out, f"done_{r['name']}.txt"))
    queue = [r for r in runs if not done(r)]
    running = {}
    print(f"{len(runs)} runs in grid, {len(queue)} still to do", flush=True)

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
    parser.add_argument("--out", required=True)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    main(parser.parse_args())
