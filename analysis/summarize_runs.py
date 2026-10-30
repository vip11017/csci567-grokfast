"""Summarize saved runs: steps to a val-accuracy threshold per run, and val curves.

Usage:
    python analysis/summarize_runs.py results/replication --plot docs/figures/replication_val_acc.png

Run files are the res_<label>.pt files written by main.py. Runs are grouped by
everything in the label after the seed tag (e.g. rep_s0_none -> none).
"""
import argparse
import glob
import os
import re
import statistics

import matplotlib.pyplot as plt
import torch


def first_step_at(its, accs, threshold):
    return next((it for it, acc in zip(its, accs) if acc >= threshold), None)


def load_runs(folder):
    runs = []
    for path in sorted(glob.glob(os.path.join(folder, "res_*.pt"))):
        name = os.path.basename(path)[4:-3]
        match = re.search(r"_s(\d+)_(.+)$", name)
        if not match:
            continue
        r = torch.load(path, weights_only=True)
        runs.append({"name": name, "seed": int(match.group(1)), "group": match.group(2), **r})
    return runs


def main(args):
    runs = load_runs(args.folder)
    groups = {}
    for run in runs:
        groups.setdefault(run["group"], []).append(run)

    print(f"| Group | Seeds | Steps to {args.threshold:.0%} val (per seed) | Median | Not reached |")
    print("|---|---|---|---|---|")
    for group, members in sorted(groups.items()):
        members.sort(key=lambda r: r["seed"])
        hits = [first_step_at(r["its"], r["val_acc"], args.threshold) for r in members]
        reached = [h for h in hits if h is not None]
        median = f"{statistics.median(reached):,.0f}" if reached else "-"
        per_seed = ", ".join("-" if h is None else f"{h:,}" for h in hits)
        print(f"| {group} | {len(members)} | {per_seed} | {median} | {hits.count(None)} |")

    if args.plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for color, (group, members) in zip(colors, sorted(groups.items())):
            for i, run in enumerate(members):
                ax.plot(run["its"], run["val_acc"], color=color, alpha=0.8, lw=1,
                        label=group if i == 0 else None)
        ax.axhline(args.threshold, color="grey", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Optimization steps")
        ax.set_ylabel("Validation accuracy")
        ax.set_title(f"x·y mod 97, {len(runs)} runs")
        ax.legend()
        ax.grid(alpha=0.3)
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"saved {args.plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--plot", default=None)
    main(parser.parse_args())
