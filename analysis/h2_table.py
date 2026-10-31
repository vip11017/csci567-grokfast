"""Build the H2 results table from experiments/report.py output.

    python analysis/h2_table.py results/h2/final_report.txt

Reads lines like
    h2_s0_adamw_none_wd10e+00 first95=1170 last=78830 val=0.8061
and prints a markdown table of steps to 95% val accuracy by weight decay (rows) and
optimizer x filter (columns): per-seed values and the median over seeds that reached
the threshold. The Adam / no filter / weight decay 0 cell comes from the replication
(results/replication), since the sweep doesn't rerun it.
"""
import re
import statistics
import sys
from collections import defaultdict

LINE = re.compile(r"^h2_s(\d)_(adamw?)_(none|ema)\S*?(?:_wd(\d)0e([+-]\d\d))?\s+first95=(\w+)\s+last=(\d+)")
REPLICATION_BASELINE = [56800, 79640, 48650]  # results/replication, seeds 0-2
WEIGHT_DECAYS = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
COLUMNS = [("adam", "none"), ("adam", "ema"), ("adamw", "none"), ("adamw", "ema")]


def parse(path):
    cells = defaultdict(dict)  # (opt, filter, wd) -> {seed: first95 or None}
    for line in open(path, encoding="utf-8"):
        m = LINE.match(line.strip())
        if not m:
            continue
        seed, opt, filt, mant, exp, first, _ = m.groups()
        wd = 0 if mant is None else round(int(mant) * 10 ** int(exp), 6)
        cells[(opt, filt, wd)][int(seed)] = None if first == "None" else int(first)
    for seed, steps in enumerate(REPLICATION_BASELINE):
        cells[("adam", "none", 0)][seed] = steps
    return cells


def fmt(results):
    if not results:
        return ""
    per_seed = "/".join("-" if results.get(s) is None else f"{results[s] / 1000:.1f}k" for s in sorted(results))
    reached = [v for v in results.values() if v is not None]
    median = f"**{statistics.median(reached) / 1000:.1f}k**" if len(reached) * 2 > len(results) else "**fail**"
    return f"{median} ({per_seed})"


def main(path):
    cells = parse(path)
    header = " | ".join(f"{opt.replace('adamw', 'AdamW').replace('adam', 'Adam')} {'+ EMA' if f == 'ema' else 'no filter'}" for opt, f in COLUMNS)
    print(f"| Weight decay | {header} |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    for wd in WEIGHT_DECAYS:
        row = [fmt(cells.get((opt, f, wd), {})) for opt, f in COLUMNS]
        if wd == 0:
            row[2] = row[3] = "(same as Adam)"
        print(f"| {wd:g} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
