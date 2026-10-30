"""Print a compact summary of runs in a folder, or export downsampled curves.

    python experiments/report.py /content/h2            # one line per run
    python experiments/report.py /content/h2 --export   # gzip+base64 JSON of all curves

The export is for copying results out of a Colab runtime through cell output when
Google Drive can't be mounted. Decode it with:
    python experiments/report.py --decode export.txt --out results/h2
which writes one res_<name>.json per run.
"""
import argparse
import base64
import glob
import gzip
import json
import os

THRESHOLD = 0.95
MAX_POINTS = 300


def first_at(its, accs, threshold=THRESHOLD):
    return next((it for it, acc in zip(its, accs) if acc >= threshold), None)


def collect(folder):
    import torch

    runs = {}
    for path in sorted(glob.glob(os.path.join(folder, "res_*.pt"))):
        name = os.path.basename(path)[4:-3]
        r = torch.load(path, weights_only=True)
        step = max(1, len(r["its"]) // MAX_POINTS)
        keep = list(range(0, len(r["its"]), step))
        if keep[-1] != len(r["its"]) - 1:
            keep.append(len(r["its"]) - 1)
        runs[name] = {
            "first_95": first_at(r["its"], r["val_acc"]),
            "last_step": r["its"][-1],
            "its": [r["its"][i] for i in keep],
            "val_acc": [round(r["val_acc"][i], 4) for i in keep],
            "train_acc": [round(r["train_acc"][i], 4) for i in keep],
            "val_loss": [round(r["val_loss"][i], 4) for i in keep],
        }
    return runs


def main(args):
    if args.decode:
        data = json.loads(gzip.decompress(base64.b64decode(open(args.decode).read().strip())))
        os.makedirs(args.out, exist_ok=True)
        for name, run in data.items():
            with open(os.path.join(args.out, f"res_{name}.json"), "w") as f:
                json.dump(run, f)
        print(f"wrote {len(data)} runs to {args.out}")
        return

    runs = {k: v for k, v in collect(args.folder).items() if args.match in k}
    if args.export:
        blob = base64.b64encode(gzip.compress(json.dumps(runs).encode())).decode()
        print(f"EXPORT {len(runs)} runs {len(blob)} chars")
        print(blob)
        return
    for name, run in runs.items():
        print(f"{name} first95={run['first_95']} last={run['last_step']} val={run['val_acc'][-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--match", default="", help="only runs whose name contains this")
    parser.add_argument("--decode")
    parser.add_argument("--out", default="results/h2")
    main(parser.parse_args())
