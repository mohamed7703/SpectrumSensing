"""Enumerate attainable decisions on the frozen GAP scores; run from repository root."""
import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Write every tied-score operating point, the last feasible point and a figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/recovery/R005-runtime-stress/scores.npz"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    best = []
    columns = ["runtime", "gain_db", "snr_db", "threshold", "false_alarms", "noise_frames", "detections", "signal_frames"]
    with np.load(args.input, allow_pickle=False) as data, (args.out/"all_thresholds.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for runtime in ["native", "int8"]:
            for gain in [-1, 0, 1]:
                scores = data[f"gap_{runtime}_gain_{gain}"]
                noise = scores[data["labels"] == 0]
                for snr in [-10, 0, 10]:
                    signal = scores[(data["labels"] == 1) & (data["snr"] == snr)]
                    feasible = []
                    for threshold in np.r_[np.inf, np.unique(scores)[::-1]]:
                        fa = int((noise >= threshold).sum())
                        hits = int((signal >= threshold).sum())
                        row = [runtime, gain, snr, threshold, fa, len(noise), hits, len(signal)]
                        writer.writerow(row)
                        if fa <= .05 * len(noise):
                            feasible.append(row)
                    best.append(max(feasible, key=lambda r: (r[6], -r[4], r[3])))
    with (args.out/"best_points.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows([columns, *best])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True, constrained_layout=True)
    for ax, snr in zip(axes, [-10, 0, 10]):
        for mode, marker in [("native", "o"), ("int8", "s")]:
            rows = [r for r in best if r[0] == mode and r[2] == snr]
            ax.plot([r[1] for r in rows], [100*r[6]/r[7] for r in rows], marker=marker, label=mode.upper())
        ax.set(title=f"SNR {snr} dB", xlabel="Gain drift (dB)", xticks=[-1, 0, 1], ylim=(-3, 103))
    axes[0].set_ylabel("Best observed detection (%)")
    axes[-1].legend()
    fig.suptitle("Frozen synthetic GAP scores · at most 15 false alarms in 300 trials")
    fig.savefig(args.out/"feasibility.png", dpi=165)
    print("Score archive SHA-256", hashlib.sha256(args.input.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
