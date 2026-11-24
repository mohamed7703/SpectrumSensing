"""Summarize a 1 MS/s little-endian complex float32 USRP recording by second."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import welch


RATE = 1_000_000
FFT_SIZE = 16_384
TONE_HZ = -26_733.4


def db(power):
    return 10 * np.log10(power) if power > 0 else float("-inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    path = args.recording
    size = path.stat().st_size
    if size == 0 or size % 8:
        parser.error("Expected a nonempty file containing whole 8-byte I/Q samples.")
    output = path.with_suffix(".per_second.csv")
    fields = ["start_s", "samples", "nonfinite_samples", "near_rail_samples",
              "mean_digital_power_db", "median_psd_db_per_hz",
              "earlier_tone_band_pct", "strongest_bin_offset_hz"]
    with path.open("rb") as source, output.open("x", newline="") as report:
        writer = csv.DictWriter(report, fieldnames=fields)
        writer.writeheader()
        offset = 0
        while offset * 8 < size:
            samples = np.fromfile(source, dtype="<c8", count=min(RATE, size // 8 - offset))
            if not samples.size:
                raise OSError("Recording ended early. Run this on a completed capture.")
            x = samples.astype(np.complex128)
            bad = np.count_nonzero(~np.isfinite(x))
            near_rail = (np.abs(x.real) >= 0.99) | (np.abs(x.imag) >= 0.99)
            row = dict(start_s=offset // RATE, samples=x.size, nonfinite_samples=bad,
                       near_rail_samples=np.count_nonzero(near_rail))
            if bad == 0:
                row["mean_digital_power_db"] = db(np.mean(np.abs(x) ** 2))
                if x.size >= FFT_SIZE:
                    freq, psd = welch(x, fs=RATE, nperseg=FFT_SIZE, detrend=False,
                                      return_onesided=False, window="hann")
                    row["median_psd_db_per_hz"] = db(np.median(psd))
                    if psd.sum() > 0:
                        band = np.abs(freq - TONE_HZ) <= 200
                        row["earlier_tone_band_pct"] = 100 * psd[band].sum() / psd.sum()
                        row["strongest_bin_offset_hz"] = freq[np.argmax(psd)]
            writer.writerow(row)
            offset += x.size
    print(f"Analyzed {offset:,} samples. Saved {output}")


if __name__ == "__main__":
    main()
