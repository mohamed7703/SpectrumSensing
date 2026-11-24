"""Fetch verified HTTP ranges from the shared capture, keeping completed bytes on retry."""
import concurrent.futures as futures
import os
from pathlib import Path
import time
import urllib.request

os.environ.pop("SSLKEYLOGFILE", None)
ROOT = Path(__file__).resolve().parent
URL = "https://drive.usercontent.google.com/download?id=1_pmfCVqJa41YR3L2EwPTCnhy2a_BL8Po&export=download&confirm=t"
SIZE = 1855763448
SECOND = 8000000
PREFIX = 524288
PARTS = ROOT / "ranges"


def get_range(bounds):
    start, stop = bounds
    path = PARTS / f"{start}-{stop}.bin"
    for attempt in range(4):
        have = path.stat().st_size if path.exists() else 0
        if have == stop-start:
            return True
        first = start + have
        request = urllib.request.Request(URL, headers={"Range": f"bytes={first}-{stop-1}"})
        try:
            with urllib.request.urlopen(request, timeout=16) as res:
                if res.status != 206 or res.headers.get("Content-Range") != f"bytes {first}-{stop-1}/{SIZE}":
                    raise ValueError("Unexpected remote range or size")
                with path.open("ab") as out:
                    while chunk := res.read(32768):
                        out.write(chunk)
            if path.stat().st_size == stop-start:
                return True
        except (OSError, ValueError) as exc:
            if attempt == 3:
                print(start, str(exc), flush=True)
            time.sleep(attempt+0.25)
    return False


def main():
    PARTS.mkdir(exist_ok=True)
    prefixes = []
    rest = []
    for start in range(0, SIZE, SECOND):
        end = min(start+SECOND, SIZE)
        split = min(start+PREFIX, end)
        prefixes.append((start, split))
        for pos in range(split, end, 2_000_000):
            rest.append((pos, min(pos+2_000_000, end)))
    with futures.ThreadPoolExecutor(max_workers=24) as pool:
        for label, ranges in [("prefixes",prefixes),("remaining",rest)]:
            jobs = [pool.submit(get_range, r) for r in ranges]
            ok = 0
            for i, job in enumerate(futures.as_completed(jobs),1):
                ok += job.result()
                if i%100 == 0 or i == len(jobs):
                    print(label,i, "of",len(jobs), "valid",ok,flush=True)
                if i == 20 and ok == 0:
                    raise RuntimeError("No successful ranges in first 20")
    ranges = sorted(prefixes+rest)
    if any(not (PARTS/f"{a}-{b}.bin").exists() or (PARTS/f"{a}-{b}.bin").stat().st_size != b-a for a,b in ranges):
        raise RuntimeError("Incomplete capture; rerun to resume.")
    import shutil
    with (ROOT/"ampient_turned_off.iq.partial").open("wb") as out:
        for a,b in ranges:
            with (PARTS/f"{a}-{b}.bin").open("rb") as src:
                shutil.copyfileobj(src,out)
    (ROOT/"ampient_turned_off.iq.partial").replace(ROOT/"ampient_turned_off.iq")


if __name__ == "__main__":
    main()
