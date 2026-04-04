"""
Simple baseline benchmark: image & video detection across CPU/CUDA
for comparing performance across different py-feat versions.
"""

import time, sys, os, itertools, gc
import torch

from feat.detector import Detector

# ── assets ──────────────────────────────────────────────────────────────
DATA = os.path.join(os.path.dirname(__file__), "feat", "tests", "data")
IMAGE_INPUT = [os.path.join(DATA, "single_face.jpg")]
VIDEO_INPUT = os.path.join(DATA, "single_face.mp4")

N_RUNS = 3          # repeats per config
SKIP_FRAMES = 24    # for video

# ── configs to test ─────────────────────────────────────────────────────
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

configs = []
for device in devices:
    configs.append({"device": device, "label": f"{device} | baseline"})


def run_once(detector, inputs, data_type, skip_frames=None):
    """Run a single detection and return wall-clock seconds."""
    kw = dict(data_type=data_type, progress_bar=False)
    if skip_frames is not None:
        kw["skip_frames"] = skip_frames

    start = time.perf_counter()
    with torch.inference_mode():
        detector.detect(inputs, **kw)
    return time.perf_counter() - start


def bench(label, detector, inputs, data_type, skip_frames=None):
    """Warm-up + N_RUNS timed runs; return stats dict."""
    # warm-up
    run_once(detector, inputs, data_type, skip_frames)

    times = []
    for _ in range(N_RUNS):
        t = run_once(detector, inputs, data_type, skip_frames)
        times.append(t)

    mean_t = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    return {"label": label, "mean": mean_t, "min": min_t, "max": max_t, "runs": times}


# ── main ────────────────────────────────────────────────────────────────
if __name__ != "__main__":
    raise SystemExit(0)

results = []

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Config: {cfg['label']}")
    print(f"  device={cfg['device']}")
    print(f"{'='*60}")

    # Build detector
    det = Detector(device=cfg["device"])

    # Image benchmark
    print("  [image] running …", end=" ", flush=True)
    r = bench(cfg["label"] + " | image", det, IMAGE_INPUT, "image")
    print(f"mean={r['mean']:.3f}s  min={r['min']:.3f}s  max={r['max']:.3f}s")
    results.append(r)

    # Video benchmark
    print("  [video] running …", end=" ", flush=True)
    r = bench(cfg["label"] + " | video", det, VIDEO_INPUT, "video", skip_frames=SKIP_FRAMES)
    print(f"mean={r['mean']:.3f}s  min={r['min']:.3f}s  max={r['max']:.3f}s")
    results.append(r)

    # Cleanup
    del det
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── summary table ───────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print(f"{'Configuration':<45} {'Mean':>8} {'Min':>8} {'Max':>8}")
print("-" * 80)
for r in results:
    print(f"{r['label']:<45} {r['mean']:>7.3f}s {r['min']:>7.3f}s {r['max']:>7.3f}s")
print("=" * 80)
