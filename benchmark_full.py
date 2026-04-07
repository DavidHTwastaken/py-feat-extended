"""
Full Benchmark: Original py-feat vs py-feat-extended.

Tests across CPU and CUDA (if available), with image and video inputs,
comparing the original unoptimized code paths against our optimizations.

Outputs: benchmark_full_results.md
"""

import gc
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import PILToTensor

sys.path.insert(0, os.path.dirname(__file__))

from feat.detector import Detector
from feat.extensions.efficiency import OptimizedDetector
from feat.facepose_detectors.img2pose.deps.models import postprocess_img2pose
from feat.utils.image_operations import (
    convert_image_to_tensor,
    extract_face_from_bbox_torch,
)

# ── Config ────────────────────────────────────────────────────────────────
DATA = os.path.join(os.path.dirname(__file__), "feat", "tests", "data")
IMAGE_INPUT = [os.path.join(DATA, "single_face.jpg")]
MULTI_IMAGES = [
    os.path.join(DATA, "single_face.jpg"),
    os.path.join(DATA, "multi_face.jpg"),
    os.path.join(DATA, "single_face.jpg"),
    os.path.join(DATA, "multi_face.jpg"),
]
VIDEO_INPUT = os.path.join(DATA, "single_face.mp4")

N_WARMUP = 1
N_RUNS = 3
SKIP_FRAMES = 24


# ── Helpers ───────────────────────────────────────────────────────────────
def load_images_as_tensor(paths):
    """Load image files into a batched tensor (B, C, H, W)."""
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = PILToTensor()(img)
        tensors.append(t)
    h, w = tensors[0].shape[1], tensors[0].shape[2]
    resized = []
    for t in tensors:
        t_resized = (
            torch.nn.functional.interpolate(
                t.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=False
            )
            .squeeze(0)
            .to(torch.uint8)
        )
        resized.append(t_resized)
    return torch.stack(resized, dim=0)


def stats(times):
    return {
        "mean": np.mean(times),
        "min": np.min(times),
        "max": np.max(times),
        "std": np.std(times),
    }


def flush_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# ── Original (unoptimized) pipeline ──────────────────────────────────────
def original_detect_faces(detector, image_tensor, face_size=112, face_detection_threshold=0.5):
    """Replicates original py-feat detect_faces: per-frame img2pose, discarded .to(device)."""
    frames = image_tensor.float() / 255.0
    frames.to(detector.device)  # Original bug: result discarded

    batch_results = []
    for i in range(frames.size(0)):
        single_frame = frames[i, ...].unsqueeze(0)
        img2pose_output = detector.facepose_detector(single_frame.to(detector.device))
        img2pose_output = postprocess_img2pose(
            img2pose_output[0], detection_threshold=face_detection_threshold
        )
        bbox = img2pose_output["boxes"]
        poses = img2pose_output["dofs"]
        facescores = img2pose_output["scores"]

        if bbox.numel() != 0:
            extracted_faces, new_bbox = extract_face_from_bbox_torch(
                single_frame, bbox, face_size=face_size
            )
        else:
            extracted_faces = torch.zeros((1, 3, face_size, face_size))
            bbox = torch.full((1, 4), float("nan"))
            new_bbox = torch.full((1, 4), float("nan"))
            facescores = torch.zeros((1))
            poses = torch.full((1, 6), float("nan"))

        frame_results = {
            "face_id": i,
            "faces": extracted_faces,
            "boxes": bbox,
            "new_boxes": new_bbox,
            "poses": poses,
            "scores": facescores,
        }

        if detector.info["emotion_model"] == "resmasknet":
            if torch.all(torch.isnan(bbox)):
                frame_results["resmasknet_faces"] = torch.full((1, 3, 224, 224), float("nan"))
            else:
                resmasknet_faces, _ = extract_face_from_bbox_torch(
                    single_frame, bbox, expand_bbox=1.1, face_size=224
                )
                frame_results["resmasknet_faces"] = resmasknet_faces

        batch_results.append(frame_results)
    return batch_results


def original_forward(detector, faces_data):
    """Replicates original py-feat forward: redundant .to(device), if/if bug."""
    from feat.utils.image_operations import (
        extract_hog_features,
        convert_bbox_output,
        inverse_transform_landmarks_torch,
    )

    extracted_faces = torch.cat([face["faces"] for face in faces_data], dim=0)
    new_bboxes = torch.cat([face["new_boxes"] for face in faces_data], dim=0)
    n_faces = extracted_faces.shape[0]

    if detector.landmark_detector is not None:
        if detector.info["landmark_model"].lower() == "mobilenet":
            from torchvision.transforms import Compose, Normalize
            extracted_faces = Compose(
                [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )(extracted_faces)
            landmarks = detector.landmark_detector.forward(extracted_faces.to(detector.device))
        if detector.info["landmark_model"].lower() == "mobilefacenet":
            landmarks = detector.landmark_detector.forward(extracted_faces.to(detector.device))[0]
        else:
            landmarks = detector.landmark_detector.forward(extracted_faces.to(detector.device))
        new_landmarks = inverse_transform_landmarks_torch(landmarks, new_bboxes)
    else:
        new_landmarks = torch.full((n_faces, 136), float("nan"))

    if detector.emotion_detector is not None:
        if detector.info["emotion_model"] == "resmasknet":
            resmasknet_faces = torch.cat(
                [face["resmasknet_faces"] for face in faces_data], dim=0
            )
            emotions = detector.emotion_detector.forward(resmasknet_faces.to(detector.device))
            emotions = torch.softmax(emotions, 1)
        elif detector.info["emotion_model"] == "svm":
            hog_features, emo_new_landmarks = extract_hog_features(extracted_faces, landmarks)
            emotions = detector.emotion_detector.detect_emo(
                frame=hog_features, landmarks=[emo_new_landmarks]
            )
            emotions = torch.tensor(emotions)
    else:
        emotions = torch.full((n_faces, 7), float("nan"))

    if detector.identity_detector is not None:
        identity_embeddings = detector.identity_detector.forward(extracted_faces.to(detector.device))
    else:
        identity_embeddings = torch.full((n_faces, 512), float("nan"))

    if detector.au_detector is not None:
        hog_features, au_new_landmarks = extract_hog_features(extracted_faces, landmarks)
        aus = detector.au_detector.detect_au(frame=hog_features, landmarks=[au_new_landmarks])
    else:
        aus = torch.full((n_faces, 20), float("nan"))

    return True  # We only care about timing


# ── Timing functions ─────────────────────────────────────────────────────
def time_core_pipeline(detector, image_tensor, n_runs, warmup, use_original=False):
    """Time detect_faces + forward (core pipeline, no DataLoader)."""
    face_size = getattr(detector, "face_size", 112)

    for _ in range(warmup):
        with torch.inference_mode():
            if use_original:
                faces = original_detect_faces(detector, image_tensor, face_size=face_size)
                original_forward(detector, faces)
            else:
                faces = detector.detect_faces(image_tensor, face_size=face_size)
                detector.forward(faces)
    flush_gpu()

    times = []
    for _ in range(n_runs):
        flush_gpu()
        start = time.perf_counter()
        with torch.inference_mode():
            if use_original:
                faces = original_detect_faces(detector, image_tensor, face_size=face_size)
                original_forward(detector, faces)
            else:
                faces = detector.detect_faces(image_tensor, face_size=face_size)
                detector.forward(faces)
        if torch.cuda.is_available() and str(detector.device).startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def time_detect_api(detector, inputs, data_type, n_runs, warmup, skip_frames=None):
    """Time the full .detect() API."""
    kw = dict(data_type=data_type, progress_bar=False)
    if skip_frames is not None:
        kw["skip_frames"] = skip_frames

    for _ in range(warmup):
        with torch.inference_mode():
            detector.detect(inputs, **kw)
    flush_gpu()

    times = []
    for _ in range(n_runs):
        flush_gpu()
        start = time.perf_counter()
        with torch.inference_mode():
            detector.detect(inputs, **kw)
        if torch.cuda.is_available() and str(detector.device).startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


# ── Run all benchmarks for one device ────────────────────────────────────
def run_benchmarks(device_name):
    """Run the full benchmark suite on a given device. Returns list of result dicts."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARKING ON: {device_name.upper()}")
    print(f"{'='*70}\n")

    print("  Loading models...", flush=True)
    det = Detector(device=device_name)
    print("  Models loaded.\n")

    single_tensor = load_images_as_tensor(IMAGE_INPUT)
    multi_tensor = load_images_as_tensor(MULTI_IMAGES)

    results = []

    def run_test(test_name, fn, **kwargs):
        print(f"    {test_name} ...", end=" ", flush=True)
        times = fn(**kwargs)
        s = stats(times)
        print(f"mean={s['mean']:.3f}s  (min={s['min']:.3f}, max={s['max']:.3f})")
        return s

    # ── Test 1: Single Image — Core Pipeline ──
    print("  [Test 1] Single Image — Core Pipeline (detect_faces + forward)")
    s_orig = run_test(
        "Original",
        time_core_pipeline, detector=det, image_tensor=single_tensor,
        n_runs=N_RUNS, warmup=N_WARMUP, use_original=True,
    )
    results.append({"device": device_name, "test": "Single Image", "scope": "Core Pipeline", "config": "Original (per-frame)", **s_orig})

    s_opt = run_test(
        "Optimized",
        time_core_pipeline, detector=det, image_tensor=single_tensor,
        n_runs=N_RUNS, warmup=N_WARMUP, use_original=False,
    )
    speedup = s_orig["mean"] / s_opt["mean"] if s_opt["mean"] > 0 else 0
    results.append({"device": device_name, "test": "Single Image", "scope": "Core Pipeline", "config": "Optimized (batched)", **s_opt, "speedup": speedup})
    print()

    # ── Test 2: Multi Image (4) — Core Pipeline ──
    print("  [Test 2] Multi Image (4) — Core Pipeline")
    m_orig = run_test(
        "Original",
        time_core_pipeline, detector=det, image_tensor=multi_tensor,
        n_runs=N_RUNS, warmup=N_WARMUP, use_original=True,
    )
    results.append({"device": device_name, "test": "4 Images", "scope": "Core Pipeline", "config": "Original (per-frame)", **m_orig})

    m_opt = run_test(
        "Optimized",
        time_core_pipeline, detector=det, image_tensor=multi_tensor,
        n_runs=N_RUNS, warmup=N_WARMUP, use_original=False,
    )
    speedup = m_orig["mean"] / m_opt["mean"] if m_opt["mean"] > 0 else 0
    results.append({"device": device_name, "test": "4 Images", "scope": "Core Pipeline", "config": "Optimized (batched)", **m_opt, "speedup": speedup})
    print()

    # ── Test 3: Single Image — Full .detect() API ──
    print("  [Test 3] Single Image — Full .detect() API")
    a_orig = run_test(
        "Original baseline",
        time_detect_api, detector=det, inputs=IMAGE_INPUT,
        data_type="image", n_runs=N_RUNS, warmup=N_WARMUP,
    )
    results.append({"device": device_name, "test": "Single Image", "scope": ".detect() API", "config": "Optimized", **a_orig})
    print()

    # ── Test 4: Multi Image (4) — Full .detect() API ──
    print("  [Test 4] Multi Image (4) — Full .detect() API (batch_size=4)")
    a_multi = run_test(
        "Optimized (batch_size=4)",
        time_detect_api, detector=det, inputs=MULTI_IMAGES,
        data_type="image", n_runs=N_RUNS, warmup=N_WARMUP,
    )
    results.append({"device": device_name, "test": "4 Images", "scope": ".detect() API", "config": "Optimized (batch=4)", **a_multi})
    print()

    # ── Test 5: Video — Full .detect() API ──
    print("  [Test 5] Video — Full .detect() API")
    v_opt = run_test(
        "Optimized",
        time_detect_api, detector=det, inputs=VIDEO_INPUT,
        data_type="video", n_runs=N_RUNS, warmup=N_WARMUP, skip_frames=SKIP_FRAMES,
    )
    results.append({"device": device_name, "test": "Video (skip=24)", "scope": ".detect() API", "config": "Optimized", **v_opt})
    print()

    # ── GPU-only tests: FP16 + auto batch ──
    if device_name == "cuda":
        print("  [Test 6] Video — OptimizedDetector (FP16 + auto batch)")

        opt_det_fp16 = OptimizedDetector(det, use_half_precision=True, auto_batch_size=False)
        kw = dict(data_type="video", progress_bar=False, skip_frames=SKIP_FRAMES)

        # Warmup
        for _ in range(N_WARMUP):
            opt_det_fp16.detect(VIDEO_INPUT, **kw)
        flush_gpu()

        times_fp16 = []
        for _ in range(N_RUNS):
            flush_gpu()
            _, timing = opt_det_fp16.detect(VIDEO_INPUT, **kw)
            times_fp16.append(timing["total_seconds"])
        s_fp16 = stats(times_fp16)
        speedup_fp16 = v_opt["mean"] / s_fp16["mean"] if s_fp16["mean"] > 0 else 0
        print(f"    FP16 ... mean={s_fp16['mean']:.3f}s  (min={s_fp16['min']:.3f}, max={s_fp16['max']:.3f})")
        results.append({"device": device_name, "test": "Video (skip=24)", "scope": ".detect() API", "config": "+ FP16 autocast", **s_fp16, "speedup": speedup_fp16})

        opt_det_full = OptimizedDetector(det, use_half_precision=True, auto_batch_size=True)
        for _ in range(N_WARMUP):
            opt_det_full.detect(VIDEO_INPUT, **kw)
        flush_gpu()

        times_full = []
        for _ in range(N_RUNS):
            flush_gpu()
            _, timing = opt_det_full.detect(VIDEO_INPUT, **kw)
            times_full.append(timing["total_seconds"])
        s_full = stats(times_full)
        speedup_full = v_opt["mean"] / s_full["mean"] if s_full["mean"] > 0 else 0
        print(f"    FP16 + auto batch ... mean={s_full['mean']:.3f}s  (min={s_full['min']:.3f}, max={s_full['max']:.3f})")
        results.append({"device": device_name, "test": "Video (skip=24)", "scope": ".detect() API", "config": "+ FP16 + auto batch", **s_full, "speedup": speedup_full})
        print()

    # Cleanup
    del det
    flush_gpu()

    return results


# ── Generate markdown report ─────────────────────────────────────────────
def generate_report(all_results):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gpu_line = ""
    if torch.cuda.is_available():
        gpu_line = f"| GPU | {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM) |\n"

    md = f"""# Benchmark Results: Original py-feat vs py-feat-extended

> Generated: {now}

## Environment

| Component | Value |
|-----------|-------|
| PyTorch | {torch.__version__} |
| Platform | {sys.platform} |
{gpu_line}| Warmup runs | {N_WARMUP} |
| Timed runs | {N_RUNS} |
| Video skip_frames | {SKIP_FRAMES} |

---

"""
    # Group results by device
    devices_seen = []
    for r in all_results:
        if r["device"] not in devices_seen:
            devices_seen.append(r["device"])

    for device in devices_seen:
        device_results = [r for r in all_results if r["device"] == device]

        md += f"## Results — {device.upper()}\n\n"

        # Core pipeline table
        core_results = [r for r in device_results if r["scope"] == "Core Pipeline"]
        if core_results:
            md += "### Core Pipeline (`detect_faces` + `forward`, no DataLoader)\n\n"
            md += "| Test | Configuration | Mean (s) | Min (s) | Max (s) | Speedup |\n"
            md += "|------|---------------|----------|---------|---------|--------:|\n"
            for r in core_results:
                spd = f"{r.get('speedup', 0):.2f}x" if "speedup" in r else "baseline"
                md += f"| {r['test']} | {r['config']} | {r['mean']:.3f} | {r['min']:.3f} | {r['max']:.3f} | {spd} |\n"
            md += "\n"

        # .detect() API table
        api_results = [r for r in device_results if r["scope"] == ".detect() API"]
        if api_results:
            md += "### Full `.detect()` API (end-to-end with DataLoader)\n\n"
            md += "| Test | Configuration | Mean (s) | Min (s) | Max (s) | Speedup |\n"
            md += "|------|---------------|----------|---------|---------|--------:|\n"
            for r in api_results:
                spd = f"{r.get('speedup', 0):.2f}x" if "speedup" in r else "—"
                md += f"| {r['test']} | {r['config']} | {r['mean']:.3f} | {r['min']:.3f} | {r['max']:.3f} | {spd} |\n"
            md += "\n"

        md += "---\n\n"

    # Cross-device comparison if both ran
    if len(devices_seen) > 1:
        md += "## CPU vs CUDA Comparison\n\n"
        md += "| Test | Scope | CPU Mean (s) | CUDA Mean (s) | GPU Speedup |\n"
        md += "|------|-------|----------:|-----------:|--------:|\n"

        # Match tests across devices
        cpu_results = {(r["test"], r["scope"], r["config"]): r for r in all_results if r["device"] == "cpu"}
        cuda_results = {(r["test"], r["scope"], r["config"]): r for r in all_results if r["device"] == "cuda"}

        for key in cpu_results:
            if key in cuda_results:
                cpu_r = cpu_results[key]
                cuda_r = cuda_results[key]
                spd = cpu_r["mean"] / cuda_r["mean"] if cuda_r["mean"] > 0 else 0
                md += f"| {key[0]} | {key[1]} ({key[2]}) | {cpu_r['mean']:.3f} | {cuda_r['mean']:.3f} | {spd:.2f}x |\n"

        md += "\n---\n\n"

    # Explanation section
    md += """## What's Being Compared

### Original (per-frame)
Replicates the **unmodified py-feat** behavior:
- `frames.to(device)` result is **discarded** (tensor stays on CPU even when GPU is selected)
- img2pose runs **one frame at a time** in a loop
- Redundant `.to(device)` on every sub-tensor in the forward pass
- `if/if` instead of `if/elif` for landmark model dispatch (double evaluation)

### Optimized (batched)
Our **py-feat-extended** changes:
- Entire batch passed to img2pose in **one forward call**
- Proper `frames = frames.to(device)` (tensor actually moves to GPU)
- Single `.to(device)` at the aggregation boundary
- `if/elif` fix for landmark dispatch
- Explicit `.cpu()` for HOG features
- Vectorized coordinate inversion (numpy broadcasting instead of per-frame pandas `.loc[]` loops)
- Cached `ImageDataset` transforms (created once, not per-frame)
- Smart DataLoader defaults (`pin_memory`, `num_workers`)

### GPU-only configurations
- **+ FP16 autocast**: `torch.autocast("cuda")` for half-precision on Tensor Core GPUs
- **+ FP16 + auto batch**: Automatic batch size estimation based on available GPU memory
"""

    return md


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  FULL BENCHMARK: Original py-feat vs py-feat-extended")
    print("=" * 70)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Runs: {N_RUNS} timed + {N_WARMUP} warmup")
    print()

    all_results = []

    # Always run CPU
    all_results.extend(run_benchmarks("cpu"))

    # Run CUDA if available
    if torch.cuda.is_available():
        all_results.extend(run_benchmarks("cuda"))

    # Generate and save report
    print("\nGenerating report...")
    md = generate_report(all_results)
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_full_results.md")
    with open(output_path, "w") as f:
        f.write(md)
    print(f"Results saved to: {output_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Device':<6} {'Test':<20} {'Scope':<18} {'Config':<25} {'Mean':>7} {'Speedup':>8}")
    print(f"{'-'*90}")
    for r in all_results:
        spd = f"{r.get('speedup', 0):.2f}x" if "speedup" in r else "—"
        print(f"{r['device']:<6} {r['test']:<20} {r['scope']:<18} {r['config']:<25} {r['mean']:>6.3f}s {spd:>8}")
