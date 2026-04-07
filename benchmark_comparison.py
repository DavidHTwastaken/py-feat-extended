"""
Benchmark: Original py-feat vs py-feat-extended optimizations.

Compares detection speed under multiple configurations:
  1. Original (sequential per-frame face detection, no DataLoader tuning)
  2. Optimized (batched face detection, smart DataLoader defaults)
  3. Optimized + FP16 autocast  (GPU only)
  4. Optimized + auto batch size (GPU only)
  5. Optimized + FP16 + auto batch (GPU only)

Outputs a formatted results table to benchmark_results.md
"""

import copy
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

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


def load_images_as_tensor(paths):
    """Load image files into a batched tensor (B, C, H, W) for direct pipeline testing."""
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = PILToTensor()(img)  # (C, H, W) uint8
        tensors.append(t)
    # Resize all to same size (use first image's size) for batching
    h, w = tensors[0].shape[1], tensors[0].shape[2]
    resized = []
    for t in tensors:
        t_resized = torch.nn.functional.interpolate(
            t.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0).to(torch.uint8)
        resized.append(t_resized)
    return torch.stack(resized, dim=0)  # (B, C, H, W)

N_WARMUP = 1
N_RUNS = 3
SKIP_FRAMES = 24


# ── Original (unoptimized) detect_faces ──────────────────────────────────
def original_detect_faces(detector, image_tensor, face_size=112, face_detection_threshold=0.5):
    """
    Replicates the ORIGINAL py-feat detect_faces logic:
    - frames.to(device) result is DISCARDED (tensor stays on CPU)
    - img2pose runs ONE FRAME AT A TIME in a loop

    Args:
        image_tensor: pre-loaded (B, C, H, W) uint8 tensor
    """
    frames = image_tensor.float() / 255.0
    # Original bug: result of .to() was not assigned back
    frames.to(detector.device)

    batch_results = []
    for i in range(frames.size(0)):
        single_frame = frames[i, ...].unsqueeze(0)
        # Original: per-frame forward pass
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
                frame_results["resmasknet_faces"] = torch.full(
                    (1, 3, 224, 224), float("nan")
                )
            else:
                resmasknet_faces, _ = extract_face_from_bbox_torch(
                    single_frame, bbox, expand_bbox=1.1, face_size=224
                )
                frame_results["resmasknet_faces"] = resmasknet_faces

        batch_results.append(frame_results)

    return batch_results


# ── Original (unoptimized) forward pass ──────────────────────────────────
def original_forward(detector, faces_data):
    """
    Replicates the ORIGINAL py-feat forward logic:
    - No .to(device) on concatenated extracted_faces
    - Redundant .to(device) calls scattered everywhere
    - if/if instead of if/elif for landmark dispatch
    - No explicit .cpu() for hog features
    """
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
            landmarks = detector.landmark_detector.forward(
                extracted_faces.to(detector.device)
            )
        # Original bug: if instead of elif — always evaluates both
        if detector.info["landmark_model"].lower() == "mobilefacenet":
            landmarks = detector.landmark_detector.forward(
                extracted_faces.to(detector.device)
            )[0]
        else:
            landmarks = detector.landmark_detector.forward(
                extracted_faces.to(detector.device)
            )
        new_landmarks = inverse_transform_landmarks_torch(landmarks, new_bboxes)
    else:
        new_landmarks = torch.full((n_faces, 136), float("nan"))

    if detector.emotion_detector is not None:
        if detector.info["emotion_model"] == "resmasknet":
            resmasknet_faces = torch.cat(
                [face["resmasknet_faces"] for face in faces_data], dim=0
            )
            # Original: .to(device) on resmasknet_faces at forward time
            emotions = detector.emotion_detector.forward(resmasknet_faces.to(detector.device))
            emotions = torch.softmax(emotions, 1)
        elif detector.info["emotion_model"] == "svm":
            # Original: no .cpu() call
            hog_features, emo_new_landmarks = extract_hog_features(
                extracted_faces, landmarks
            )
            emotions = detector.emotion_detector.detect_emo(
                frame=hog_features, landmarks=[emo_new_landmarks]
            )
            emotions = torch.tensor(emotions)
    else:
        emotions = torch.full((n_faces, 7), float("nan"))

    if detector.identity_detector is not None:
        # Original: redundant .to(device)
        identity_embeddings = detector.identity_detector.forward(
            extracted_faces.to(detector.device)
        )
    else:
        identity_embeddings = torch.full((n_faces, 512), float("nan"))

    if detector.au_detector is not None:
        # Original: no .cpu() call
        hog_features, au_new_landmarks = extract_hog_features(
            extracted_faces, landmarks
        )
        aus = detector.au_detector.detect_au(
            frame=hog_features, landmarks=[au_new_landmarks]
        )
    else:
        aus = torch.full((n_faces, 20), float("nan"))

    # Original: .to(device) on each sub-tensor
    bboxes = torch.cat(
        [
            convert_bbox_output(
                face_output["new_boxes"].to(detector.device),
                face_output["scores"].to(detector.device),
            )
            for face_output in faces_data
        ],
        dim=0,
    )
    poses = torch.cat(
        [face_output["poses"].to(detector.device) for face_output in faces_data], dim=0
    )

    return {"emotions": emotions, "landmarks": new_landmarks, "aus": aus, "bboxes": bboxes, "poses": poses, "identities": identity_embeddings}


# ── Timing helpers ────────────────────────────────────────────────────────
def time_original_pipeline(detector, image_tensor, n_runs, warmup=1):
    """Time the original (unoptimized) detect_faces + forward pipeline."""
    face_size = getattr(detector, "face_size", 112)

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            faces = original_detect_faces(detector, image_tensor, face_size=face_size)
            original_forward(detector, faces)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.inference_mode():
            faces = original_detect_faces(detector, image_tensor, face_size=face_size)
            original_forward(detector, faces)
        times.append(time.perf_counter() - start)
    return times


def time_optimized_pipeline(detector, image_tensor, n_runs, warmup=1):
    """Time the optimized detect_faces + forward pipeline."""
    face_size = getattr(detector, "face_size", 112)

    for _ in range(warmup):
        with torch.inference_mode():
            faces = detector.detect_faces(image_tensor, face_size=face_size)
            detector.forward(faces)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.inference_mode():
            faces = detector.detect_faces(image_tensor, face_size=face_size)
            detector.forward(faces)
        times.append(time.perf_counter() - start)
    return times


def time_detect_api(detector, inputs, data_type, n_runs, warmup=1, skip_frames=None):
    """Time the full .detect() API (includes DataLoader)."""
    kw = dict(data_type=data_type, progress_bar=False)
    if skip_frames is not None:
        kw["skip_frames"] = skip_frames

    for _ in range(warmup):
        with torch.inference_mode():
            detector.detect(inputs, **kw)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.inference_mode():
            detector.detect(inputs, **kw)
        times.append(time.perf_counter() - start)
    return times


def time_optimized_detect_api(opt_det, inputs, data_type, n_runs, warmup=1, skip_frames=None):
    """Time the OptimizedDetector .detect() wrapper."""
    kw = dict(data_type=data_type, progress_bar=False)
    if skip_frames is not None:
        kw["skip_frames"] = skip_frames

    for _ in range(warmup):
        opt_det.detect(inputs, **kw)

    times = []
    for _ in range(n_runs):
        _, timing = opt_det.detect(inputs, **kw)
        times.append(timing["total_seconds"])
    return times


def stats(times):
    mean = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    return mean, mn, mx


def speedup_str(baseline_mean, optimized_mean):
    if optimized_mean == 0 or baseline_mean == 0:
        return "N/A"
    ratio = baseline_mean / optimized_mean
    return f"{ratio:.2f}x"


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ != "__main__":
    raise SystemExit(0)

has_cuda = torch.cuda.is_available()
device = "cuda" if has_cuda else "cpu"

print(f"Device: {device}")
if has_cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Runs per config: {N_RUNS} (+ {N_WARMUP} warmup)")
print()

# Build detector once
print("Loading models...", flush=True)
det = Detector(device=device)
print("Models loaded.\n")

# Pre-load images as tensors for core pipeline tests
print("Pre-loading images as tensors...")
single_tensor = load_images_as_tensor(IMAGE_INPUT)
multi_tensor = load_images_as_tensor(MULTI_IMAGES)
print(f"  Single: {single_tensor.shape}  Multi: {multi_tensor.shape}\n")

results = []

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Single Image — Core Pipeline (detect_faces + forward)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Single Image — Core Pipeline")
print("=" * 60)

print("  [original] ...", end=" ", flush=True)
t = time_original_pipeline(det, single_tensor, N_RUNS, N_WARMUP)
orig_single_mean, orig_single_min, orig_single_max = stats(t)
print(f"mean={orig_single_mean:.3f}s")
results.append(("Single Image — Core Pipeline", "Original (per-frame)", orig_single_mean, orig_single_min, orig_single_max, "—"))

print("  [optimized] ...", end=" ", flush=True)
t = time_optimized_pipeline(det, single_tensor, N_RUNS, N_WARMUP)
opt_single_mean, opt_single_min, opt_single_max = stats(t)
print(f"mean={opt_single_mean:.3f}s")
results.append(("Single Image — Core Pipeline", "Optimized (batched)", opt_single_mean, opt_single_min, opt_single_max, speedup_str(orig_single_mean, opt_single_mean)))

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Multi Image (4 images) — Core Pipeline
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: Multi Image (4 images) — Core Pipeline")
print("=" * 60)

print("  [original] ...", end=" ", flush=True)
t = time_original_pipeline(det, multi_tensor, N_RUNS, N_WARMUP)
orig_multi_mean, orig_multi_min, orig_multi_max = stats(t)
print(f"mean={orig_multi_mean:.3f}s")
results.append(("Multi Image (4) — Core Pipeline", "Original (per-frame)", orig_multi_mean, orig_multi_min, orig_multi_max, "—"))

print("  [optimized] ...", end=" ", flush=True)
t = time_optimized_pipeline(det, multi_tensor, N_RUNS, N_WARMUP)
opt_multi_mean, opt_multi_min, opt_multi_max = stats(t)
print(f"mean={opt_multi_mean:.3f}s")
results.append(("Multi Image (4) — Core Pipeline", "Optimized (batched)", opt_multi_mean, opt_multi_min, opt_multi_max, speedup_str(orig_multi_mean, opt_multi_mean)))

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Full .detect() API — Single Image
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Full .detect() API — Single Image")
print("=" * 60)

print("  [optimized baseline] ...", end=" ", flush=True)
t = time_detect_api(det, IMAGE_INPUT, "image", N_RUNS, N_WARMUP)
api_img_mean, api_img_min, api_img_max = stats(t)
print(f"mean={api_img_mean:.3f}s")
results.append(("Single Image — .detect() API", "Optimized baseline", api_img_mean, api_img_min, api_img_max, "—"))

if has_cuda:
    opt_det = OptimizedDetector(det, use_half_precision=True, auto_batch_size=False)
    print("  [+ FP16 autocast] ...", end=" ", flush=True)
    t = time_optimized_detect_api(opt_det, IMAGE_INPUT, "image", N_RUNS, N_WARMUP)
    fp16_img_mean, fp16_img_min, fp16_img_max = stats(t)
    print(f"mean={fp16_img_mean:.3f}s")
    results.append(("Single Image — .detect() API", "+ FP16 autocast", fp16_img_mean, fp16_img_min, fp16_img_max, speedup_str(api_img_mean, fp16_img_mean)))

    opt_det2 = OptimizedDetector(det, use_half_precision=True, auto_batch_size=True)
    print("  [+ FP16 + auto batch] ...", end=" ", flush=True)
    t = time_optimized_detect_api(opt_det2, IMAGE_INPUT, "image", N_RUNS, N_WARMUP)
    fp16ab_img_mean, fp16ab_img_min, fp16ab_img_max = stats(t)
    print(f"mean={fp16ab_img_mean:.3f}s")
    results.append(("Single Image — .detect() API", "+ FP16 + auto batch", fp16ab_img_mean, fp16ab_img_min, fp16ab_img_max, speedup_str(api_img_mean, fp16ab_img_mean)))

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Full .detect() API — Video
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: Full .detect() API — Video")
print("=" * 60)

print("  [optimized baseline] ...", end=" ", flush=True)
t = time_detect_api(det, VIDEO_INPUT, "video", N_RUNS, N_WARMUP, skip_frames=SKIP_FRAMES)
api_vid_mean, api_vid_min, api_vid_max = stats(t)
print(f"mean={api_vid_mean:.3f}s")
results.append(("Video — .detect() API", "Optimized baseline", api_vid_mean, api_vid_min, api_vid_max, "—"))

if has_cuda:
    opt_det = OptimizedDetector(det, use_half_precision=True, auto_batch_size=False)
    print("  [+ FP16 autocast] ...", end=" ", flush=True)
    t = time_optimized_detect_api(opt_det, VIDEO_INPUT, "video", N_RUNS, N_WARMUP, skip_frames=SKIP_FRAMES)
    fp16_vid_mean, fp16_vid_min, fp16_vid_max = stats(t)
    print(f"mean={fp16_vid_mean:.3f}s")
    results.append(("Video — .detect() API", "+ FP16 autocast", fp16_vid_mean, fp16_vid_min, fp16_vid_max, speedup_str(api_vid_mean, fp16_vid_mean)))

    opt_det2 = OptimizedDetector(det, use_half_precision=True, auto_batch_size=True)
    print("  [+ FP16 + auto batch] ...", end=" ", flush=True)
    t = time_optimized_detect_api(opt_det2, VIDEO_INPUT, "video", N_RUNS, N_WARMUP, skip_frames=SKIP_FRAMES)
    fp16ab_vid_mean, fp16ab_vid_min, fp16ab_vid_max = stats(t)
    print(f"mean={fp16ab_vid_mean:.3f}s")
    results.append(("Video — .detect() API", "+ FP16 + auto batch", fp16ab_vid_mean, fp16ab_vid_min, fp16ab_vid_max, speedup_str(api_vid_mean, fp16ab_vid_mean)))

# ═══════════════════════════════════════════════════════════════════════════
# Write results to markdown
# ═══════════════════════════════════════════════════════════════════════════
print("\n\nWriting results to benchmark_results.md ...")

gpu_info = ""
if has_cuda:
    gpu_info = f"  - **GPU:** {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)\n"

md = f"""# Benchmark Results: Original py-feat vs py-feat-extended

> Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Environment

  - **Device:** {device.upper()}
{gpu_info}  - **PyTorch:** {torch.__version__}
  - **Runs per config:** {N_RUNS} (+ {N_WARMUP} warmup)
  - **Video skip_frames:** {SKIP_FRAMES}

---

## Results

| Test | Configuration | Mean (s) | Min (s) | Max (s) | Speedup |
|------|---------------|----------|---------|---------|---------|
"""

current_test = None
for test, config, mean, mn, mx, spd in results:
    if test != current_test:
        if current_test is not None:
            md += f"| | | | | | |\n"
        current_test = test
    md += f"| {test} | {config} | {mean:.3f} | {mn:.3f} | {mx:.3f} | {spd} |\n"

md += f"""
---

## What Each Configuration Tests

### Core Pipeline Tests (Tests 1 & 2)
These isolate the **detect_faces + forward** pipeline (no DataLoader overhead):

| Configuration | Description |
|---------------|-------------|
| **Original (per-frame)** | Replicates the original py-feat behavior: img2pose runs one frame at a time in a loop, `frames.to(device)` result is discarded (tensor stays on CPU), redundant `.to(device)` calls on every sub-tensor, `if/if` instead of `if/elif` for landmark dispatch |
| **Optimized (batched)** | Our changes: entire batch passed to img2pose in one forward call, proper `frames = frames.to(device)`, single `.to(device)` at aggregation, `if/elif` fix, explicit `.cpu()` for HOG features |

### Full .detect() API Tests (Tests 3 & 4)
These test the **complete detection pipeline** including DataLoader with smart defaults:

| Configuration | Description |
|---------------|-------------|
| **Optimized baseline** | All core optimizations + smart DataLoader (`pin_memory=True` on CUDA, auto `num_workers`) |
| **+ FP16 autocast** | Adds `torch.autocast("cuda")` for half-precision inference on Tensor Core GPUs |
| **+ FP16 + auto batch** | Adds automatic batch size estimation based on available GPU/system memory |

---

## Key Optimizations Benchmarked

1. **Batched img2pose inference** — Single GPU forward pass for entire batch vs N separate calls
2. **Device transfer fix** — `frames = frames.to(device)` actually moves tensor to GPU (original discarded the result)
3. **Redundant transfer elimination** — Tensors placed on device once at aggregation boundary
4. **Smart DataLoader defaults** — `pin_memory=True` + parallel `num_workers` on CUDA
5. **FP16 autocast** — Half-precision via `torch.autocast("cuda")` for ~2x throughput on Tensor Core GPUs
6. **Auto batch sizing** — Memory-aware batch size to maximize throughput without OOM
7. **`torch.inference_mode()`** — Disables autograd tracking for lower memory and faster passes
"""

output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.md")
with open(output_path, "w") as f:
    f.write(md)

print(f"Done! Results saved to: {output_path}")

# Also print the table to console
print("\n")
print(f"{'Test':<35} {'Configuration':<25} {'Mean':>8} {'Min':>8} {'Max':>8} {'Speedup':>8}")
print("-" * 95)
for test, config, mean, mn, mx, spd in results:
    print(f"{test:<35} {config:<25} {mean:>7.3f}s {mn:>7.3f}s {mx:>7.3f}s {spd:>8}")
