"""Profile each stage of the detection pipeline on GPU."""

import time
import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("Loading detector...")
from feat import Detector

detector = Detector(device=DEVICE)

# Warmup GPU
frame = torch.randn(1, 3, 480, 640)
with torch.inference_mode():
    for _ in range(3):
        faces_data = detector.detect_faces(frame, face_size=112)
        detector.forward(faces_data)
if DEVICE == "cuda":
    torch.cuda.synchronize()

print(f"\nProfiling pipeline on {DEVICE} with 640x480 input...")
print("=" * 60)

# Full pipeline timing (average of 10 runs)
times = []
for _ in range(10):
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        faces_data = detector.detect_faces(frame, face_size=112)
        fex = detector.forward(faces_data)
    if DEVICE == "cuda": torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

avg = np.mean(times) * 1000
std = np.std(times) * 1000
print(f"TOTAL pipeline (avg 10):    {avg:7.1f} ms  (±{std:.1f})")
print(f"Theoretical max FPS:        {1000/avg:7.1f}")

# Per-stage breakdown
print("\nPer-stage breakdown:")
print("-" * 60)

# Face detection
times_fd = []
for _ in range(10):
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        faces_data = detector.detect_faces(frame, face_size=112)
    if DEVICE == "cuda": torch.cuda.synchronize()
    times_fd.append(time.perf_counter() - t0)
print(f"Face detection (img2pose):  {np.mean(times_fd)*1000:7.1f} ms")

# Forward stages
extracted_faces = torch.cat([f["faces"] for f in faces_data], dim=0).to(DEVICE)
resmasknet_faces = torch.cat([f["resmasknet_faces"] for f in faces_data], dim=0).to(DEVICE)

with torch.inference_mode():
    # Landmarks
    times_lm = []
    for _ in range(10):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        landmarks = detector.landmark_detector.forward(extracted_faces)[0]
        if DEVICE == "cuda": torch.cuda.synchronize()
        times_lm.append(time.perf_counter() - t0)
    print(f"Landmarks (mobilefacenet):  {np.mean(times_lm)*1000:7.1f} ms")

    # Emotions
    times_emo = []
    for _ in range(10):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        emotions = detector.emotion_detector.forward(resmasknet_faces)
        if DEVICE == "cuda": torch.cuda.synchronize()
        times_emo.append(time.perf_counter() - t0)
    print(f"Emotions (resmasknet):      {np.mean(times_emo)*1000:7.1f} ms")

    # Identity
    times_id = []
    for _ in range(10):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        identity = detector.identity_detector.forward(extracted_faces)
        if DEVICE == "cuda": torch.cuda.synchronize()
        times_id.append(time.perf_counter() - t0)
    print(f"Identity (facenet):         {np.mean(times_id)*1000:7.1f} ms")

    # HOG + AU (CPU-bound)
    from feat.utils.image_operations import extract_hog_features
    faces_cpu = extracted_faces.cpu()
    landmarks_cpu = landmarks.cpu()
    times_hog = []
    for _ in range(10):
        t0 = time.perf_counter()
        hog_features, hog_landmarks = extract_hog_features(faces_cpu, landmarks_cpu)
        times_hog.append(time.perf_counter() - t0)
    print(f"HOG + AU (cpu):             {np.mean(times_hog)*1000:7.1f} ms")

    times_au = []
    for _ in range(10):
        t0 = time.perf_counter()
        aus = detector.au_detector.detect_au(frame=hog_features, landmarks=[hog_landmarks])
        times_au.append(time.perf_counter() - t0)
    print(f"AU detection (xgb):         {np.mean(times_au)*1000:7.1f} ms")

# DataFrame construction overhead
print("-" * 60)
with torch.inference_mode():
    faces_data = detector.detect_faces(frame, face_size=112)
    # Time just forward (includes DataFrame construction)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    fex = detector.forward(faces_data)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0
print(f"Full forward() w/ DataFrame:{t_fwd*1000:6.1f} ms")

# Test with half precision
print(f"\n\nTesting with half precision (float16)...")
print("=" * 60)
detector3 = Detector(device=DEVICE, half_precision=True)
with torch.inference_mode():
    for _ in range(3):
        fd = detector3.detect_faces(frame, face_size=112)
        detector3.forward(fd)
if DEVICE == "cuda": torch.cuda.synchronize()

times3 = []
for _ in range(10):
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        fd = detector3.detect_faces(frame, face_size=112)
        fex = detector3.forward(fd)
    if DEVICE == "cuda": torch.cuda.synchronize()
    times3.append(time.perf_counter() - t0)

avg3 = np.mean(times3) * 1000
print(f"TOTAL w/ half precision:    {avg3:7.1f} ms")
print(f"Theoretical max FPS:        {1000/avg3:7.1f}")
print(f"Speedup vs baseline:        {avg/avg3:5.2f}x")
