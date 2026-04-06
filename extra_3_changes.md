  1. Coordinate inversion vectorization (detector.py:654-680)

  - Replaced the nested loop (per-frame × 68 landmarks × 2 coordinates = 136+ individual .loc[] assignments) with
  vectorized numpy operations
  - Pre-computes per-row padding/scale arrays via a frame-to-index mapping, then applies them to all columns at once
  using broadcasting
  - The landmark columns (x_0-x_67, y_0-y_67) are now adjusted in two bulk operations instead of 136 individual ones

  2. ImageDataset transform caching (data.py:2290-2302, 2322)

  - The Rescale + Compose transform is now created once in __init__ and reused, instead of being recreated on every
  __getitem__ call
  - Removed the logging.info() call that fired on every single frame — this was doing string formatting and I/O overhead
   per image

  3. torch.compile() enabled with guards (detector.py:119-123, 172-176, 247-251, 301-305)

  - All four neural network models (img2pose, landmark, emotion, identity) now get torch.compile() applied
  - Guarded by hasattr(torch, "compile") (requires PyTorch 2.0+) and excludes MPS devices (known compatibility issues)
  - Wrapped in try/except so it silently falls back to eager mode if compilation fails for any reason
  - First inference will be slower due to JIT compilation, but subsequent calls will be faster