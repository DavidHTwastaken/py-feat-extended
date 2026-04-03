"""
Computational Efficiency Optimizations for Py-Feat.

Provides an OptimizedDetector wrapper that adds:
- Automatic inference-mode context (torch.no_grad / torch.inference_mode)
- Optional half-precision (FP16) for GPU inference
- Automatic batch size tuning based on available memory
- Performance timing and benchmarking
"""

import time
import torch
import psutil
from contextlib import contextmanager, nullcontext as _nullcontext
from typing import Optional, List, Union
from pathlib import Path


class OptimizedDetector:
    """Wrapper around feat.Detector that adds performance optimizations."""

    def __init__(
        self,
        detector,
        use_half_precision: bool = False,
        auto_batch_size: bool = False,
        max_memory_usage: float = 0.8,
    ):
        """
        Args:
            detector: An initialized feat.Detector instance.
            use_half_precision: Convert model to FP16 (GPU only, faster but slightly less accurate).
            auto_batch_size: Automatically determine batch size based on available memory.
            max_memory_usage: Maximum fraction of available memory to use (0-1).
        """
        self.detector = detector
        self.use_half_precision = use_half_precision
        self.auto_batch_size = auto_batch_size
        self.max_memory_usage = max_memory_usage
        self._timings = []

        if use_half_precision and str(detector.device) != "cpu":
            self._convert_to_half()

    def _convert_to_half(self):
        """Enable FP16 via autocast for faster GPU inference.

        Rather than converting model weights directly (which causes dtype
        mismatches in ops like torchvision::nms that expect input and weight
        types to match), we set a flag and apply torch.autocast during
        detection. This lets CUDA kernels choose FP16 where safe while
        keeping ops like NMS in FP32 automatically.
        """
        # We no longer call self.detector.half() because it causes
        # dtype mismatches. Instead, autocast is applied in detect().
        pass

    def _estimate_batch_size(self, sample_size: tuple = (3, 512, 512)) -> int:
        """Estimate optimal batch size based on available memory."""
        if str(self.detector.device) == "cpu":
            available_mb = psutil.virtual_memory().available / (1024 ** 2)
            per_image_mb = 50  # rough estimate per image
        else:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                gpu_used = torch.cuda.memory_allocated(0)
                available_mb = (gpu_mem - gpu_used) / (1024 ** 2)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't expose memory info easily; use system RAM as proxy
                available_mb = psutil.virtual_memory().available / (1024 ** 2)
            else:
                available_mb = 4096  # default fallback
            per_image_mb = 100  # GPU images take more memory due to model activations

        usable_mb = available_mb * self.max_memory_usage
        batch = max(1, int(usable_mb / per_image_mb))
        return min(batch, 32)  # cap at 32

    @contextmanager
    def _timed(self, label: str):
        """Context manager to time operations."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self._timings.append({"label": label, "seconds": elapsed})

    def detect(
        self,
        inputs,
        data_type: str = "image",
        batch_size: Optional[int] = None,
        face_detection_threshold: float = 0.5,
        skip_frames: Optional[int] = None,
        progress_bar: bool = True,
        **kwargs,
    ):
        """Run detection with optimizations applied.

        Returns:
            tuple: (results_fex, timing_info_dict)
        """
        self._timings = []

        # Auto batch size
        if batch_size is None and self.auto_batch_size:
            with self._timed("batch_size_estimation"):
                batch_size = self._estimate_batch_size()
        elif batch_size is None:
            batch_size = 1

        # Run detection under inference mode (with autocast for half precision)
        with self._timed("detection"):
            with torch.inference_mode():
                ctx = torch.autocast("cuda") if self.use_half_precision and str(self.detector.device) != "cpu" else _nullcontext()
                with ctx:
                    results = self.detector.detect(
                        inputs,
                        data_type=data_type,
                        batch_size=batch_size,
                        face_detection_threshold=face_detection_threshold,
                        skip_frames=skip_frames,
                        progress_bar=progress_bar,
                        **kwargs,
                    )

        timing = self.get_timing_summary()
        return results, timing

    def get_timing_summary(self) -> dict:
        """Return a summary of timing information."""
        total = sum(t["seconds"] for t in self._timings)
        return {
            "total_seconds": total,
            "steps": self._timings,
            "settings": {
                "half_precision": self.use_half_precision,
                "auto_batch_size": self.auto_batch_size,
                "device": str(self.detector.device),
            },
        }

    def benchmark(self, image_path: str, n_runs: int = 5) -> dict:
        """Benchmark detection speed on a single image.

        Returns:
            dict with mean, min, max, std of processing time.
        """
        times = []
        for _ in range(n_runs):
            with torch.inference_mode():
                ctx = torch.autocast("cuda") if self.use_half_precision and str(self.detector.device) != "cpu" else _nullcontext()
                with ctx:
                    start = time.perf_counter()
                    self.detector.detect([image_path], data_type="image", progress_bar=False)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)

        return {
            "mean_seconds": sum(times) / len(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
            "std_seconds": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
            "n_runs": n_runs,
            "device": str(self.detector.device),
            "half_precision": self.use_half_precision,
        }
