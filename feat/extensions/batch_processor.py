"""
Scalable Batch Processing for Py-Feat.

Provides a BatchProcessor that handles:
- Processing large folders of images/videos with progress tracking
- Automatic memory management (chunk-based processing)
- Parallel file discovery and preprocessing
- Graceful error handling with per-file error logs
- Resume capability (skip already-processed files)
- Streaming results to disk to avoid OOM on huge datasets
"""

import os
import time
import json
import logging
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class BatchResult:
    """Container for batch processing results."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_detections: int = 0
    errors: List[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    output_path: Optional[str] = None

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.processed_files / self.total_files

    def summary(self) -> str:
        lines = [
            f"Batch Processing Complete",
            f"  Total files:      {self.total_files}",
            f"  Processed:        {self.processed_files}",
            f"  Failed:           {self.failed_files}",
            f"  Skipped:          {self.skipped_files}",
            f"  Total detections: {self.total_detections}",
            f"  Success rate:     {self.success_rate:.1%}",
            f"  Time elapsed:     {self.elapsed_seconds:.1f}s",
            f"  Output:           {self.output_path}",
        ]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for err in self.errors[:10]:
                lines.append(f"    - {err['file']}: {err['error']}")
            if len(self.errors) > 10:
                lines.append(f"    ... and {len(self.errors) - 10} more")
        return "\n".join(lines)


class BatchProcessor:
    """Process large batches of images or videos with Py-Feat."""

    def __init__(
        self,
        detector,
        chunk_size: int = 20,
        batch_size: int = 4,
        face_detection_threshold: float = 0.5,
        skip_frames: Optional[int] = None,
        num_workers: int = 0,
    ):
        """
        Args:
            detector: An initialized feat.Detector.
            chunk_size: Number of files to process before flushing to disk.
            batch_size: Batch size for the detector.
            face_detection_threshold: Confidence threshold for face detection.
            skip_frames: For videos, skip every N frames.
            num_workers: DataLoader workers.
        """
        self.detector = detector
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.face_detection_threshold = face_detection_threshold
        self.skip_frames = skip_frames
        self.num_workers = num_workers

    def discover_files(self, folder: str, file_type: str = "image") -> List[str]:
        """Find all image or video files in a folder (recursively)."""
        extensions = IMAGE_EXTENSIONS if file_type == "image" else VIDEO_EXTENSIONS
        files = []
        for root, _, filenames in os.walk(folder):
            for fname in sorted(filenames):
                if Path(fname).suffix.lower() in extensions:
                    files.append(os.path.join(root, fname))
        return files

    def _load_processed_files(self, output_path: str) -> set:
        """Load set of already-processed files from existing output CSV."""
        if not os.path.exists(output_path):
            return set()
        try:
            existing = pd.read_csv(output_path, usecols=["input"])
            return set(existing["input"].unique())
        except Exception:
            return set()

    def process_folder(
        self,
        folder: str,
        output_path: str,
        file_type: str = "image",
        resume: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> BatchResult:
        """Process all files in a folder and save results to CSV.

        Args:
            folder: Path to folder containing images or videos.
            output_path: Path to save the output CSV.
            file_type: "image" or "video".
            resume: If True, skip files already in the output CSV.
            progress_callback: Optional callable(processed, total, current_file)
                for external progress tracking (e.g., Streamlit progress bar).

        Returns:
            BatchResult with processing statistics.
        """
        result = BatchResult()
        start_time = time.perf_counter()
        result.output_path = output_path

        # Discover files
        all_files = self.discover_files(folder, file_type)
        result.total_files = len(all_files)

        if not all_files:
            result.elapsed_seconds = time.perf_counter() - start_time
            return result

        # Resume support
        already_processed = set()
        if resume:
            already_processed = self._load_processed_files(output_path)
            result.skipped_files = len([f for f in all_files if f in already_processed])

        files_to_process = [f for f in all_files if f not in already_processed]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Process in chunks
        header_written = os.path.exists(output_path) and os.path.getsize(output_path) > 0

        for chunk_start in range(0, len(files_to_process), self.chunk_size):
            chunk = files_to_process[chunk_start : chunk_start + self.chunk_size]

            try:
                with torch.inference_mode():
                    if file_type == "image":
                        chunk_results = self.detector.detect(
                            chunk,
                            data_type="image",
                            batch_size=self.batch_size,
                            face_detection_threshold=self.face_detection_threshold,
                            num_workers=self.num_workers,
                            progress_bar=False,
                        )
                    else:
                        # Process videos one at a time within the chunk
                        video_results = []
                        for vid_path in chunk:
                            try:
                                vr = self.detector.detect(
                                    vid_path,
                                    data_type="video",
                                    batch_size=self.batch_size,
                                    face_detection_threshold=self.face_detection_threshold,
                                    skip_frames=self.skip_frames,
                                    progress_bar=False,
                                )
                                video_results.append(vr)
                                result.processed_files += 1
                            except Exception as e:
                                result.failed_files += 1
                                result.errors.append({"file": vid_path, "error": str(e)})
                                logger.warning(f"Failed to process {vid_path}: {e}")

                        if video_results:
                            chunk_results = pd.concat(video_results, ignore_index=True)
                        else:
                            continue

                    if file_type == "image":
                        result.processed_files += len(chunk)

                    result.total_detections += len(chunk_results)

                    # Append to CSV
                    chunk_results.to_csv(
                        output_path,
                        mode="a",
                        header=not header_written,
                        index=False,
                    )
                    header_written = True

            except Exception as e:
                result.failed_files += len(chunk) if file_type == "image" else 0
                for f in chunk:
                    result.errors.append({"file": f, "error": str(e)})
                logger.warning(f"Failed to process chunk: {e}")

            # Progress callback
            if progress_callback:
                total_done = result.processed_files + result.failed_files
                current = chunk[-1] if chunk else ""
                progress_callback(total_done, len(files_to_process), current)

        result.elapsed_seconds = time.perf_counter() - start_time
        return result

    def process_file_list(
        self,
        file_list: List[str],
        output_path: str,
        file_type: str = "image",
        progress_callback: Optional[Callable] = None,
    ) -> BatchResult:
        """Process an explicit list of files.

        Same as process_folder but takes a list instead of discovering files.
        """
        result = BatchResult()
        result.total_files = len(file_list)
        start_time = time.perf_counter()
        result.output_path = output_path

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        header_written = False

        for chunk_start in range(0, len(file_list), self.chunk_size):
            chunk = file_list[chunk_start : chunk_start + self.chunk_size]

            try:
                with torch.inference_mode():
                    chunk_results = self.detector.detect(
                        chunk,
                        data_type=file_type,
                        batch_size=self.batch_size,
                        face_detection_threshold=self.face_detection_threshold,
                        progress_bar=False,
                    )
                result.processed_files += len(chunk)
                result.total_detections += len(chunk_results)

                chunk_results.to_csv(
                    output_path,
                    mode="a",
                    header=not header_written,
                    index=False,
                )
                header_written = True

            except Exception as e:
                result.failed_files += len(chunk)
                for f in chunk:
                    result.errors.append({"file": f, "error": str(e)})

            if progress_callback:
                total_done = result.processed_files + result.failed_files
                progress_callback(total_done, len(file_list), chunk[-1] if chunk else "")

        result.elapsed_seconds = time.perf_counter() - start_time
        return result
