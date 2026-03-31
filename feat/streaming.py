"""
Streaming video processing infrastructure for real-time facial expression analysis.

Provides threaded frame decoding, lightweight result objects, and adaptive
frame skipping to enable 60fps real-time video processing.
"""

import dataclasses
import time
import threading
import queue
from typing import Iterator, Optional, Callable, Union

import numpy as np
import torch

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import av

    HAS_AV = True
except ImportError:
    HAS_AV = False


@dataclasses.dataclass
class FaceResult:
    """Lightweight container for a single detected face's results."""

    bbox: np.ndarray  # (5,) — FaceRectX, FaceRectY, FaceRectWidth, FaceRectHeight, FaceScore
    landmarks: np.ndarray  # (136,) — x_0..x_67, y_0..y_67
    pose: np.ndarray  # (6,) — Pitch, Roll, Yaw, X, Y, Z
    aus: np.ndarray  # (20,) — action unit intensities
    emotions: np.ndarray  # (7,) — anger, disgust, fear, happiness, sadness, surprise, neutral
    identity_embedding: np.ndarray  # (512,) — face identity embedding


@dataclasses.dataclass
class StreamingResult:
    """Lightweight container for per-frame detection results.

    Much faster to construct than a full Fex DataFrame. Use `to_fex()` on
    a list of StreamingResults when you need DataFrame-based analysis.
    """

    frame_idx: int
    timestamp: float  # seconds
    faces: list  # list[FaceResult]

    @staticmethod
    def to_fex(results: list) -> "Fex":
        """Convert a list of StreamingResult into a Fex DataFrame.

        Args:
            results: List of StreamingResult objects to convert.

        Returns:
            Fex DataFrame compatible with existing Py-FEAT analysis methods.
        """
        from feat.data import Fex
        from feat.pretrained import AU_LANDMARK_MAP
        from feat.utils import (
            openface_2d_landmark_columns,
            FEAT_EMOTION_COLUMNS,
            FEAT_FACEBOX_COLUMNS,
            FEAT_FACEPOSE_COLUMNS_6D,
            FEAT_IDENTITY_COLUMNS,
        )
        import pandas as pd

        rows = []
        for result in results:
            for face in result.faces:
                row = {}
                # Facebox
                for col, val in zip(FEAT_FACEBOX_COLUMNS, face.bbox):
                    row[col] = val
                # Landmarks
                for col, val in zip(openface_2d_landmark_columns, face.landmarks):
                    row[col] = val
                # Pose
                for col, val in zip(FEAT_FACEPOSE_COLUMNS_6D, face.pose):
                    row[col] = val
                # AUs
                for col, val in zip(AU_LANDMARK_MAP["Feat"], face.aus):
                    row[col] = val
                # Emotions
                for col, val in zip(FEAT_EMOTION_COLUMNS, face.emotions):
                    row[col] = val
                # Identity
                for col, val in zip(FEAT_IDENTITY_COLUMNS[1:], face.identity_embedding):
                    row[col] = val
                row["frame"] = result.frame_idx
                rows.append(row)

        if not rows:
            return Fex()

        df = pd.DataFrame(rows)
        return Fex(
            df,
            au_columns=AU_LANDMARK_MAP["Feat"],
            emotion_columns=FEAT_EMOTION_COLUMNS,
            facebox_columns=FEAT_FACEBOX_COLUMNS,
            landmark_columns=openface_2d_landmark_columns,
            facepose_columns=FEAT_FACEPOSE_COLUMNS_6D,
            identity_columns=FEAT_IDENTITY_COLUMNS[1:],
            detector="Feat",
        )


class StreamingVideoSource:
    """Threaded video frame producer for real-time processing.

    Decodes video frames in a background thread and pushes them into a queue,
    allowing the main thread to run model inference concurrently with I/O.

    Supports both video files (via PyAV) and webcam capture (via OpenCV).

    Args:
        source: File path (str) for video files, or integer device index for webcam.
        max_queue_size: Maximum number of frames to buffer. When full, adaptive
            frame skipping kicks in.
        skip_frames: Fixed frame skip interval. None = process every frame.
        target_fps: Target FPS cap. None = no cap (process as fast as possible).
        output_tensor: If True, pre-convert frames to torch.Tensor in the decode
            thread to overlap conversion with GPU inference.
    """

    def __init__(
        self,
        source: Union[str, int],
        max_queue_size: int = 60,
        skip_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
        output_tensor: bool = True,
    ):
        self.source = source
        self.max_queue_size = max_queue_size
        self.skip_frames = skip_frames
        self.target_fps = target_fps
        self.output_tensor = output_tensor

        self._queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread = None
        self._is_webcam = isinstance(source, int)
        self._fps = 0.0
        self._source_fps = 0.0
        self._total_frames = 0
        self._adaptive_skip = 0
        self._lock = threading.Lock()

    @property
    def fps(self) -> float:
        """Current achieved decode FPS."""
        return self._fps

    @property
    def source_fps(self) -> float:
        """Source video's native FPS (0 for webcam)."""
        return self._source_fps

    @property
    def total_frames(self) -> int:
        """Total frames in source (0 for webcam/live)."""
        return self._total_frames

    def start(self):
        """Start the background decode thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._decode_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Signal the decode thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _decode_loop(self):
        """Main decode loop running in background thread."""
        if self._is_webcam:
            self._decode_webcam()
        else:
            self._decode_video_file()

    def _decode_webcam(self):
        """Decode frames from webcam using OpenCV."""
        if not HAS_CV2:
            raise ImportError(
                "opencv-python is required for webcam capture. "
                "Install with: pip install opencv-python"
            )

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam device {self.source}")

        try:
            self._source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = 0
            min_interval = 1.0 / self.target_fps if self.target_fps else 0
            last_time = time.monotonic()

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # Fixed frame skip
                if self.skip_frames and frame_idx % (self.skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                # Adaptive frame skip
                with self._lock:
                    adaptive_skip = self._adaptive_skip
                if adaptive_skip > 0 and frame_idx % (adaptive_skip + 1) != 0:
                    frame_idx += 1
                    continue

                # FPS cap
                if min_interval > 0:
                    elapsed = time.monotonic() - last_time
                    if elapsed < min_interval:
                        frame_idx += 1
                        continue

                timestamp = frame_idx / self._source_fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.output_tensor:
                    tensor = (
                        torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                    )
                    data = tensor
                else:
                    data = frame_rgb

                # Adaptive back-pressure: if queue is almost full, increase skip
                qsize = self._queue.qsize()
                with self._lock:
                    if qsize > self.max_queue_size * 0.8:
                        self._adaptive_skip = min(self._adaptive_skip + 1, 10)
                    elif qsize < self.max_queue_size * 0.2:
                        self._adaptive_skip = max(self._adaptive_skip - 1, 0)

                try:
                    self._queue.put(
                        (frame_idx, timestamp, data), timeout=0.1
                    )
                except queue.Full:
                    pass  # Drop frame if queue is full

                now = time.monotonic()
                if now > last_time:
                    self._fps = 1.0 / (now - last_time)
                last_time = now
                frame_idx += 1
        finally:
            cap.release()
            self._queue.put(None)  # Sentinel

    def _decode_video_file(self):
        """Decode frames from a video file using PyAV."""
        if not HAS_AV:
            raise ImportError(
                "PyAV is required for video file processing. "
                "Install with: pip install av"
            )

        container = av.open(str(self.source))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"  # Enable multithreaded decode
            self._source_fps = float(stream.average_rate or stream.rate or 30)
            self._total_frames = stream.frames or 0

            frame_idx = 0
            min_interval = 1.0 / self.target_fps if self.target_fps else 0
            last_time = time.monotonic()

            for av_frame in container.decode(stream):
                if self._stop_event.is_set():
                    break

                # Fixed frame skip
                if self.skip_frames and frame_idx % (self.skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                # Adaptive frame skip
                with self._lock:
                    adaptive_skip = self._adaptive_skip
                if adaptive_skip > 0 and frame_idx % (adaptive_skip + 1) != 0:
                    frame_idx += 1
                    continue

                timestamp = float(av_frame.pts * stream.time_base) if av_frame.pts else frame_idx / self._source_fps

                frame_rgb = av_frame.to_ndarray(format="rgb24")

                if self.output_tensor:
                    tensor = (
                        torch.from_numpy(frame_rgb.copy())
                        .permute(2, 0, 1)
                        .float()
                        / 255.0
                    )
                    data = tensor
                else:
                    data = frame_rgb

                # Adaptive back-pressure
                qsize = self._queue.qsize()
                with self._lock:
                    if qsize > self.max_queue_size * 0.8:
                        self._adaptive_skip = min(self._adaptive_skip + 1, 10)
                    elif qsize < self.max_queue_size * 0.2:
                        self._adaptive_skip = max(self._adaptive_skip - 1, 0)

                try:
                    self._queue.put(
                        (frame_idx, timestamp, data), timeout=0.5
                    )
                except queue.Full:
                    pass  # Drop frame

                now = time.monotonic()
                if now > last_time:
                    self._fps = 1.0 / (now - last_time)
                last_time = now
                frame_idx += 1
        finally:
            container.close()
            self._queue.put(None)  # Sentinel

    def __iter__(self) -> Iterator:
        """Iterate over decoded frames. Yields (frame_idx, timestamp, data)."""
        self.start()
        while True:
            try:
                item = self._queue.get(timeout=5.0)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            if item is None:  # Sentinel
                break
            yield item

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()
