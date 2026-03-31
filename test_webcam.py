"""Quick webcam test for the real-time streaming pipeline."""

import time
import cv2
import numpy as np
import torch
from feat import Detector

EMOTION_NAMES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Initializing Detector on {DEVICE}...")
    detector = Detector(device=DEVICE, half_precision=(DEVICE == "cuda"))
    print("Detector ready. Opening webcam...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    fps_start = time.monotonic()
    display_fps = 0.0
    detect_ms = 0.0

    print(f"Running on {DEVICE}... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR to RGB tensor for detector
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

        # Run detection using fast raw path (no DataFrame)
        t0 = time.perf_counter()
        with torch.inference_mode():
            faces_data = detector.detect_faces(
                frame_tensor.unsqueeze(0),
                face_size=detector.face_size if hasattr(detector, "face_size") else 112,
                face_detection_threshold=0.5,
            )
            raw = detector.forward_raw(faces_data)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        detect_ms = (time.perf_counter() - t0) * 1000

        # Calculate FPS
        frame_count += 1
        elapsed = time.monotonic() - fps_start
        if elapsed >= 1.0:
            display_fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.monotonic()

        # Draw FPS and timing
        has_face = not np.all(np.isnan(raw["bboxes"]))
        n_faces = raw["bboxes"].shape[0] if has_face else 0
        cv2.putText(
            frame,
            f"FPS: {display_fps:.1f} | Detect: {detect_ms:.0f}ms | Faces: {n_faces}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Draw per-face results
        if has_face:
            for i in range(raw["bboxes"].shape[0]):
                fx, fy, fw, fh = raw["bboxes"][i, :4]
                if np.isnan(fx):
                    continue
                x, y, w, h = int(fx), int(fy), int(fw), int(fh)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Emotion label
                emotions = raw["emotions"][i]
                if not np.any(np.isnan(emotions)):
                    emo_idx = np.argmax(emotions)
                    emo_name = EMOTION_NAMES[emo_idx]
                    emo_conf = emotions[emo_idx]
                    cv2.putText(
                        frame,
                        f"{emo_name} ({emo_conf:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                # Key landmarks
                lm = raw["landmarks"][i]
                for pt_idx in [36, 45, 30, 48, 54]:
                    lx, ly = lm[pt_idx], lm[68 + pt_idx]
                    if not (np.isnan(lx) or np.isnan(ly)):
                        cv2.circle(frame, (int(lx), int(ly)), 3, (0, 0, 255), -1)

        cv2.imshow("Py-FEAT Real-Time", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
