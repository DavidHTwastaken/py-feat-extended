"""
Py-Feat Web GUI — Streamlit-based interface for facial expression analysis.

Integrates all four proposed enhancements:
1. Computational Efficiency — OptimizedDetector with FP16, auto-batch, benchmarking
2. GUI — This entire application
3. Scalability — BatchProcessor with progress, resume, error handling
4. Interpretable Outputs — ReportGenerator with summaries, comparisons, HTML export

Launch with:
    streamlit run gui_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import io
import time
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from feat.detector import Detector


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Py-Feat — Facial Expression Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMOTION_COLS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
AU_COLS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
]
AU_NAMES = {
    "AU01": "Inner Brow Raiser", "AU02": "Outer Brow Raiser",
    "AU04": "Brow Lowerer", "AU05": "Upper Lid Raiser",
    "AU06": "Cheek Raiser", "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler", "AU10": "Upper Lip Raiser",
    "AU11": "Nasolabial Deepener", "AU12": "Lip Corner Puller",
    "AU14": "Dimpler", "AU15": "Lip Corner Depressor",
    "AU17": "Chin Raiser", "AU20": "Lip Stretcher",
    "AU23": "Lip Tightener", "AU24": "Lip Pressor",
    "AU25": "Lips Part", "AU26": "Jaw Drop",
    "AU28": "Lip Suck", "AU43": "Eyes Closed",
}

EMOTION_DESCRIPTIONS = {
    "anger": "Anger — often indicated by brow lowering (AU04), lid tightening (AU07), and lip pressing (AU24).",
    "disgust": "Disgust — characterized by nose wrinkling (AU09) and upper lip raising (AU10).",
    "fear": "Fear — marked by inner brow raising (AU01), brow lowering (AU04), and lip stretching (AU20).",
    "happiness": "Happiness — typically shown through cheek raising (AU06) and lip corner pulling (AU12).",
    "sadness": "Sadness — indicated by inner brow raising (AU01), brow lowering (AU04), and lip corner depression (AU15).",
    "surprise": "Surprise — characterized by inner brow raising (AU01), outer brow raising (AU02), and jaw drop (AU26).",
    "neutral": "Neutral — no strong emotional expression detected.",
}


# ---------------------------------------------------------------------------
# Cached detector loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Py-Feat models (first time may take a minute)...")
def load_detector(landmark_model, au_model, emotion_model, identity_model, device):
    # Ensure classifiers are in __main__ for skops deserialization under Streamlit.
    # The .skops model files reference these classes as '__main__.XGBClassifier' etc.
    # because they were serialized from a script where these classes were in __main__.
    # Under Streamlit, __main__ is Streamlit's bootstrapper, not our script, so the
    # patching in detector.py (line 51-53) targets the wrong module. We fix it here.
    import sys
    from feat.au_detectors.StatLearning.SL_test import XGBClassifier, SVMClassifier
    from feat.emo_detectors.StatLearning.EmoSL_test import EmoSVMClassifier
    sys.modules["__main__"].__dict__["XGBClassifier"] = XGBClassifier
    sys.modules["__main__"].__dict__["SVMClassifier"] = SVMClassifier
    sys.modules["__main__"].__dict__["EmoSVMClassifier"] = EmoSVMClassifier

    det = Detector(
        landmark_model=landmark_model if landmark_model != "None" else None,
        au_model=au_model if au_model != "None" else None,
        emotion_model=emotion_model if emotion_model != "None" else None,
        identity_model=identity_model if identity_model != "None" else None,
        device=device,
    )
    return det



def get_optimized_detector(detector, use_half, auto_batch):
    """Wrap detector in OptimizedDetector if efficiency options are enabled."""
    from feat.extensions.efficiency import OptimizedDetector
    return OptimizedDetector(
        detector,
        use_half_precision=use_half,
        auto_batch_size=auto_batch,
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def draw_detections_on_image(img_path, fex_row):
    """Return a matplotlib figure with bounding box + landmarks overlaid."""
    img = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)

    # Bounding box
    try:
        x = fex_row["FaceRectX"]
        y = fex_row["FaceRectY"]
        w = fex_row["FaceRectWidth"]
        h = fex_row["FaceRectHeight"]
        from matplotlib.patches import Rectangle
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
    except Exception:
        pass

    # Landmarks
    try:
        lm_x = [fex_row[f"x_{i}"] for i in range(68)]
        lm_y = [fex_row[f"y_{i}"] for i in range(68)]
        ax.scatter(lm_x, lm_y, s=8, c="cyan", zorder=5)
    except Exception:
        pass

    ax.axis("off")
    plt.tight_layout()
    return fig


def emotion_bar_chart(emotions_series):
    """Return a matplotlib figure with a horizontal bar chart of emotions."""
    fig, ax = plt.subplots(figsize=(5, 3))
    cols = [c for c in EMOTION_COLS if c in emotions_series.index]
    vals = emotions_series[cols].values.astype(float)
    colors_arr = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(cols)))
    ax.barh(cols, vals, color=colors_arr)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Emotion Probabilities")
    plt.tight_layout()
    return fig


def au_bar_chart(au_series):
    """Return a matplotlib figure with AU activations."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cols = [c for c in AU_COLS if c in au_series.index]
    vals = au_series[cols].values.astype(float)
    labels = [f"{c} ({AU_NAMES.get(c, '')})" for c in cols]
    colors_arr = plt.cm.OrRd(vals / max(vals.max(), 1))
    ax.barh(labels, vals, color=colors_arr)
    ax.set_xlabel("Activation")
    ax.set_title("Action Unit Activations")
    plt.tight_layout()
    return fig


def generate_summary(fex_row):
    """Generate a plain-language summary of the detection results."""
    lines = []

    emo_cols = [c for c in EMOTION_COLS if c in fex_row.index]
    if emo_cols:
        emo_vals = {c: float(fex_row[c]) for c in emo_cols}
        dominant = max(emo_vals, key=emo_vals.get)
        confidence = emo_vals[dominant]
        lines.append(f"**Dominant emotion:** {dominant.capitalize()} ({confidence:.1%} confidence)")
        lines.append(f"> {EMOTION_DESCRIPTIONS.get(dominant, '')}")
        lines.append("")

        sorted_emos = sorted(emo_vals.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_emos) > 1 and sorted_emos[1][1] > 0.15:
            sec = sorted_emos[1]
            lines.append(f"**Secondary emotion:** {sec[0].capitalize()} ({sec[1]:.1%})")
            lines.append("")

    au_cols = [c for c in AU_COLS if c in fex_row.index]
    if au_cols:
        au_vals = {c: float(fex_row[c]) for c in au_cols}
        active = [(k, v) for k, v in sorted(au_vals.items(), key=lambda x: x[1], reverse=True) if v > 0.5]
        if active:
            lines.append("**Active Action Units (>0.5):**")
            for au, val in active[:5]:
                lines.append(f"- {au} — {AU_NAMES.get(au, 'Unknown')} (activation: {val:.2f})")
            lines.append("")

    pose_cols = ["Pitch", "Roll", "Yaw"]
    if all(c in fex_row.index for c in pose_cols):
        pitch = float(fex_row["Pitch"])
        roll = float(fex_row["Roll"])
        yaw = float(fex_row["Yaw"])
        lines.append("**Head Pose:**")
        direction_parts = []
        if abs(yaw) > 10:
            direction_parts.append("left" if yaw > 0 else "right")
        if abs(pitch) > 10:
            direction_parts.append("down" if pitch > 0 else "up")
        if direction_parts:
            lines.append(f"- Face is turned slightly {' and '.join(direction_parts)}")
        else:
            lines.append("- Face is approximately frontal")
        lines.append(f"- Pitch: {pitch:.1f}, Roll: {roll:.1f}, Yaw: {yaw:.1f}")

    return "\n".join(lines) if lines else "No detection results to summarize."


# ---------------------------------------------------------------------------
# Sidebar — Model & Efficiency Configuration
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")

st.sidebar.subheader("Models")
device_option = st.sidebar.selectbox("Device", ["cpu", "cuda", "mps", "auto"], index=0)
landmark_option = st.sidebar.selectbox("Landmark Model", ["mobilefacenet", "mobilenet", "pfld", "None"], index=0)
au_option = st.sidebar.selectbox("AU Model", ["xgb", "svm", "None"], index=0)
emotion_option = st.sidebar.selectbox("Emotion Model", ["resmasknet", "svm", "None"], index=0)
identity_option = st.sidebar.selectbox("Identity Model", ["facenet", "None"], index=0)

st.sidebar.divider()
st.sidebar.subheader("Detection Settings")
face_threshold = st.sidebar.slider("Face Detection Threshold", 0.0, 1.0, 0.5, 0.05)
batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=64, value=1)

st.sidebar.divider()
st.sidebar.subheader("Efficiency Options")
use_half_precision = st.sidebar.checkbox(
    "Half Precision (FP16)",
    value=False,
    help="Use FP16 for faster GPU inference. Only effective on CUDA/MPS devices.",
)
auto_batch = st.sidebar.checkbox(
    "Auto Batch Size",
    value=False,
    help="Automatically determine batch size based on available memory.",
)

st.sidebar.divider()
st.sidebar.caption("Py-Feat v0.7.0 Extended")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("Py-Feat — Facial Expression Analysis")
st.markdown(
    "Upload images or videos to detect faces, emotions, action units, and landmarks. "
    "Configure models and efficiency settings in the sidebar."
)

tab_images, tab_video, tab_batch, tab_report, tab_results = st.tabs([
    "Image Analysis", "Video Analysis", "Batch Processing", "Report Generator", "Saved Results"
])

# ===================== IMAGE ANALYSIS TAB =====================
with tab_images:
    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        key="img_upload",
    )

    if uploaded_files:
        cols = st.columns(min(len(uploaded_files), 4))
        for i, uf in enumerate(uploaded_files):
            with cols[i % len(cols)]:
                st.image(uf, caption=uf.name, use_container_width=True)

        if st.button("Run Detection", key="run_img", type="primary"):
            tmp_dir = tempfile.mkdtemp()
            img_paths = []
            for uf in uploaded_files:
                path = os.path.join(tmp_dir, uf.name)
                with open(path, "wb") as f:
                    f.write(uf.getbuffer())
                img_paths.append(path)

            with st.spinner("Loading detector..."):
                detector = load_detector(
                    landmark_option, au_option, emotion_option, identity_option, device_option
                )

            # Use OptimizedDetector if efficiency options are set
            use_optimized = use_half_precision or auto_batch
            if use_optimized:
                opt_det = get_optimized_detector(detector, use_half_precision, auto_batch)

            progress_bar = st.progress(0, text="Detecting faces...")
            t_start = time.perf_counter()

            try:
                if use_optimized:
                    results, timing = opt_det.detect(
                        img_paths,
                        batch_size=batch_size if not auto_batch else None,
                        face_detection_threshold=face_threshold,
                        data_type="image",
                        progress_bar=False,
                    )
                else:
                    import torch
                    with torch.inference_mode():
                        results = detector.detect(
                            img_paths,
                            batch_size=batch_size,
                            face_detection_threshold=face_threshold,
                            data_type="image",
                            progress_bar=False,
                        )
                    timing = None
                progress_bar.progress(100, text="Detection complete!")
            except Exception as e:
                st.error(f"Detection failed: {e}")
                st.stop()

            elapsed = time.perf_counter() - t_start
            st.success(f"Detected {len(results)} face(s) across {len(img_paths)} image(s) in {elapsed:.2f}s.")

            # Show timing info if optimized
            if timing:
                with st.expander("Performance Details"):
                    st.json(timing)

            st.session_state["last_results"] = results
            st.session_state["last_img_paths"] = img_paths

            # Per-face results
            for idx, (_, row) in enumerate(results.iterrows()):
                with st.expander(f"Face {idx + 1} — {Path(row.get('input', 'unknown')).name}", expanded=(idx == 0)):
                    col_img, col_emo, col_au = st.columns([1.2, 1, 1])

                    with col_img:
                        st.subheader("Detection")
                        input_path = row.get("input", "")
                        matching = [p for p in img_paths if Path(p).name == Path(str(input_path)).name]
                        if matching:
                            fig = draw_detections_on_image(matching[0], row)
                            st.pyplot(fig)
                            plt.close(fig)

                    with col_emo:
                        st.subheader("Emotions")
                        emo_data = {c: float(row[c]) for c in EMOTION_COLS if c in row.index}
                        if emo_data:
                            fig = emotion_bar_chart(row)
                            st.pyplot(fig)
                            plt.close(fig)

                    with col_au:
                        st.subheader("Action Units")
                        au_data = {c: float(row[c]) for c in AU_COLS if c in row.index}
                        if au_data:
                            fig = au_bar_chart(row)
                            st.pyplot(fig)
                            plt.close(fig)

                    # Interpretable summary using ReportGenerator
                    st.subheader("Interpretable Summary")
                    try:
                        from feat.extensions.report_generator import ReportGenerator
                        rg = ReportGenerator(results.iloc[[idx]])
                        st.markdown(rg.summarize_face(0))
                    except Exception:
                        st.markdown(generate_summary(row))

            # Download CSV
            csv_buf = io.StringIO()
            results.to_csv(csv_buf, index=False)
            st.download_button(
                "Download Results (CSV)",
                csv_buf.getvalue(),
                file_name="pyfeat_image_results.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload one or more images to get started.")


# ===================== VIDEO ANALYSIS TAB =====================
with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
    )

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        skip_frames = st.number_input("Skip frames (0 = process all)", min_value=0, max_value=100, value=24, key="skip_frames")
    with col_v2:
        vid_batch = st.number_input("Batch size (# of frames)", min_value=1, max_value=64, value=4, key="vid_batch")

    if uploaded_video:
        st.video(uploaded_video)

        if st.button("Run Video Detection", key="run_vid", type="primary"):
            tmp_dir = tempfile.mkdtemp()
            vid_path = os.path.join(tmp_dir, uploaded_video.name)
            with open(vid_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            with st.spinner("Loading detector..."):
                detector = load_detector(
                    landmark_option, au_option, emotion_option, identity_option, device_option
                )

            t_start = time.perf_counter()
            with st.spinner("Processing video — this may take a while..."):
                try:
                    use_optimized = use_half_precision or auto_batch
                    if use_optimized:
                        opt_det = get_optimized_detector(detector, use_half_precision, auto_batch)
                        results, timing = opt_det.detect(
                            vid_path,
                            data_type="video",
                            batch_size=vid_batch if not auto_batch else None,
                            face_detection_threshold=face_threshold,
                            skip_frames=skip_frames if skip_frames > 0 else None,
                            progress_bar=False,
                        )
                    else:
                        import torch
                        with torch.inference_mode():
                            results = detector.detect(
                                vid_path,
                                data_type="video",
                                batch_size=vid_batch,
                                face_detection_threshold=face_threshold,
                                skip_frames=skip_frames if skip_frames > 0 else None,
                                progress_bar=True,
                            )
                        timing = None
                except Exception as e:
                    st.error(f"Video detection failed: {e}")
                    st.stop()

            elapsed = time.perf_counter() - t_start
            st.success(f"Processed video: {len(results)} detections in {elapsed:.1f}s ({len(results)/max(elapsed,0.01):.1f} det/s).")
            st.session_state["last_results"] = results

            if timing:
                with st.expander("Performance Details"):
                    st.json(timing)

            # Emotion timeline
            st.subheader("Emotion Timeline")
            emo_cols_present = [c for c in EMOTION_COLS if c in results.columns]
            if emo_cols_present and "frame" in results.columns:
                chart_data = results[["frame"] + emo_cols_present].copy()
                chart_data = chart_data.set_index("frame")
                st.line_chart(chart_data)

            # AU timeline
            st.subheader("Action Unit Timeline")
            au_cols_present = [c for c in AU_COLS if c in results.columns]
            if au_cols_present and "frame" in results.columns:
                au_chart = results[["frame"] + au_cols_present].copy()
                au_chart = au_chart.set_index("frame")
                top_aus = au_chart.mean().nlargest(5).index.tolist()
                st.line_chart(au_chart[top_aus])

            # Video summary using ReportGenerator
            st.subheader("Video Summary")
            try:
                from feat.extensions.report_generator import ReportGenerator
                rg = ReportGenerator(results)
                st.markdown(rg.summarize_dataset())
            except Exception:
                if emo_cols_present:
                    mean_emos = results[emo_cols_present].mean()
                    dominant = mean_emos.idxmax()
                    st.markdown(f"**Average dominant emotion:** {dominant.capitalize()} ({mean_emos[dominant]:.1%})")

            # Download
            csv_buf = io.StringIO()
            results.to_csv(csv_buf, index=False)
            st.download_button(
                "Download Video Results (CSV)",
                csv_buf.getvalue(),
                file_name="pyfeat_video_results.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload a video file to analyze facial expressions over time.")


# ===================== BATCH PROCESSING TAB =====================
with tab_batch:
    st.markdown("""
    **Scalable batch processing** for large image/video datasets. Features:
    - Automatic chunked processing with progress tracking
    - Resume capability (skip already-processed files)
    - Graceful error handling with per-file error logs
    - Recursive folder scanning
    - Streaming results to disk (no OOM on huge datasets)
    """)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        folder_path = st.text_input("Path to folder", placeholder="/path/to/images/")
        file_type = st.selectbox("File type", ["image", "video"], key="batch_file_type")
    with col_b2:
        output_csv = st.text_input("Output CSV path", placeholder="/path/to/output/results.csv")
        chunk_size = st.number_input("Chunk size", min_value=1, max_value=100, value=20,
                                     help="Number of files to process before flushing to disk.")

    resume_enabled = st.checkbox("Resume from previous run", value=True,
                                 help="Skip files already in the output CSV.")

    if st.button("Run Batch Processing", key="run_batch", type="primary"):
        if not folder_path or not os.path.isdir(folder_path):
            st.error("Please provide a valid folder path.")
        elif not output_csv:
            st.error("Please provide an output CSV path.")
        else:
            with st.spinner("Loading detector..."):
                detector = load_detector(
                    landmark_option, au_option, emotion_option, identity_option, device_option
                )

            from feat.extensions.batch_processor import BatchProcessor

            bp = BatchProcessor(
                detector=detector,
                chunk_size=chunk_size,
                batch_size=batch_size,
                face_detection_threshold=face_threshold,
                skip_frames=skip_frames if file_type == "video" and skip_frames > 0 else None,
            )

            # Discover files first
            files = bp.discover_files(folder_path, file_type)
            if not files:
                st.warning(f"No {file_type} files found in `{folder_path}`.")
            else:
                st.info(f"Found **{len(files)}** {file_type} files. Starting batch processing...")

                progress = st.progress(0, text="Starting batch...")
                status_text = st.empty()

                def progress_callback(done, total, current_file):
                    pct = done / max(total, 1)
                    fname = Path(current_file).name if current_file else ""
                    progress.progress(pct, text=f"Processed {done}/{total}")
                    status_text.text(f"Last processed: {fname}")

                result = bp.process_folder(
                    folder=folder_path,
                    output_path=output_csv,
                    file_type=file_type,
                    resume=resume_enabled,
                    progress_callback=progress_callback,
                )

                progress.progress(1.0, text="Complete!")

                # Show results
                st.success(f"Batch complete!")
                st.code(result.summary(), language="text")

                if result.errors:
                    with st.expander(f"Errors ({len(result.errors)})"):
                        for err in result.errors:
                            st.warning(f"**{err['file']}**: {err['error']}")

                # Load and show summary
                if os.path.exists(output_csv):
                    df = pd.read_csv(output_csv)
                    st.session_state["last_results"] = df

                    emo_cols_present = [c for c in EMOTION_COLS if c in df.columns]
                    if emo_cols_present:
                        st.subheader("Batch Summary")
                        try:
                            from feat.extensions.report_generator import ReportGenerator
                            rg = ReportGenerator(df)
                            st.markdown(rg.summarize_dataset())
                        except Exception:
                            st.dataframe(
                                df[emo_cols_present].describe().T.style.format("{:.3f}"),
                                use_container_width=True,
                            )


# ===================== REPORT GENERATOR TAB =====================
with tab_report:
    st.markdown("""
    **Interpretable Analysis Reports** — Generate plain-language summaries, comparative analyses,
    and exportable HTML reports from Py-Feat results.
    """)

    report_source = st.radio(
        "Data source",
        ["Last detection results (from Image/Video/Batch tab)", "Upload CSV"],
        key="report_source",
    )

    report_df = None
    if report_source == "Upload CSV":
        report_csv = st.file_uploader("Upload results CSV", type=["csv"], key="report_csv_upload")
        if report_csv:
            report_df = pd.read_csv(report_csv)
    else:
        if "last_results" in st.session_state:
            report_df = st.session_state["last_results"]
            if not isinstance(report_df, pd.DataFrame):
                report_df = pd.DataFrame(report_df)
            st.success(f"Using last results: {len(report_df)} detections")
        else:
            st.info("No results available yet. Run a detection first, or upload a CSV.")

    if report_df is not None and len(report_df) > 0:
        from feat.extensions.report_generator import ReportGenerator
        rg = ReportGenerator(report_df)

        st.subheader("Dataset Summary")
        st.markdown(rg.summarize_dataset())

        # Per-face explorer
        if len(report_df) > 1:
            st.divider()
            st.subheader("Individual Detection Explorer")
            face_idx = st.slider("Select detection", 0, len(report_df) - 1, 0, key="report_face_idx")
            st.markdown(rg.summarize_face(face_idx))

        # Comparative analysis
        st.divider()
        st.subheader("Comparative Analysis")
        available_group_cols = [c for c in ["input", "frame"] if c in report_df.columns]
        # Also offer any string columns
        for c in report_df.columns:
            if report_df[c].dtype == "object" and c not in available_group_cols:
                available_group_cols.append(c)

        if available_group_cols:
            group_col = st.selectbox("Group by", available_group_cols, key="report_group_col")
            n_groups = report_df[group_col].nunique()
            if n_groups > 100:
                st.warning(f"Column `{group_col}` has {n_groups} unique values — comparison may be verbose. Consider grouping differently.")
            if n_groups <= 100:
                st.markdown(rg.compare_groups(group_col))
        else:
            st.info("No suitable grouping columns found.")

        # Export HTML report
        st.divider()
        st.subheader("Export HTML Report")
        report_title = st.text_input("Report title", value="Py-Feat Analysis Report", key="report_title")
        if st.button("Generate HTML Report", key="gen_html_report"):
            tmp_dir = tempfile.mkdtemp()
            html_path = os.path.join(tmp_dir, "pyfeat_report.html")
            rg.export_html(html_path, title=report_title)

            with open(html_path, "r") as f:
                html_content = f.read()

            st.download_button(
                "Download HTML Report",
                html_content,
                file_name="pyfeat_report.html",
                mime="text/html",
            )
            st.success("Report generated! Click above to download.")


# ===================== SAVED RESULTS TAB =====================
with tab_results:
    st.markdown("Load a previously saved Py-Feat CSV to explore results interactively.")

    csv_file = st.file_uploader("Upload a Py-Feat results CSV", type=["csv"], key="csv_upload")

    if csv_file:
        df = pd.read_csv(csv_file)
        st.dataframe(df.head(50), use_container_width=True)

        emo_cols_present = [c for c in EMOTION_COLS if c in df.columns]
        au_cols_present = [c for c in AU_COLS if c in df.columns]

        if emo_cols_present:
            st.subheader("Emotion Distribution")
            mean_emos = df[emo_cols_present].mean()
            fig, ax = plt.subplots(figsize=(6, 3))
            mean_emos.plot(kind="bar", ax=ax, color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(emo_cols_present))))
            ax.set_ylabel("Mean Probability")
            ax.set_title("Average Emotions Across All Detections")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if au_cols_present:
            st.subheader("Action Unit Distribution")
            mean_aus = df[au_cols_present].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            mean_aus.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_xlabel("Mean Activation")
            ax.set_title("Average AU Activations")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if len(df) > 1:
            st.subheader("Row Explorer")
            row_idx = st.slider("Select detection", 0, len(df) - 1, 0)
            row = df.iloc[row_idx]
            st.markdown(generate_summary(row))
    else:
        st.info("Upload a CSV file exported from Py-Feat to explore results.")


# ===================== BENCHMARKING (sidebar) =====================
with st.sidebar.expander("Benchmark"):
    st.markdown("Test detection speed on a sample image.")
    bench_file = st.file_uploader("Upload test image", type=["jpg", "png"], key="bench_img")
    bench_runs = st.number_input("Number of runs", min_value=1, max_value=20, value=3, key="bench_runs")

    if bench_file and st.button("Run Benchmark", key="run_bench"):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(bench_file.getbuffer())
        tmp.close()

        with st.spinner("Running benchmark..."):
            detector = load_detector(
                landmark_option, au_option, emotion_option, identity_option, device_option
            )
            opt_det = get_optimized_detector(detector, use_half_precision, auto_batch)
            bench_result = opt_det.benchmark(tmp.name, n_runs=bench_runs)

        st.json(bench_result)
        os.unlink(tmp.name)
