"""
Interpretable Analysis Outputs for Py-Feat.

Provides a ReportGenerator that creates:
- Plain-language summaries of detection results
- Contextual explanations of AU activations and emotion scores
- Comparative reports across conditions, subjects, or groups
- Exportable HTML reports with embedded charts
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import json
from datetime import datetime

EMOTION_COLS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

AU_COLS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
    "AU25", "AU26", "AU28", "AU43",
]

AU_DESCRIPTIONS = {
    "AU01": ("Inner Brow Raiser", "The inner portions of the eyebrows are pulled upward. Associated with sadness, fear, and surprise."),
    "AU02": ("Outer Brow Raiser", "The outer portions of the eyebrows are pulled upward. Common in surprise expressions."),
    "AU04": ("Brow Lowerer", "The eyebrows are pulled downward and together. Associated with anger, sadness, and concentration."),
    "AU05": ("Upper Lid Raiser", "The upper eyelids are raised, showing more of the white above the iris. Common in fear and surprise."),
    "AU06": ("Cheek Raiser", "The cheeks are pushed upward, often creating crow's feet wrinkles. A key marker of genuine (Duchenne) smiles."),
    "AU07": ("Lid Tightener", "The eyelids are tightened. Often accompanies anger or intense focus."),
    "AU09": ("Nose Wrinkler", "The nose is wrinkled, pulling the skin upward. Strongly associated with disgust."),
    "AU10": ("Upper Lip Raiser", "The upper lip is raised, showing the upper teeth. Associated with disgust and contempt."),
    "AU11": ("Nasolabial Deepener", "The nasolabial furrow is deepened. Can indicate sadness or disgust."),
    "AU12": ("Lip Corner Puller", "The corners of the lips are pulled upward (smiling). The primary marker for happiness."),
    "AU14": ("Dimpler", "The lip corners are tightened, creating dimples. Can indicate contempt or doubt."),
    "AU15": ("Lip Corner Depressor", "The corners of the lips are pulled downward. Associated with sadness and frowning."),
    "AU17": ("Chin Raiser", "The chin boss is pushed upward, wrinkling the chin. Often accompanies pouting or sadness."),
    "AU20": ("Lip Stretcher", "The lips are pulled laterally (stretched). Associated with fear."),
    "AU23": ("Lip Tightener", "The lips are tightened, making them appear thinner. Associated with anger."),
    "AU24": ("Lip Pressor", "The lips are pressed together firmly. Associated with anger or determination."),
    "AU25": ("Lips Part", "The lips are separated. Common in speech, surprise, and fear."),
    "AU26": ("Jaw Drop", "The jaw is lowered, opening the mouth. Associated with surprise or shock."),
    "AU28": ("Lip Suck", "The lips are rolled inward, sucked between the teeth. Can indicate nervousness or suppression."),
    "AU43": ("Eyes Closed", "The eyelids are closed relaxedly. Indicates blinking or eyes-closed states."),
}

EMOTION_INTERPRETATIONS = {
    "anger": {
        "description": "Anger is characterized by lowered and drawn-together brows, tightened eyelids, and pressed or thinned lips.",
        "key_aus": ["AU04", "AU07", "AU23", "AU24"],
        "context": "High anger scores may reflect frustration, irritation, or intensity. In some contexts, similar facial configurations can indicate concentration or determination.",
    },
    "disgust": {
        "description": "Disgust involves nose wrinkling and upper lip raising, sometimes with the lower lip pushed forward.",
        "key_aus": ["AU09", "AU10", "AU25"],
        "context": "Disgust can be a response to physical stimuli (bad taste/smell) or moral/social disapproval. Mild activations may simply reflect distaste or skepticism.",
    },
    "fear": {
        "description": "Fear is shown through raised inner brows, widened eyes, and horizontally stretched lips.",
        "key_aus": ["AU01", "AU04", "AU05", "AU20"],
        "context": "Fear expressions can range from mild anxiety to intense terror. Similar configurations sometimes appear during surprise or apprehension.",
    },
    "happiness": {
        "description": "Happiness is primarily expressed through smiling — lip corners pulled up with cheeks raised (Duchenne smile).",
        "key_aus": ["AU06", "AU12"],
        "context": "The presence of AU06 (cheek raiser) alongside AU12 (smile) is often considered a marker of genuine enjoyment, as opposed to a polite or social smile (AU12 alone).",
    },
    "sadness": {
        "description": "Sadness involves raised inner brows, lowered lip corners, and sometimes a raised chin.",
        "key_aus": ["AU01", "AU04", "AU15", "AU17"],
        "context": "Sadness expressions can be subtle and brief. They sometimes overlap with empathy or compassion displays.",
    },
    "surprise": {
        "description": "Surprise is marked by raised brows, widened eyes, and a dropped jaw.",
        "key_aus": ["AU01", "AU02", "AU05", "AU26"],
        "context": "Surprise is typically brief and can quickly transition to another emotion (e.g., fear, happiness, or anger) depending on the stimulus.",
    },
    "neutral": {
        "description": "A neutral expression indicates no strong emotional display — the face is at rest.",
        "key_aus": [],
        "context": "High neutral scores indicate the absence of strong emotional signals. This is common in resting faces, attentive listening, or deliberate emotional suppression.",
    },
}


class ReportGenerator:
    """Generate interpretable analysis reports from Py-Feat results."""

    def __init__(self, results: pd.DataFrame):
        """
        Args:
            results: A Py-Feat Fex DataFrame or any DataFrame with emotion/AU columns.
        """
        self.results = results
        self._emo_cols = [c for c in EMOTION_COLS if c in results.columns]
        self._au_cols = [c for c in AU_COLS if c in results.columns]

    # ------------------------------------------------------------------
    # Single-face summary
    # ------------------------------------------------------------------
    def summarize_face(self, row_index: int = 0) -> str:
        """Generate a plain-language summary for a single detection."""
        row = self.results.iloc[row_index]
        parts = []

        # Emotion interpretation
        if self._emo_cols:
            emo_vals = {c: float(row[c]) for c in self._emo_cols}
            sorted_emos = sorted(emo_vals.items(), key=lambda x: x[1], reverse=True)
            dominant = sorted_emos[0]

            parts.append(f"## Emotion Analysis")
            parts.append(f"The dominant detected emotion is **{dominant[0].capitalize()}** "
                         f"with {dominant[1]:.1%} probability.")

            interp = EMOTION_INTERPRETATIONS.get(dominant[0], {})
            if interp:
                parts.append(f"\n> {interp['description']}")
                parts.append(f"\n*Context:* {interp.get('context', '')}")

            if len(sorted_emos) > 1 and sorted_emos[1][1] > 0.10:
                parts.append(f"\nSecondary emotion: **{sorted_emos[1][0].capitalize()}** ({sorted_emos[1][1]:.1%})")
            parts.append("")

        # AU interpretation
        if self._au_cols:
            au_vals = {c: float(row[c]) for c in self._au_cols}
            active_aus = [(k, v) for k, v in sorted(au_vals.items(), key=lambda x: x[1], reverse=True) if v > 0.5]

            parts.append(f"## Action Unit Analysis")
            if active_aus:
                parts.append(f"**{len(active_aus)} action unit(s) show significant activation (>0.5):**\n")
                for au, val in active_aus:
                    name, desc = AU_DESCRIPTIONS.get(au, (au, ""))
                    parts.append(f"- **{au} — {name}** (activation: {val:.2f}): {desc}")
            else:
                parts.append("No action units show strong activation (>0.5), suggesting a relatively neutral expression.")
            parts.append("")

        # Head pose
        pose_cols = ["Pitch", "Roll", "Yaw"]
        if all(c in row.index for c in pose_cols):
            parts.append(f"## Head Pose")
            pitch, roll, yaw = float(row["Pitch"]), float(row["Roll"]), float(row["Yaw"])
            desc_parts = []
            if abs(yaw) > 15:
                desc_parts.append(f"turned {'left' if yaw > 0 else 'right'} ({abs(yaw):.0f} degrees)")
            if abs(pitch) > 15:
                desc_parts.append(f"tilted {'down' if pitch > 0 else 'up'} ({abs(pitch):.0f} degrees)")
            if abs(roll) > 10:
                desc_parts.append(f"rotated {'counter-clockwise' if roll > 0 else 'clockwise'} ({abs(roll):.0f} degrees)")
            if desc_parts:
                parts.append(f"The head is {', '.join(desc_parts)}.")
            else:
                parts.append("The head is in an approximately frontal position.")
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Dataset-level summary
    # ------------------------------------------------------------------
    def summarize_dataset(self) -> str:
        """Generate a summary report for the entire dataset."""
        parts = []
        n = len(self.results)
        parts.append(f"# Dataset Summary Report")
        parts.append(f"**Total detections:** {n}")
        if "input" in self.results.columns:
            n_files = self.results["input"].nunique()
            parts.append(f"**Unique source files:** {n_files}")
        parts.append("")

        # Emotion summary
        if self._emo_cols:
            parts.append(f"## Emotion Overview")
            mean_emos = self.results[self._emo_cols].mean()
            dominant = mean_emos.idxmax()
            parts.append(f"The most prevalent emotion across all detections is **{dominant.capitalize()}** "
                         f"(mean probability: {mean_emos[dominant]:.3f}).\n")

            parts.append("| Emotion | Mean | Std | Min | Max |")
            parts.append("|---------|------|-----|-----|-----|")
            for c in self._emo_cols:
                vals = self.results[c].astype(float)
                parts.append(f"| {c.capitalize()} | {vals.mean():.3f} | {vals.std():.3f} | {vals.min():.3f} | {vals.max():.3f} |")
            parts.append("")

        # AU summary
        if self._au_cols:
            parts.append(f"## Action Unit Overview")
            mean_aus = self.results[self._au_cols].mean().sort_values(ascending=False)
            top3 = mean_aus.head(3)
            parts.append("Most frequently activated AUs:\n")
            for au, val in top3.items():
                name = AU_DESCRIPTIONS.get(au, (au, ""))[0]
                parts.append(f"- **{au} ({name})**: mean activation {val:.3f}")
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Comparative report
    # ------------------------------------------------------------------
    def compare_groups(
        self,
        group_column: str,
        group_labels: Optional[Dict[str, str]] = None,
    ) -> str:
        """Compare emotion and AU patterns across groups (e.g., conditions, subjects).

        Args:
            group_column: Column name to group by (e.g., "input", "condition", "subject").
            group_labels: Optional mapping from group values to display labels.

        Returns:
            Markdown-formatted comparative report.
        """
        if group_column not in self.results.columns:
            return f"Column '{group_column}' not found in results."

        groups = self.results.groupby(group_column)
        parts = []
        parts.append(f"# Comparative Report by `{group_column}`")
        parts.append(f"**Number of groups:** {len(groups)}\n")

        if self._emo_cols:
            parts.append("## Emotion Comparison\n")
            # Build comparison table
            header = "| Group | " + " | ".join(c.capitalize() for c in self._emo_cols) + " | Dominant |"
            sep = "|" + "---|" * (len(self._emo_cols) + 2)
            parts.append(header)
            parts.append(sep)

            for gname, gdf in groups:
                label = group_labels.get(str(gname), str(gname)) if group_labels else str(gname)
                # Truncate long labels
                if len(label) > 30:
                    label = label[:27] + "..."
                means = gdf[self._emo_cols].mean()
                dominant = means.idxmax()
                vals = " | ".join(f"{means[c]:.3f}" for c in self._emo_cols)
                parts.append(f"| {label} | {vals} | {dominant.capitalize()} |")
            parts.append("")

        if self._au_cols:
            parts.append("## Top Action Units by Group\n")
            for gname, gdf in groups:
                label = group_labels.get(str(gname), str(gname)) if group_labels else str(gname)
                if len(label) > 40:
                    label = label[:37] + "..."
                mean_aus = gdf[self._au_cols].mean().sort_values(ascending=False)
                top3 = mean_aus.head(3)
                au_strs = ", ".join(f"{au} ({val:.2f})" for au, val in top3.items())
                parts.append(f"- **{label}**: {au_strs}")
            parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # HTML report export
    # ------------------------------------------------------------------
    def export_html(self, output_path: str, title: str = "Py-Feat Analysis Report") -> str:
        """Export a self-contained HTML report with embedded charts.

        Args:
            output_path: Path to write the HTML file.
            title: Report title.

        Returns:
            Path to the generated HTML file.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        charts_html = []

        # Emotion distribution chart
        if self._emo_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            mean_emos = self.results[self._emo_cols].mean()
            colors = ['#e74c3c', '#9b59b6', '#f39c12', '#2ecc71', '#3498db', '#e67e22', '#95a5a6']
            mean_emos.plot(kind="bar", ax=ax, color=colors[:len(self._emo_cols)])
            ax.set_ylabel("Mean Probability")
            ax.set_title("Average Emotion Distribution")
            ax.set_xticklabels([c.capitalize() for c in self._emo_cols], rotation=45)
            plt.tight_layout()

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            charts_html.append(f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;">')

        # AU distribution chart
        if self._au_cols:
            fig, ax = plt.subplots(figsize=(8, 6))
            mean_aus = self.results[self._au_cols].mean().sort_values(ascending=True)
            labels = [f"{au} ({AU_DESCRIPTIONS.get(au, (au,))[0]})" for au in mean_aus.index]
            mean_aus.index = labels
            mean_aus.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_xlabel("Mean Activation")
            ax.set_title("Action Unit Activations")
            plt.tight_layout()

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            charts_html.append(f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;">')

        # Build HTML
        summary_md = self.summarize_dataset()
        # Simple markdown-to-html conversion for the summary
        summary_html = summary_md.replace("# ", "<h1>").replace("\n## ", "\n<h2>")
        # More robust: just use <pre> for the markdown
        summary_html = f"<pre style='white-space:pre-wrap;font-family:sans-serif;'>{summary_md}</pre>"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #f4f6f7; }}
        img {{ margin: 15px 0; border: 1px solid #eee; border-radius: 4px; }}
        .meta {{ color: #7f8c8d; font-size: 0.9em; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="meta">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(self.results)} detections</p>

    {summary_html}

    <h2>Visualizations</h2>
    {''.join(charts_html)}

    <h2>Detection Statistics</h2>
    {self.results.describe().to_html() if len(self.results) > 0 else '<p>No data</p>'}

</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html)

        return output_path
