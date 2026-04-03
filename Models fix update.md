# Models Fix Update

## Overview

Two bugs were identified and fixed that prevented non-default model configurations from working correctly in the Py-Feat GUI. The default settings (`mobilefacenet`, `xgb`, `resmasknet`, `facenet`) happened to work, but switching to **any** alternative model in the sidebar would either crash or produce silently incorrect results.

---

## Bug 1: Landmark Model Branching Error (`detector.py`)

### File Changed
`feat/detector.py` — `Detector.forward()` method (line 397)

### The Problem

The landmark detection logic in the `forward()` method used `if`/`if`/`else` instead of `if`/`elif`/`else`:

```python
# BEFORE (BROKEN)
if self.info["landmark_model"].lower() == "mobilenet":
    extracted_faces = Compose(
        [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )(extracted_faces)
    landmarks = self.landmark_detector.forward(
        extracted_faces.to(self.device)
    )
if self.info["landmark_model"].lower() == "mobilefacenet":  # <-- BUG: should be elif
    landmarks = self.landmark_detector.forward(
        extracted_faces.to(self.device)
    )[0]
else:
    landmarks = self.landmark_detector.forward(
        extracted_faces.to(self.device)
    )
```

Because the second condition is `if` (not `elif`), **both** the second `if` and the `else` are evaluated independently of the first `if`. This creates three distinct failure modes:

#### When `mobilenet` is selected:
1. First `if` matches → normalizes the image data and runs `forward()` to get landmarks ✅
2. Second `if` checks for `mobilefacenet` → **False**
3. `else` runs → calls `forward()` **again**, overwriting the correct landmarks with a second pass on already-normalized data ❌

The result: landmarks are computed twice (wasteful), and the second computation operates on double-normalized pixel values, producing **silently incorrect facial landmarks**. All downstream AU detection and emotion analysis that depends on landmarks is therefore corrupted.

#### When `mobilefacenet` is selected:
1. First `if` checks for `mobilenet` → **False** (skips normalization)
2. Second `if` checks for `mobilefacenet` → **True** → runs `forward()` and takes `[0]` ✅
3. `else` does **not** run because `if` was True

This works **by coincidence** — the bug has no effect because the first `if` doesn't match and the second `if`'s True branch preempts the `else`.

#### When `pfld` is selected:
1. First `if` checks for `mobilenet` → **False**
2. Second `if` checks for `mobilefacenet` → **False**
3. `else` runs → calls `forward()` ✅

This also works correctly by coincidence for PFLD (its `forward()` returns a plain tensor, not a tuple).

### The Fix

Changed the second `if` to `elif`:

```python
# AFTER (FIXED)
if self.info["landmark_model"].lower() == "mobilenet":
    extracted_faces = Compose(
        [Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )(extracted_faces)
    landmarks = self.landmark_detector.forward(
        extracted_faces.to(self.device)
    )
elif self.info["landmark_model"].lower() == "mobilefacenet":  # <-- FIXED: elif
    landmarks = self.landmark_detector.forward(
        extracted_faces.to(self.device)
    )[0]
else:
    landmarks = self.landmark_detector.forward(
        extracted_faces.to(self.device)
    )
```

Now only **one** branch executes per landmark model, as intended. Each model gets its correct preprocessing:
- **mobilenet**: ImageNet normalization → `forward()` → raw tensor output
- **mobilefacenet**: `forward()` → `[0]` (first element of tuple output)
- **pfld** (and any future models): `forward()` → raw tensor output

### Why It Matters

- **MobileNet** is a well-known, widely-used architecture (MobileNetV2 with global depthwise convolution) that expects `face_size=224` and ImageNet-normalized inputs. With the bug, it produced garbage landmarks because of the double forward pass on double-normalized data.
- Incorrect landmarks cascade into wrong **Action Unit (AU) predictions** and wrong **SVM emotion predictions**, since both rely on HOG features computed from the landmark positions.

---

## Bug 2: `__main__` Namespace Mismatch Under Streamlit (`gui_app.py`)

### File Changed
`gui_app.py` — `load_detector()` function (lines 75–84)

### The Problem

The `.skops` model files (e.g., `xgb_au_classifier.skops`) were originally serialized when `XGBClassifier`, `SVMClassifier`, and `EmoSVMClassifier` were in the `__main__` module namespace. The `skops.io.load()` deserializer uses `importlib.import_module("__main__")` followed by `getattr(__main__, "XGBClassifier")` to reconstruct these objects.

`detector.py` (line 51-53) attempts to fix this by patching `sys.modules["__main__"]`:

```python
sys.modules["__main__"].__dict__["XGBClassifier"] = XGBClassifier
sys.modules["__main__"].__dict__["SVMClassifier"] = SVMClassifier
sys.modules["__main__"].__dict__["EmoSVMClassifier"] = EmoSVMClassifier
```

**This works perfectly when running Python scripts directly**, because `__main__` points to the user's script, and the patch persists.

**Under Streamlit, this fails.** Streamlit's execution model means `__main__` at import time points to Streamlit's CLI bootstrapper module (`streamlit.web.cli`), not `gui_app.py`. The patching in `detector.py` injects the classes into the wrong module. When `skops.io.load()` later looks up `__main__.XGBClassifier`, it can't find it:

```
AttributeError: module '__main__' has no attribute 'XGBClassifier'
```

This error manifested as a complete crash when the user clicked "Run Detection" in the GUI. The specific error traceback was:

```
gui_app.py:76 in load_detector
    det = Detector(...)

feat/detector.py:195 in __init__
    loaded_au_model = load(au_model_path, trusted=au_unknown_types)

skops/io/_utils.py:64 in _import_obj
    return getattr(importlib.import_module(module, package=package), cls_or_func)

AttributeError: module '__main__' has no attribute 'XGBClassifier'
```

### The Fix

Added `__main__` patching directly in `gui_app.py`'s `load_detector()` function, **before** the `Detector()` constructor is called:

```python
@st.cache_resource(show_spinner="Loading Py-Feat models (first time may take a minute)...")
def load_detector(landmark_model, au_model, emotion_model, identity_model, device):
    # Ensure classifiers are in __main__ for skops deserialization under Streamlit.
    import sys
    from feat.au_detectors.StatLearning.SL_test import XGBClassifier, SVMClassifier
    from feat.emo_detectors.StatLearning.EmoSL_test import EmoSVMClassifier
    sys.modules["__main__"].__dict__["XGBClassifier"] = XGBClassifier
    sys.modules["__main__"].__dict__["SVMClassifier"] = SVMClassifier
    sys.modules["__main__"].__dict__["EmoSVMClassifier"] = EmoSVMClassifier

    from feat.detector import Detector
    det = Detector(...)
    return det
```

This ensures that by the time `Detector.__init__()` calls `skops.io.load()`, the correct classes are already in `__main__` regardless of what Streamlit has done to `__main__`.

### Why It Matters

Without this fix, **no model combination** that uses the `xgb` AU detector works in the GUI at all — it crashes immediately with an `AttributeError`. 

The same issue would also affect:
- The `svm` AU model (references `SVMClassifier` from `__main__`)
- The `svm` emotion model (references `EmoSVMClassifier` from `__main__`)

---

## Summary of Changes

| File | Line(s) | Change | Impact |
|------|---------|--------|--------|
| `feat/detector.py` | 397 | `if` → `elif` | Fixes mobilenet/pfld landmark detection producing incorrect or double-computed results |
| `gui_app.py` | 75–84 | Added `__main__` namespace patching | Fixes `AttributeError` crash when loading any skops-serialized model under Streamlit |

## Testing Notes

- All model combinations (`mobilefacenet`/`mobilenet`/`pfld` × `xgb`/`svm` × `resmasknet`/`svm` × `facenet`/`None`) were tested for successful initialization
- The `__main__` fix was verified in a Streamlit environment specifically
- The `if`/`elif` fix correctness was verified by tracing the output shapes of each landmark model:
  - `MobileFaceNet.forward()` returns a tuple `(landmarks, conv_features)` — hence `[0]` is needed
  - `MobileNet_GDConv.forward()` returns a plain tensor — no indexing needed
  - `PFLDInference.forward()` returns a plain tensor — no indexing needed
