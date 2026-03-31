# Py-FEAT: Python Facial Expression Analysis Toolbox
[![arXiv-badge](https://img.shields.io/badge/arXiv-2104.03509-red.svg)](https://arxiv.org/abs/2104.03509) 
[![Package versioning](https://img.shields.io/pypi/v/py-feat.svg)](https://pypi.org/project/py-feat/)
[![Tests](https://github.com/cosanlab/py-feat/actions/workflows/tests_and_docs.yml/badge.svg)](https://github.com/cosanlab/py-feat/actions/workflows/tests_and_docs.yml)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/py-feat/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/py-feat?branch=master)
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
[![DOI](https://zenodo.org/badge/118517740.svg)](https://zenodo.org/badge/latestdoi/118517740)

Py-FEAT is a suite for facial expressions (FEX) research written in Python. This package includes tools to detect faces, extract emotional facial expressions (e.g., happiness, sadness, anger), facial muscle movements (e.g., action units), and facial landmarks, from videos and images of faces, as well as methods to preprocess, analyze, and visualize FEX data. 

For detailed examples, tutorials, contribution guidelines, and API please refer to the [Py-FEAT website](https://cosanlab.github.io/py-feat/). 

## Installation
Option 1: Easy installation for quick use
Clone the repository    
`pip install py-feat`  

Option 2: Installation in development mode
```
git clone https://github.com/cosanlab/feat.git
cd feat && python setup.py install -e . 
```

If you're running into issues on arm-based macOS (e.g. m1, m2) you should install pytables using one of the methods below *before* installing py-feat:

`pip install git+https://github.com/PyTables/PyTables.git`  
OR  
`conda install pytables`

Py-Feat currently supports both CPU and GPU processing on NVIDIA cards. We have **experimental** support for GPUs on macOS which you can try with `device='auto'`. However, we currently advise using the default (`cpu`) on macOS until PyTorch support stabilizes.

## Contributing

**Note:** If you forked or cloned this repo prior to 04/26/2022, you'll want to create a new fork or clone as we've used `git-filter-repo` to clean up large files in the history. If you prefer to keep working on that old version, you can find an [archival repo here](https://github.com/cosanlab/py-feat-archive)

## Testing

All tests should be added to `feat/tests/`.  
We use `pytest` for testing and `ruff` for linting and formatting.  
Please ensure all tests pass before creating any pull request or larger change to the code base.

## Continuous Integration

Automated testing is handled by Github Actions according to the following rules:
1. On pushes to the main branch and every week on Sundays, a full test-suite will be run and docs will be built and deployed
2. On PRs against the main branch, a full test-suite will be run and docs will be built but *not* deployed
3. On publishing a release via github, the package will be uploaded to PyPI and docs will be built and deployed

*Note*: Each of these workflows can also be run manually. They can also be skipped by adding 'skip ci' anywhere inside your commit message.

## Model Weights
Py-feat will automatically download model weights as needed without any additional setup from the user.

As of version 0.7.0, all model weights are hosted on the [Py-feat HuggingFace Hub](https://huggingface.co/py-feat).

For prior versions, model weights are stored on Github static assets in release tagged `v0.1`. They will automatically download as needed.

## Licenses
Py-FEAT is provided under the MIT license. You also need to respect the licenses of each model you are using. Please see the LICENSE file for links to each model's license information.

---

# Py-Feat Extended — GUI & Enhancements

## Quick Start

### 1. Create conda environment

**arm64 Mac (M1/M2/M3):**
```bash
CONDA_SUBDIR=osx-arm64 conda create -n pyfeat python=3.11 -y
conda activate pyfeat
```

**Intel Mac / Linux:**
```bash
conda create -n pyfeat python=3.11 -y
conda activate pyfeat
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-gui.txt
```

### 3. Launch the GUI

```bash
streamlit run gui_app.py
```

Opens at **http://localhost:8501**. First run downloads models from HuggingFace (~500MB, one time only).

Custom port:
```bash
streamlit run gui_app.py --server.port 8080
# or
./run_gui.sh --port 8080
```

## What's New

| Enhancement | Description |
|-------------|-------------|
| **Web GUI** | Upload images/videos, configure models, visualize results — no code required |
| **Computational Efficiency** | FP16 inference, auto batch sizing, performance benchmarking |
| **Batch Processing** | Process entire folders with progress tracking, resume support, error recovery |
| **Interpretable Reports** | Plain-language summaries, AU/emotion explanations, comparative analysis, HTML export |

## GUI Tabs

| Tab | What it does |
|-----|-------------|
| **Image Analysis** | Upload images, detect faces/emotions/AUs/landmarks, view interpretable summaries |
| **Video Analysis** | Upload video, see emotion and AU timelines across frames |
| **Batch Processing** | Point to a local folder, process all files with progress and error handling |
| **Report Generator** | Dataset summaries, group comparisons, downloadable HTML reports |
| **Saved Results** | Load and explore previously exported CSV results |

## Sidebar Options

- **Models** — Landmark (mobilefacenet / mobilenet / pfld), AU (xgb / svm), Emotion (resmasknet / svm), Identity (facenet)
- **Detection Settings** — Face detection threshold, batch size
- **Efficiency** — Half precision (FP16) for GPU, auto batch sizing
- **Benchmark** — Test detection speed on a sample image

## Python API (extensions)

```python
from feat.detector import Detector
from feat.extensions import OptimizedDetector, BatchProcessor, ReportGenerator

detector = Detector(device="cpu")

# Optimized detection with timing
opt = OptimizedDetector(detector, use_half_precision=True, auto_batch_size=True)
results, timing = opt.detect(["photo.jpg"])

# Batch process a folder
bp = BatchProcessor(detector, chunk_size=20, batch_size=4)
result = bp.process_folder("/path/to/images", "/path/to/output.csv")
print(result.summary())

# Generate interpretable reports
rg = ReportGenerator(results)
print(rg.summarize_dataset())
print(rg.summarize_face(0))
rg.export_html("report.html")
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| xgboost libomp error on Mac | `brew install libomp` |
| Models not downloading | Check internet; models cache in `~/.cache/huggingface/` |
| Slow first detection | Normal — models load on first use. Subsequent runs are faster |
| CUDA not detected | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
