# Deep Learning Project — Reproduction & Demo

This repository contains the reproduction code, data pointers, and demo notebook for the Deep Active Learning for Multi-Label project (submission: Taimoor Taha Ammar).

Contents
- `reproduce.ipynb` — main Jupyter notebook that contains data loading, model definition (BERT multi-label wrapper), training, resume logic, evaluation and visualization cells.
- `train.csv`, `dev.csv`, `test.csv` — dataset CSVs used in the notebook (note: `train.csv` is ~56 MB).
- `Deep active learning for multi label.pdf` — project write-up.
- `models/` (not included in this push by default) — checkpoints. Large; keep out of Git or use Git LFS.
- `results/` (not included in this push by default) — evaluation outputs and visualizations.

Quick Overview
-------------
This repo is designed to support both a "paper-scale" run and a much faster demo run. The notebook supports:
- FAST demo mode via `FAST_CONFIG`/`FAST_VALIDATION` (shorter epochs, smaller subsets) to get results quickly.
- Resume / warm-start behavior: automatic detection/load of latest checkpoint and options to warm-start and run extra epochs.
- Visualization cells that produce threshold sweeps, probability histograms, loss curves and reliability diagrams.

Requirements
------------
Recommended Python environment (tested with Python 3.10+):

- torch (PyTorch)
- transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- jupyter
- nbconvert

You can install the most common dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn jupyter nbconvert
```

(If you prefer, create a `requirements.txt` from this list.)

Notes about large files
----------------------
- `train.csv` is ~56 MB; Git pushed it but GitHub warns about files >50 MB. For a cleaner repository, consider using Git LFS for `train.csv` and any large checkpoints in `models/`.
- To enable Git LFS for large files, run:

```powershell
git lfs install
git lfs track "train.csv"
git lfs track "models/*"
git add .gitattributes
git add train.csv
git commit -m "Track large files with git-lfs"
git push
```

If you already pushed large files and want them removed from history, consider using the BFG Repo Cleaner or rewriting history carefully.

How to run (interactive)
-------------------------
1. Open Jupyter and run the notebook interactively:

```powershell
cd D:\deeplearning_project
jupyter notebook reproduce.ipynb
```

2. Recommended order: run import/data/model-definition cells first (top of notebook), then:
- If you only want a fast demo: enable FAST mode (cell named `FAST_CONFIG`) and run the main training cell.
- To resume from existing checkpoints: run the cell that sets `LOAD_CHECKPOINT_IF_EXISTS = True` and configure `WARM_START` / `EXTRA_EPOCHS_ON_CHECKPOINT` as needed.

How to run (headless)
---------------------
You can execute the notebook headless using `nbconvert` (useful for CI or timed runs):

```powershell
cd D:\deeplearning_project
jupyter nbconvert --to notebook --execute reproduce.ipynb --output reproduce_run.ipynb --ExecutePreprocessor.timeout=3600
```

Adjust `--ExecutePreprocessor.timeout` as needed.

Reproducibility & checkpoints
-----------------------------
- Checkpoint paths used by the notebook: `models/BEAL_seed{seed}_round{n}.pt` and `models/BEAL_seed{seed}_resumed.pt`. When resuming, the notebook looks for the latest checkpoint for the configured seed.
- Behavior flags (defined in the notebook):
  - `LOAD_CHECKPOINT_IF_EXISTS` — if true, loads latest available checkpoint and optionally continues training.
  - `WARM_START` — reuse model weights but reset optimizer (or vice versa) (see notebook comments).
  - `EXTRA_EPOCHS_ON_CHECKPOINT` — number of extra epochs to run on top of resumed checkpoint.

Diagnostics & visualizations
----------------------------
The notebook contains several diagnostic cells to help explain low F1 (e.g., micro-F1 = 0 at threshold 0.5):
- Probability histograms on dev/test sets
- Threshold sweep (compute micro-F1 for multiple thresholds)
- Training loss curves and F1 curves with standard-deviation bands
- Reliability diagrams / calibration checks

Look for `results/visualizations/` for PNG/JSON output (these directories may not be present in the pushed copy).

If something is not running
--------------------------
- If you hit errors in the aggressive fine-tune runner, it is usually caused by heterogeneous batch formats returned by the `DataLoader`. The notebook includes a helper `unpack_batch` to normalize batch tuples/dicts to tensors — ensure the top data/model cells are run in the kernel before calling fine-tune cells.
- If Git operations fail locally due to space issues, use another drive (e.g., `D:`) as done for this copy.

Recommended workflow for experiments
-----------------------------------
1. Use `FAST_CONFIG` for quick iterations.
2. Inspect `results/statistics.json` after a successful run to check dev/test micro-F1 values and threshold behavior.
3. If metrics are poor, try targeted resume/warm-start from the best checkpoint with a low learning rate for a few epochs.
4. If you need to share checkpoints or large artifacts, upload them separately (GitHub Releases, Google Drive, or S3) or enable Git LFS.

Contributing & contact
----------------------
If you want changes, open an issue or contact the author (Taimoor Taha Ammar). This repository is intended as a reproduction artifact for the accompanying report.

License
-------
This repository does not include an explicit license file in this push. If you want an open-source license, add a `LICENSE` (e.g., MIT) and include license text here.

Acknowledgements & references
-----------------------------
- HuggingFace Transformers (BERT)
- PyTorch
- scikit-learn


---
Generated on Nov 29, 2025. If you want any section expanded (detailed run commands, exact config parameters, or a `requirements.txt`), tell me which part and I'll add it.
