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

## Detailed Methods, Model & Training (full reproducibility)

1) Model architecture
- Base encoder: `bert-base-uncased` (HuggingFace Transformers).
- Wrapper: `BertForMultiLabel` — a lightweight classification head on top of BERT:
  - Pooling: uses BERT's `[CLS]` embedding (hidden state at index 0) or mean-pooling depending on the notebook cell.
  - Classification head: a single linear layer mapping encoder hidden-size (768) to `num_labels` (dataset dependent), followed by optional dropout (p=0.1).
  - Loss: binary cross-entropy with logits (`torch.nn.functional.binary_cross_entropy_with_logits`) for multi-label prediction.

2) Data preprocessing
- Input: CSVs with columns for text and multi-hot label vectors (or label lists). The notebook contains a preprocessing cell that:
  - tokenizes text using `BertTokenizer` with `truncation=True` and `max_length` (default 512 for full runs, lower for FAST runs).
  - builds `input_ids` and `attention_mask` tensors.
  - collate function: the notebook uses a custom `collate_fn`/DataLoader that may return dicts or tuples; the helper `unpack_batch` normalizes this to `(input_ids, attention_mask, labels)`.

3) Training loop and checkpointing
- Optimizer: `AdamW` (weight decay enabled) with learning rates typical for BERT fine-tuning:
  - Paper-scale runs: 2e-5 — 5e-5 for head-only; 1e-5 — 5e-6 when unfreezing encoder layers.
  - Demo runs used small LR (e.g., 5e-6) to stabilize short fine-tuning.
- Scheduler: optional linear warmup + decay (via `transformers.get_linear_schedule_with_warmup`).
- Batch size: depends on GPU; typical values: 8, 16, 32. FAST mode uses smaller batches.
- Epochs: full runs 10–40; FAST demo 1–5.
- Checkpointing:
  - Epoch-level checkpoints saved under `models/` with names like `BEAL_seed{seed}_round{n}.pt`.
  - Resume logic: the notebook's `LOAD_CHECKPOINT_IF_EXISTS` finds the latest matching checkpoint and loads model weights; `WARM_START` and `EXTRA_EPOCHS_ON_CHECKPOINT` control optimizer/extra training.

4) Evaluation & metrics
- Model outputs logits; probabilities are `sigmoid(logits)`.
- Default threshold: 0.5; the notebook includes threshold sweeps (0.1–0.6 or wider) to find better operating points.
- Metrics: micro-F1 (primary), macro-F1, per-class precision/recall, and calibration plots.

5) Exact hyperparameters used in demo runs (example)
- Model: `bert-base-uncased`
- Batch size: 16
- Learning rate: 5e-6 (head + last layers)
- Weight decay: 0.01
- Dropout: 0.1
- Optimizer: AdamW
- Epochs (FAST demo): 3
- Aggressive fine-tune example: up to 40 epochs with early stopping (patience=2)

6) Reproduce a FAST demo (exact notebook steps)
- Open `reproduce.ipynb` and run the setup cells.
- In the `FAST_CONFIG` cell set `FAST_VALIDATION=True` (the notebook merges FAST onto PAPER config).
- Optionally set `LOAD_CHECKPOINT_IF_EXISTS=False` to train from scratch.
- Run the main training cell — FAST mode will use smaller datasets and fewer epochs for a quick run.

7) Reproduce the targeted resume run used for debugging
- Place `models/BEAL_seed42_round*.pt` into `models/` in the D: copy or point the notebook to your checkpoints.
- Set `LOAD_CHECKPOINT_IF_EXISTS=True`, `WARM_START=True`, and `EXTRA_EPOCHS_ON_CHECKPOINT=5` (or desired value).
- Run the resume/fine-tune cell — it will load the latest checkpoint and run the extra epochs.

8) Headless commands for exact reproducibility
```powershell
cd D:\deeplearning_project
jupyter nbconvert --to notebook --execute reproduce.ipynb --output reproduce_run.ipynb --ExecutePreprocessor.timeout=14400
```

9) Files to add for stronger reproducibility
- `requirements.txt` (pinned versions)
- `configs/paper.yaml` and `configs/fast.yaml` with the exact config values used in the notebook
- `scripts/download_data.ps1` or `download_data.sh` to download large datasets / checkpoints

10) Troubleshooting notes
- If `micro-F1==0` at threshold 0.5 but loss decreases: run a threshold sweep; plot probability histograms; try longer fine-tuning or lower the LR.
- If the aggressive runner errors on batch shapes, run the data/model setup cells first and use the `unpack_batch` helper; inspect one batch (`repr(next(iter(train_loader)))`) to see exact structure.

11) Reproducibility checklist
- Fix seeds: `random`, `numpy`, `torch` (+ `torch.cuda.manual_seed_all`)
- Save the git commit hash and full config to `results/statistics.json` for each run
- Export `pip freeze > requirements-frozen.txt` once you lock the environment

If you'd like, I can now add a pinned `requirements.txt` and a `configs/` folder with `paper.yaml` and `fast.yaml` to the repo and push them. Tell me which you want me to add next.
