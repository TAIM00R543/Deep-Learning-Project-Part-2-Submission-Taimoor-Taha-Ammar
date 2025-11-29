# 🎨 Deep Learning Project — Reproducibility, Diagnostics & Visual Gallery

![demo-ready](https://img.shields.io/badge/demo-ready-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-%E2%9D%A4-red)
![transformers](https://img.shields.io/badge/transformers-HuggingFace-orange)

This README now includes an extended visual gallery, interpretation tips for each diagnostic, example configs and commands, and small helper scripts to generate and embed real visual artifacts into the repository. The intention is to make it obvious what to look for when reproducing or debugging model behavior.

Table of contents
- Project overview
- Visual gallery (with interpretation & actionable checks)
- How visuals are generated (notebook & script examples)
- Quick start (interactive & headless)
- Example configs (paper / fast) and recommended hyperparameters
- Pinned dependencies (suggested requirements.txt)
- Reproducibility checklist & tips
- Contributing, artifacts & storage recommendations
- Next steps I can add

Project overview (short)
------------------------
Reproduction artifact for "Deep Active Learning for Multi-Label". Main entry point: `reproduce.ipynb` — contains data loading, BERT-based multi-label model, training/resume logic, and diagnostic visualizations.

Visual gallery — what you should expect and how to read it
----------------------------------------------------------

Note: These inline images are illustrative (QuickChart generated). After you run the notebook you'll get real PNGs in `results/visualizations/`; replace these sample URLs by the actual files (preferably via relative links).

1) Probability histogram (per-class or aggregated)
![Probability Histogram](https://quickchart.io/chart?c=%7B%22type%22%3A%22bar%22%2C%22data%22%3A%7B%22labels%22%3A%5B%220.0-0.1%22%2C%220.1-0.2%22%2C%220.2-0.3%22%2C%220.3-0.4%22%2C%220.4-0.5%22%2C%220.5-0.6%22%2C%220.6-0.7%22%2C%220.7-0.8%22%2C%220.8-0.9%22%2C%220.9-1.0%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Probability%20count%22%2C%22data%22%3A%5B5%2C10%2C18%2C30%2C45%2C40%2C25%2C15%2C8%2C4%5D%2C%22backgroundColor%22%3A%22rgba(75%2C192%2C192%2C0.7)%22%7D%5D%7D%2C%22options%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Sample%20Probability%20Histogram%22%7D%7D%7D)

- What it shows: distribution of predicted probabilities across buckets.
- Actionable checks:
  - If almost all probabilities are near 0 or 1 (peaked) — model is confident; ensure calibration.
  - If probabilities cluster near 0 with low positives but loss decreasing — thresholding or class imbalance issues.
  - Inspect per-class histograms for imbalance or collapsed predictions.

2) Threshold sweep (micro-F1 vs threshold)
![Threshold Sweep](https://quickchart.io/chart?c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22labels%22%3A%5B%220.1%22%2C%220.2%22%2C%220.3%22%2C%220.4%22%2C%220.5%22%2C%220.6%22%2C%220.7%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22micro-F1%22%2C%22data%22%3A%5B0.42%2C0.47%2C0.53%2C0.56%2C0.51%2C0.44%2C0.30%5D%2C%22borderColor%22%3A%22%233e95cd%22%2C%22fill%22%3Afalse%7D%5D%7D%2C%22options%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Threshold%20Sweep%20%28sample%29%22%7D%7D%7D)

- What it shows: how micro-F1 varies as you sweep the binary threshold applied to sigmoid outputs.
- Actionable checks:
  - If best threshold ≠ 0.5, prefer the better threshold for final metrics.
  - Use this to set operational trade-offs (precision vs recall).

3) Reliability diagram / calibration plot
![Reliability Diagram](https://quickchart.io/chart?c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22labels%22%3A%5B%220.0%22%2C%220.1%22%2C%220.2%22%2C%220.3%22%2C%220.4%22%2C%220.5%22%2C%220.6%22%2C%220.7%22%2C%220.8%22%2C%220.9%22%2C%221.0%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22observed%22%2C%22data%22%3A%5B0.02%2C0.08%2C0.15%2C0.25%2C0.38%2C0.52%2C0.61%2C0.75%2C0.82%2C0.92%2C0.99%5D%2C%22borderColor%22%3A%22%23FF6F61%22%2C%22fill%22%3Afalse%7D%2C%7B%22label%22%3A%22perfect%22%2C%22data%22%3A%5B0%2C0.1%2C0.2%2C0.3%2C0.4%2C0.5%2C0.6%2C0.7%2C0.8%2C0.9%2C1.0%5D%2C%22borderColor%22%3A%22%2300A86B%22%2C%22borderDash%22%3A%5B5%2C5%5D%2C%22fill%22%3Afalse%7D%5D%7D%2C%22options%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Reliability%20Diagram%20%28sample%29%22%7D%7D%7D)

- What it shows: average observed frequency vs predicted probability (calibration).
- Actionable checks:
  - If observed < predicted (curve below diagonal): model over-confident — consider temperature scaling or calibration techniques.
  - If observed > predicted: model under-confident — may need better training or a different threshold.

4) Loss & F1 curves with std bands (training/validation)
![Loss & F1 Curves](https://quickchart.io/chart?c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22labels%22%3A%5B%221%22%2C%222%22%2C%223%22%2C%224%22%2C%225%22%2C%226%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22train_loss%22%2C%22data%22%3A%5B1.2%2C0.9%2C0.8%2C0.75%2C0.72%2C0.70%5D%2C%22borderColor%22%3A%22%237b9acc%22%7D%2C%7B%22label%22%3A%22val_micro_f1%22%2C%22data%22%3A%5B0.30%2C0.40%2C0.45%2C0.50%2C0.51%2C0.49%5D%2C%22borderColor%22%3A%22%23e76f51%22%7D%5D%7D%7D)

- What it shows: training loss declining and validation F1. Std bands (if multiple seeds) give uncertainty estimate.
- Actionable checks:
  - Diverging loss/F1 patterns → overfitting or data mismatch.
  - No validation improvement but training loss decreases → try lower LR, regularize, or inspect labels.

5) Per-class precision/recall & co-occurrence heatmap
- What it shows: which labels the model struggles with; which label pairs commonly co-occur.
- Actionable checks:
  - Low precision & high recall for a class → many false positives, perhaps noisy labels.
  - Co-occurrence heatmap can reveal label dependencies worth modeling explicitly.

How visuals are generated (notebook & script examples)
------------------------------------------------------
The notebook contains cells that compute metrics and save plots using matplotlib/seaborn. Example snippet (from notebook) that saves a histogram:

```python
# inside reproduce.ipynb
plt.figure(figsize=(8,4))
sns.histplot(probabilities.flatten(), bins=20, kde=False, color='teal')
plt.title("Probability histogram (aggregated)")
plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/visualizations/prob_hist.png", dpi=150)
```

Small helper script to regenerate all visualizations from a serialized eval file (example: `results/eval_outputs.json`):

```bash
# scripts/generate_visuals.sh
#!/usr/bin/env bash
set -e
python scripts/plot_from_eval.py --eval results/eval_outputs.json --outdir results/visualizations
```

- `scripts/plot_from_eval.py` should load JSON or npz saved by the notebook and create the charts (examples included in notebook). If you want, I can add this script for you.

Embedding real visuals in README
-------------------------------
Two recommended ways:
1. Commit generated PNGs under `results/visualizations/` and reference them with relative links:
   - `![Probability Histogram](results/visualizations/prob_hist.png)`
   - Pros: simple, self-contained.
   - Cons: increases repo size; use Git LFS for many/large images.

2. Host images externally (GitHub Releases, S3, or an image CDN) and link to absolute URLs:
   - Pros: keeps repo small.
   - Cons: external dependency.

Quick start (interactive)
-------------------------
1. Create & activate venv (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt  # if you added it
```

2. Open the notebook:

```bash
cd <repo-root>
jupyter notebook reproduce.ipynb
```

3. Recommended notebook order:
- Run top-level imports and dataset loader cells.
- Configure mode: set FAST vs PAPER config cell.
- Run training cell; run evaluation cell that saves `results/eval_outputs.json` and PNGs.

Quick start (headless / CI)
---------------------------
Execute headlessly with nbconvert:

```bash
jupyter nbconvert --to notebook --execute reproduce.ipynb \
  --output reproduce_run.ipynb \
  --ExecutePreprocessor.timeout=14400
```

Batch script to run FAST demo and then generate visuals:

```bash
# scripts/run_fast_and_visualize.sh
jupyter nbconvert --to notebook --execute reproduce.ipynb --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=python3 \
  --TagRemovePreprocessor.enabled=False
bash scripts/generate_visuals.sh
```

Example configs (recommended YAML files)
--------------------------------------
Store `configs/paper.yaml` and `configs/fast.yaml` and load them in the notebook to avoid re-editing cells.

Example: configs/fast.yaml

```yaml
mode: fast
seed: 42
max_length: 128
batch_size: 16
learning_rate: 5e-6
weight_decay: 0.01
dropout: 0.1
epochs: 3
fast_validation: true
dataset_subsample: 0.05  # use 5% of training set for FAST demo
```

Example: configs/paper.yaml

```yaml
mode: paper
seed: 42
max_length: 512
batch_size: 16
learning_rate: 2e-5
weight_decay: 0.01
dropout: 0.1
epochs: 20
fast_validation: false
dataset_subsample: 1.0
```

Suggested pinned dependencies (example requirements.txt)
--------------------------------------------------------
Create `requirements.txt` to freeze the tested environment. Example (you may want to update versions):

```text
torch>=1.12.0,<2.1
transformers>=4.15.0,<5.0
scikit-learn>=1.0.2
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
nbconvert>=6.0.0
tqdm
pyyaml
```

Reproducibility checklist & tips
--------------------------------
- Fix seeds: random, numpy, torch, torch.cuda.manual_seed_all.
- Record commit SHA and config used in `results/statistics.json`.
- Save env: `pip freeze > requirements-frozen.txt`.
- Save model checkpoints with descriptive names (seed, round, mode).
- For downstream reporting, save raw predictions and label indexes to JSON/npz for post-hoc analysis.

Storage & artifact hygiene
--------------------------
- Use Git LFS for large CSVs, PNGs, or model checkpoints (files > 50 MB).
- Preferred options for sharing large artifacts: GitHub Releases, Google Drive, S3, or Zenodo for citation.
- If you accidentally pushed large files and want to remove them, consider the BFG Repo Cleaner.

Contributing & next steps
-------------------------
I can:
- Add `configs/paper.yaml` and `configs/fast.yaml` to the repo and update the notebook to load them.
- Add `requirements.txt` with pinned versions and `requirements-frozen.txt`.
- Add `scripts/generate_visuals.sh` and `scripts/plot_from_eval.py` (to produce the gallery PNGs from notebook outputs).
- Generate a small set of sample visuals, commit them under `results/visualizations/`, and open a PR.

Tell me which of the above you want next (configs, requirements, scripts, or sample visual artifacts) and I'll prepare the files and a PR for you.
