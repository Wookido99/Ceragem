# Ceragem Sleep Stage Toolkit

이 저장소는 웨어러블 및 PSG 데이터를 사용한 수면 단계 분류 실험을 위한 코드 베이스입니다.  
It consolidates the end-to-end training pipeline and dataset adapters used for DREAMT (CSV) as well as the PhysioNet CAPSLPDB/SleepBRL (EDF) corpora.

## Highlights
- Unified training entry point (`main.py`) with configurable datasets, features, and models.
- Dataset loaders for DREAMT CSV exports, CAPSLPDB EDF files, and SleepBRL EDF files (`datasets/`).
- Model zoo including an LSTM baseline, SleepTransformer, and DeepSleepNet implementations (`model.py`).
- Utility scripts for feature engineering, quality scoring, and raw signal alignment (`feature_engineering.py`, `calculate_quality_score.py`, `read_raw_e4.py`).

## Repository Layout
- `main.py` – CLI for training and evaluating sleep-stage models.
- `preprocess.py` – wraps dataset loaders into PyTorch `DataLoader` objects and manages default feature sets.
- `model.py` – definitions of supported neural architectures.
- `datasets/` – dataset-specific readers and preprocessing logic.
- `dataset_sample/` – lightweight DREAMT-style CSV samples plus metadata (`participant_info.csv`, `features_df/`, `E4_aggregate_subsample/`).
- `results/` – example outputs such as `quality_score_per_subject.csv`.
- `environment.yml` – conda environment spec (env name: `ceragem`).

## Quick Start
1. Clone the repository and move into the project directory.
2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ceragem
   ```
3. (Optional) Install this package in editable mode if you plan to import it elsewhere:
   ```bash
   pip install -e .
   ```

## Training Examples
Use `--data` to pick the dataset family and adjust directories as needed.

```bash
# DREAMT CSVs (expects Sleep_Stage column). Point --data_dir to your CSV folder.
python main.py --data dreamt --data_dir ./dataset_sample/E4_aggregate_subsample \
               --model lstm --epochs 5 --batch_size 16

# CAPSLPDB EDFs (uses default channel list; requires annotation files in the same folder)
python main.py --data capslpdb --data_dir /data/ceragem/physionet.org/files/capslpdb/1.0.0 \
               --model sleep_transformer --seq_len 15 --epochs 10

# SleepBRL EDFs with DeepSleepNet
python main.py --data sleepbrl --data_dir /data/ceragem/physionet.org/files/sleepbrl/1.0.0 \
               --model deepsleepnet --seq_len 20 --epochs 10
```

Helpful flags:
- `--features` – comma-separated override for feature columns (falls back to dataset defaults).
- `--epoch_seconds`, `--sample_rate` – control windowing when raw signals are re-sampled.
- `--frame_size`, `--frame_stride`, `--num_layers`, etc. – fine-tune architecture-specific hyperparameters.
- `python main.py --help` – displays the full list of options.

## Data Preparation
- DREAMT CSVs must include a `Sleep_Stage` column and the feature columns specified in `datasets/dreamt.py`.
- CAPSLPDB and SleepBRL loaders expect EDF signal files plus matching annotation text files in the same directory.  
  The defaults point to `/data/ceragem/physionet.org/files/...`; override with `--data_dir` if your paths differ.
- `dataset_sample/` contains miniature DREAMT-style artifacts for quick smoke tests but is not a full training set.

## Utility Scripts
- `feature_engineering.py` – generates feature-level CSVs from processed signals.
- `calculate_quality_score.py` – aggregates artifact ratios into `results/quality_score_per_subject.csv`.
- `read_raw_e4.py` – aligns Empatica E4 raw signals with staged labels and sleep metrics.

## Notes
- Results (models, metrics, checkpoints) are not committed; create your own output folder or reuse `results/`.
- Large PhysioNet assets are ignored by Git. Download them manually or mount the shared dataset directory before training.
