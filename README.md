# Parkinson's Disease Detection from Speech via Domain Adaptation

Systematic benchmark of domain adaptation (DA) methods for detecting Parkinson's disease (PD) from speech recordings across distinct vocal tasks on the **PC-GITA** corpus, comparing handcrafted acoustic features, log-mel spectrogram CNNs, and frozen **wav2vec 2.0 / HuBERT** speech embeddings under strict patient-wise evaluation.

## Overview

Models trained to detect Parkinson's disease from one speech task (e.g., sustained vowels or diadochokinetic syllable repetition) often degrade when evaluated on another speech task (e.g., sentence reading or spontaneous monologue). This project implements and compares classical classifiers, deep feature-alignment methods, adversarial domain adaptation, and pretrained self-supervised speech representations across **pairwise cross-domain** and **leave-one-domain-out multi-source** protocols.

| Component | Implementation |
| --- | --- |
| **Acoustic Feature Pipeline** | 47 time/frequency/cepstral features (`MFCC`, `RMS`, `ZCR`, spectral centroid/bandwidth/rolloff/flatness/entropy/flux, voiced/silence ratios) + `LogReg`, `SVM`, `ExtraTrees`, `MLP` |
| **Feature-Level Domain Adaptation** | `Linear CORAL`, `Deep CORAL`, `MMD`, `DANN` (Domain-Adversarial Neural Network), `CDAN` |
| **Spectrogram CNNs** | Log-mel spectrograms (`4s`, `64 mels`) + `CNN-Source`, `CNN-DANN`, `CNN-MMD`, `CNN-CORAL` |
| **Pretrained Speech Embeddings** | Frozen `facebook/wav2vec2-base` / `hubert-base-ls960` representations (`mean+std` pooling, 1536-dim) evaluated across all tabular DA algorithms |
| **Evaluation Protocol** | Strict **patient-wise** splits (zero speaker overlap between train and test; target labels never used during adaptation) |

Detailed experimental tables and analysis are available in [`PCGITA_EXPERIMENTS.md`](PCGITA_EXPERIMENTS.md).

## Key Results (5-Seed Patient-Wise Evaluation)

### In-Domain 5-Fold Patient-Wise Cross-Validation (Upper Bound)

| Speech Task (Domain) | Mean ROC AUC |
| --- | ---: |
| `monologue` | 0.920 |
| `words` | 0.878 |
| `sentences2` | 0.862 |
| `sentences` | 0.858 |
| `read_text` | 0.838 |
| `vowels` | 0.794 |
| `ddk` | 0.768 |

### Pairwise Cross-Task Transfer (Source → Target)

| Source → Target | Best Method | Mean AUC | Mean F1 |
| --- | --- | ---: | ---: |
| `sentences2 -> words` | `coral_linear` | 0.870 | 0.762 |
| `sentences -> read_text` | `extra_trees` | 0.803 | 0.812 |
| `read_text -> sentences` | `extra_trees` | 0.779 | 0.654 |
| `words -> sentences2` | `extra_trees` | 0.776 | 0.769 |
| `vowels -> ddk` | `logreg_source` | 0.739 | 0.708 |
| `ddk -> vowels` | `extra_trees` | 0.702 | 0.600 |

### Multi-Source Leave-One-Task-Out Transfer

| Target Domain | Best Method | Mean AUC | Mean F1 |
| --- | --- | ---: | ---: |
| `monologue` | `extra_trees` | 0.900 | 0.830 |
| `sentences` | `extra_trees` | 0.850 | 0.805 |
| `sentences2` | `extra_trees` | 0.816 | 0.814 |
| `read_text` | `extra_trees` | 0.782 | 0.781 |
| `words` | `extra_trees` | 0.775 | 0.624 |
| `ddk` | `logreg_source` | 0.734 | 0.696 |
| `vowels` | `extra_trees` | 0.690 | 0.619 |

## Report Plots (`reports/pcgita/`)

Pre-rendered figures and summary CSVs are stored in [`reports/pcgita/`](reports/pcgita/):
- `01_pairwise_best_auc.png` — Best model AUC per `source -> target` pair
- `02_multisource_best_auc.png` — Leave-one-task-out multi-source AUC per target domain
- `03_spectrogram_cnn_best_auc.png` — Spectrogram CNN adaptation performance
- `04_pairwise_all_methods_heatmap.png` — Full method × pair comparison matrix
- `05_domain_gap_auc.png` — In-domain vs. out-of-domain generalization gap
- `06_cnn_vs_acoustic_auc.png` — Spectrogram CNNs vs. handcrafted acoustic features
- `09_wav2vec2_vs_acoustic_auc.png` — Pretrained wav2vec 2.0 embeddings vs. acoustic features

## Getting Started

```sh
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# Regenerate all report plots from cached experiment CSVs
python make_report_plots.py

# List all experiment runner commands
python main.py --commands
```

## License

[MIT](LICENSE)
