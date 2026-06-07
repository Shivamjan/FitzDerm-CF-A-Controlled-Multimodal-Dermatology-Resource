# FitzDerm-CF: A Controlled Multimodal Dermatology Resource for Skin-Tone Counterfactual and Leakage Evaluation

> A controlled multimodal dermatology benchmark that augments Fitzpatrick17k and DDI with temperature-controlled synthetic clinical notes and validated skin-tone counterfactuals for studying fairness, diagnostic leakage, retrieval, and multimodal robustness.

---

## Table of Contents

* [Overview](#overview)
* [Repository Structure](#repository-structure)
* [Datasets](#datasets)
* [Getting Started](#getting-started)
* [Dataset Generation Pipeline](#dataset-generation-pipeline)
* [Benchmark Tasks](#benchmark-tasks)
* [Reproducing Experiments](#reproducing-experiments)
* [License](#license)

---

## Overview

Modern dermatology vision-language models rely on paired image-text data. However, existing dermatology datasets often lack realistic clinical notes and provide limited control over demographic confounding and diagnostic leakage.

**FitzDerm-CF** addresses these limitations by augmenting dermatology image datasets with synthetic clinical descriptions generated from structured metadata while carefully suppressing explicit diagnostic disclosure.

Key features include:

* Temperature-controlled clinical note generation
* Explicit control of semantic capacity through generation temperature
* Validated skin-tone counterfactual descriptions
* Diagnostic leakage benchmarking
* Retrieval-based fairness evaluation
* Multimodal classification benchmarks
* Counterfactual consistency evaluation

The resource is designed to support research on:

* Fairness in dermatology AI
* Multimodal learning
* Counterfactual robustness
* Representation leakage
* Skin-tone bias auditing

---

## Repository Structure

```text
FitzDerm-CF/
├── Generating Texts and Their  Counterfactuals/
│   ├── generating text.py
│   ├── generating counterfactual.py
│
├── Text Embedding/
│   ├── getting embeddings.ipynb
│
├── Experiments/
│   ├── probe_benchmark.py
│   ├── generate_embddings.py
│   ├── retrieval_leakage.py
|   ├── multimodal_classification.py
│   
│
├── Cleaning Metadata for Training/
│   ├── clean_fitz.ipynb
│   ├── clean_ddi.ipynb
│
├── LICENSE
└── README.md
```

---

## Datasets

FitzDerm-CF is constructed from two publicly available dermatology datasets.

### Fitzpatrick17k

* ~17,000 dermatology images
* Fitzpatrick skin type annotations (I–VI)
* Diverse dermatological conditions
* Significant skin-tone imbalance

Dataset:
https://github.com/mattgroh/fitzpatrick17k

---

### DDI (Diverse Dermatology Images)

* Curated dermatology benchmark
* Broad skin-tone diversity
* Expert-verified diagnoses
* Designed for fairness evaluation

Dataset:
https://ddi-dataset.github.io/

---

### Required Metadata

For every image:

* Fitzpatrick skin type
* Disease label
* Malignancy status

These attributes are used solely for controlled note generation.

<!-- ---

## Data Directory Layout

```text
data/
├── Fitzpatrick17k/
│   ├── images/
│   └── metadata.csv
│
└── DDI/
    ├── images/
    └── metadata.csv
```

--- -->

## Getting Started

### Prerequisites

* Python 3.10+
* CUDA-enabled GPU (recommended)
* HuggingFace Transformers
* PyTorch
* Jupyter Notebook

<!-- ### Installation

```bash
git clone https://github.com/<username>/FitzDerm-CF.git

cd FitzDerm-CF

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
``` -->

---

## Dataset Generation Pipeline

The complete generation pipeline consists of four stages.

### Stage 1 — Metadata Preparation

Clean and standardize metadata from both datasets.

```bash
metadata_cleaning/
├── clean_fitzpatrick17k.ipynb
└── clean_ddi.ipynb
```

Outputs:

* Unified metadata schema
* Disease labels
* Malignancy annotations
* Fitzpatrick type mappings

---

### Stage 2 — Factual Clinical Note Generation

Generate synthetic dermatology notes using BioMistral-7B.

Generation is conditioned on:

* Fitzpatrick type
* Disease label
* Malignancy status

The following information is intentionally excluded:

* Explicit disease names
* Patient identifiers
* Treatment recommendations
* Diagnostic certainty statements

Temperature settings:

```text
τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
```

Result:

```text
5 factual notes per image
```

---

### Stage 3 — Counterfactual Note Generation

For each accepted factual note:

1. Change only the Fitzpatrick skin type.
2. Preserve disease information.
3. Preserve malignancy context.
4. Rewrite only skin-tone-dependent descriptors.

Example:

```text
Original:
Lightly pigmented erythematous lesion...

Counterfactual:
Darkly pigmented erythematous lesion...
```

The image itself is never modified.

---

### Stage 4 — Validation Pipeline

Every counterfactual passes four validation stages.

#### 1. No-op Rejection

Reject unchanged notes.

#### 2. Semantic Similarity Bounds

Require:

```text
0.85 < cosine_similarity < 0.93
```

using MiniLM embeddings.

#### 3. Causal Invariance

Reject changes that alter:

* disease risk
* epidemiological associations
* causal claims

#### 4. Skin-Tone Intervention Check

Require at least one valid skin-tone-related modification.

Samples failing validation after multiple retries are discarded.

<!-- --- -->
<!-- 
## Benchmark Tasks

### 1. Leakage Probing

Measure recoverability of:

* Disease labels
* Malignancy status
* Fitzpatrick skin type

using:

* Logistic Regression
* MLP
* KNN

Embeddings:

* MiniLM
* BioMedCLIP

Metrics:

* Accuracy
* Macro-F1
* ROC-AUC

---

### 2. Retrieval Leakage Analysis

Evaluate demographic clustering in embedding space.

Metrics:

* Purity@k
* Excess Purity@k
* Hit@k
* NDCG@k

Neighborhood sizes:

```text
k ∈ {1, 5, 10}
```

Primary metric:

```text
Purity@10
```

---

### 3. Multimodal Classification

Benchmark multimodal prediction using:

* Images
* Generated clinical notes

Tasks:

* Disease classification
* Malignancy prediction

Metrics:

* Balanced Accuracy
* Macro-F1

---

### 4. Counterfactual Consistency

Evaluate prediction stability under:

```text
Factual Note
        ↓
Counterfactual Note
```

while holding:

* image
* disease
* malignancy

constant.

Performance shifts quantify sensitivity to skin-tone language.

---

## Reproducing Experiments

### Generate Notes

```bash
python generate_factual_notes.py
```

### Generate Counterfactuals

```bash
python generate_counterfactuals.py
```

### Run Validation

```bash
python validate_counterfactuals.py
```

### Create Embeddings

```bash
python generate_embeddings.py
```

### Leakage Probing

```bash
python experiments/leakage_probing/run_probe.py
```

### Retrieval Evaluation

```bash
python experiments/retrieval_analysis/run_retrieval.py
```

### Multimodal Classification

```bash
python experiments/multimodal_classification/train_classifier.py
```

### Counterfactual Evaluation

```bash
python experiments/counterfactual_consistency/run_cf_eval.py
```

---





## Citation

```bibtex
@inproceedings{jangid2026fitzdermcf,
  title={FitzDerm-CF: A Controlled Multimodal Dermatology Resource for Skin-Tone Counterfactual and Leakage Evaluation},
  author={Jangid, Shivam and Ansari, Faizanuddin and Das, Swagatam},
  booktitle={Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM)},
  year={2026}
}
```

---
 -->


## License

This project is released under the MIT License.

---

## Contact

For questions regarding the dataset, benchmarks, or implementation:

* Shivam Jangid
* Indian Statistical Institute, Kolkata

Please open an issue in the repository or contact the authors through GitHub.

---

*This repository accompanies the CIKM 2026 paper(under review) "FitzDerm-CF: A Controlled Multimodal Dermatology Resource for Skin-Tone Counterfactual and Leakage Evaluation."*
