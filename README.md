# A Prototypical Signature Approach for Writer-Independent Offline Signature Verification

This repository contains the official implementation of the experiments presented in our paper.

It provides tools for:

* Generating **dissimilarity-based datasets** using prototypical signatures
* Training and testing **writer-independent classifiers** (SVM or SGD)
* Reproducing **validation and test experiments** reported in the paper
* Automatically producing **prediction files** and **EER metrics** for all datasets

---

## 1. Installation

This repository requires **Python ≥ 3.8**.

Install the required dependency:

```bash
pip install git+https://github.com/kdmoura/stream_hsv.git
```

Clone this repository:

```bash
git clone https://github.com/kdmoura/proto_hsv.git
cd proto_hsv
```

---

## 2. Repository Structure

```
proto_hsv/
│
├── main_process.py        # Main training/validation/test pipeline
├── prototype_model.py     # Prototype model 
├── reproduce.py           # Full reproduction of paper results
├── util.py                # Helper functions
├── README.md              # This file
```

---

## 3. Datasets

Experiments are performed on the three datasets used in the paper: **GPDS-S**, **CEDAR**, and **MCYT**.

Each dataset must be preprocessed to extract deep features using the feature extractor described in the paper. They should:
1. Undergo preprocessing as outlined in https://github.com/tallesbrito/contrastive_sigver.
2. Have features extracted using the SigNet-S available in the same repository.
3. The resulting data should be a single .NPZ file containing:
   - `features`: shape (samples, features)
   - `y`: writer ID
   - `yforg`: 1 for forgery, 0 otherwise
   
---

## 4. Running Individual Experiments

To:

* compute prototypes
* generate dissimilarity training/validation/test data
* train the chosen classifier
* produce prediction files
* compute **EER (global + user thresholds)**

### 4.1. Validation (for selecting *k*)

Example:

```bash
python main_process.py \
    --cluster-algo kmeans \
    --n-clusters 150 \
    --model-choice svm \
    --dist-type poscentroid \
    --f-pred-path /path/to/pred_val_folder \
    --f-metric-path /path/to/metric_val_folder \
    --input-feat-path /path/to/features.npz \
    --dev-users 300 581 \
    --perform-validation
```

### 4.2. Testing (final experiment)

Example:

```bash
python main_process.py \
    --cluster-algo kmeans \
    --n-clusters 100 \
    --model-choice sgd \
    --dist-type poscentroid \
    --f-pred-path /path/to/pred_test_folder \
    --f-metric-path /path/to/metric_test_folder \
    --input-feat-path /path/to/features.npz \
    --exp-users 0 300 \
    --dev-users 300 581
```


---

## 5. Full Reproducibility Pipeline

The **complete reproduction** of results (validation + test for all datasets, and all model choices) is handled by:

```bash
python reproduce.py
```

This script:

1. Creates the required folders:

   ```
   pred_test/
   pred_val/
   metric_test/
   metric_val/
   ```
2. Runs the **validation protocol** for all datasets
3. Runs the **test protocol** with the selected best-*k* values
4. Produces prediction files and metrics as reported in the paper

To use it, update the paths in `reproduce.py`:

```python
gpdss_npz_path = "path/to/gpdss_features.npz"
mcyt_npz_path  = "path/to/mcyt_features.npz"
cedar_npz_path = "path/to/cedar_features.npz"
```

### Important Note:

While the script is fully configured and can be run as-is, it is **not advisable** to execute it sequentially due to the significant amount of time this would take. To optimize your workflow, it is recommended to adapt the execution by parallelizing the processes.

---


