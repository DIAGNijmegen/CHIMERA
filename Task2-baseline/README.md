# CHIMERA 2025 - Task 2 Baseline: ABMIL Fusion (WSI + Clinical)

This repository provides the implementation of a baseline model for **CHIMERA 2025 MICCAI Grand Challenge – Task 2**, which focuses on predicting **Bacillus Calmette-Guérin (BCG) response subtypes (BRS1, BRS2, BRS3)** using:

- Histopathology whole slide images (WSIs)
- Clinical metadata (tabular format)

The baseline method uses **Attention-Based Multiple Instance Learning (ABMIL)** with a **clinical MLP encoder** and late fusion.

---

📁 Project Structure

```text
Task2-baseline/
├── configs/               # YAML configs for ABMIL model
├── data_factory/          # Dataset configuration for classification
├── mil_models/            # ABMIL_Fusion model definitions
├── training/              # Training loop and trainer logic
├── utils/                 # Helper utilities (metrics, schedulers, etc.)
├── wsi_datasets/          # WSI + clinical dataset loader
├── main_classification.py # Entry point for training
├── train_folds.py         # Script to run training over all folds (cross-validation)
├── inference.py           # Script for test-time inference
├── requirements.txt       # Python dependencies
└── README.md              # You're here
```

               


---

##  Getting Started

### 1. Install Requirements


pip install -r requirements.txt


### 2. Prepare Data


WSIs must be preprocessed into feature representations using slide2vec (https://github.com/clemsgrs/slide2vec/tree/master/slide2vec)

Clinical data must be in per-patient .json format or as a consolidated CSV with the required fields.


### 3. Train the Model

Model Overview:

The baseline model ABMIL_Fusion includes:

WSI branch: Attention-based MIL for WSI features.

Clinical branch: MLP (SNN-style with SELU and AlphaDropout).

Fusion: Concatenated latent representation from both branches before classification.


⚠️ Before running train_folds.py, make sure to:

Place feature files under data/features/

Place clinical fold CSVs under data/clinical_splits/fold_X/

Modify train_folds.py if your directory names differ

Task2-baseline/
├── data/
│   ├── clinical_splits/
│   │   ├── fold_0/
│   │   │   ├── train.csv
│   │   │   └── test.csv
│   └── features/
├── results/



## 📄 Citation


If you use this baseline in your work, please cite the CHIMERA challenge:

@misc{chimera2025,
  title={CHIMERA MICCAI 2025 Grand Challenge},
  author={DIAG Nijmegen},
  howpublished={\url{https://chimera.grand-challenge.org}},
  year={2025}
}



## 🙏 Acknowledgements


Model structure inspired by MIL frameworks used in computational pathology.

Feature extraction via slide2vec.


## 👩‍💻 Maintainer

``` text
📧 Nadieh Khalili       nadieh.khalili@radboudumc.nl
📧 Maryam Mohammadlou   maryam.mohammadlou@tuni.fi
```

