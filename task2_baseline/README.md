ABMIL_CHIMERA – Multimodal Fusion for CHIMERA Task 2
This repository contains a PyTorch implementation of an Attention-Based Multiple Instance Learning (ABMIL) fusion model for CHIMERA Challenge Task 2.
The model predicts BRS subtypes (BRS3 vs. BRS1/2) from histopathology whole-slide image (WSI) features and structured clinical data.

📌 Key Features
ABMIL_Fusion architecture: Combines pathology features (WSI embeddings) with clinical features.

Configurable Dropout & Gating for clinical feature scaling.

Weighted Sampling to address class imbalance.

Training, Validation, and Evaluation pipelines with early stopping and TensorBoard logging.

Bag Size Control for patch-level WSI features.

Ready for Grand Challenge-style inference containers.

📂 Repository Structure
yaml
Copy
Edit
ABMIL_CHIMERA/
│
├── configs/                     # Model configuration files
│   └── ABMIL_default/
│       └── config.json
│
├── data_factory/                 # Dataset configs & label mapping
│   └── cls_default.py
│
├── mil_models/                   # Model architectures
│   ├── model_abmil_fusion.py
│   ├── model_factory.py
│   └── tabular_snn.py
│
├── training/                     # Training scripts
│   ├── main_classification.py
│   └── trainer.py
│
├── utils/                        # Utility functions
│   ├── file_utils.py
│   ├── pandas_helper_funcs.py
│   ├── scheduler.py
│   └── utils.py
│
├── wsi_datasets/                 # WSI + clinical dataset loading
│   ├── dataset_utils.py
│   └── wsi_classification.py
│
├── training.py                   # Example runner for training
├── inference.py                  # Example inference script
├── check_val_predictions.py      # Check predictions on validation set



⚙️ Installation

git clone https://github.com/.../ABMIL_CHIMERA.git
cd ABMIL_CHIMERA
pip install -r requirements.txt


📑 Data Preparation
WSI Features: Extracted (e.g., with UNI / Slide2Vec) and stored as .pt or .h5 files.
Directory structure:

features/
├── 1001.pt
├── 1002.pt
└── ...


Clinical CSV: Contains per-sample clinical attributes and labels.
Example:


sample_id,label,age,sex,stage,...
1001,BRS3,65,M,T2,...
1002,BRS1,58,F,T1,...



🚀 Training

python -m training.main_classification \
  --train_csv /path/to/train.csv \
  --train_feats_dir /path/to/train_features \
  --val_csv /path/to/val.csv \
  --val_feats_dir /path/to/val_features \
  --model_config configs/ABMIL_default/config.json \
  --results_dir ./results/ \
  --bag_size 500 \
  --batch_size 1 \
  --max_epochs 20 \
  --model_type ABMIL_Fusion \
  --early_stopping



📊 Model Configuration
configs/ABMIL_default/config.json:

{
  "gate": true,
  "in_dim": 1024,
  "n_classes": 2
}

gate: Whether to scale clinical embeddings before fusion.

in_dim: Dimensionality of WSI features.

n_classes: Number of output classes.



📈 Monitoring
Training progress and metrics can be visualized via TensorBoard:

tensorboard --logdir ./results/


🧪 Inference
After training, you can run inference on new data:

python inference.py \
  --test_csv /path/to/test.csv \
  --feats_dir /path/to/features \
  --checkpoint ./results/s_checkpoint.pth \
  --output_csv ./predictions.csv

  
📄 Citation
If you use this code for research, please cite the CHIMERA Challenge and relevant ABMIL/UNI/Slide2Vec works.
