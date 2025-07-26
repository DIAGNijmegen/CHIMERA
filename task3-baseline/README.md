Aggregators/
│
├── configs/                      ← Contains model hyperparameter settings
│   ├── ABMIL_default/
│   │   └── config.json           ← Used by ABMILConfig to build model
│   └── ABMIL_tiny/               ← Smaller config version
│
├── mil_models/                   ← All MIL model definitions
│   ├── model_abmil.py           ← ABMIL architecture (you improve this!)
│   ├── model_factory.py         ← Links model_type (e.g. "ABMIL") to model class
│   └── components.py            ← Reusable layers: attention, MLP, etc.
│
├── training/                     ← Scripts to train and evaluate models
│   ├── main_classification.py   ← Main CLI script to train ABMIL
│   └── trainer.py               ← Training loop, validation logic
│
├── data_factory/                 ← Dataset loader factory
│   └── cls_default.py           ← Creates datasets using WSI features and labels
│
├── wsi_datasets/                ← Dataset definitions
│   └── wsi_classification.py    ← Defines how bags and labels are handled
│
├── utils/                        ← Helper functions and tools
│   ├── file_utils.py            ← For saving/loading models and pkl files
│   ├── losses.py                ← Loss functions (e.g., cross-entropy)
│   └── scheduler.py             ← Learning rate scheduling utilities
│
├── visualization/               ← Optional attention or embedding plots
│   └── prototype_visualization_utils.py
│
├── runABMIL_chimera.ipynb       ← Notebook to run ABMIL interactively
└── test_survival.ipynb          ← Notebook for testing survival analysis


