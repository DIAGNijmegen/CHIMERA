# CHIMERA: Combining HIstology, Medical Imaging (Radiology), and molEcular Data for Medical pRognosis and diAgnosis.

Welcome to the CHIMERA challenge repository! This challenge aims to accelerate research in multimodal AI for cancer prognosis by providing standardized tasks integrating histopathology, radiology, clinical, and transcriptomic data.

## 🚀 Overview

CHIMERA is a MICCAI 2025 challenge consisting of **three multimodal tasks**:

| Task | Description                                                                 |
|------|-----------------------------------------------------------------------------|
| 1    | Predict biochemical recurrence in prostate cancer using H&E + MRI + clinical data |
| 2    | Predict BCG response subtype (BRS) in bladder cancer from H&E + clinical data |
| 3    | Predict recurrence in NMIBC using H&E + RNA-seq + clinical dat          |

All tasks involve gigapixel WSIs, structured data, and standard clinical endpoints.

## ⚖️ Model Weights

The pre-trained model weights required to run the inference pipelines are not included directly in this repository due to their size. You must download the necessary files and place them in the correct directory structure as explained in the `task1_baseline/README.md`.

### Pathology Feature Extractor (UNI)

*   **Description:** These are the weights for the `UNI` model used for extracting features from Whole Slide Images (WSIs).
*   **Download:** [MahmoodLab/UNI on Hugging Face](https://huggingface.co/MahmoodLab/UNI/tree/main)
*   **Files to download:** `model.bin` and `config.json`.

### Radiology Feature Extractor (nnU-Net)

*   **Description:** These are the weights for the `nnU-Net v1` model ensemble used for extracting features from MRI scans.
*   **Download:** [PI-CAI nnU-Net Baseline on GitHub](https://github.com/DIAGNijmegen/picai_nnunet_semi_supervised_gc_algorithm/tree/master/results/nnUNet/3d_fullres/Task2203_picai_baseline/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1)
*   **Files to download:** `plans.pkl` and the `model_best.model` file from each of the five `fold_*` subdirectories.

### Prediction Models (ABMIL)

*   **Description:** The final prediction models (e.g., ABMIL) for each task are not provided pre-trained. You must train them yourself using the provided scripts.
*   **Instructions:** Please refer to the `README.md` file within each task's directory for detailed instructions on how to train the model and generate the necessary weight files:
    *   **Task 1:** See `task1_baseline/README.md`
    *   **Task 2:** See `task2-baseline/README.md`
    *   **Task 3:** See `task3-baseline/README.md`

After downloading the feature extractor weights and training your own prediction models, please follow the instructions in the task-specific `README.md` files to place all assets in the correct `common/model/` subdirectories.