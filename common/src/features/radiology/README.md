# Radiology Feature Extraction Module 🧠

This module (`common/src/features/radiology/`) is responsible for extracting features from multimodal Magnetic Resonance Imaging (MRI) scans using a pre-trained nnU-Net v1 model ensemble. It serves as a critical preprocessing step for downstream multimodal fusion models.

## Purpose

The primary goal of this module is to:
1.  **Process MRI Scans:** Discover and handle different MRI modalities (T2-weighted, ADC, and HBV).
2.  **Extract Deep Features:** Utilize an ensemble of pre-trained nnU-Net models to generate feature vectors from the encoder bottleneck.
3.  **Prepare Features for Downstream Tasks:** Output the extracted features in a `.pt` file, ready for use in machine learning models for tasks like survival prediction or classification.

## Key Components

-   **`main.py`**: Contains the `run_radiology_feature_extraction` function, which serves as the main entry point for the Grand Challenge inference pipeline. It orchestrates the entire pipeline, from case discovery to processing and saving results.
-   **`stand_alone_feature_extractor.py`**: A script for running feature extraction on a local dataset (e.g., the training data). This script is more flexible for handling different local data structures.
-   **`processing.py`**: Provides the `process_single_case` function, which manages the core logic for a single patient case. It handles preprocessing, invokes the feature extraction, and saves the output.
-   **`feature_extraction.py`**: Implements `run_ensemble_feature_extraction`, which loads multiple model checkpoints (folds) and computes the averaged bottleneck features from the nnU-Net encoder.
-   **`dataset.py`**: Contains functions for data handling. `discover_cases` robustly finds patient cases from various directory structures, including the grand-challenge `inputs.json` format. `load_and_preprocess_case` prepares the MRI scans for the model, and `save_features` writes the final tensor to disk.
-   **`models.py`**: Defines functions to load the nnU-Net model configuration (`plans.pkl`) and construct the `Generic_UNet` architecture (`create_network_from_params`).

## Standalone Feature Extraction for Training Data

To extract features from your local training dataset, you can use the `stand_alone_feature_extractor.py` script. This script is designed to be run from the command line.

After downloading the radiology model weights and placing them in the `common/model/radiology/` directory (as explained in the `task1_baseline/README.md`), you can run the script as follows:

**Usage:**

```bash
python common/src/features/radiology/stand_alone_feature_extractor.py \
    --input_dir /path/to/your/training/mri_scans/ \
    --output_dir /path/to/your/radiology_features/ \
    --model_dir common/model/radiology/ \
    --mode features
```

This command will process the MRI scans, extract features using the downloaded model, and save them as `.pt` files in your specified output directory.

## Integration with Inference Pipelines

The `common/src/features/radiology/` module is designed to be called as a preliminary step in a larger inference workflow. A master script would typically invoke `run_radiology_feature_extraction`, providing paths to the input MRI data, the pre-trained model weights, and the desired output directory.

Here's how it integrates:

1.  **Initialization:** The main inference script calls `run_radiology_feature_extraction`.
2.  **Case Discovery:** The module automatically discovers all patient cases in the input directory.
3.  **Feature Extraction Loop:** For each case, `process_single_case` is called to load the MRI files, preprocess them into a tensor, and pass them through the model ensemble.
4.  **Output:** The `run_ensemble_feature_extraction` function generates a single, averaged feature vector for each case. These are collected and saved into a single `features.pt` file in the specified output directory, ready for the next stage of the pipeline.

This modular design ensures that the complex logic of radiology feature extraction is self-contained and easily reusable.

## Input Data

The module expects:
-   **MRI Scans:** `.mha` files for three modalities: T2-weighted (`*_t2w.mha`), ADC (`*_adc.mha`), and HBV (`*_hbv.mha`).
-   **Model Files:** A directory containing pre-trained nnU-Net v1 weights (e.g., `fold_0/model_best.model`) and a `plans.pkl` configuration file.
-   **Input Structure:** The module can flexibly handle different input layouts, from simple case folders to the `inputs.json` format used in grand-challenge.

## Output Data

The main output of this module is:
-   **Feature Tensor (`features.pt`):** A single PyTorch tensor file containing the extracted features for all processed cases. This file is saved in the specified output directory.
