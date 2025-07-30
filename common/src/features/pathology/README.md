# Pathology Feature Extraction Module 🔬

This module (`common/src/features/pathology/`) is dedicated to the robust extraction of visual features from Whole Slide Images (WSIs) for pathology tasks. It serves as a foundational component for downstream prediction models, such as those found in `task1_baseline/prediction_model/Aggregators/inference/inference.py`.

## Purpose

The primary goal of this module is to:
1.  **Process Whole Slide Images:** Handle WSI loading, tissue segmentation, and tile extraction.
2.  **Extract Tile-Level Features:** Utilize pre-trained vision models to generate high-dimensional feature vectors from individual image tiles.
3.  **Prepare Features for Downstream Tasks:** Output these extracted features in a format suitable for machine learning models that perform tasks like survival prediction or classification.

## Key Components

-   **`main.py`**: Contains the `run_pathology_vision_task` function, which orchestrates the entire feature extraction process for a given WSI. This function handles WSI loading, tissue mask processing, tile coordinate extraction, and feature computation.
-   **`models.py`**: Defines the `UNI` class, a tile-level feature extractor. This class leverages pre-trained vision transformers (e.g., from `timm`) to generate embeddings from image tiles.
-   **`feature_extraction.py`**: Provides functions like `extract_tile_features` and `extract_features` that manage the inference process of the `UNI` model over batches of image tiles.
-   **`dataset.py`**: Implements `TileDataset` and `TileDatasetFromDisk` for efficient loading of WSI tiles, either directly from the WSI or from pre-saved tile images.
-   **`wsi.py`**: Contains classes and functions for WSI handling, including `WholeSlideImage` class, `TilingParams`, and `FilterParams` for managing tile extraction and tissue detection.
-   **`wsi_utils.py`**: Provides helper functions for tissue detection and filtering.

## Integration with Inference Pipelines

The `common/src/features/pathology/` module is a critical dependency for the main inference scripts across different tasks. For instance, in `task1_baseline/prediction_model/Aggregators/inference/inference.py`, the `run_complete_pipeline` function explicitly calls `common.src.features.pathology.main.run_pathology_vision_task`.

Here's how it typically integrates:

1.  **Initialization:** An instance of a pathology feature extractor (e.g., `UNI` from `models.py`) is initialized within the main inference script.
2.  **Feature Extraction Loop:** For each WSI requiring processing, the `run_pathology_vision_task` function is invoked. This function takes the WSI path, its corresponding tissue mask path, and the initialized feature extractor model as input.
3.  **Output:** The `run_pathology_vision_task` function generates a `.pt` file containing the extracted features for the WSI. These features are then collected and used by the downstream prediction model (e.g., the ABMIL model in `task1_baseline`).

This modular design ensures that the complex process of pathology WSI feature extraction is encapsulated and reusable across various tasks (e.g., `task2-baseline`, `task3-baseline`), promoting consistency and maintainability.

## Input Data

The module primarily expects:
-   **Whole Slide Image (WSI) files:** Typically in formats supported by `wholeslidedata` (e.g., SVS, TIFF).
-   **Tissue Mask files:** Binary masks corresponding to the WSIs, indicating tissue regions. These are crucial for efficient and relevant tile extraction.

## Output Data

The main output of this module, when used within an inference pipeline, is:
-   **Feature Tensors (`.pt` files):** PyTorch tensor files containing the extracted visual features for each WSI. These are typically saved to a temporary directory during the inference process and then aggregated for the final prediction.
