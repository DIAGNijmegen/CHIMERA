# Vision Pathology 🧑‍🔬🔬

This repository provides a baseline solution for pathology vision tasks. Currently, it supports the following models:

- **TITAN** as a slide-level encoder for classification and regression tasks.
- **CONCH** as a patch-level encoder for detection and segmentation tasks.

At the moment, the code supports Task 1: Classifying HE Prostate Biopsies into ISUP Scores. Support for additional pathology vision tasks will be included soon.

### Input Data 📥
The code for encoding takes as input:
- ``unicorn-task-description.json``: This file contains the task-specific metadata. For more details, check the ``example-data`` folder. Based on the information provided in this file, the code will load the necessary input images from the correct folder in the archives.

### Output Data 📤
The code generates the following output files:
- ``image-neural-representation.json`` for slide-level tasks.
- ``patch-neural-representation.json`` for patch-level tasks.
For more details on the structure of these files, please refer to the ``example-data`` folder.

⚠️ **This code is under development, and additional features and support for more tasks will be added soon.**
