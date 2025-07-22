# CHIMERA Baseline Template

This repository provides a **minimal working template** for participating in the [CHIMERA Challenge](https://chimera.grand-challenge.org/).  
It serves as a **starting point** for your own submission and implements the required boilerplate to run across all tasks in the challenge.

---

## 📁 Structure

Each task follows this structure:

- `inference.py`: Main entry point for processing inputs and generating outputs.
- `model/`: Placeholder for model-related resources.
  - `README.md`: Instructions for uploading or including models.
  - `a_tarball_subdirectory/`: Example subdirectory for tarball-based resources.
- `resources/`: Placeholder for any additional resources.
- `requirements.in` and `requirements.txt`: Define the Python dependencies for the project.
- `Dockerfile`: Specifies the container environment for running the algorithm.
- `do_build.sh`: Script to build the Docker container.
- `do_test_run.sh`: Script to test the container locally.
- `do_save.sh`: Script to save the container image and optional tarball for upload.

---

## 🚀 Getting Started

**System requirements:**  
- Linux-based OS (e.g., Ubuntu 22.04)  
- Python 3.10+  
- Docker installed

Depending on your preferred development setup, you can follow one of our tutorials:

## ⚙️ Running the Baseline Models Locally
To try out the baseline models on your local system, follow these steps:

Open a terminal or command prompt
Navigate to the directory where you want to clone the repository:
```
cd /path/to/your/desired/location
```
Clone the repository:
```
git clone https://github.com/nadieh/CHIMERA_minimal_baseline.git
```
Change to the task directory you want to run (e.g., Task1, Task2 or Task3):
```
cd /path/to/each/task
```
Follow the instructions provided [here](https://github.com/nadieh/CHIMERA_minimal_baseline/blob/main/CHIMERA-bladder-brs/model/README.md) to set up the necessary files. Then, to test the container locally, run:
```
./do_test_run.sh
```
This script launches Docker to execute the inference.py script.

## 🛠️ Customization
Modify inference.py to implement your own feature extraction or prediction logic.
Add your model weights to the model/ directory or upload them as a tarball to Grand Challenge.
Update requirements.in to include additional Python dependencies and regenerate requirements.txt using pip-compile.
## 📄 License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

