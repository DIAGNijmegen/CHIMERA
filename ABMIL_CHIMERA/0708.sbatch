#!/bin/bash
#SBATCH --job-name=nnunet_cpu_job
#SBATCH --qos=low
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/maryammohamm/task2_CHIMERA2/ABMIL_CHIMERA/log/0408/train_%j.out
#SBATCH --error=/home/maryammohamm/task2_CHIMERA2/ABMIL_CHIMERA/log/0408/train_%j.err
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dodrio1.umcn.nl#catherine/nvidiapytorch_asap_wsd:npt2410_asap22_wsd241115"

pip install transformers lifelines openpyxl \
    pandas==2.1.4 numpy==1.26.4 scipy==1.12.0 matplotlib==3.8.4
cd /home/maryammohamm/task2_CHIMERA2/ABMIL_CHIMERA &&
python3 training.py
