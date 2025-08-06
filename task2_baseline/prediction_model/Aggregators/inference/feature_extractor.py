# Aggregators/inference/feature_extractor.py
import torch
from pathlib import Path
from common.src.features.pathology.main import run_pathology_vision_task

def extract_pathology_features(input_dir: Path, output_dir: Path, model_dir: Path):
    """
    Extract pathology features using Slide2Vec or pre-trained MIL feature extractor.
    In GC, you can either run this or skip if you pre-extract features.
    """
    pathology_output_dir = output_dir / "pathology_features"
    pathology_output_dir.mkdir(parents=True, exist_ok=True)

    run_pathology_vision_task(
        input_dir=input_dir,
        output_dir=pathology_output_dir,
        model_dir=model_dir
    )

    return pathology_output_dir

