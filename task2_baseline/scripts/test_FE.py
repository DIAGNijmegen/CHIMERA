import sys
from pathlib import Path

# Add prediction_model to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "prediction_model"))

from inference import run_inference

if __name__ == "__main__":
    input_dir = Path("/input")
    output_dir = Path("/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print(" Starting Task 2 Classification Inference")
    print(f" Input directory: {input_dir}")
    print(f" Output directory: {output_dir}")
    print("========================================")

    # Run inference
    run_inference(input_dir=input_dir, output_dir=output_dir, debug=False)

    print("✅ Inference finished.")
