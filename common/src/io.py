import json
import orjson
from glob import glob
from pathlib import Path
from typing import Any
import numpy as np


def load_inputs(input_path: Path = Path("/input/inputs.json")) -> list[dict[str, Any]]:
    """
    Read information from inputs.json
    """
    input_information_path = Path(input_path)
    base_dir = input_information_path.parent
    with input_information_path.open("r") as f:
        input_information = json.load(f)

    for item in input_information:
        relative_path = item["interface"]["relative_path"]
        item["input_location"] = base_dir / relative_path

    return input_information


def resolve_image_path(*, location: str | Path) -> Path:
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    if len(input_files) != 1:
        raise ValueError(f"Expected one image file, got {len(input_files)}")

    input_file = Path(input_files[0])
    return input_file


if __name__ == "__main__":
    # Example usage for local testing
    print(load_inputs("task1_baseline/test/input/interf0/inputs.json"))