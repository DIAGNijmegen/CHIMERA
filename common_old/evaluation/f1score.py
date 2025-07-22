
"""
The following is a simple example evaluation method.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the evaluation, reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import json
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat, pprint

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("ground_truth")

def main():
    print_inputs()

    metrics = {}

    result_df = pd.DataFrame(columns=["case_id", "case_id_gt", "case_id_pred"])
    predictions = read_predictions()

    with Pool(processes=4) as pool:
        metrics["results"] = pool.map(process, predictions)
    
    result_df = pd.DataFrame(metrics["results"])
    
    y_true = result_df["case_id_gt"].astype(int).values
    y_prob = result_df["case_id_pred"].astype(float).values
    y_pred = (y_prob > 0.5).astype(int)

    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    metrics["aggregates"] = {
        "f1_score": f1,
        "auc": auc
    }

    write_metrics(metrics=metrics)

    return 0

def process(job):
    report = "Processing:\n"
    report += pformat(job) + "\n"

    location_prediction = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="brs3-probability",
    )

    pred_result = load_json_file(location=location_prediction)

    image_name = get_image_name(
        values=job["inputs"],
        slug="prostatectomy-tissue-whole-slide-image",
    )

    ground_truth_df = pd.read_csv(GROUND_TRUTH_DIRECTORY / "ground_truth.csv")
    gt_row = ground_truth_df[ground_truth_df["case_id"] == image_name]

    y_true = gt_row["brs3"].item()
    y_pred = pred_result

    return {
        "case_id": image_name,
        "case_id_gt": y_true,
        "case_id_pred": y_pred
    }

def print_inputs():
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]
    print("Input Files:")
    pprint(input_files)
    print("")

def read_predictions():
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())

def get_image_name(*, values, slug):
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]
    raise RuntimeError(f"Image with interface {slug} not found!")

def get_interface_relative_path(*, values, slug):
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]
    raise RuntimeError(f"Value with interface {slug} not found!")

def get_file_location(*, job_pk, values, slug):
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path

def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())

def write_metrics(*, metrics):
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    raise SystemExit(main())

