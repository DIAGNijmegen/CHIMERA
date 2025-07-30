

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
import random
from multiprocessing import Pool
from statistics import mean
from pathlib import Path
from pprint import pformat, pprint

import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_censored


INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("ground_truth")

#INPUT_DIRECTORY = Path("test/input")
#OUTPUT_DIRECTORY = Path("test/output")
#GROUND_TRUTH_DIRECTORY = Path("ground_truth")


def main():
    print_inputs()

    metrics = {}


    result_df = pd.DataFrame(columns=["case_id", "case_id_gt_time", "case_id_gt_event","case_id_prediction_years_to_recurrence"])
    predictions = read_predictions()



    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Start a number of process workers, using multiprocessing
    # The optimal number of workers ultimately depends on how many
    # resources each process() would call upon
    with Pool(processes=4) as pool:
        metrics["results"] = pool.map(process, predictions)
    
    result_df = pd.DataFrame(metrics["results"])
    
    survival_times = np.array(result_df["case_id_gt_time"])
    events = np.array(result_df["case_id_gt_event"], dtype=bool)
    predicted_times = np.array(result_df["case_id_prediction_years_to_recurrence"])
    #print("events, survival_times, predicted_times",events, survival_times, predicted_times)
  
    # Negating the predicted times to convert them into risk scores before calculating the concordance index. 
    # This ensures the concordance index correctly interprets higher scores as lower risk.
    c_index = concordance_index_censored(events, survival_times, -predicted_times)

    print("c_index",c_index)
    



    # Now generate an overall score(s) for this submission

    # here use c-indexx function
    metrics["aggregates"] = {
        "c_index": c_index[0]
    }

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def process(job):
    # Processes a single algorithm job, looking at the outputs
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"


    # Firstly, find the location of the results
    location_overall_survival_years = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="overall-survival-years",
        )
    

    # Secondly, read the results
    result_overall_survival_years = load_json_file(
        location=location_overall_survival_years,
    )
    
    #print('result_overall_survival_years',result_overall_survival_years)

    # Thirdly, retrieve the input image name to match it with an image in your ground truth
    image_name_prostatectomy_tissue_whole_slide_image = get_image_name(
            values=job["inputs"],
            slug="prostatectomy-tissue-whole-slide-image",
    )

    #print('image_name_prostatectomy_tissue_whole_slide_image',image_name_prostatectomy_tissue_whole_slide_image)

    # Fourthly, your load your ground truth
    # Include it in your evaluation container by placing it in ground_truth/
    ground_truth_df =  pd.read_csv(GROUND_TRUTH_DIRECTORY / "ground_truth.csv")

    gt_image_name_df = ground_truth_df[ground_truth_df["case_id"]==image_name_prostatectomy_tissue_whole_slide_image]

    

    gt_image_name_time = gt_image_name_df["follow_up_years"].item()

    print('gt_image_name_time',gt_image_name_time)

    gt_image_name_event = gt_image_name_df["event"].item()

    print('gt_image_name_event',gt_image_name_event)


    

    


    # Finally, calculate by comparing the ground truth to the actual results
    return {
        "case_id": image_name_prostatectomy_tissue_whole_slide_image,
        "case_id_gt_time": gt_image_name_time, 
        "case_id_gt_event": gt_image_name_event,
        "case_id_prediction_years_to_recurrence": result_overall_survival_years
    }


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
