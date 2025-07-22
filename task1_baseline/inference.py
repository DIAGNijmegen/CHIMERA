"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
from glob import glob
import random
import pyvips
import SimpleITK
import numpy

INPUT_PATH = Path("/test/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        (
            "axial-adc-prostate-mri",
            "axial-t2-prostate-mri",
            "chimera-clinical-data-of-prostate-cancer-patients",
            "prostate-tissue-mask-for-axial-t2-prostate-mri",
            "prostatectomy-tissue-mask",
            "prostatectomy-tissue-whole-slide-image",
            "transverse-hbv-prostate-mri",
        ): interf0_handler,
        (
            "axial-adc-prostate-mri",
            "axial-t2-prostate-mri",
            "chimera-clinical-data-of-prostate-cancer-patients",
            "prostate-tissue-mask-for-axial-t2-prostate-mri",
            "prostatectomy-tissue-mask",
            "prostatectomy-tissue-whole-slide-image",
            "prostatectomy-tissue-whole-slide-image-1",
            "prostatectomy-tissue-whole-slide-image-1-2",
            "transverse-hbv-prostate-mri",
        ): interf1_handler,
        (
            "axial-adc-prostate-mri",
            "axial-t2-prostate-mri",
            "chimera-clinical-data-of-prostate-cancer-patients",
            "prostate-tissue-mask-for-axial-t2-prostate-mri",
            "prostatectomy-tissue-mask",
            "prostatectomy-tissue-whole-slide-image",
            "prostatectomy-tissue-whole-slide-image-1",
            "prostatectomy-tissue-whole-slide-image-1-2",
            "prostatectomy-tissue-whole-slide-image-2",
            "prostatectomy-tissue-whole-slide-image-2-2",
            "transverse-hbv-prostate-mri",
        ): interf2_handler,
        (
            "axial-adc-prostate-mri",
            "axial-t2-prostate-mri",
            "chimera-clinical-data-of-prostate-cancer-patients",
            "prostate-tissue-mask-for-axial-t2-prostate-mri",
            "prostatectomy-tissue-mask",
            "prostatectomy-tissue-whole-slide-image",
            "prostatectomy-tissue-whole-slide-image-1",
            "prostatectomy-tissue-whole-slide-image-1-2",
            "prostatectomy-tissue-whole-slide-image-2",
            "prostatectomy-tissue-whole-slide-image-2-2",
            "prostatectomy-tissue-whole-slide-image-3",
            "prostatectomy-tissue-whole-slide-image-3-2",
            "transverse-hbv-prostate-mri",
        ): interf3_handler,
        (
            "axial-adc-prostate-mri",
            "axial-t2-prostate-mri",
            "chimera-clinical-data-of-prostate-cancer-patients",
            "prostate-tissue-mask-for-axial-t2-prostate-mri",
            "prostatectomy-tissue-mask",
            "prostatectomy-tissue-whole-slide-image",
            "prostatectomy-tissue-whole-slide-image-1",
            "prostatectomy-tissue-whole-slide-image-1-2",
            "prostatectomy-tissue-whole-slide-image-2",
            "prostatectomy-tissue-whole-slide-image-2-2",
            "prostatectomy-tissue-whole-slide-image-3",
            "prostatectomy-tissue-whole-slide-image-3-2",
            "prostatectomy-tissue-whole-slide-image-4",
            "prostatectomy-tissue-whole-slide-image-4-2",
            "transverse-hbv-prostate-mri",
        ): interf4_handler,
        (
            "axial-adc-prostate-mri",
            "axial-t2-prostate-mri",
            "chimera-clinical-data-of-prostate-cancer-patients",
            "prostate-tissue-mask-for-axial-t2-prostate-mri",
            "prostatectomy-tissue-mask",
            "prostatectomy-tissue-whole-slide-image",
            "prostatectomy-tissue-whole-slide-image-1",
            "prostatectomy-tissue-whole-slide-image-1-2",
            "prostatectomy-tissue-whole-slide-image-2",
            "prostatectomy-tissue-whole-slide-image-2-2",
            "prostatectomy-tissue-whole-slide-image-3",
            "prostatectomy-tissue-whole-slide-image-3-2",
            "prostatectomy-tissue-whole-slide-image-4",
            "prostatectomy-tissue-whole-slide-image-4-2",
            "prostatectomy-tissue-whole-slide-image-5",
            "prostatectomy-tissue-whole-slide-image-5-2",
            "transverse-hbv-prostate-mri",
        ): interf5_handler,
        (
            "axial-adc-prostate-mri",
            "axial-t2-prostate-mri",
            "chimera-clinical-data-of-prostate-cancer-patients",
            "prostate-tissue-mask-for-axial-t2-prostate-mri",
            "prostatectomy-tissue-mask",
            "prostatectomy-tissue-whole-slide-image",
            "prostatectomy-tissue-whole-slide-image-1",
            "prostatectomy-tissue-whole-slide-image-1-2",
            "prostatectomy-tissue-whole-slide-image-2",
            "prostatectomy-tissue-whole-slide-image-2-2",
            "prostatectomy-tissue-whole-slide-image-3",
            "prostatectomy-tissue-whole-slide-image-3-2",
            "prostatectomy-tissue-whole-slide-image-4",
            "prostatectomy-tissue-whole-slide-image-4-2",
            "prostatectomy-tissue-whole-slide-image-5",
            "prostatectomy-tissue-whole-slide-image-5-2",
            "prostatectomy-tissue-whole-slide-image-6",
            "prostatectomy-tissue-whole-slide-image-6-2",
            "transverse-hbv-prostate-mri",
        ): interf6_handler,
        (
            "axial-adc-prostate-mri",
            "axial-t2-prostate-mri",
            "chimera-clinical-data-of-prostate-cancer-patients",
            "prostate-tissue-mask-for-axial-t2-prostate-mri",
            "prostatectomy-tissue-mask",
            "prostatectomy-tissue-whole-slide-image",
            "prostatectomy-tissue-whole-slide-image-1",
            "prostatectomy-tissue-whole-slide-image-1-2",
            "prostatectomy-tissue-whole-slide-image-2",
            "prostatectomy-tissue-whole-slide-image-2-2",
            "prostatectomy-tissue-whole-slide-image-3",
            "prostatectomy-tissue-whole-slide-image-3-2",
            "prostatectomy-tissue-whole-slide-image-4",
            "prostatectomy-tissue-whole-slide-image-4-2",
            "prostatectomy-tissue-whole-slide-image-5",
            "prostatectomy-tissue-whole-slide-image-5-2",
            "prostatectomy-tissue-whole-slide-image-6",
            "prostatectomy-tissue-whole-slide-image-6-2",
            "prostatectomy-tissue-whole-slide-image-7",
            "prostatectomy-tissue-whole-slide-image-7-2",
            "transverse-hbv-prostate-mri",
        ): interf7_handler,
        (
            "axial-adc-prostate-mri",
            "axial-t2-prostate-mri",
            "chimera-clinical-data-of-prostate-cancer-patients",
            "prostate-tissue-mask-for-axial-t2-prostate-mri",
            "prostatectomy-tissue-mask",
            "prostatectomy-tissue-whole-slide-image",
            "prostatectomy-tissue-whole-slide-image-1",
            "prostatectomy-tissue-whole-slide-image-1-2",
            "prostatectomy-tissue-whole-slide-image-2",
            "prostatectomy-tissue-whole-slide-image-2-2",
            "prostatectomy-tissue-whole-slide-image-3",
            "prostatectomy-tissue-whole-slide-image-3-2",
            "prostatectomy-tissue-whole-slide-image-4",
            "prostatectomy-tissue-whole-slide-image-4-2",
            "prostatectomy-tissue-whole-slide-image-5",
            "prostatectomy-tissue-whole-slide-image-5-2",
            "prostatectomy-tissue-whole-slide-image-6",
            "prostatectomy-tissue-whole-slide-image-6-2",
            "prostatectomy-tissue-whole-slide-image-7",
            "prostatectomy-tissue-whole-slide-image-7-2",
            "prostatectomy-tissue-whole-slide-image-8",
            "prostatectomy-tissue-whole-slide-image-8-2",
            "transverse-hbv-prostate-mri",
        ): interf8_handler,
        (
            "axial-adc-prostate-mri",
            "axial-t2-prostate-mri",
            "chimera-clinical-data-of-prostate-cancer-patients",
            "prostate-tissue-mask-for-axial-t2-prostate-mri",
            "prostatectomy-tissue-mask",
            "prostatectomy-tissue-whole-slide-image",
            "prostatectomy-tissue-whole-slide-image-1",
            "prostatectomy-tissue-whole-slide-image-1-2",
            "prostatectomy-tissue-whole-slide-image-2",
            "prostatectomy-tissue-whole-slide-image-2-2",
            "prostatectomy-tissue-whole-slide-image-3",
            "prostatectomy-tissue-whole-slide-image-3-2",
            "prostatectomy-tissue-whole-slide-image-4",
            "prostatectomy-tissue-whole-slide-image-4-2",
            "prostatectomy-tissue-whole-slide-image-5",
            "prostatectomy-tissue-whole-slide-image-5-2",
            "prostatectomy-tissue-whole-slide-image-6",
            "prostatectomy-tissue-whole-slide-image-6-2",
            "prostatectomy-tissue-whole-slide-image-7",
            "prostatectomy-tissue-whole-slide-image-7-2",
            "prostatectomy-tissue-whole-slide-image-8",
            "prostatectomy-tissue-whole-slide-image-8-2",
            "prostatectomy-tissue-whole-slide-image-9",
            "prostatectomy-tissue-whole-slide-image-9-2",
            "transverse-hbv-prostate-mri",
        ): interf9_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input
    input_transverse_hbv_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/transverse-hbv-prostate-mri",
    )
    input_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-t2-prostate-mri",
    )
    input_axial_adc_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-adc-prostate-mri",
    )
    # Use thumbnail loading for large WSI to avoid memory issues
    input_prostatectomy_tissue_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi",
        max_size=1024,
    )
    input_prostatectomy_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask",
        max_size=1024,
    )
    input_chimera_clinical_data_of_prostate_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json",
    )
    input_prostate_tissue_mask_for_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri",
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Debug: Print information about loaded data
    print("=+=" * 10)
    print("Data Loading Summary:")
    print(f"MRI T2 shape: {input_axial_t2_prostate_mri.shape}")
    print(f"MRI ADC shape: {input_axial_adc_prostate_mri.shape}")
    print(f"MRI HBV shape: {input_transverse_hbv_prostate_mri.shape}")
    print(f"Pathology WSI type: {type(input_prostatectomy_tissue_whole_slide_image)}")
    if hasattr(input_prostatectomy_tissue_whole_slide_image, 'width'):
        print(f"Pathology WSI size: {input_prostatectomy_tissue_whole_slide_image.width}x{input_prostatectomy_tissue_whole_slide_image.height}")
    
    # Handle both PyVips images and numpy arrays
    if hasattr(input_prostatectomy_tissue_mask, 'width'):
        print(f"Tissue mask size: {input_prostatectomy_tissue_mask.width}x{input_prostatectomy_tissue_mask.height}")
    else:
        print(f"Tissue mask shape: {input_prostatectomy_tissue_mask.shape}")
    
    print(f"Prostate mask shape: {input_prostate_tissue_mask_for_axial_t2_prostate_mri.shape}")
    print(f"Clinical data keys: {list(input_chimera_clinical_data_of_prostate_cancer_patients.keys()) if input_chimera_clinical_data_of_prostate_cancer_patients else 'None'}")
    print("=+=" * 10)

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    try:
        with open(
            model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
        ) as f:
            print(f.read())
    except FileNotFoundError:
        print("Model resource file not found - this is expected in test environment")

    # For now, let us make bogus predictions
    # Generate random float between 0 and 80 with one decimal place
    output_time_to_biochemical_recurrence_for_prostate_cancer = round(random.uniform(0, 80), 1)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


def interf1_handler():
    # Read the input
    input_transverse_hbv_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/transverse-hbv-prostate-mri",
    )
    input_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-t2-prostate-mri",
    )
    input_axial_adc_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-adc-prostate-mri",
    )
    # Use thumbnail loading for large WSI to avoid memory issues
    input_prostatectomy_tissue_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi",
        max_size=1024,
    )
    input_prostatectomy_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask",
        max_size=1024,
    )
    input_chimera_clinical_data_of_prostate_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json",
    )
    # Use thumbnail loading for large WSI to avoid memory issues
    input_prostatectomy_tissue_whole_slide_image_1 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-1",
        max_size=1024,
    )
    input_prostate_tissue_mask_for_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image_1_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-1",
        max_size=1024,
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    try:
        with open(
            model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
        ) as f:
            print(f.read())
    except FileNotFoundError:
        print("Model resource file not found - this is expected in test environment")

    # For now, let us make bogus predictions
    # Generate random float between 0 and 80 with one decimal place
    output_time_to_biochemical_recurrence_for_prostate_cancer = round(random.uniform(0, 80), 1)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


def interf2_handler():
    # Read the input
    input_transverse_hbv_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/transverse-hbv-prostate-mri",
    )
    input_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-t2-prostate-mri",
    )
    input_axial_adc_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-adc-prostate-mri",
    )
    # Use thumbnail loading for large WSI to avoid memory issues
    input_prostatectomy_tissue_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi",
        max_size=1024,
    )
    input_prostatectomy_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask",
        max_size=1024,
    )
    input_chimera_clinical_data_of_prostate_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json",
    )
    # Use thumbnail loading for large WSI to avoid memory issues
    input_prostatectomy_tissue_whole_slide_image_1 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-1",
        max_size=1024,
    )
    # Use thumbnail loading for large WSI to avoid memory issues  
    input_prostatectomy_tissue_whole_slide_image_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-2",
        max_size=1024,
    )
    input_prostate_tissue_mask_for_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image_1_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-2",
        max_size=1024,
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    try:
        with open(
            model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
        ) as f:
            print(f.read())
    except FileNotFoundError:
        print("Model resource file not found - this is expected in test environment")

    # For now, let us make bogus predictions
    # Generate random float between 0 and 80 with one decimal place
    output_time_to_biochemical_recurrence_for_prostate_cancer = round(random.uniform(0, 80), 1)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


def interf3_handler():
    # Read the input
    input_transverse_hbv_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/transverse-hbv-prostate-mri",
    )
    input_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-t2-prostate-mri",
    )
    input_axial_adc_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-adc-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi",
        max_size=1024,
    )
    input_prostatectomy_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask",
        max_size=1024,
    )
    input_chimera_clinical_data_of_prostate_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json",
    )
    input_prostatectomy_tissue_whole_slide_image_1 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-3",
        max_size=1024,
    )
    input_prostate_tissue_mask_for_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image_1_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-3",
        max_size=1024,
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    try:
        with open(
            model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
        ) as f:
            print(f.read())
    except FileNotFoundError:
        print("Model resource file not found - this is expected in test environment")

    # For now, let us make bogus predictions
    # Generate random float between 0 and 80 with one decimal place
    output_time_to_biochemical_recurrence_for_prostate_cancer = round(random.uniform(0, 80), 1)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


def interf4_handler():
    # Read the input
    input_transverse_hbv_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/transverse-hbv-prostate-mri",
    )
    input_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-t2-prostate-mri",
    )
    input_axial_adc_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-adc-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi",
        max_size=1024,
    )
    input_prostatectomy_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask",
        max_size=1024,
    )
    input_chimera_clinical_data_of_prostate_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json",
    )
    input_prostatectomy_tissue_whole_slide_image_1 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-4",
        max_size=1024,
    )
    input_prostate_tissue_mask_for_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image_1_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-4",
        max_size=1024,
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    try:
        with open(
            model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
        ) as f:
            print(f.read())
    except FileNotFoundError:
        print("Model resource file not found - this is expected in test environment")

    # For now, let us make bogus predictions
    # Generate random float between 0 and 80 with one decimal place
    output_time_to_biochemical_recurrence_for_prostate_cancer = round(random.uniform(0, 80), 1)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


def interf5_handler():
    # Read the input
    input_transverse_hbv_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/transverse-hbv-prostate-mri",
    )
    input_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-t2-prostate-mri",
    )
    input_axial_adc_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-adc-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi",
        max_size=1024,
    )
    input_prostatectomy_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask",
        max_size=1024,
    )
    input_chimera_clinical_data_of_prostate_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json",
    )
    input_prostatectomy_tissue_whole_slide_image_1 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-4",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_5 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-5",
        max_size=1024,
    )
    input_prostate_tissue_mask_for_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image_1_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-4",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_5_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-5",
        max_size=1024,
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    try:
        with open(
            model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
        ) as f:
            print(f.read())
    except FileNotFoundError:
        print("Model resource file not found - this is expected in test environment")

    # For now, let us make bogus predictions
    # Generate random float between 0 and 80 with one decimal place
    output_time_to_biochemical_recurrence_for_prostate_cancer = round(random.uniform(0, 80), 1)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


def interf6_handler():
    # Read the input
    input_transverse_hbv_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/transverse-hbv-prostate-mri",
    )
    input_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-t2-prostate-mri",
    )
    input_axial_adc_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-adc-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi",
        max_size=1024,
    )
    input_prostatectomy_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask",
        max_size=1024,
    )
    input_chimera_clinical_data_of_prostate_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json",
    )
    input_prostatectomy_tissue_whole_slide_image_1 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-4",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_5 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-5",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_6 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-6",
        max_size=1024,
    )
    input_prostate_tissue_mask_for_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image_1_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-4",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_5_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-5",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_6_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-6",
        max_size=1024,
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    try:
        with open(
            model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
        ) as f:
            print(f.read())
    except FileNotFoundError:
        print("Model resource file not found - this is expected in test environment")

    # For now, let us make bogus predictions
    # Generate random float between 0 and 80 with one decimal place
    output_time_to_biochemical_recurrence_for_prostate_cancer = round(random.uniform(0, 80), 1)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


def interf7_handler():
    # Read the input
    input_transverse_hbv_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/transverse-hbv-prostate-mri",
    )
    input_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-t2-prostate-mri",
    )
    input_axial_adc_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-adc-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi",
        max_size=1024,
    )
    input_prostatectomy_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask",
        max_size=1024,
    )
    input_chimera_clinical_data_of_prostate_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json",
    )
    input_prostatectomy_tissue_whole_slide_image_1 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-4",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_5 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-5",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_6 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-6",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_7 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-7",
        max_size=1024,
    )
    input_prostate_tissue_mask_for_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image_1_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-4",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_5_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-5",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_6_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-6",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_7_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-7",
        max_size=1024,
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    try:
        with open(
            model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
        ) as f:
            print(f.read())
    except FileNotFoundError:
        print("Model resource file not found - this is expected in test environment")

    # For now, let us make bogus predictions
    # Generate random float between 0 and 80 with one decimal place
    output_time_to_biochemical_recurrence_for_prostate_cancer = round(random.uniform(0, 80), 1)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


def interf8_handler():
    # Read the input
    input_transverse_hbv_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/transverse-hbv-prostate-mri",
    )
    input_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-t2-prostate-mri",
    )
    input_axial_adc_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-adc-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi",
        max_size=1024,
    )
    input_prostatectomy_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask",
        max_size=1024,
    )
    input_chimera_clinical_data_of_prostate_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json",
    )
    input_prostatectomy_tissue_whole_slide_image_1 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-4",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_5 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-5",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_6 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-6",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_7 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-7",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_8 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-8",
        max_size=1024,
    )
    input_prostate_tissue_mask_for_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image_1_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-4",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_5_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-5",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_6_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-6",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_7_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-7",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_8_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-8",
        max_size=1024,
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    try:
        with open(
            model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
        ) as f:
            print(f.read())
    except FileNotFoundError:
        print("Model resource file not found - this is expected in test environment")

    # For now, let us make bogus predictions
    # Generate random float between 0 and 80 with one decimal place
    output_time_to_biochemical_recurrence_for_prostate_cancer = round(random.uniform(0, 80), 1)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


def interf9_handler():
    # Read the input
    input_transverse_hbv_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/transverse-hbv-prostate-mri",
    )
    input_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-t2-prostate-mri",
    )
    input_axial_adc_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/axial-adc-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi",
        max_size=1024,
    )
    input_prostatectomy_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask",
        max_size=1024,
    )
    input_chimera_clinical_data_of_prostate_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json",
    )
    input_prostatectomy_tissue_whole_slide_image_1 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-4",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_5 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-5",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_6 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-6",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_7 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-7",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_8 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-8",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_9 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-wsi-z-9",
        max_size=1024,
    )
    input_prostate_tissue_mask_for_axial_t2_prostate_mri = load_image_file_as_array(
        location=INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri",
    )
    input_prostatectomy_tissue_whole_slide_image_1_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-1",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_2_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-2",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_3_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-3",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_4_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-4",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_5_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-5",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_6_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-6",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_7_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-7",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_8_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-8",
        max_size=1024,
    )
    input_prostatectomy_tissue_whole_slide_image_9_2 = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/prostatectomy-tissue-mask-wsi-9",
        max_size=1024,
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    try:
        with open(
            model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
        ) as f:
            print(f.read())
    except FileNotFoundError:
        print("Model resource file not found - this is expected in test environment")

    # For now, let us make bogus predictions
    # Generate random float between 0 and 80 with one decimal place
    output_time_to_biochemical_recurrence_for_prostate_cancer = round(random.uniform(0, 80), 1)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    """
    Load image files using appropriate library based on file type:
    - PyVips for pathology images: .tif, .tiff, .mrxs, .svs, .ndpi
    - SimpleITK for radiology images: .mha
    """
    # Find all compatible files
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
        + glob(str(location / "*.mrxs"))
        + glob(str(location / "*.svs"))
        + glob(str(location / "*.ndpi"))
    )
    
    if not input_files:
        raise FileNotFoundError(f"No compatible image files found in {location}")
    
    file_path = input_files[0]
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.mha':
        # Use SimpleITK for radiology images (.mha)
        print(f"Loading radiology image using SimpleITK: {file_path}")
        image = SimpleITK.ReadImage(file_path)
        array = SimpleITK.GetArrayFromImage(image)
        return array
    
    else:
        # Use PyVips for pathology images (.tif, .tiff, .mrxs, .svs, .ndpi)
        print(f"Loading pathology image using PyVips: {file_path}")
        image = pyvips.Image.new_from_file(file_path)
        
        # For very large images, you might want to downsample first
        # Uncomment the next line to downsample by factor of 4 to save memory
        # image = image.resize(0.25)
        
        # Convert to numpy array
        # Note: This will load the entire image into memory
        # For production use, consider processing tiles instead
        memory_image = image.write_to_memory()
        array = numpy.frombuffer(memory_image, dtype=numpy.uint8)
        
        # Reshape based on image dimensions and bands
        height = image.height
        width = image.width
        bands = image.bands
        
        if bands == 1:
            # Grayscale image
            array = array.reshape((height, width))
        else:
            # Multi-channel image (RGB, RGBA, etc.)
            array = array.reshape((height, width, bands))
        
        return array


def load_image_file_as_thumbnail(*, location, max_size=1024):
    """
    Returns the PyVips image object directly for memory efficiency
    """
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
        + glob(str(location / "*.mrxs"))
        + glob(str(location / "*.svs"))
        + glob(str(location / "*.ndpi"))
    )
    
    if not input_files:
        raise FileNotFoundError(f"No compatible image files found in {location}")
    
    file_path = input_files[0]
    print(f"Loading pathology image as thumbnail using PyVips: {file_path}")
    
    # Load image with PyVips
    image = pyvips.Image.new_from_file(file_path)
    
    # Calculate downsampling factor to fit within max_size
    scale_factor = min(max_size / image.width, max_size / image.height)
    if scale_factor < 1.0:
        print(f"Downsampling image by factor {scale_factor:.3f} (from {image.width}x{image.height} to {int(image.width*scale_factor)}x{int(image.height*scale_factor)})")
        image = image.resize(scale_factor)
    else:
        print(f"Image size {image.width}x{image.height} is within max_size={max_size}, no downsampling needed")
    
    # Return the PyVips image object directly (much more memory efficient)
    return image


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
