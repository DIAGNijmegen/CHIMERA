import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
import torch
import wholeslidedata as wsd
from PIL import Image

from common.src.io import resolve_image_path
from common.src.features.pathology.feature_extraction import extract_features
from common.src.features.pathology.models import UNI
from common.src.features.pathology.wsi import (
    TilingParams,
    FilterParams,
    WholeSlideImage,
)


def extract_coordinates(
    *,
    wsi_path: Path,
    tissue_mask_path: Path,
    tiling_params: TilingParams,
    filter_params: FilterParams,
    max_number_of_tiles: int | None = None,
    seed: int = 0,
    num_workers: int = 1,
):
    """
    Extracts tile coordinates from a whole slide image (wsi) based on the given parameters.

    Args:
        wsi_path (str): File path to the wsi.
        tissue_mask_path (str): File path to the tissue mask associated with the wsi.
        tiling_params (TilingParams): Parameters for tiling the wsi.
        filter_params (FilterParams): Parameters for filtering the tiles.
        max_number_of_tiles (int, optional): Maximum number of tiles to keep. If None, all tiles are kept. Defaults to None.
        num_workers (int, optional): Number of workers to use for parallel processing. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - sorted_coordinates (list): List of tile coordinates sorted by tissue percentage.
            - sorted_tissue_percentages (list): List of tissue percentages corresponding to the sorted coordinates.
            - tile_level (int): Tile level used for extraction.
            - resize_factor (float): Resize factor applied to the tiles.
            - tile_size_lv0 (int): Tile size at level 0 in the wsi.
    """
    wsi = WholeSlideImage(wsi_path, tissue_mask_path)
    (
        coordinates,
        tissue_percentages,
        tile_level,
        resize_factor,
        tile_size_lv0,
    ) = wsi.get_tile_coordinates(
        tiling_params=tiling_params,
        filter_params=filter_params,
        num_workers=num_workers,
    )

    image_spacings = wsi.spacings[0]
    required_level, _ = wsi.get_best_level_for_spacing(
        tiling_params.spacing, tiling_params.tolerance
    )
    image_size = wsi.level_dimensions[required_level]

    # sort coordinates by tissue percentage
    sorted_coordinates, sorted_tissue_percentages = select_coordinates_with_tissue(
        coordinates=coordinates,
        tissue_percentages=tissue_percentages,
        max_number_of_tiles=max_number_of_tiles,
        seed=seed,
    )

    return (
        sorted_coordinates,
        sorted_tissue_percentages,
        tile_level,
        resize_factor,
        tile_size_lv0,
        image_spacings,
        image_size,
    )


def save_coordinates(
    *,
    wsi_path: Path,
    coordinates,
    tile_level,
    tile_size,
    resize_factor,
    tile_size_lv0,
    target_spacing,
    save_dir: str,
):
    """
    Saves tile coordinates and associated metadata into a .npy file.

    Args:
        wsi_path (Path): File path to the whole slide image (wsi).
        coordinates (list of tuples): List of (x, y) coordinates of tiles, defined with respect to level 0.
        tile_level (int): Level of the image pyramid at which tiles have been extracted.
        tile_size (int): Desired tile size.
        resize_factor (float): Factor by which the tile size must be resized.
        tile_size_lv0 (int): Size of the tile at level 0.
        target_spacing (float): Target spacing for the tiles.
        save_dir (str): Directory where the output .npy file will be saved.

    Returns:
        Path: Path to the saved .npy file containing tile coordinates and associated metadata.

    Notes:
        - The output file is saved with the same name as the wsi file stem.
        - The metadata includes the resized tile size, tile level, resize factor, tile size at level 0,
          and target spacing for each tile.
    """
    wsi_name = wsi_path.stem
    output_path = Path(save_dir, f"{wsi_name}.npy")
    x = [c[0] for c in coordinates]  # defined w.r.t level 0
    y = [c[1] for c in coordinates]  # defined w.r.t level 0
    ntile = len(x)
    tile_size_resized = int(tile_size * resize_factor)

    dtype = [
        ("x", int),
        ("y", int),
        ("tile_size_resized", int),
        ("tile_level", int),
        ("resize_factor", float),
        ("tile_size_lv0", int),
        ("target_spacing", float),
    ]
    data = np.zeros(ntile, dtype=dtype)
    for i in range(ntile):
        data[i] = (
            x[i],
            y[i],
            tile_size_resized,
            tile_level,
            resize_factor,
            tile_size_lv0,
            target_spacing,
        )

    data_arr = np.array(data)
    np.save(output_path, data_arr)
    return output_path


def select_coordinates_with_tissue(
    *,
    coordinates: list[tuple],
    tissue_percentages: list[float],
    max_number_of_tiles: int | None = None,
    seed: int = 0,
) -> tuple[list[tuple], list[float]]:
    """
    Select coordinates and their corresponding tissue percentages based on a
    maximum number of tiles. If more than the maximum number of tiles are found
    with full tissue content (100%), a random selection of tiles is made.
    Otherwise, the tiles are sorted by tissue percentage and the top tiles are
    selected.

    Args:
        coordinates (list of tuple): A list of tuples representing coordinates,
            where each tuple contains two integers (x, y).
        tissue_percentages (list of float): A list of tissue percentages
            corresponding to the tile for each coordinate.
        max_number_of_tiles (int | None): The maximum number of tiles to select.
            If None, all tiles above the minimum tissue percentage will be selected.

    Returns:
        tuple: A tuple containing two lists:
            - sorted_coordinates (list of tuple): The coordinates sorted based
              on the tissue percentages.
            - sorted_tissue_percentages (list of float): The tissue percentages
              corresponding to the sorted coordinates.
    """

    # separate perfect tissue tiles
    perfect = [
        (coord, perc)
        for coord, perc in zip(coordinates, tissue_percentages)
        if perc == 1.0
    ]
    if max_number_of_tiles is not None and len(perfect) > max_number_of_tiles:
        rng = random.Random(seed)
        selected = rng.sample(perfect, max_number_of_tiles)
    else:
        # Sort by descending tissue percentage and take top N if needed
        all = [(coord, perc) for coord, perc in zip(coordinates, tissue_percentages)]
        all.sort(key=lambda x: x[1], reverse=True)
        selected = all[:max_number_of_tiles] if max_number_of_tiles is not None else all

    selected_coordinates, selected_percentages = zip(*selected)
    return list(selected_coordinates), list(selected_percentages)


def save_tile(
    *,
    x: int,
    y: int,
    wsi_path: Path,
    spacing: float,
    tile_size: int,
    resize_factor: int | float,
    save_dir: Path,
    tile_format: str,
    backend: str = "asap",
):
    tile_size_resized = int(tile_size * resize_factor)
    wsi = wsd.WholeSlideImage(wsi_path, backend=backend)
    tile_arr = wsi.get_patch(
        x, y, tile_size_resized, tile_size_resized, spacing=spacing, center=False
    )
    tile = Image.fromarray(tile_arr).convert("RGB")
    if resize_factor != 1:
        tile = tile.resize((tile_size, tile_size))
    tile_fp = save_dir / f"{int(x)}_{int(y)}.{tile_format}"
    tile.save(tile_fp)
    return tile_fp


def save_tile_mp(args):
    coord, wsi_path, spacing, tile_size, resize_factor, tile_dir, tile_format = args
    x, y = coord
    return save_tile(
        x=x,
        y=y,
        wsi_path=wsi_path,
        spacing=spacing,
        tile_size=tile_size,
        resize_factor=resize_factor,
        save_dir=tile_dir,
        tile_format=tile_format,
    )


def save_tiles(
    *,
    wsi_path: Path,
    coordinates: list[tuple[int, int]],
    tile_level: int,
    tile_size: int,
    resize_factor: float,
    save_dir: Path,
    tile_format: str,
    backend: str = "asap",
    num_workers: int = 1,
):
    wsi_name = wsi_path.stem
    wsi = wsd.WholeSlideImage(wsi_path, backend=backend)
    tile_spacing = wsi.spacings[tile_level]
    tile_dir = save_dir / wsi_name
    tile_dir.mkdir(parents=True, exist_ok=True)
    iterable = [
        (coord, wsi_path, tile_spacing, tile_size, resize_factor, tile_dir, tile_format)
        for coord in coordinates
    ]
    with mp.Pool(num_workers) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(save_tile_mp, iterable),
            desc=f"Saving tiles for {wsi_path.stem}",
            unit=" tile",
            total=len(iterable),
            leave=True,
            file=sys.stdout,
        ):
            pass


def save_features_to_pt(feature: torch.Tensor, output_path: Path):
    """
    Saves the final feature tensor to the specified .pt file.

    Args:
        feature (torch.Tensor): The feature tensor to save.
        output_path (Path): The full path for the output file.
    """
    # Create the parent output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the tensor to the specified path
    torch.save(feature, output_path)
    
    print(f"Feature tensor saved to {output_path}")


def run_pathology_vision_task(
    *,
    input_information: dict[str, Any],
    model_dir: Path,
    output_dir: Path,
):
    wsi_path = None
    tissue_mask_path = None

    # Loop through inputs to find the specific pathology WSI and its mask by slug
    for input_socket in input_information:
        slug = input_socket["interface"]["slug"]
        
        # Uniquely identify the WSI by its slug
        if slug == "prostatectomy-tissue-whole-slide-image":
            wsi_path = resolve_image_path(location=input_socket["input_location"])
            print(f"Found WSI via slug: {wsi_path}")
            
        # Uniquely identify the WSI's tissue mask by its slug
        elif slug == "prostatectomy-tissue-mask":
            tissue_mask_path = resolve_image_path(
                location=input_socket["input_location"]
            )
            print(f"Found WSI tissue mask via slug: {tissue_mask_path}")

    # Add a check to ensure we found the required files
    if wsi_path is None:
        raise FileNotFoundError("Could not find the 'prostatectomy-tissue-whole-slide-image' in the input files.")
    if tissue_mask_path is None:
        raise FileNotFoundError("Could not find the 'prostatectomy-tissue-mask' in the input files.")

    batch_size = 32
    use_mixed_precision = False
    spacing = 0.5
    tolerance = 0.05 # tolerance to consider two spacings equal (e.g. if tolerance is 0.10, any spacing between [0.45, 0.55] is considered equal to 0.5)
    tile_size = 224
    max_number_of_tiles = 30000 # limit number of tiles to comply with time limits and GPU memory

    num_workers = min(mp.cpu_count(), 8)
    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        num_workers = min(num_workers, int(os.environ["SLURM_JOB_CPUS_PER_NODE"]))

    tiling_params = TilingParams(
        spacing=spacing,
        tolerance=tolerance,
        tile_size=tile_size,
        overlap=0.0,
        drop_holes=False,
        min_tissue_ratio=0.1,  # Using a lenient value for comprehensive feature extraction
        use_padding=True,
    )
    filter_params = FilterParams(ref_tile_size=64, a_t=1, a_h=1, max_n_holes=8)

    # create output directories
    coordinates_dir = Path("/tmp/coordinates/")
    coordinates_dir.mkdir(parents=True, exist_ok=True)

    # Extract tile coordinates
    coordinates, _, level, resize_factor, tile_size_lv0, image_spacing, image_size = (
        extract_coordinates(
            wsi_path=wsi_path,
            tissue_mask_path=tissue_mask_path,
            tiling_params=tiling_params,
            filter_params=filter_params,
            max_number_of_tiles=max_number_of_tiles,
            num_workers=num_workers,
        )
    )

    save_coordinates(
        wsi_path=wsi_path,
        coordinates=coordinates,
        tile_level=level,
        tile_size=tiling_params.tile_size,
        resize_factor=resize_factor,
        tile_size_lv0=tile_size_lv0,
        target_spacing=tiling_params.spacing,
        save_dir=coordinates_dir,
    )

    print("=+=" * 10)

    feature_extractor = UNI(model_dir=model_dir)
    print(f"Initialized feature extractor: {type(feature_extractor).__name__}")

    # Extract tile or slide features
    feature = extract_features(
        wsi_path=wsi_path,
        model=feature_extractor,
        coordinates_dir=coordinates_dir,
        backend="asap",
        batch_size=batch_size,
        num_workers=num_workers,
        use_mixed_precision=use_mixed_precision,
    )

    output_path = output_dir / "features.pt"
    save_features_to_pt(feature=feature, output_path=output_path)

