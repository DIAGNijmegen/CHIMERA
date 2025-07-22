from pathlib import Path

import numpy as np
import torch
import wholeslidedata as wsd
from PIL import Image


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, wsi_path, coordinates_dir, backend, transforms=None):
        self.path = wsi_path
        self.backend = backend
        self.name = wsi_path.stem.replace(" ", "_")
        self.load_coordinates(coordinates_dir)
        self.transforms = transforms

    def set_coordinates(self, coordinates):
        self.coordinates = (np.array([coordinates["x"], coordinates["y"]]).T).astype(
            int
        )

    def load_coordinates(self, coordinates_dir):
        coordinates_file = coordinates_dir / f"{self.name}.npy"
        coordinates = np.load(coordinates_file, allow_pickle=True)
        self.x = coordinates["x"]
        self.y = coordinates["y"]
        self.tile_size_resized = coordinates["tile_size_resized"]
        self.tile_level = coordinates["tile_level"]
        self.resize_factor = coordinates["resize_factor"]
        self.tile_size_lv0 = coordinates["tile_size_lv0"][0]
        self.set_coordinates(coordinates)

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        wsi = wsd.WholeSlideImage(self.path, backend=self.backend)
        tile_level = self.tile_level[idx]
        tile_spacing = wsi.spacings[tile_level]
        tile_arr = wsi.get_patch(
            self.x[idx],
            self.y[idx],
            self.tile_size_resized[idx],
            self.tile_size_resized[idx],
            spacing=tile_spacing,
            center=False,
        )
        tile = Image.fromarray(tile_arr).convert("RGB")
        if self.resize_factor[idx] != 1:
            tile_size = int(self.tile_size_resized[idx] / self.resize_factor[idx])
            tile = tile.resize((tile_size, tile_size))
        if self.transforms:
            tile = self.transforms(tile)
        return idx, tile


class TileDatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, wsi_path, tile_dir, tile_format, transforms=None):
        self.name = wsi_path.stem.replace(" ", "_")
        self.tiles = sorted(
            [x for x in Path(tile_dir, self.name).glob(f"*.{tile_format}")]
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        pil_tile = Image.open(self.tiles[idx])
        if self.transforms is not None:
            img = self.transforms(pil_tile)
        else:
            img = pil_tile
        return idx, img
