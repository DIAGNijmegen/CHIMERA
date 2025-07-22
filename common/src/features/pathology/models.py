from pathlib import Path
import os
import json

import torch
import torch.nn as nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

from common.src.features.pathology.model_utils import update_state_dict

class UNI(nn.Module):
    """
    Tile-level feature extractor.
    """

    def __init__(self, model_dir, input_size=224):
        super().__init__()
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model configuration
        with open(os.path.join(self.model_dir, "uni-config.json"), "r") as f:
            self.config = json.load(f)

        if input_size == 256:
            self.config["pretrained_cfg"]["crop_pct"] = (
                224 / 256
            )  # Ensure Resize is 256

        # # Initialize tile encoder
        # model_name = self.config.pop("model_name", "vit_large_patch16_224") # model name not found in config
        # self.config.pop("architecture", None) # not supported in timm
        # self.config.pop("num_features", None) # not supported in timm
        # self.tile_encoder = timm.create_model(
        #     model_name,
        #     **self.config,
        #     mlp_layer=SwiGLUPacked,
        #     act_layer=torch.nn.SiLU
        # )

        # Initialize tile encoder
        self.tile_encoder = model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )

        self.load_weights()
        self.transforms = self.get_transforms()

    def load_weights(self):
        """Load pretrained weights for the tile encoder."""
        checkpoint_path = os.path.join(self.model_dir, "pytorch_model.bin")
        print(f"Loading tile encoder weights from {checkpoint_path}...")
        weights = torch.load(checkpoint_path, map_location=self.device)
        updated_sd, msg = update_state_dict(
            model_dict=self.tile_encoder.state_dict(), state_dict=weights
        )
        print(msg)
        self.tile_encoder.load_state_dict(updated_sd, strict=False) # turn to False to allow for missing keys
        self.tile_encoder.to(self.device)
        self.tile_encoder.eval()

    def get_transforms(self):
        """Retrieve the transformation pipeline for input images."""
        data_config = resolve_data_config(
            self.config["pretrained_cfg"], model=self.tile_encoder
        )
        return create_transform(**data_config)

    def forward(self, x):
        """Extract tile-level embeddings."""
        x = x.to(self.device)
        with torch.no_grad():
            output = self.tile_encoder(x)

        # Directly return the full embedding
        return output


# class Virchow(nn.Module):
#     """
#     Tile-level feature extractor.
#     """

#     def __init__(self, model_dir, mode: str, input_size=224):
#         super().__init__()
#         self.model_dir = model_dir
#         self.input_size = input_size
#         self.mode = mode
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Load model configuration
#         with open(os.path.join(self.model_dir, "virchow-config.json"), "r") as f:
#             self.config = json.load(f)

#         if input_size == 256:
#             self.config["pretrained_cfg"]["crop_pct"] = (
#                 224 / 256
#             )  # Ensure Resize is 256

#         # Initialize tile encoder
#         self.tile_encoder = timm.create_model(
#             **self.config, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
#         )

#         self.load_weights()
#         self.transforms = self.get_transforms()

#     def load_weights(self):
#         """Load pretrained weights for the tile encoder."""
#         checkpoint_path = os.path.join(self.model_dir, "virchow-tile-encoder.pth")
#         print(f"Loading tile encoder weights from {checkpoint_path}...")
#         weights = torch.load(checkpoint_path, map_location=self.device)
#         updated_sd, msg = update_state_dict(
#             model_dict=self.tile_encoder.state_dict(), state_dict=weights
#         )
#         print(msg)
#         self.tile_encoder.load_state_dict(updated_sd, strict=True)
#         self.tile_encoder.to(self.device)
#         self.tile_encoder.eval()

#     def get_transforms(self):
#         """Retrieve the transformation pipeline for input images."""
#         data_config = resolve_data_config(
#             self.config["pretrained_cfg"], model=self.tile_encoder
#         )
#         return create_transform(**data_config)

#     def forward(self, x):
#         """Extract tile-level embeddings."""
#         x = x.to(self.device)
#         with torch.no_grad():
#             output = self.tile_encoder(x)

#         # Extract class and patch tokens
#         class_token = output[:, 0]
#         patch_tokens = output[:, 1:]
#         embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)

#         if self.mode == "full":
#             return embedding

#         elif self.mode == "patch_tokens":
#             return patch_tokens

#         elif self.mode == "class_token":
#             return class_token

#         else:
#             raise ValueError(f"Unknown mode: {self.mode}. Choose from 'full', 'patch_tokens', or 'class_token'.")