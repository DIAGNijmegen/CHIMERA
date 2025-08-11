#!/usr/bin/env python3
"""
Neural Network Module

Defines functions for loading nnU-Net model configuration and creating the network.
"""
import torch.nn as nn
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import load_pickle

# --- nnU-Net v1 IMPORTS ---
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He

def load_model_configuration(model_folder: Path) -> dict:
    """Load and validate nnU-Net model configuration."""
    print("--- Loading nnU-Net configuration ---")
    plans_path = model_folder / "plans.pkl"
    if not plans_path.exists():
        raise FileNotFoundError(f"plans.pkl not found: {plans_path}")
        
    plans = load_pickle(str(plans_path))
    print("   ✅ Plans loaded successfully")
    return plans

def get_network_parameters(plans: dict) -> dict:
    """Extract network parameters from plans."""
    net_params = plans['plans_per_stage'][0]
    return {
        'net_params': net_params,
        'base_num_features': plans['base_num_features'],
        'num_input_channels': plans['num_modalities'],
        'num_classes': plans['num_classes'] + 1,
        'num_pool': len(net_params['pool_op_kernel_sizes']),
        'conv_op': nn.Conv3d,
        'norm_op': nn.InstanceNorm3d,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': nn.Dropout3d,
        'dropout_op_kwargs': {'p': 0, 'inplace': True},
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True}
    }

def create_network_from_params(params: dict) -> Generic_UNet:
    """Create a Generic_UNet instance from a parameters dictionary."""
    return Generic_UNet(
        input_channels=params['num_input_channels'],
        base_num_features=params['base_num_features'],
        num_classes=params['num_classes'],
        num_pool=params['num_pool'],
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=params['conv_op'],
        norm_op=params['norm_op'],
        norm_op_kwargs=params['norm_op_kwargs'],
        dropout_op=params['dropout_op'],
        dropout_op_kwargs=params['dropout_op_kwargs'],
        nonlin=params['nonlin'],
        nonlin_kwargs=params['nonlin_kwargs'],
        deep_supervision=True,
        dropout_in_localization=False,
        final_nonlin=lambda x: x,
        weightInitializer=InitWeights_He(1e-2),
        pool_op_kernel_sizes=params['net_params']['pool_op_kernel_sizes'],
        conv_kernel_sizes=params['net_params']['conv_kernel_sizes'],
        upscale_logits=False,
        convolutional_pooling=False,
        convolutional_upsampling=True
    )