#!/usr/bin/env python3
"""
Minimal debug script to test basic functionality
"""
import os
import sys

def main():
    print("🐍 PYTHON DEBUG SCRIPT", flush=True)
    print("="*40, flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Current directory: {os.getcwd()}", flush=True)
    print(f"Script file: {__file__}", flush=True)
    
    # Test the main directory paths
    folds_dir = "/data/temporary/chimera/Baseline_models/Task1_ABMIL/folds"
    print(f"\nTesting paths:", flush=True)
    print(f"FOLDS_DIR exists: {os.path.exists(folds_dir)}", flush=True)
    
    if os.path.exists(folds_dir):
        try:
            contents = os.listdir(folds_dir)
            print(f"FOLDS_DIR contents: {contents[:10]}...", flush=True)  # First 10 items
        except Exception as e:
            print(f"Error listing directory: {e}", flush=True)
    
    # Test importing basic libraries
    try:
        import pandas as pd
        print("✅ pandas imported successfully", flush=True)
    except Exception as e:
        print(f"❌ pandas import failed: {e}", flush=True)
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully", flush=True)
    except Exception as e:
        print(f"❌ matplotlib import failed: {e}", flush=True)
    
    try:
        import numpy as np
        print("✅ numpy imported successfully", flush=True)
    except Exception as e:
        print(f"❌ numpy import failed: {e}", flush=True)
    
    print("\n✅ DEBUG SCRIPT COMPLETE", flush=True)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
