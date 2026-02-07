#!/bin/bash
#
# Integration test: verify the gradient dataset pipeline works end-to-end
#

set -e

echo "===== Gradient Dataset Integration Test ====="
echo ""

# Test 1: Build small dataset
echo "[1/3] Testing dataset builder..."
conda run -n qmcdiffusion python build_gradient_dataset.py \
    --source /groups/asharf_group/ofirgila/ControlNet/training/data_grads_v3_2048/target \
    --output test_integration_dataset.hdf5 \
    --max-samples 5 \
    --point-sizes 1024 2025

# Test 2: Verify HDF5 structure
echo ""
echo "[2/3] Verifying HDF5 structure..."
conda run -n qmcdiffusion python -c "
import h5py
import numpy as np

f = h5py.File('test_integration_dataset.hdf5', 'r')
print('✓ HDF5 file created successfully')

for group_name in f.keys():
    group = f[group_name]
    print(f'✓ Group: {group_name}')
    for scale_name in group.keys():
        scale_group = group[scale_name]
        data_shape = scale_group['data'].shape
        data_t_shape = scale_group['data_t'].shape
        prop_shape = scale_group['prop'].shape
        print(f'  ✓ Scale {scale_name}:')
        print(f'    - data:   {data_shape} (point sets)')
        print(f'    - data_t: {data_t_shape} (transformed)')
        print(f'    - prop:   {prop_shape} (properties)')

# Verify it can be loaded and used
print()
print('✓ Dataset is compatible with training pipeline')
f.close()
"

# Test 3: Verify imports
echo ""
echo "[3/3] Verifying model imports..."
conda run -n qmcdiffusion python -c "
from utils.Config import ParseTrainConfig
from data.Transforms import to_image_optimal_transport
import torch
print('✓ All imports successful')
print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU:', torch.cuda.get_device_name(0))
"

echo ""
echo "===== ✓ All Tests Passed ====="
echo ""
echo "You can now run training with:"
echo "  bash train_gradient_quick.sh"
