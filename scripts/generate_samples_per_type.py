#!/usr/bin/env python
"""Generate samples for each gradient type using the trained conditional model."""

import json
import os
import torch
import numpy as np
from utils.Config import parse_model, parse_diffusion

def main():
    # Setup
    model_path = 'outputs/models/gradient_models_balanced/model.ckpt'
    model_config_path = 'outputs/models/gradient_models_balanced/config.json'
    results_dir = 'outputs/results_balanced'
    os.makedirs(results_dir, exist_ok=True)

    TYPES = [
        'Radial_Cosine_Gradient',
        'Noise',
        'Wave',
        'Radial_Wave',
        'Cosine_Gradient',
        'Sinusoidal_Gradient',
        'Combined_Shape',
        'Linear_Gradient',
        'Radial_Sinusoidal_Gradient'
    ]

    # Load model
    print("Loading model...")
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    model, _, _ = parse_model(model_config['model'])
    model = model.to('cuda:0')
    diffusion = parse_diffusion(model_config['diffusion'], model)
    
    # Load checkpoint (contains full diffusion state dict)
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cuda:0')
    diffusion.load_state_dict(checkpoint['diffu'])
    
    # Set to eval mode
    model.eval()
    diffusion.eval()
    print("✅ Checkpoint loaded!")

    print('✅ Model loaded! Generating samples...')

    # Generate samples per type
    for type_idx, gtype in enumerate(TYPES):
        print(f'[{type_idx+1}/9] Generating 5 samples for {gtype}...')
        cond_vec = np.zeros(9)
        cond_vec[type_idx] = 1.0
        cond_tensor = torch.from_numpy(cond_vec).float().to('cuda:0').unsqueeze(0)
        
        for sample_idx in range(5):
            with torch.no_grad():
                x_t = torch.randn(1, 2, 32, 32, device='cuda:0')
                # Use p_sample_loop for generation with conditioning
                x_sample = diffusion.p_sample_loop(None, img=x_t, cond=cond_tensor, with_tqdm=False)
                sample_file = f'{results_dir}/{gtype}_sample_{sample_idx}.npy'
                np.save(sample_file, x_sample.cpu().numpy())
                print(f'  ✓ {gtype}_sample_{sample_idx}.npy')

    print('\n✅ Generated 45 samples (5 per type) in outputs/results_balanced/')

if __name__ == '__main__':
    main()
