#!/usr/bin/env python3
"""Generate 30 JSON config files for the field map NN grid search."""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, 'configs')
OUTPUT_BASE = os.path.join(BASE_DIR, 'trained_models')
FIELD_MAP = '/data/bfys/gscriven/TrackExtrapolation/experiments/field_maps/twodip.rtf'

os.makedirs(CONFIG_DIR, exist_ok=True)

widths = [32, 64, 128, 256, 512]
depths = [1, 2, 3]
activations = ['relu', 'silu']

configs = []
for activation in activations:
    for depth in depths:
        for width in widths:
            name = f'field_nn_{activation}_{depth}L_{width}H'
            hidden_dims = [width] * depth

            config = {
                'name': name,
                'hidden_dims': hidden_dims,
                'activation': activation,
                'lr': 1e-3,
                'epochs': 200,
                'batch_size': 4096,
                'patience': 20,
                'n_val': 100_000,
                'val_seed': 42,
                'field_map_path': FIELD_MAP,
                'output_dir': os.path.join(OUTPUT_BASE, name),
            }

            config_path = os.path.join(CONFIG_DIR, f'{name}.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            configs.append(name)
            print(f'  {name:40s}  dims={hidden_dims}')

# Write job list for HTCondor
jobs_path = os.path.join(BASE_DIR, 'cluster', 'field_nn_jobs.txt')
with open(jobs_path, 'w') as f:
    f.write('# Field map NN grid search -- one config per line\n')
    f.write(f'# Generated: {len(configs)} jobs\n')
    for name in configs:
        f.write(name + '\n')

print(f'\nGenerated {len(configs)} configs in {CONFIG_DIR}/')
print(f'Job list written to {jobs_path}')
