# V2 Training

Training configurations for V2 shallow-wide experiments.

## Key Changes from V1

1. **Shallow-wide architectures**: 1-3 layers with 256-1024 neurons
2. **PINN residual architecture**: Fixes IC constraint
3. **More epochs**: 20 epochs (vs 10 in V1)

## Configurations

Located in `configs/`:

### MLP Configs
- `mlp_v2_single_256.json` - Single layer, 256 neurons
- `mlp_v2_single_512.json` - Single layer, 512 neurons
- `mlp_v2_shallow_512.json` - [512, 256] architecture
- `mlp_v2_shallow_1024_512.json` - [1024, 512] architecture

### PINN Configs
- `pinn_v2_shallow_256.json` - Residual PINN
- `pinn_v2_shallow_1024_256.json` - Large residual PINN

## Usage

```bash
python models/train.py --config configs/mlp_v2_shallow_512.json
```

## Cluster Submission

See [V2/cluster](../cluster/) for HTCondor job files.
