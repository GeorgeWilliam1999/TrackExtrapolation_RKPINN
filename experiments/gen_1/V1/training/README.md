# V1 Training

HTCondor job submission scripts for V1 model training.

## Training Jobs

### Core Experiments (12 models)
- MLP: tiny, small, medium, wide
- PINN: tiny, small, medium, wide  
- RK-PINN: tiny, small, medium, wide

### Physics Ablations (8 models)
- PINN/RK-PINN with λ_pde: 0, 0.01, 1.0, 10.0

### Momentum Studies (9 models)
- Low-p (0.5-5 GeV), mid-p (5-20 GeV), high-p (20-100 GeV)

## Submission

```bash
# Individual job
condor_submit jobs/mlp_medium.sub

# All V1 jobs
./submit_all.sh
```

## Training Wrapper

`train_wrapper.sh` handles:
- Environment activation
- Argument parsing
- Training script execution

## Job Files

Located in `jobs/`:
- `mlp_*.sub` - MLP training jobs
- `pinn_*.sub` - PINN training jobs
- `rkpinn_*.sub` - RK-PINN training jobs

## See Also

- [V1/trained_models](../trained_models/) - Trained model checkpoints
- [V2/training](../../V2/training/) - V2 shallow-wide training
