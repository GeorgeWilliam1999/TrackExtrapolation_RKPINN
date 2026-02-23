#!/usr/bin/env python3
"""
================================================================================
Submit Training Jobs to HTCondor
================================================================================

Submits neural network training jobs to HTCondor cluster for parallel execution.

Usage:
    # Submit single model training
    python submit_training.py --model mlp --preset medium

    # Submit architecture sweep for one model
    python submit_training.py --experiment pinn_architecture_sweep

    # Submit all experiments
    python submit_training.py --experiment all_architecture_sweeps

    # Dry run (show what would be submitted)
    python submit_training.py --experiment quick_test --dry-run

Author: G. Scriven
Date: January 2026
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import experiment definitions
from run_experiments import EXPERIMENTS, DEFAULT_MAX_EPOCHS, DEFAULT_PATIENCE, DEFAULT_MIN_DELTA

# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = Path(__file__).parent.absolute()
DATA_DIR = MODELS_DIR.parent / 'data_generation' / 'data'
CONDA_ENV = 'TE'

# HTCondor settings
CONDOR_CONFIG = {
    'request_cpus': 4,
    'request_memory': '16GB',
    'request_gpus': 1,  # Request GPU if available
    'max_runtime_hours': 24,
}


# =============================================================================
# Job Generation
# =============================================================================

def generate_job_script(
    model: str,
    preset: str,
    name: str,
    epochs: int,
    data_path: str,
    extra_args: Dict,
    job_dir: Path,
) -> Path:
    """Generate a shell script for a single training job."""
    
    script_path = job_dir / f"{name}.sh"
    
    # Build training command
    cmd_parts = [
        f'python {MODELS_DIR}/train.py',
        f'--model {model}',
        f'--preset {preset}',
        f'--name {name}',
        f'--epochs {epochs}',
        f'--data_path {data_path}',
        f'--checkpoint_dir {job_dir}/checkpoints',
    ]
    
    for key, value in extra_args.items():
        if isinstance(value, bool) and value:
            cmd_parts.append(f'--{key}')
        elif isinstance(value, list):
            cmd_parts.append(f'--{key} ' + ' '.join(str(v) for v in value))
        elif value is not None:
            cmd_parts.append(f'--{key} {value}')
    
    train_cmd = ' \\\n    '.join(cmd_parts)
    
    script_content = f'''#!/bin/bash
# Training job: {name}
# Generated: {datetime.now().isoformat()}

set -e  # Exit on error

echo "=========================================="
echo "Training Job: {name}"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Activate conda environment
source /data/bfys/gscriven/miniforge3/etc/profile.d/conda.sh
conda activate {CONDA_ENV}

# Set up environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}}

# Check GPU availability
python -c "import torch; print(f'PyTorch version: {{torch.__version__}}'); print(f'CUDA available: {{torch.cuda.is_available()}}'); print(f'GPU: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}}')"

# Create output directory
mkdir -p {job_dir}/checkpoints
mkdir -p {job_dir}/logs

# Run training
echo ""
echo "Starting training..."
echo ""

{train_cmd}

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    return script_path


def generate_condor_submit(
    jobs: List[Dict],
    job_dir: Path,
    use_gpu: bool = True,
) -> Path:
    """Generate HTCondor submit file for all jobs."""
    
    submit_path = job_dir / 'submit.sub'
    
    # GPU requirements
    if use_gpu:
        gpu_req = f'''
# GPU requirements
request_gpus = {CONDOR_CONFIG['request_gpus']}
+WantGPU = true
requirements = (TARGET.GPUSlot == true)
'''
    else:
        gpu_req = '# No GPU requested'
    
    submit_content = f'''# HTCondor submit file for training jobs
# Generated: {datetime.now().isoformat()}
# Jobs: {len(jobs)}

universe = vanilla
getenv = true

# Resource requirements
request_cpus = {CONDOR_CONFIG['request_cpus']}
request_memory = {CONDOR_CONFIG['request_memory']}
{gpu_req}

# Runtime limit
+MaxRuntime = {CONDOR_CONFIG['max_runtime_hours'] * 3600}

# File transfer
should_transfer_files = NO

# Notifications
notification = Error
notify_user = gscriven@nikhef.nl

'''
    
    # Add each job
    for job in jobs:
        submit_content += f'''
# Job: {job['name']}
executable = {job['script']}
output = {job_dir}/logs/{job['name']}.out
error = {job_dir}/logs/{job['name']}.err
log = {job_dir}/logs/{job['name']}.log
queue 1

'''
    
    with open(submit_path, 'w') as f:
        f.write(submit_content)
    
    return submit_path


def generate_jobs_from_experiment(
    experiment_name: str,
    data_path: str,
    job_dir: Path,
) -> List[Dict]:
    """Generate job definitions from an experiment configuration."""
    
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    exp = EXPERIMENTS[experiment_name]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs = []
    
    common_args = exp.get('common_args', {})
    epochs = exp.get('epochs', DEFAULT_MAX_EPOCHS)
    
    # Ensure convergence settings
    if 'patience' not in common_args:
        common_args['patience'] = DEFAULT_PATIENCE
    if 'min_delta' not in common_args:
        common_args['min_delta'] = DEFAULT_MIN_DELTA
    
    # Handle preset sweep
    if 'presets' in exp:
        for preset in exp['presets']:
            for model in exp['models']:
                name = f"{experiment_name}_{model}_{preset}_{timestamp}"
                
                script = generate_job_script(
                    model=model,
                    preset=preset,
                    name=name,
                    epochs=epochs,
                    data_path=data_path,
                    extra_args=common_args,
                    job_dir=job_dir,
                )
                
                jobs.append({
                    'name': name,
                    'model': model,
                    'preset': preset,
                    'script': script,
                })
    
    # Handle parameter sweep
    elif 'sweep_param' in exp:
        param = exp['sweep_param']
        preset = exp.get('preset', 'medium')
        
        for value in exp['sweep_values']:
            for model in exp['models']:
                name = f"{experiment_name}_{model}_{param}_{value}_{timestamp}"
                
                sweep_args = common_args.copy()
                sweep_args[param] = value
                
                script = generate_job_script(
                    model=model,
                    preset=preset,
                    name=name,
                    epochs=epochs,
                    data_path=data_path,
                    extra_args=sweep_args,
                    job_dir=job_dir,
                )
                
                jobs.append({
                    'name': name,
                    'model': model,
                    'preset': preset,
                    param: value,
                    'script': script,
                })
    
    # Standard multi-model run
    else:
        preset = exp.get('preset', 'medium')
        for model in exp['models']:
            name = f"{experiment_name}_{model}_{timestamp}"
            
            script = generate_job_script(
                model=model,
                preset=preset,
                name=name,
                epochs=epochs,
                data_path=data_path,
                extra_args=common_args,
                job_dir=job_dir,
            )
            
            jobs.append({
                'name': name,
                'model': model,
                'preset': preset,
                'script': script,
            })
    
    return jobs


# =============================================================================
# Main Functions
# =============================================================================

def submit_experiment(
    experiment_name: str,
    data_path: Optional[str] = None,
    use_gpu: bool = True,
    dry_run: bool = False,
) -> None:
    """Submit an experiment to HTCondor."""
    
    # Default data path
    if data_path is None:
        data_path = str(DATA_DIR / 'training_50M.npz')
    
    # Check data exists
    if not Path(data_path).exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Please wait for data generation to complete or specify --data-path")
        sys.exit(1)
    
    # Create job directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = MODELS_DIR / 'condor_jobs' / f"{experiment_name}_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / 'logs').mkdir(exist_ok=True)
    (job_dir / 'checkpoints').mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Preparing HTCondor submission: {experiment_name}")
    print(f"{'='*60}")
    print(f"Data: {data_path}")
    print(f"Job directory: {job_dir}")
    
    # Generate jobs
    jobs = generate_jobs_from_experiment(experiment_name, data_path, job_dir)
    print(f"Jobs to submit: {len(jobs)}")
    
    for job in jobs:
        print(f"  - {job['name']}: {job['model']} / {job.get('preset', 'custom')}")
    
    # Generate submit file
    submit_file = generate_condor_submit(jobs, job_dir, use_gpu=use_gpu)
    print(f"\nSubmit file: {submit_file}")
    
    if dry_run:
        print("\n[DRY RUN] Would submit with:")
        print(f"  condor_submit {submit_file}")
        return
    
    # Submit to HTCondor
    print(f"\nSubmitting {len(jobs)} jobs to HTCondor...")
    result = subprocess.run(
        ['condor_submit', str(submit_file)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"\nSuccessfully submitted {len(jobs)} training jobs!")
        print(f"\nMonitor with: condor_q gscriven")
        print(f"Logs in: {job_dir}/logs/")
        print(f"Models in: {job_dir}/checkpoints/")
    else:
        print(f"ERROR: Submission failed!")
        print(result.stderr)
        sys.exit(1)


def submit_single_model(
    model: str,
    preset: str,
    name: Optional[str] = None,
    data_path: Optional[str] = None,
    epochs: int = DEFAULT_MAX_EPOCHS,
    extra_args: Optional[Dict] = None,
    use_gpu: bool = True,
    dry_run: bool = False,
) -> None:
    """Submit a single model training job to HTCondor."""
    
    if data_path is None:
        data_path = str(DATA_DIR / 'training_50M.npz')
    
    if not Path(data_path).exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name is None:
        name = f"{model}_{preset}_{timestamp}"
    
    job_dir = MODELS_DIR / 'condor_jobs' / name
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / 'logs').mkdir(exist_ok=True)
    (job_dir / 'checkpoints').mkdir(exist_ok=True)
    
    if extra_args is None:
        extra_args = {
            'patience': DEFAULT_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA,
        }
    
    print(f"\n{'='*60}")
    print(f"Preparing HTCondor submission: {name}")
    print(f"{'='*60}")
    
    script = generate_job_script(
        model=model,
        preset=preset,
        name=name,
        epochs=epochs,
        data_path=data_path,
        extra_args=extra_args,
        job_dir=job_dir,
    )
    
    jobs = [{'name': name, 'model': model, 'preset': preset, 'script': script}]
    submit_file = generate_condor_submit(jobs, job_dir, use_gpu=use_gpu)
    
    print(f"Model: {model}")
    print(f"Preset: {preset}")
    print(f"Data: {data_path}")
    print(f"Submit file: {submit_file}")
    
    if dry_run:
        print("\n[DRY RUN] Would submit with:")
        print(f"  condor_submit {submit_file}")
        return
    
    result = subprocess.run(
        ['condor_submit', str(submit_file)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"\nJob submitted! Monitor with: condor_q gscriven")
    else:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)


def list_experiments():
    """Print available experiments."""
    print("\n" + "=" * 70)
    print("Available Experiments for HTCondor Submission")
    print("=" * 70)
    
    for name, exp in EXPERIMENTS.items():
        n_jobs = len(exp['models'])
        if 'presets' in exp:
            n_jobs *= len(exp['presets'])
        elif 'sweep_values' in exp:
            n_jobs *= len(exp['sweep_values'])
        
        print(f"\n  {name}: ({n_jobs} jobs)")
        print(f"    {exp['description']}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Submit training jobs to HTCondor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit single model
  python submit_training.py --model mlp --preset medium

  # Submit architecture sweep
  python submit_training.py --experiment pinn_architecture_sweep

  # Submit all experiments (15 jobs)
  python submit_training.py --experiment all_architecture_sweeps

  # Dry run
  python submit_training.py --experiment quick_test --dry-run

  # List available experiments
  python submit_training.py --list
"""
    )
    
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model', type=str, choices=['mlp', 'pinn', 'rk_pinn'],
                      help='Submit single model training')
    mode.add_argument('--experiment', type=str,
                      help='Submit predefined experiment')
    mode.add_argument('--list', action='store_true',
                      help='List available experiments')
    
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['tiny', 'small', 'medium', 'large', 'wide'],
                        help='Architecture preset (for --model)')
    parser.add_argument('--name', type=str, default=None,
                        help='Job name')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to training data')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Do not request GPU')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be submitted without submitting')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.list:
        list_experiments()
        return
    
    use_gpu = not args.no_gpu
    
    if args.model:
        submit_single_model(
            model=args.model,
            preset=args.preset,
            name=args.name,
            data_path=args.data_path,
            use_gpu=use_gpu,
            dry_run=args.dry_run,
        )
    
    elif args.experiment:
        submit_experiment(
            experiment_name=args.experiment,
            data_path=args.data_path,
            use_gpu=use_gpu,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    main()
