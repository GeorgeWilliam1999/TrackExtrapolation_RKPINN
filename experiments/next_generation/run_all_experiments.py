#!/usr/bin/env python3
"""
================================================================================
UNIFIED EXPERIMENT RUNNER
================================================================================

Master script for running ALL experiments in the Neural Network Track 
Extrapolator project. This script consolidates all training, evaluation, 
and analysis workflows into a single reusable interface.

Experiments:
    1. Architecture Comparison - MLP/PINN/RK_PINN × tiny/small/medium/wide
    2. Physics Loss Ablation - Varying λ_PDE for PINN/RK_PINN
    3. Momentum Range Studies - Low/Mid/High momentum specialized models
    4. Timing Benchmarks - NN inference vs C++ RK extrapolator
    5. Generalization Tests - Out-of-distribution evaluation
    6. Learning Dynamics - Loss landscape and convergence analysis

Usage:
    # List all available experiments
    python run_all_experiments.py --list

    # Run specific experiment
    python run_all_experiments.py --experiment architecture_comparison

    # Run all experiments
    python run_all_experiments.py --all

    # Run experiments locally (not HTCondor)
    python run_all_experiments.py --experiment architecture_comparison --local

    # Dry run (show what would be done)
    python run_all_experiments.py --all --dry-run

    # Run analysis after training completes
    python run_all_experiments.py --analyze

Author: G. Scriven
Date: January 2026
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
TRAINING_DIR = BASE_DIR / "training"
DATA_DIR = BASE_DIR / "data_generation" / "data"
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
ANALYSIS_DIR = BASE_DIR / "analysis"
RESULTS_DIR = BASE_DIR / "results"

# Conda environment
CONDA_ENV = "TE"
CONDA_PATH = "/data/bfys/gscriven/conda"

# =============================================================================
# Experiment Definitions
# =============================================================================

EXPERIMENTS = {
    # =========================================================================
    # EXPERIMENT 1: Architecture Comparison (12 jobs)
    # =========================================================================
    "architecture_comparison": {
        "description": "Compare MLP, PINN, RK_PINN across architecture sizes",
        "category": "core",
        "jobs": [
            # MLP variants
            {"name": "mlp_tiny", "model": "mlp", "hidden_dims": [64, 64]},
            {"name": "mlp_small", "model": "mlp", "hidden_dims": [128, 128]},
            {"name": "mlp_medium", "model": "mlp", "hidden_dims": [256, 256, 128]},
            {"name": "mlp_wide", "model": "mlp", "hidden_dims": [512, 512, 256, 128]},
            # PINN variants
            {"name": "pinn_tiny", "model": "pinn", "hidden_dims": [64, 64], "lambda_pde": 1.0},
            {"name": "pinn_small", "model": "pinn", "hidden_dims": [128, 128], "lambda_pde": 1.0},
            {"name": "pinn_medium", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0},
            {"name": "pinn_wide", "model": "pinn", "hidden_dims": [512, 512, 256, 128], "lambda_pde": 1.0},
            # RK_PINN variants
            {"name": "rkpinn_tiny", "model": "rk_pinn", "hidden_dims": [64, 64], "lambda_pde": 1.0},
            {"name": "rkpinn_small", "model": "rk_pinn", "hidden_dims": [128, 128], "lambda_pde": 1.0},
            {"name": "rkpinn_medium", "model": "rk_pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0},
            {"name": "rkpinn_wide", "model": "rk_pinn", "hidden_dims": [512, 512, 256, 128], "lambda_pde": 1.0},
        ],
        "data_path": "training_50M.npz",
    },
    
    # =========================================================================
    # EXPERIMENT 2: Physics Loss Ablation (8 jobs)
    # =========================================================================
    "physics_ablation": {
        "description": "Study effect of physics loss weight on PINN/RK_PINN",
        "category": "ablation",
        "jobs": [
            # PINN ablations
            {"name": "pinn_medium_data_only", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 0.0, "lambda_ic": 0.0},
            {"name": "pinn_medium_pde_weak", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 0.01, "lambda_ic": 0.01},
            {"name": "pinn_medium_pde_strong", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 10.0, "lambda_ic": 10.0},
            {"name": "pinn_medium_pde_dominant", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 100.0, "lambda_ic": 100.0},
            # RK_PINN ablations
            {"name": "rkpinn_medium_data_only", "model": "rk_pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 0.0, "lambda_ic": 0.0},
            {"name": "rkpinn_medium_pde_weak", "model": "rk_pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 0.01, "lambda_ic": 0.01},
            {"name": "rkpinn_medium_pde_strong", "model": "rk_pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 10.0, "lambda_ic": 10.0},
            {"name": "rkpinn_medium_pde_dominant", "model": "rk_pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 100.0, "lambda_ic": 100.0},
        ],
        "data_path": "training_50M.npz",
    },
    
    # =========================================================================
    # EXPERIMENT 3: Momentum Range Studies (9 jobs)
    # =========================================================================
    "momentum_studies": {
        "description": "Train specialized models for different momentum ranges",
        "category": "momentum",
        "jobs": [
            # Low momentum (0.5-5 GeV) - most challenging
            {"name": "mlp_medium_low_p", "model": "mlp", "hidden_dims": [256, 256, 128], "data_path": "training_low_p.npz"},
            {"name": "pinn_medium_low_p", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0, "data_path": "training_low_p.npz"},
            {"name": "rkpinn_medium_low_p", "model": "rk_pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0, "data_path": "training_low_p.npz"},
            # Mid momentum (5-20 GeV) - typical LHCb
            {"name": "mlp_medium_mid_p", "model": "mlp", "hidden_dims": [256, 256, 128], "data_path": "training_mid_p.npz"},
            {"name": "pinn_medium_mid_p", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0, "data_path": "training_mid_p.npz"},
            {"name": "rkpinn_medium_mid_p", "model": "rk_pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0, "data_path": "training_mid_p.npz"},
            # High momentum (20-100 GeV) - small bending
            {"name": "mlp_medium_high_p", "model": "mlp", "hidden_dims": [256, 256, 128], "data_path": "training_high_p.npz"},
            {"name": "pinn_medium_high_p", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0, "data_path": "training_high_p.npz"},
            {"name": "rkpinn_medium_high_p", "model": "rk_pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0, "data_path": "training_high_p.npz"},
        ],
    },
    
    # =========================================================================
    # EXPERIMENT 4: Activation Function Study (4 jobs)
    # =========================================================================
    "activation_study": {
        "description": "Compare activation functions (ReLU, Tanh, SiLU, GELU)",
        "category": "hyperparameter",
        "jobs": [
            {"name": "pinn_medium_relu", "model": "pinn", "hidden_dims": [256, 256, 128], "activation": "relu", "lambda_pde": 1.0},
            {"name": "pinn_medium_tanh", "model": "pinn", "hidden_dims": [256, 256, 128], "activation": "tanh", "lambda_pde": 1.0},
            {"name": "pinn_medium_silu", "model": "pinn", "hidden_dims": [256, 256, 128], "activation": "silu", "lambda_pde": 1.0},
            {"name": "pinn_medium_gelu", "model": "pinn", "hidden_dims": [256, 256, 128], "activation": "gelu", "lambda_pde": 1.0},
        ],
        "data_path": "training_50M.npz",
    },
    
    # =========================================================================
    # EXPERIMENT 5: Data Efficiency Study (5 jobs)
    # =========================================================================
    "data_efficiency": {
        "description": "Study effect of training data volume",
        "category": "data",
        "jobs": [
            {"name": "pinn_medium_1M", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0, "max_samples": 1_000_000},
            {"name": "pinn_medium_5M", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0, "max_samples": 5_000_000},
            {"name": "pinn_medium_10M", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0, "max_samples": 10_000_000},
            {"name": "pinn_medium_25M", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0, "max_samples": 25_000_000},
            {"name": "pinn_medium_50M", "model": "pinn", "hidden_dims": [256, 256, 128], "lambda_pde": 1.0, "max_samples": 50_000_000},
        ],
        "data_path": "training_50M.npz",
    },
}

# Default training parameters
DEFAULT_TRAINING_PARAMS = {
    "epochs": 500,
    "patience": 30,
    "batch_size": 2048,
    "lr": 1e-3,
    "activation": "silu",
    "lambda_pde": 1.0,
    "lambda_ic": 1.0,
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_data_path(job: dict, experiment: dict) -> Path:
    """Resolve data path for a job."""
    data_file = job.get("data_path", experiment.get("data_path", "training_50M.npz"))
    return DATA_DIR / data_file


def build_train_command(job: dict, experiment: dict, local: bool = False) -> List[str]:
    """Build the training command for a job."""
    data_path = get_data_path(job, experiment)
    
    cmd = [
        sys.executable,
        str(MODELS_DIR / "train.py"),
        "--model", job["model"],
        "--name", job["name"],
        "--hidden_dims", *[str(d) for d in job["hidden_dims"]],
        "--epochs", str(job.get("epochs", DEFAULT_TRAINING_PARAMS["epochs"])),
        "--patience", str(job.get("patience", DEFAULT_TRAINING_PARAMS["patience"])),
        "--batch_size", str(job.get("batch_size", DEFAULT_TRAINING_PARAMS["batch_size"])),
        "--lr", str(job.get("lr", DEFAULT_TRAINING_PARAMS["lr"])),
        "--activation", job.get("activation", DEFAULT_TRAINING_PARAMS["activation"]),
        "--checkpoint_dir", str(TRAINED_MODELS_DIR),
        "--data_path", str(data_path),
    ]
    
    # Add physics loss parameters for PINN/RK_PINN
    if job["model"] in ["pinn", "rk_pinn"]:
        lambda_pde = job.get("lambda_pde", DEFAULT_TRAINING_PARAMS["lambda_pde"])
        lambda_ic = job.get("lambda_ic", DEFAULT_TRAINING_PARAMS["lambda_ic"])
        cmd.extend(["--lambda_pde", str(lambda_pde)])
        cmd.extend(["--lambda_ic", str(lambda_ic)])
    
    # Add max_samples if specified
    if "max_samples" in job:
        cmd.extend(["--max_samples", str(job["max_samples"])])
    
    return cmd


def submit_condor_job(job: dict, experiment: dict, dry_run: bool = False) -> Optional[int]:
    """Submit a job to HTCondor."""
    job_file = TRAINING_DIR / "jobs" / f"{job['name']}.sub"
    
    if not job_file.exists():
        print(f"  ⚠️  Job file not found: {job_file}")
        return None
    
    if dry_run:
        print(f"  [DRY RUN] Would submit: {job_file}")
        return None
    
    result = subprocess.run(
        ["condor_submit", str(job_file)],
        capture_output=True,
        text=True,
        cwd=str(TRAINING_DIR)
    )
    
    if result.returncode == 0:
        # Extract cluster ID from output
        for line in result.stdout.split("\n"):
            if "submitted to cluster" in line:
                cluster_id = int(line.split()[-1].rstrip("."))
                return cluster_id
    else:
        print(f"  ❌ Failed to submit: {result.stderr}")
    
    return None


def run_local_job(job: dict, experiment: dict, dry_run: bool = False) -> bool:
    """Run a job locally (not on HTCondor)."""
    cmd = build_train_command(job, experiment, local=True)
    
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True
    
    print(f"  Running: {job['name']}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    return result.returncode == 0


def list_experiments():
    """Print all available experiments."""
    print("\n" + "=" * 70)
    print("AVAILABLE EXPERIMENTS")
    print("=" * 70)
    
    for name, exp in EXPERIMENTS.items():
        n_jobs = len(exp["jobs"])
        category = exp.get("category", "other")
        print(f"\n  {name}")
        print(f"    Category: {category}")
        print(f"    Jobs: {n_jobs}")
        print(f"    Description: {exp['description']}")
    
    print("\n" + "-" * 70)
    total = sum(len(e["jobs"]) for e in EXPERIMENTS.values())
    print(f"Total: {len(EXPERIMENTS)} experiments, {total} jobs")
    print()


def run_experiment(
    name: str,
    local: bool = False,
    dry_run: bool = False,
    force: bool = False
) -> Dict[str, Any]:
    """Run a single experiment."""
    if name not in EXPERIMENTS:
        print(f"❌ Unknown experiment: {name}")
        print(f"   Available: {list(EXPERIMENTS.keys())}")
        return {"success": False, "error": "Unknown experiment"}
    
    exp = EXPERIMENTS[name]
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 70}")
    print(f"Description: {exp['description']}")
    print(f"Jobs: {len(exp['jobs'])}")
    print(f"Mode: {'Local' if local else 'HTCondor'}")
    if dry_run:
        print(f"[DRY RUN MODE]")
    print()
    
    results = {"experiment": name, "jobs": [], "submitted": 0, "skipped": 0, "failed": 0}
    
    for job in exp["jobs"]:
        job_name = job["name"]
        model_dir = TRAINED_MODELS_DIR / job_name
        
        # Check if already trained
        if model_dir.exists() and not force:
            print(f"  ⏭️  {job_name} - already exists (use --force to retrain)")
            results["skipped"] += 1
            results["jobs"].append({"name": job_name, "status": "skipped"})
            continue
        
        if local:
            success = run_local_job(job, exp, dry_run)
            status = "success" if success else "failed"
            if success:
                results["submitted"] += 1
            else:
                results["failed"] += 1
        else:
            cluster_id = submit_condor_job(job, exp, dry_run)
            if cluster_id:
                print(f"  ✅ {job_name} - submitted (cluster {cluster_id})")
                results["submitted"] += 1
                status = "submitted"
            elif dry_run:
                status = "dry_run"
                results["submitted"] += 1
            else:
                results["failed"] += 1
                status = "failed"
        
        results["jobs"].append({"name": job_name, "status": status})
    
    return results


def run_all_experiments(local: bool = False, dry_run: bool = False, force: bool = False):
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 70)
    
    all_results = []
    for name in EXPERIMENTS:
        result = run_experiment(name, local=local, dry_run=dry_run, force=force)
        all_results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_submitted = sum(r["submitted"] for r in all_results)
    total_skipped = sum(r["skipped"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)
    
    print(f"  Submitted: {total_submitted}")
    print(f"  Skipped:   {total_skipped}")
    print(f"  Failed:    {total_failed}")
    
    # Save results
    results_file = RESULTS_DIR / f"experiment_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")


def check_job_status():
    """Check status of HTCondor jobs."""
    result = subprocess.run(["condor_q", "-nobatch"], capture_output=True, text=True)
    print(result.stdout)


def collect_results() -> Dict[str, Any]:
    """Collect results from all trained models."""
    results = {}
    
    for model_dir in TRAINED_MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        
        history_file = model_dir / "history.json"
        config_file = model_dir / "config.json"
        
        if not history_file.exists():
            continue
        
        with open(history_file) as f:
            history = json.load(f)
        
        config = {}
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
        
        # Extract final metrics
        final_metrics = {
            "train_loss": history["train_loss"][-1] if history.get("train_loss") else None,
            "val_loss": history["val_loss"][-1] if history.get("val_loss") else None,
            "epochs_trained": len(history.get("train_loss", [])),
            "best_val_loss": min(history.get("val_loss", [float("inf")])),
        }
        
        results[model_dir.name] = {
            "config": config,
            "metrics": final_metrics,
            "history": history,
        }
    
    return results


def generate_summary_table(results: Dict[str, Any]) -> str:
    """Generate a summary table of results."""
    lines = [
        "=" * 90,
        f"{'Model':<30} {'Val Loss':>12} {'Epochs':>8} {'Architecture':>25}",
        "-" * 90,
    ]
    
    # Sort by validation loss
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"].get("best_val_loss", float("inf"))
    )
    
    for name, data in sorted_results:
        val_loss = data["metrics"].get("best_val_loss", float("inf"))
        epochs = data["metrics"].get("epochs_trained", 0)
        arch = str(data["config"].get("hidden_dims", "N/A"))
        
        lines.append(f"{name:<30} {val_loss:>12.6f} {epochs:>8} {arch:>25}")
    
    lines.append("=" * 90)
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Experiment Runner for Neural Network Track Extrapolators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all_experiments.py --list
    python run_all_experiments.py --experiment architecture_comparison
    python run_all_experiments.py --all
    python run_all_experiments.py --all --local
    python run_all_experiments.py --analyze
        """
    )
    
    parser.add_argument("--list", action="store_true", help="List all available experiments")
    parser.add_argument("--experiment", "-e", type=str, help="Run specific experiment by name")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--local", action="store_true", help="Run locally instead of HTCondor")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    parser.add_argument("--force", action="store_true", help="Force retraining even if model exists")
    parser.add_argument("--status", action="store_true", help="Check HTCondor job status")
    parser.add_argument("--analyze", action="store_true", help="Collect and analyze results")
    parser.add_argument("--summary", action="store_true", help="Print summary table of results")
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
    elif args.status:
        check_job_status()
    elif args.analyze or args.summary:
        results = collect_results()
        if not results:
            print("No trained models found.")
            return
        
        print(f"\nFound {len(results)} trained models\n")
        print(generate_summary_table(results))
        
        if args.analyze:
            # Save full results
            output_file = RESULTS_DIR / "all_results.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nFull results saved to: {output_file}")
    elif args.experiment:
        run_experiment(args.experiment, local=args.local, dry_run=args.dry_run, force=args.force)
    elif args.all:
        run_all_experiments(local=args.local, dry_run=args.dry_run, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
