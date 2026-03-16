#!/usr/bin/env python3
"""
ONNX Export Utilities for Track Extrapolation Models

This module handles:
1. PyTorch to ONNX conversion
2. Normalization parameter export
3. Model validation (PyTorch vs ONNX consistency)
4. C++ integration helpers

Author: G. Scriven
Date: January 2026

Usage:
    python export_onnx.py --checkpoint checkpoints/mlp_best/best_model.pt --output exports/mlp_v1.onnx
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))
from architectures import create_model, MODEL_REGISTRY


# =============================================================================
# ONNX Export
# =============================================================================

def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    opset_version: int = 14,
    input_names: list = None,
    output_names: list = None,
    dynamic_axes: dict = None
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path for ONNX file
        opset_version: ONNX opset version
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axis specification
        
    Returns:
        Path to exported ONNX file
    """
    model.eval()
    
    # Default names
    if input_names is None:
        input_names = ['track_state']
    if output_names is None:
        output_names = ['extrapolated_state']
    if dynamic_axes is None:
        dynamic_axes = {
            'track_state': {0: 'batch_size'},
            'extrapolated_state': {0: 'batch_size'}
        }
    
    # Create dummy input
    batch_size = 1
    input_dim = model.input_dim
    dummy_input = torch.randn(batch_size, input_dim)
    
    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"✓ Exported ONNX model to: {output_path}")
    
    return str(output_path)


def export_normalization(
    model: torch.nn.Module,
    output_path: str
) -> str:
    """
    Export normalization parameters to JSON for C++ usage.
    
    Args:
        model: Model with normalization buffers
        output_path: Path for JSON file
        
    Returns:
        Path to exported JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    norm = {
        'input_mean': model.input_mean.cpu().tolist(),
        'input_std': model.input_std.cpu().tolist(),
        'output_mean': model.output_mean.cpu().tolist(),
        'output_std': model.output_std.cpu().tolist(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(norm, f, indent=2)
    
    print(f"✓ Exported normalization to: {output_path}")
    
    return str(output_path)


# =============================================================================
# Validation
# =============================================================================

def validate_onnx_export(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    n_samples: int = 100,
    tolerance: float = 1e-5
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate ONNX export matches PyTorch model.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX file
        n_samples: Number of test samples
        tolerance: Maximum allowed difference
        
    Returns:
        Tuple of (passed, metrics_dict)
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime not installed, skipping validation")
        return True, {}
    
    pytorch_model.eval()
    
    # Create ONNX runtime session
    session = ort.InferenceSession(onnx_path)
    
    # Generate test data
    test_input = torch.randn(n_samples, pytorch_model.input_dim)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX inference
    onnx_output = session.run(None, {'track_state': test_input.numpy()})[0]
    
    # Compare
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    passed = max_diff < tolerance
    
    metrics = {
        'max_diff': float(max_diff),
        'mean_diff': float(mean_diff),
        'tolerance': tolerance,
        'passed': passed,
    }
    
    if passed:
        print(f"✓ ONNX validation passed (max diff: {max_diff:.2e})")
    else:
        print(f"✗ ONNX validation FAILED (max diff: {max_diff:.2e} > {tolerance:.2e})")
    
    return passed, metrics


# =============================================================================
# Model Loading
# =============================================================================

def load_checkpoint(checkpoint_path: str) -> Tuple[torch.nn.Module, dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model_kwargs = {
        'hidden_dims': config['hidden_dims'],
        'activation': config['activation'],
    }
    
    if config['model_type'] == 'mlp':
        model_kwargs['dropout'] = config.get('dropout', 0.0)
    elif config['model_type'] == 'pinn':
        model_kwargs['lambda_pde'] = config.get('lambda_pde', 1e-3)
    elif config['model_type'] == 'rk_pinn':
        model_kwargs['n_collocation'] = config.get('n_collocation', 10)
        model_kwargs['lambda_pde'] = config.get('lambda_pde', 1e-3)
    
    model = create_model(config['model_type'], **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load normalization if available
    checkpoint_dir = Path(checkpoint_path).parent
    norm_path = checkpoint_dir / 'normalization.json'
    if norm_path.exists():
        model.load_normalization(str(norm_path))
    
    return model, config


# =============================================================================
# Full Export Pipeline
# =============================================================================

def full_export(
    checkpoint_path: str,
    output_dir: str,
    model_name: str = None,
    validate: bool = True
) -> Dict[str, str]:
    """
    Full export pipeline: checkpoint → ONNX + normalization.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Output directory for exports
        model_name: Name for exported files (default: from config)
        validate: Whether to validate ONNX export
        
    Returns:
        Dictionary of exported file paths
    """
    print("\n" + "=" * 60)
    print("ONNX EXPORT PIPELINE")
    print("=" * 60)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model, config = load_checkpoint(checkpoint_path)
    
    # Determine output name
    if model_name is None:
        model_name = config.get('experiment_name', config['model_type'])
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export ONNX
    onnx_path = output_dir / f"{model_name}.onnx"
    export_to_onnx(model, str(onnx_path))
    
    # Export normalization
    norm_path = output_dir / f"{model_name}_norm.json"
    export_normalization(model, str(norm_path))
    
    # Export model config
    config_path = output_dir / f"{model_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'model_config': model.get_config(),
            'training_config': config,
        }, f, indent=2)
    print(f"✓ Exported config to: {config_path}")
    
    # Validate
    validation_results = None
    if validate:
        print("\nValidating ONNX export...")
        passed, validation_results = validate_onnx_export(model, str(onnx_path))
        
        val_path = output_dir / f"{model_name}_validation.json"
        with open(val_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"  ONNX model:     {onnx_path}")
    print(f"  Normalization:  {norm_path}")
    print(f"  Config:         {config_path}")
    
    return {
        'onnx': str(onnx_path),
        'normalization': str(norm_path),
        'config': str(config_path),
    }


# =============================================================================
# C++ Code Generation
# =============================================================================

def generate_cpp_header(
    model_name: str,
    onnx_path: str,
    norm_path: str
) -> str:
    """
    Generate C++ header with model paths and normalization constants.
    
    This can be included in C++ code for easy integration.
    """
    with open(norm_path, 'r') as f:
        norm = json.load(f)
    
    # Format arrays for C++
    def format_array(arr):
        return '{' + ', '.join(f'{x:.8f}f' for x in arr) + '}'
    
    cpp_code = f'''// Auto-generated header for {model_name}
// Generated by export_onnx.py

#pragma once

#include <array>
#include <string>

namespace TrackML {{
namespace {model_name.replace('-', '_')} {{

// ONNX model path
constexpr const char* ONNX_PATH = "{onnx_path}";

// Normalization parameters
constexpr std::array<float, {len(norm['input_mean'])}> INPUT_MEAN = {format_array(norm['input_mean'])};
constexpr std::array<float, {len(norm['input_std'])}> INPUT_STD = {format_array(norm['input_std'])};
constexpr std::array<float, {len(norm['output_mean'])}> OUTPUT_MEAN = {format_array(norm['output_mean'])};
constexpr std::array<float, {len(norm['output_std'])}> OUTPUT_STD = {format_array(norm['output_std'])};

// Model info
constexpr int INPUT_DIM = {len(norm['input_mean'])};
constexpr int OUTPUT_DIM = {len(norm['output_mean'])};

}} // namespace {model_name.replace('-', '_')}
}} // namespace TrackML
'''
    
    return cpp_code


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export track extrapolation model to ONNX',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='exports',
                       help='Output directory')
    parser.add_argument('--name', type=str, default=None,
                       help='Model name for output files')
    parser.add_argument('--no_validate', action='store_true',
                       help='Skip ONNX validation')
    parser.add_argument('--generate_cpp', action='store_true',
                       help='Generate C++ header')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Export
    paths = full_export(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        model_name=args.name,
        validate=not args.no_validate
    )
    
    # Generate C++ header if requested
    if args.generate_cpp:
        model_name = args.name or Path(args.checkpoint).parent.name
        cpp_code = generate_cpp_header(
            model_name=model_name,
            onnx_path=paths['onnx'],
            norm_path=paths['normalization']
        )
        
        cpp_path = Path(args.output) / f"{model_name}.h"
        with open(cpp_path, 'w') as f:
            f.write(cpp_code)
        print(f"✓ Generated C++ header: {cpp_path}")


if __name__ == '__main__':
    main()
