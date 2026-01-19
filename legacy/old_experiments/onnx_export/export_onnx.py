#!/usr/bin/env python3
"""
ONNX Export for Track Extrapolation Models

Export trained PyTorch models to ONNX format for production C++ inference.

Features:
- Export MLP and PINN models to ONNX
- Include normalization in the model graph
- Optimize for inference (constant folding, etc.)
- Validate exported models against PyTorch originals

Author: Auto-generated
Date: 2024-12-21
"""

import numpy as np
import torch
import torch.nn as nn
import struct
from pathlib import Path
import json

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx/onnxruntime not installed. Install with:")
    print("  pip install onnx onnxruntime")


class TrackMLPForExport(nn.Module):
    """MLP model for ONNX export with built-in normalization."""
    def __init__(self, hidden_dims=[256, 256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = 6
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.Tanh()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 4))
        self.network = nn.Sequential(*layers)
        
        # Normalization parameters (will be loaded from binary)
        self.register_buffer('input_mean', torch.zeros(6))
        self.register_buffer('input_std', torch.ones(6))
        self.register_buffer('output_mean', torch.zeros(4))
        self.register_buffer('output_std', torch.ones(4))
        
    def forward(self, x):
        # Normalize input
        x_norm = (x - self.input_mean) / self.input_std
        # Network forward
        out_norm = self.network(x_norm)
        # Denormalize output
        out = out_norm * self.output_std + self.output_mean
        return out


def load_model_from_binary(filepath, model_class=TrackMLPForExport):
    """Load model from binary file format."""
    model = model_class()
    
    with open(filepath, 'rb') as f:
        n_layers = struct.unpack('i', f.read(4))[0]
        
        linear_layers = [m for m in model.network if isinstance(m, nn.Linear)]
        
        for layer in linear_layers:
            rows, cols = struct.unpack('ii', f.read(8))
            W = np.frombuffer(f.read(rows * cols * 8), dtype=np.float64).reshape(rows, cols)
            b = np.frombuffer(f.read(rows * 8), dtype=np.float64)
            layer.weight.data = torch.FloatTensor(W)
            layer.bias.data = torch.FloatTensor(b)
        
        # Load normalization parameters
        n = struct.unpack('i', f.read(4))[0]
        model.input_mean = torch.FloatTensor(np.frombuffer(f.read(n * 8), dtype=np.float64))
        
        n = struct.unpack('i', f.read(4))[0]
        model.input_std = torch.FloatTensor(np.frombuffer(f.read(n * 8), dtype=np.float64))
        
        n = struct.unpack('i', f.read(4))[0]
        model.output_mean = torch.FloatTensor(np.frombuffer(f.read(n * 8), dtype=np.float64))
        
        n = struct.unpack('i', f.read(4))[0]
        model.output_std = torch.FloatTensor(np.frombuffer(f.read(n * 8), dtype=np.float64))
    
    return model


def export_to_onnx(model, output_path, opset_version=14, optimize=True):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path for ONNX file
        opset_version: ONNX opset version
        optimize: Whether to optimize the exported model
    
    Returns:
        Path to exported ONNX file
    """
    if not ONNX_AVAILABLE:
        raise ImportError("onnx not installed")
    
    model.eval()
    
    # Create dummy input
    # Input: [x, y, tx, ty, qop, dz]
    dummy_input = torch.randn(1, 6)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Exported to {output_path}")
    
    # Optimize if requested
    if optimize:
        from onnx import optimizer
        onnx_model = onnx.load(output_path)
        
        # Basic optimization passes
        passes = [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'eliminate_nop_pad',
            'eliminate_unused_initializer',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
        ]
        
        try:
            optimized_model = optimizer.optimize(onnx_model, passes)
            onnx.save(optimized_model, output_path)
            print(f"Optimized model saved to {output_path}")
        except Exception as e:
            print(f"Optimization failed (using unoptimized): {e}")
    
    return output_path


def validate_onnx_export(pytorch_model, onnx_path, n_tests=100, atol=1e-5):
    """
    Validate ONNX export against PyTorch model.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX file
        n_tests: Number of random inputs to test
        atol: Absolute tolerance for comparison
    
    Returns:
        dict with validation results
    """
    if not ONNX_AVAILABLE:
        raise ImportError("onnxruntime not installed")
    
    pytorch_model.eval()
    
    # Load ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))
    
    max_diff = 0
    all_close = True
    
    for i in range(n_tests):
        # Generate random input in realistic range
        test_input = torch.FloatTensor([
            [np.random.uniform(-900, 900),      # x
             np.random.uniform(-750, 750),      # y
             np.random.uniform(-0.3, 0.3),      # tx
             np.random.uniform(-0.25, 0.25),    # ty
             np.random.uniform(-0.001, 0.001),  # qop
             4000.0]                             # dz
        ])
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # ONNX inference
        onnx_output = ort_session.run(
            None, 
            {'input': test_input.numpy()}
        )[0]
        
        diff = np.abs(pytorch_output - onnx_output).max()
        max_diff = max(max_diff, diff)
        
        if not np.allclose(pytorch_output, onnx_output, atol=atol):
            all_close = False
    
    return {
        'valid': all_close,
        'max_diff': float(max_diff),
        'n_tests': n_tests,
        'atol': atol
    }


def benchmark_inference(onnx_path, n_samples=10000, batch_size=1):
    """
    Benchmark ONNX inference speed.
    
    Returns:
        dict with timing results
    """
    if not ONNX_AVAILABLE:
        raise ImportError("onnxruntime not installed")
    
    import time
    
    # Create session with optimizations
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    ort_session = ort.InferenceSession(str(onnx_path), sess_options)
    
    # Generate test data
    test_data = np.random.randn(n_samples, 6).astype(np.float32)
    
    # Warmup
    for i in range(100):
        _ = ort_session.run(None, {'input': test_data[i:i+1]})
    
    # Benchmark single inference
    start = time.time()
    for i in range(n_samples):
        _ = ort_session.run(None, {'input': test_data[i:i+1]})
    single_time = (time.time() - start) / n_samples * 1e6  # microseconds
    
    # Benchmark batched inference
    n_batches = n_samples // batch_size
    start = time.time()
    for i in range(n_batches):
        batch = test_data[i*batch_size:(i+1)*batch_size]
        _ = ort_session.run(None, {'input': batch})
    batch_time = (time.time() - start) / n_samples * 1e6  # microseconds per sample
    
    return {
        'single_inference_us': single_time,
        'batched_inference_us': batch_time,
        'batch_size': batch_size,
        'n_samples': n_samples
    }


def main():
    if not ONNX_AVAILABLE:
        print("ERROR: onnx and onnxruntime are required")
        print("Install with: pip install onnx onnxruntime")
        return
    
    base_dir = Path(__file__).parent.parent.parent
    models_dir = base_dir / 'ml_models' / 'models'
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("ONNX EXPORT")
    print("=" * 60)
    
    results = {}
    
    # Export each model
    models_to_export = [
        ('mlp_full_domain.bin', 'mlp_full_domain.onnx'),
        ('pinn_full_domain.bin', 'pinn_full_domain.onnx'),
    ]
    
    for bin_name, onnx_name in models_to_export:
        bin_path = models_dir / bin_name
        if not bin_path.exists():
            print(f"\nSkipping {bin_name} (not found)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {bin_name}")
        print(f"{'='*60}")
        
        # Load model
        print("Loading model...")
        model = load_model_from_binary(str(bin_path))
        
        # Export to ONNX
        onnx_path = output_dir / onnx_name
        print(f"Exporting to ONNX...")
        export_to_onnx(model, str(onnx_path), optimize=False)  # Skip optimization for compatibility
        
        # Validate
        print("Validating export...")
        validation = validate_onnx_export(model, onnx_path)
        print(f"  Valid: {validation['valid']}")
        print(f"  Max diff: {validation['max_diff']:.2e}")
        
        # Benchmark
        print("Benchmarking...")
        benchmark = benchmark_inference(str(onnx_path))
        print(f"  Single inference: {benchmark['single_inference_us']:.2f} μs")
        print(f"  Batched (100): ", end="")
        benchmark_batch = benchmark_inference(str(onnx_path), batch_size=100)
        print(f"{benchmark_batch['batched_inference_us']:.2f} μs/sample")
        
        # Get file sizes
        bin_size = bin_path.stat().st_size / 1024  # KB
        onnx_size = onnx_path.stat().st_size / 1024  # KB
        
        results[bin_name] = {
            'onnx_path': str(onnx_path),
            'validation': validation,
            'benchmark_single_us': benchmark['single_inference_us'],
            'benchmark_batched_us': benchmark_batch['batched_inference_us'],
            'binary_size_kb': bin_size,
            'onnx_size_kb': onnx_size
        }
    
    # Save results
    results_path = output_dir / 'onnx_export_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Valid':<8} {'Binary KB':<12} {'ONNX KB':<12} {'Time (μs)':<12}")
    print("-" * 69)
    for name, res in results.items():
        print(f"{name:<25} {str(res['validation']['valid']):<8} "
              f"{res['binary_size_kb']:<12.1f} {res['onnx_size_kb']:<12.1f} "
              f"{res['benchmark_single_us']:<12.2f}")
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
