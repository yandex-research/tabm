"""Benchmark inference-only folding of LinearBatchEnsemble.

The folded layer materializes per-ensemble weights and evaluates the same function
with LinearEnsemble:

    y[b, k, o] = sum_i x[b, k, i] * r[k, i] * weight[o, i] * s[k, o] + bias[k, o]

Examples:
    python benchmarks/benchmark_fold_linear_batchensemble.py --device cpu --quick
    python benchmarks/benchmark_fold_linear_batchensemble.py --device all \
        --output results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import tabm


@dataclass(frozen=True)
class Shape:
    batch_size: int
    k: int
    in_features: int
    out_features: int
    bias: bool


@dataclass
class BenchmarkResult:
    device: str
    dtype: str
    batch_size: int
    k: int
    in_features: int
    out_features: int
    bias: bool
    original_eager_ms: float
    original_compiled_ms: float | None
    folded_eager_ms: float
    folded_compiled_ms: float | None
    folded_eager_speedup: float
    folded_compiled_speedup: float | None
    original_compiled_speedup: float | None
    original_params_mb: float
    folded_params_mb: float
    folded_param_ratio: float
    compile_error: str | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        '--device',
        action='append',
        choices=['auto', 'all', 'cpu', 'cuda', 'mps'],
        help=(
            'Device to benchmark. Pass multiple times to run several devices. '
            'Use "all" for every available device.'
        ),
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Optional path to write benchmark results as JSON.',
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup iterations before timing.',
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=30,
        help='Number of measured iterations per repeat.',
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=3,
        help='Number of timing repeats. The median is reported.',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run a small subset of shapes for quick smoke testing.',
    )
    return parser.parse_args()


def _is_mps_available() -> bool:
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def _available_devices(requested_devices: Iterable[str]) -> list[torch.device]:
    requested = set(requested_devices)
    if 'all' in requested:
        requested = {'cpu', 'cuda', 'mps'}
    elif 'auto' in requested:
        requested = {'cuda' if torch.cuda.is_available() else 'mps'}
        if 'mps' in requested and not _is_mps_available():
            requested = {'cpu'}

    devices: list[torch.device] = []
    for name in ['cpu', 'mps', 'cuda']:
        if name not in requested:
            continue
        if name == 'cuda' and not torch.cuda.is_available():
            print('Skipping CUDA because it is not available.')
            continue
        if name == 'mps' and not _is_mps_available():
            print('Skipping MPS because it is not available.')
            continue
        devices.append(torch.device(name))

    if not devices:
        raise RuntimeError('No requested devices are available.')
    return devices


def _make_shapes(quick: bool) -> list[Shape]:
    if quick:
        return [
            Shape(64, 8, 128, 128, True),
            Shape(256, 16, 128, 512, True),
            Shape(256, 16, 128, 512, False),
        ]

    return [
        Shape(
            batch_size=batch_size,
            k=k,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        for batch_size, k, in_features, out_features, bias in itertools.product(
            [64, 256, 1024],
            [8, 16, 32],
            [128, 256, 512],
            [128, 512],
            [True, False],
        )
    ]


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def _reset_compile_cache() -> None:
    if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'reset'):
        torch.compiler.reset()
    elif hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'reset'):
        torch._dynamo.reset()


def _compile(module: torch.nn.Module) -> Any:
    if not hasattr(torch, 'compile'):
        raise RuntimeError('torch.compile is not available in this PyTorch version.')
    _reset_compile_cache()
    return torch.compile(module)


def fold_linear_batchensemble(
    module: tabm.LinearBatchEnsemble,
) -> tabm.LinearEnsemble:
    """Materialize an inference-only LinearEnsemble equivalent."""
    folded = tabm.LinearEnsemble(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        k=module.k,
        dtype=module.weight.dtype,
        device=module.weight.device,
    )
    with torch.inference_mode():
        folded.weight.copy_(
            module.r[:, :, None] * module.weight.T[None, :, :] * module.s[:, None, :]
        )
        if module.bias is not None:
            assert folded.bias is not None
            folded.bias.copy_(module.bias)
    return folded.eval()


def _benchmark(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    runs: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)

    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(runs):
            fn()
        _sync(device)
        measurements.append((time.perf_counter() - start) / runs * 1000.0)
    return statistics.median(measurements)


def _parameter_size_mb(module: torch.nn.Module) -> float:
    return (
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in module.parameters()
        )
        / 1_000_000
    )


def _check_correctness(
    original: tabm.LinearBatchEnsemble,
    folded: tabm.LinearEnsemble,
    x: torch.Tensor,
) -> None:
    with torch.inference_mode():
        torch.testing.assert_close(
            folded(x),
            original(x),
            rtol=1e-4,
            atol=1e-3,
        )


def _run_one_shape(
    shape: Shape,
    *,
    device: torch.device,
    warmup: int,
    runs: int,
    repeats: int,
) -> BenchmarkResult:
    torch.manual_seed(0)
    original = tabm.LinearBatchEnsemble(
        shape.in_features,
        shape.out_features,
        bias=shape.bias,
        k=shape.k,
        scaling_init='random-signs',
        device=device,
    ).eval()
    folded = fold_linear_batchensemble(original)
    x = torch.randn(
        shape.batch_size,
        shape.k,
        shape.in_features,
        device=device,
    )
    _check_correctness(original, folded, x)

    with torch.inference_mode():
        original_eager_ms = _benchmark(
            lambda: original(x),
            device=device,
            warmup=warmup,
            runs=runs,
            repeats=repeats,
        )
        folded_eager_ms = _benchmark(
            lambda: folded(x),
            device=device,
            warmup=warmup,
            runs=runs,
            repeats=repeats,
        )

        original_compiled_ms: float | None = None
        folded_compiled_ms: float | None = None
        compile_error: str | None = None
        try:
            compiled_original = _compile(original)
            torch.testing.assert_close(
                compiled_original(x),
                original(x),
                rtol=1e-4,
                atol=1e-3,
            )
            original_compiled_ms = _benchmark(
                lambda: compiled_original(x),
                device=device,
                warmup=warmup,
                runs=runs,
                repeats=repeats,
            )

            compiled_folded = _compile(folded)
            torch.testing.assert_close(
                compiled_folded(x),
                original(x),
                rtol=1e-4,
                atol=1e-3,
            )
            folded_compiled_ms = _benchmark(
                lambda: compiled_folded(x),
                device=device,
                warmup=warmup,
                runs=runs,
                repeats=repeats,
            )
        except Exception as err:
            compile_error = f'{type(err).__name__}: {err}'

    original_params_mb = _parameter_size_mb(original)
    folded_params_mb = _parameter_size_mb(folded)
    return BenchmarkResult(
        device=device.type,
        dtype=str(x.dtype).removeprefix('torch.'),
        batch_size=shape.batch_size,
        k=shape.k,
        in_features=shape.in_features,
        out_features=shape.out_features,
        bias=shape.bias,
        original_eager_ms=original_eager_ms,
        original_compiled_ms=original_compiled_ms,
        folded_eager_ms=folded_eager_ms,
        folded_compiled_ms=folded_compiled_ms,
        folded_eager_speedup=original_eager_ms / folded_eager_ms,
        folded_compiled_speedup=(
            None
            if folded_compiled_ms is None
            else original_eager_ms / folded_compiled_ms
        ),
        original_compiled_speedup=(
            None
            if original_compiled_ms is None
            else original_eager_ms / original_compiled_ms
        ),
        original_params_mb=original_params_mb,
        folded_params_mb=folded_params_mb,
        folded_param_ratio=folded_params_mb / original_params_mb,
        compile_error=compile_error,
    )


def _format_ms(value: float | None) -> str:
    return '-' if value is None else f'{value:.3f}'


def _format_speedup(value: float | None) -> str:
    return '-' if value is None else f'{value:.2f}x'


def _format_mb(value: float) -> str:
    return f'{value:.2f}'


def _print_table(results: list[BenchmarkResult]) -> None:
    headers = [
        'device',
        'B',
        'K',
        'Din',
        'Dout',
        'bias',
        'orig_ms',
        'orig_compile_ms',
        'folded_ms',
        'folded_compile_ms',
        'folded_speedup',
        'folded_compile_speedup',
        'orig_mb',
        'folded_mb',
        'param_ratio',
    ]
    rows = [
        [
            result.device,
            str(result.batch_size),
            str(result.k),
            str(result.in_features),
            str(result.out_features),
            str(result.bias),
            _format_ms(result.original_eager_ms),
            _format_ms(result.original_compiled_ms),
            _format_ms(result.folded_eager_ms),
            _format_ms(result.folded_compiled_ms),
            _format_speedup(result.folded_eager_speedup),
            _format_speedup(result.folded_compiled_speedup),
            _format_mb(result.original_params_mb),
            _format_mb(result.folded_params_mb),
            _format_speedup(result.folded_param_ratio),
        ]
        for result in results
    ]
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    print(' | '.join(cell.ljust(width) for cell, width in zip(headers, widths)))
    print('-+-'.join('-' * width for width in widths))
    for row in rows:
        print(' | '.join(cell.rjust(width) for cell, width in zip(row, widths)))

    errors = [result for result in results if result.compile_error is not None]
    if errors:
        print('\nCompile errors:')
        for result in errors:
            print(
                f'- device={result.device}, B={result.batch_size}, K={result.k}, '
                f'Din={result.in_features}, Dout={result.out_features}, '
                f'bias={result.bias}: {result.compile_error}'
            )


def _print_summary(results: list[BenchmarkResult]) -> None:
    folded_speedups = [result.folded_eager_speedup for result in results]
    folded_compiled_speedups = [
        result.folded_compiled_speedup
        for result in results
        if result.folded_compiled_speedup is not None
    ]
    print('\nSummary:')
    print(
        'folded eager speedup: '
        f'median={statistics.median(folded_speedups):.2f}x, '
        f'min={min(folded_speedups):.2f}x, '
        f'max={max(folded_speedups):.2f}x'
    )
    if folded_compiled_speedups:
        print(
            'folded compiled speedup vs original eager: '
            f'median={statistics.median(folded_compiled_speedups):.2f}x, '
            f'min={min(folded_compiled_speedups):.2f}x, '
            f'max={max(folded_compiled_speedups):.2f}x'
        )


def main() -> None:
    args = _parse_args()
    devices = _available_devices(args.device or ['auto'])
    shapes = _make_shapes(args.quick)

    results = []
    for device in devices:
        for shape in shapes:
            results.append(
                _run_one_shape(
                    shape,
                    device=device,
                    warmup=args.warmup,
                    runs=args.runs,
                    repeats=args.repeats,
                )
            )

    _print_table(results)
    _print_summary(results)
    if args.output is not None:
        args.output.write_text(
            json.dumps([asdict(result) for result in results], indent=2) + '\n'
        )
        print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
