import argparse
import datetime
import enum
import importlib
import inspect
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any, TypeVar, cast

import numpy as np
import tomli
import tomli_w
from loguru import logger

# NOTE
# This file must NOT import anything from lib except for `env`,
# because all other submodules are allowed to import `util`.
from . import env

# The purpose of the following snippet is to optimize import times
# when slow-to-import modules are not needed.
_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


# ======================================================================================
# Const
# ======================================================================================
WORST_SCORE = -999999.0

# ======================================================================================
# Types
# ======================================================================================
KWArgs = dict[str, Any]
JSONDict = dict[str, Any]  # Must be JSON-serializable.

DataKey = str  # 'x_num', 'x_bin', 'x_cat', 'y', ...
PartKey = str  # 'train', 'val', 'test', ...


class TaskType(enum.Enum):
    REGRESSION = 'regression'
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'


class PredictionType(enum.Enum):
    LABELS = 'labels'
    PROBS = 'probs'
    LOGITS = 'logits'


class Score(enum.Enum):
    ACCURACY = 'accuracy'
    CROSS_ENTROPY = 'cross-entropy'
    MAE = 'mae'
    R2 = 'r2'
    RMSE = 'rmse'
    ROC_AUC = 'roc-auc'


# ======================================================================================
# Tools for the `main` function.
#
# The following utilities expect that the `main` function
# has one of the following signatures:
#
# 1. main(config, output = None, *, force: bool = False) -> None | JSONDict
# 2. main(config, output = None, *, force: bool = False, continue_: bool = False) -> None | JSONDict  # noqa
#
# Notes:
# * `config` is a Python dictionary or a path to a config in the TOML format.
# * `output` is the output directory with all results of the run.
#   If not provided, it it automatically inferred from the config path.
# * Setting `force=True` means removing the already existing output.
# * Setting `continue_=True` means continuing the execution from a checkpoint.
# * The return value is `report` -- a JSON-serializable Python dictionary
#   with any information about the run.
# ======================================================================================
T = TypeVar('T')


def check(
    config, output: None | str | Path, *, config_type: type[T] = dict
) -> tuple[T, Path]:
    """Load the config and infer the path to the output directory."""
    # >>> This is a snippet for the internal infrastructure, ignore it.
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if snapshot_dir and Path(snapshot_dir).joinpath('CHECKPOINTS_RESTORED').exists():
        assert inspect.stack()[1].frame.f_locals.get('continue_')
    del snapshot_dir
    # <<<

    # >>> Check paths.
    if isinstance(config, str | Path):
        # config is a path.
        config = Path(config)
        assert config.suffix == '.toml'
        assert config.exists(), f'The config {config} does not exist'
        if output is None:
            # In this case, output is a directory located next to the config.
            output = config.with_suffix('')
        config = load_config(config)
    else:
        # config is already a dictionary.
        assert (
            output is not None
        ), 'If config is a dictionary, then the `output` directory must be provided.'
    output = Path(output).resolve()

    # >>> Check the config.
    if config_type is dict:
        pass
    elif (
        # If all conditions are True, config_type is assumed to be a TypedDict.
        issubclass(config_type, dict)
        and hasattr(config_type, '__required_keys__')
        and hasattr(config_type, '__optional_keys__')
    ):
        # >>> Check the keys.
        presented_keys = frozenset(config)
        required_keys = config_type.__required_keys__  # type: ignore[code]
        optional_keys = config_type.__optional_keys__  # type: ignore[code]
        assert presented_keys >= required_keys, (
            'The config is missing the following required keys:'
            f' {", ".join(required_keys - presented_keys)}'
        )
        assert set(config) <= (required_keys | optional_keys), (
            'The config has unknown keys:'
            f' {", ".join(presented_keys - required_keys - optional_keys)}'
        )

    return cast(T, config), output


def start(output: str | Path, *, continue_: bool = False, force: bool = False) -> bool:
    """Create the output directory (if missing).

    Returns:
        True if the caller should continue the execution.
        False if the caller should immediately return.
    """
    print_sep('=')
    output = Path(output).resolve()
    print(f'[>>>] {try_get_relative_path(output)} | {datetime.datetime.now()}')
    print_sep('=')

    if output.exists():
        if force:
            logger.warning('Removing the existing output')
            time.sleep(2.0)  # Keep the above message visible for some time.
            shutil.rmtree(output)
            output.mkdir()
            return True
        elif not continue_:
            backup_output(output)
            logger.warning('The output already exists!')
            return False
        elif output.joinpath('DONE').exists():
            backup_output(output)
            logger.info('Already done!\n')
            return False
        else:
            logger.info('Continuing with the existing output')
            return True
    else:
        logger.info('Creating the output')
        output.mkdir()
        return True


def create_report(function, config) -> JSONDict:
    return {
        'function': get_function_full_name(function),
        'gpus': get_gpu_names(),
        'config': jsonify(config),
    }


def summarize(report: JSONDict) -> JSONDict:
    """Summarize the key information from the report."""
    summary = {'function': report.get('function')}

    if 'best' in report:
        # The gpus info is collected from the best report.
        summary['best'] = summarize(report['best'])
    elif 'gpus' in report:
        summary['gpus'] = report['gpus']

    for key in ['n_parameters', 'best_stage', 'best_epoch', 'tuning_time', 'trial_id']:
        if key in report:
            summary[key] = deepcopy(report[key])

    metrics = report.get('metrics')
    if metrics is not None and 'score' in next(iter(metrics.values())):
        summary['scores'] = {part: metrics[part]['score'] for part in metrics}

    for key in ['n_completed_trials', 'time']:
        if key in report:
            summary[key] = deepcopy(report[key])

    return summary


def finish(output: Path, report: JSONDict) -> None:
    dump_report(output, report)

    # >>> A code block for the internal infrastructure, ignore it.
    JSON_OUTPUT_FILE = os.environ.get('JSON_OUTPUT_FILE')
    if JSON_OUTPUT_FILE:
        try:
            key = str(output.relative_to(env.get_project_dir()))
        except ValueError:
            pass
        else:
            json_output_path = Path(JSON_OUTPUT_FILE)
            try:
                json_data = json.loads(json_output_path.read_text())
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                json_data = {}
            json_data[key] = load_report(output)
            json_output_path.write_text(json.dumps(json_data, indent=4))
            shutil.copyfile(
                json_output_path,
                os.path.join(os.environ['SNAPSHOT_PATH'], 'json_output.json'),
            )
    # <<<

    output.joinpath('DONE').touch()
    backup_output(output)
    print()
    try:
        print_summary(output)
    except FileNotFoundError:
        pass
    print()
    print_sep('=')
    print(f'[<<<] {try_get_relative_path(output)} | {datetime.datetime.now()}')
    print_sep('=')


def run(function: Callable[..., None | JSONDict]) -> None | JSONDict:
    """Run CLI for the main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--output')
    parser.add_argument('--force', action='store_true')
    if 'continue_' in inspect.signature(function).parameters:
        parser.add_argument('--continue', action='store_true', dest='continue_')

    return function(**vars(parser.parse_args(sys.argv[1:])))


# ======================================================================================
# IO for the output directory
# ======================================================================================
def load_config(output_or_config: str | Path) -> JSONDict:
    return tomli.loads(Path(output_or_config).with_suffix('.toml').read_text())


def dump_config(
    output_or_config: str | Path, config: JSONDict, *, force: bool = False
) -> None:
    config_path = Path(output_or_config).with_suffix('.toml')
    if config_path.exists() and not force:
        raise RuntimeError(
            'The following config already exists (pass force=True to overwrite it)'
            f' {config_path}'
        )
    config_path.write_text(tomli_w.dumps(config))


def load_report(output: str | Path) -> JSONDict:
    return json.loads(Path(output).joinpath('report.json').read_text())


def dump_report(output: str | Path, report: JSONDict) -> None:
    Path(output).joinpath('report.json').write_text(json.dumps(report, indent=4))


def load_summary(output: str | Path) -> JSONDict:
    return json.loads(Path(output).joinpath('summary.json').read_text())


def print_summary(output: str | Path):
    pprint(load_summary(output), sort_dicts=False, width=60)


def dump_summary(output: str | Path, summary: JSONDict) -> None:
    Path(output).joinpath('summary.json').write_text(json.dumps(summary, indent=4))


def load_predictions(output: str | Path) -> dict[PartKey, np.ndarray]:
    x = np.load(Path(output) / 'predictions.npz')
    return {key: x[key] for key in x}


def dump_predictions(
    output: str | Path, predictions: dict[PartKey, np.ndarray]
) -> None:
    np.savez(Path(output) / 'predictions.npz', **predictions)


def get_checkpoint_path(output: str | Path) -> Path:
    return Path(output) / 'checkpoint.pt'


def load_checkpoint(output: str | Path, **kwargs) -> Any:
    return _torch().load(get_checkpoint_path(output), **kwargs)


def dump_checkpoint(output: str | Path, checkpoint: JSONDict, **kwargs) -> None:
    _torch().save(checkpoint, get_checkpoint_path(output), **kwargs)


# ======================================================================================
# Printing
# ======================================================================================
def print_sep(ch='-'):
    print(ch * 100)


def print_config(config: dict) -> None:
    print()
    pprint(config, sort_dicts=False, width=100)
    print()


def print_metrics(loss: float, metrics: dict) -> None:
    print(
        f'(val) {metrics["val"]["score"]:.3f}'
        f' (test) {metrics["test"]["score"]:.3f}'
        f' (loss) {loss:.5f}'
    )


def log_scores(metrics: dict) -> None:
    logger.debug(
        f'[val] {metrics["val"]["score"]:.4f} [test] {metrics["test"]["score"]:.4f}'
    )


# ======================================================================================
# CUDA
# ======================================================================================
def get_device():  # -> torch.device
    torch = _torch()
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def is_dataparallel_available() -> bool:
    torch = _torch()
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and 'CUDA_VISIBLE_DEVICES' in os.environ
    )


def get_gpu_names() -> list[str]:
    return [
        _torch().cuda.get_device_name(i) for i in range(_torch().cuda.device_count())
    ]


def is_oom_exception(err: RuntimeError) -> bool:
    return isinstance(err, _torch().cuda.OutOfMemoryError) or any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


# ======================================================================================
# Other
# ======================================================================================
def configure_libraries():
    torch = _torch()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[code]
    torch.backends.cudnn.allow_tf32 = False  # type: ignore[code]
    torch.backends.cudnn.benchmark = False  # type: ignore[code]
    torch.backends.cudnn.deterministic = True  # type: ignore[code]

    logger.remove()
    logger.add(sys.stderr, format='<level>{message}</level>')


def try_get_relative_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    project_dir = env.get_project_dir()
    return path.relative_to(project_dir) if project_dir in path.parents else path


def jsonify(value):
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return value
    elif isinstance(value, list):
        return [jsonify(x) for x in value]
    elif isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    else:
        return '<nonserializable>'


def are_valid_predictions(predictions: dict) -> bool:
    # predictions: dict[PartKey, np.ndarray]
    assert all(isinstance(x, np.ndarray) for x in predictions.values())
    return all(np.isfinite(x).all() for x in predictions.values())


def import_(qualname: str) -> Any:
    """
    Examples:

    >>> import_('bin.model.main')
    """
    try:
        module, name = qualname.rsplit('.', 1)
        return getattr(importlib.import_module(module), name)
    except Exception as err:
        raise ValueError(f'Cannot import "{qualname}"') from err


def get_function_full_name(function: Callable) -> str:
    """
    Examples:

    >>> # In the script bin/model.py
    >>> get_function_full_name(main) == 'bin.model.main'

    >>> # In the script a/b/c/foo.py
    >>> assert get_function_full_name(main) == 'a.b.c.foo.main'
    """
    module = inspect.getmodule(function)
    assert module is not None, 'Failed to locate the module of the function.'

    module_path = getattr(module, '__file__', None)
    assert module_path is not None, (
        'Failed to locate the module of the function.'
        ' This can happen if the code is running in a Jupyter notebook.'
    )

    module_path = Path(module_path).resolve()
    project_dir = env.get_project_dir()
    assert project_dir in module_path.parents, (
        'The module of the function must be located within the project directory: '
        f' {project_dir}'
    )

    module_full_name = str(
        module_path.relative_to(project_dir).with_suffix('')
    ).replace('/', '.')
    return f'{module_full_name}.{function.__name__}'


_LAST_SNAPSHOT_TIME = None


def backup_output(output: Path) -> None:
    """A function for the internal infrastructure, ignore it."""
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output.relative_to(env.get_project_dir())
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output = dir_ / relative_output_dir
        prev_backup_output = new_output.with_name(new_output.name + '_prev')
        new_output.parent.mkdir(exist_ok=True, parents=True)
        if new_output.exists():
            new_output.rename(prev_backup_output)
        shutil.copytree(output, new_output)
        # The case for evaluate.py which automatically creates configs.
        if output.with_suffix('.toml').exists():
            shutil.copyfile(
                output.with_suffix('.toml'), new_output.with_suffix('.toml')
            )
        if prev_backup_output.exists():
            shutil.rmtree(prev_backup_output)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        import nirvana_dl.snapshot  # type: ignore[code]

        nirvana_dl.snapshot.dump_snapshot()
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')
