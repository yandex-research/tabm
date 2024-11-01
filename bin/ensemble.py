import argparse
import shutil
import sys
from pathlib import Path

import delu
import numpy as np
from loguru import logger
from scipy.special import expit, softmax

if __name__ == '__main__':
    _cwd = Path.cwd()
    assert _cwd.joinpath(
        '.git'
    ).exists(), 'The script must be run from the root of the repository'
    sys.path.append(str(_cwd))
    del _cwd

import lib
import lib.data

DEFAULT_N_ENSEMBLES = 3
DEFAULT_ENSEMBLE_SIZE = 5


def main(
    evaluation_dir: str | Path,
    n_ensembles: int = DEFAULT_N_ENSEMBLES,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    *,
    force: bool = False,
):
    """
    Examples:

    >>> main('exp/mlp/california/0-evaluation')
    <The result will be at 'exp/mlp/california/0-ensemble-{ensemble_size}'>
    """

    evaluation_dir = Path(evaluation_dir).resolve()
    assert evaluation_dir.name.endswith('-evaluation')
    logger.info(f'Computing ensembles for {lib.try_get_relative_path(evaluation_dir)}')

    for ensemble_id in range(n_ensembles):
        seeds = range(ensemble_id * ensemble_size, (ensemble_id + 1) * ensemble_size)
        single_outputs = [evaluation_dir / str(x) for x in seeds]
        output = evaluation_dir.with_name(
            evaluation_dir.name.replace('evaluation', f'ensemble-{ensemble_size}')
        ) / str(ensemble_id)

        relative_output = lib.try_get_relative_path(output)
        if not all((x / 'DONE').exists() for x in single_outputs):
            logger.warning(f'Not enough single models for {relative_output}')
            continue
        if output.exists():
            if force:
                logger.warning(f'Removing the existing output: {relative_output}')
                shutil.rmtree(output)
            else:
                lib.backup_output(output)
                logger.warning(f'Already exists! | {relative_output}')
                continue
        del relative_output

        first_report = lib.load_report(single_outputs[0])
        output.mkdir(parents=True)
        report = {
            'function': lib.get_function_full_name(main),
            'config': {
                'seeds': list(seeds),
                'data': {'path': first_report['config']['data']['path']},
            },
        }

        delu.random.seed(0)
        report['single_model_function'] = first_report['function']
        task = lib.data.Task.from_dir(first_report['config']['data']['path'])
        report['prediction_type'] = 'labels' if task.is_regression else 'probs'
        single_predictions = [lib.load_predictions(x) for x in single_outputs]

        predictions = {}
        for part in ['train', 'val', 'test']:
            stacked_predictions = np.stack([x[part] for x in single_predictions])
            if task.is_binclass:
                # Predictions for binary classifications are expected to contain
                # only the probability of the positive label.
                assert stacked_predictions.ndim == 2
                if first_report['prediction_type'] == 'logits':
                    stacked_predictions = expit(stacked_predictions)
            elif task.is_multiclass:
                assert stacked_predictions.ndim == 3
                if first_report['prediction_type'] == 'logits':
                    stacked_predictions = softmax(stacked_predictions, -1)
            else:
                assert task.is_regression
                assert stacked_predictions.ndim == 2
            predictions[part] = stacked_predictions.mean(0)

        report['metrics'] = task.calculate_metrics(
            predictions, report['prediction_type']
        )
        lib.dump_predictions(output, predictions)
        lib.dump_summary(output, lib.summarize(report))
        lib.finish(output, report)


if __name__ == '__main__':
    lib.configure_libraries()

    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation_dir')
    parser.add_argument('--n_ensembles', type=int, default=DEFAULT_N_ENSEMBLES)
    parser.add_argument('--ensemble_size', type=int, default=DEFAULT_ENSEMBLE_SIZE)
    parser.add_argument('--force', action='store_true')
    main(**vars(parser.parse_args(sys.argv[1:])))
