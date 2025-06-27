import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path
from pprint import pprint
from typing import Any, Literal

import delu
import optuna
import optuna.samplers
import optuna.trial
from loguru import logger
from typing_extensions import NotRequired, TypedDict

if __name__ == '__main__':
    _cwd = Path.cwd()
    assert _cwd.joinpath(
        'pixi.toml'
    ).exists(), 'The script must be run from the `paper/` directory'
    sys.path.append(str(_cwd))
    del _cwd


import lib
from lib.util import KWArgs


def _suggest(trial: optuna.trial.Trial, distribution: str, label: str, *args):
    return getattr(trial, f'suggest_{distribution}')(label, *args)


def sample_config(
    trial: optuna.trial.Trial,
    space: bool | int | float | str | bytes | list | dict,
    label_parts: list,
) -> Any:
    if isinstance(space, bool | int | float | str | bytes):
        return space
    elif isinstance(space, dict):
        if '_tune_' in space or '-tune-' in space:
            distribution = space['_tune_'] if '_tune_' in space else space['-tune-']
            # for complex cases, for example:
            # [model]
            # _tune_ = "complex-custom-distribution"
            # <any keys and values for the distribution>
            if distribution == 'complex-custom-distribution':
                raise NotImplementedError()
            else:
                raise ValueError(f'Unknown distibution: "{distribution}"')
        else:
            return {
                key: sample_config(trial, subspace, [*label_parts, key])
                for key, subspace in space.items()
            }
    elif isinstance(space, list):
        if not space:
            return space
        elif space[0] not in ['_tune_', '-tune-']:
            return [
                sample_config(trial, subspace, [*label_parts, i])
                for i, subspace in enumerate(space)
            ]
        else:
            # space = ["_tune_"/"-tune-", distribution, distribution_arg_0, distribution_1, ...]  # noqa: E501
            _, distribution, *args = space
            label = '.'.join(map(str, label_parts))

            if distribution.startswith('?'):
                default, args_ = args[0], args[1:]
                if trial.suggest_categorical('?' + label, [False, True]):
                    return _suggest(trial, distribution.lstrip('?'), label, *args_)
                else:
                    return default

            elif distribution == '$list':
                size, item_distribution, *item_args = args
                return [
                    _suggest(trial, item_distribution, label + f'.{i}', *item_args)
                    for i in range(size)
                ]

            else:
                return _suggest(trial, distribution, label, *args)


class Config(TypedDict):
    seed: int
    function: str
    space: dict[str, Any]
    n_trials: NotRequired[int]
    timeout: NotRequired[int]
    sampler: NotRequired[KWArgs]
    sampler_type: NotRequired[str]
    # Using the following option can lead to high disk space consumption.
    # The supposed usage is to set it to True or 'only-reports' for
    # more expensive long-running tasks, but not for those numerous
    # tabular tasks presented in this repository.
    save_trials: NotRequired[bool | Literal['only-reports']]


def main(
    config: Config | str | Path,
    output: None | str | Path = None,
    *,
    force: bool = False,
    continue_: bool = False,
) -> None | lib.JSONDict:
    config, output = lib.check(config, output, config_type=Config)
    assert 'seed' not in config.get('sampler', {})
    if not lib.start(output, force=force, continue_=continue_):
        return None

    lib.print_config(config)  # type: ignore[code]
    delu.random.seed(config['seed'])
    report = lib.create_report(main, config)  # type: ignore[code]
    function = lib.import_(config['function'])

    n_trials = config.get('n_trials')
    timeout = config.get('timeout')

    if lib.get_checkpoint_path(output).exists():
        del report
        checkpoint = lib.load_checkpoint(output)
        report, study, trial_reports, timer = (
            checkpoint['report'],
            checkpoint['study'],
            checkpoint['trial_reports'],
            checkpoint['timer'],
        )
        delu.random.set_state(checkpoint['random_state'])
        if n_trials is not None:
            n_trials -= len(study.trials)
        if timeout is not None:
            timeout -= timer.elapsed()

        report.setdefault('continuations', []).append(len(study.trials))
        print(
            f'Resuming from checkpoint ({len(study.trials)} completed,'
            f' {n_trials or "inf"} remaining)'
        )
        time.sleep(1.0)
    else:
        study = optuna.create_study(
            direction='maximize',
            sampler=getattr(optuna.samplers, config.get('sampler_type', 'TPESampler'))(
                **config.get('sampler', {}), seed=config['seed']
            ),
        )
        trial_reports = []
        timer = delu.tools.Timer()

    def objective(trial: optuna.trial.Trial) -> float:
        trial_config = sample_config(trial, config['space'], [])
        kwargs = {}

        save_trials = config.get('save_trials', False)
        if save_trials:
            trial_output = output / 'trials' / str(trial.number)
            trial_output.parent.mkdir(exist_ok=True)
            if trial_output.exists():
                # Resuming the latest unfinished trial is not supported.
                logger.warning(
                    'Removing the latest unfinished trial'
                    f' {trial_output.relative_to(lib.env.get_project_dir())}'
                )
                shutil.rmtree(trial_output, True)
            report = function(trial_config, trial_output, **kwargs)
            if isinstance(save_trials, str) and save_trials == 'only-reports':
                for path in trial_output.iterdir():
                    if path.is_dir():
                        shutil.rmtree(path, True)
                    elif path.name not in ('DONE', 'report.json'):
                        path.unlink()
        else:
            with tempfile.TemporaryDirectory(suffix=f'_trial_{trial.number}') as tmp:
                report = function(trial_config, Path(tmp) / 'output', **kwargs)

        assert report is not None
        report['trial_id'] = trial.number
        report['tuning_time'] = str(timer)
        trial_reports.append(report)
        delu.cuda.free_memory()
        return report['metrics']['val']['score']

    def callback(*_, **__):
        report['best'] = trial_reports[study.best_trial.number]
        report['time'] = str(timer)
        report['n_completed_trials'] = len(trial_reports)
        lib.dump_checkpoint(
            output,
            {
                'report': report,
                'study': study,
                'trial_reports': trial_reports,
                'timer': timer,
                'random_state': delu.random.get_state(),
            },
        )
        lib.dump_summary(output, lib.summarize(report))
        lib.dump_report(output, report)
        lib.backup_output(output)
        print()
        lib.print_sep()
        print(f'Time: {timer}')
        print('The current best result:')
        print()
        pprint(lib.summarize(report['best']), width=60, sort_dicts=False)
        lib.print_sep()
        print()

    timer.run()
    # Ignore the progress bar warning.
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    # Ignore the warnings about the deprecated suggest_* methods.
    warnings.filterwarnings('ignore', category=FutureWarning)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[callback])
    lib.dump_summary(output, lib.summarize(report))
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run(main)
