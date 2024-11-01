# NOTE: this file must not import anything from lib.

from pathlib import Path

_PROJECT_DIR = Path.cwd()

assert _PROJECT_DIR.joinpath(
    '.git'
).exists(), 'The script must be run from the root of the repository'


def get_project_dir() -> Path:
    return _PROJECT_DIR


def get_cache_dir() -> Path:
    path = get_project_dir() / 'cache'
    path.mkdir(exist_ok=True)
    return path


def get_data_dir() -> Path:
    return get_project_dir() / 'data'


def get_exp_dir() -> Path:
    return get_project_dir() / 'exp'
