import argparse
import shutil
from pathlib import Path

import numpy as np


def prepare_tabred(
    tabred_src: Path, tabred_dst: Path, print_only: bool = False, exist_ok: bool = False
) -> None:
    assert tabred_src.exists()
    tabred_dst.mkdir(exist_ok=True)

    for src in tabred_src.iterdir():
        print('>>>', src.name)

        dst = tabred_dst / src.name
        dst.mkdir(exist_ok=exist_ok)
        shutil.copyfile(src / 'info.json', dst / 'info.json')

        idx = (
            {
                part: np.load(src / f'split-default/{part}_idx.npy')
                for part in ['train', 'val', 'test']
            }
            if not print_only
            else {}
        )

        for path in [*list(src.glob('X_*.npy')), src / 'Y.npy']:
            if path.name != 'X_meta.npy':
                print(path.name)
                if print_only:
                    continue
                x = np.load(path)
                for part in idx:
                    np.save(dst / f'{path.stem}_{part}.npy', x[idx[part]])
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=Path, help='Source folder of TabReD datasets.')
    parser.add_argument('dst', type=Path, help='Where to save TabReD datasets.')
    parser.add_argument(
        '--print_only',
        action='store_true',
        help='Only print files from source, without loading and saving.',
    )
    parser.add_argument(
        '--force', action='store_true', help='Overwrite existing dst folders.'
    )

    args = parser.parse_args()
    try:
        prepare_tabred(
            args.src, args.dst, print_only=args.print_only, exist_ok=args.force
        )
    except FileExistsError as e:
        raise FileExistsError(
            f'File exists: {e.filename}. Use --force if you want to overwrite.'
        )


if __name__ == '__main__':
    main()
