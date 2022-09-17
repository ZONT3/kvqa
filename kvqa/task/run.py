"""
Файл запуска проекта
"""
from pathlib import Path

import kvqa.task.arg as arg
from kvqa.vqa.kvqa_trainval import VQA

if __name__ == '__main__':
    args = arg.parse_args()
    vqa = VQA(args)

    if args.load_checkpoint:
        ckp_path = Path(args.load_checkpoint)
        if not ckp_path.is_file():
            raise FileNotFoundError('Checkpoint must be an existing file and not a directory')
        vqa.load_checkpoint(ckp_path)
    else:
        ckp_path = None

    if args.train:
        vqa.train()

    elif args.test:
        if not ckp_path:
            raise AssertionError('--load-checkpoint must be specified on test')
        vqa.test()

    else:
        raise AssertionError('--train or --test must be specified')
