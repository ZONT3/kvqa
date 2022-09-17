"""
Парсинг аргументов
"""

from argparse import ArgumentParser

import torch

p = ArgumentParser()

p.add_argument('--train', action='store_true', help="Do train")
p.add_argument('--test', action='store_true', help="Do test")

p.add_argument('--tiny', action='store_true', help="Tiny dataset size for experiments")
p.add_argument('--update-features', action='store_true', help="Extract and save features even if it was done before")
p.add_argument('--yes', '-y', action='store_true', help="Yes to all prompts")

p.add_argument('--dataset', type=str, required=True, help='Dataset directory. Must contain \'text\' subdirectory and '
                                                          'one of (or both) \'img\', \'feats\' subdirectories')
p.add_argument('--val-dataset', type=str, default=None,
               help='Validation dataset. If absent, validation will not be performed')
p.add_argument('--checkpoints', type=str, default='checkpoints', help='Dir to save checkpoints')
p.add_argument('--test-results', type=str, default='test-result', help='Filename to save .json test results')

p.add_argument('--vision-model-path', type=str, default='vinvl_vision', help='Path to download/read VinVL '
                                                                             'vision model weights and config')
p.add_argument('--load-checkpoint', type=str, default=None, help='Path to checkpoint file to load')

p.add_argument('--device', type=str, default=None, help='Select device explicitly. May be \'cuda\' or \'cpu\'. '
                                                        'Auto-detect by default')
p.add_argument('--batch-size', type=int, default=32, help='Batch size')
p.add_argument('--epochs', type=int, default=60, help='Epochs to train')
p.add_argument('--seed', type=int, default=1337, help='Seed for train')

p.add_argument('--hidden-dropout-prob', type=float, default=0.3, help='Hidden dropout prob')
p.add_argument('--hidden-size', type=int, default=768, help='Hidden layer size')
p.add_argument('--max-text-length', type=int, default=128, help='Max tokenized text sequence length')
p.add_argument('--max-image-length', type=int, default=50, help='Max image features sequence length')

p.add_argument('--print-delay', type=int, default=60, help='Train/val progress print delay')


def parse_args(src=None):
    args = p.parse_args(src)
    args.n_gpu = torch.cuda.device_count()
    return args
