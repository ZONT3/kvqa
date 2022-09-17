"""
Вспомогательный модуль с инструментами
"""
import time
from typing import Dict, Union, Tuple


_log_timer = -1


def log_rich(s, **kwargs):
    global _log_timer
    if _log_timer >= 0:
        log_time()
    highlight = '=' * len(s)
    print(highlight)
    print(s, **kwargs)
    print(highlight, flush=True)


def log(*args, start_timer=False, **kwargs):
    global _log_timer
    if _log_timer >= 0:
        log_time()
    if start_timer:
        if 'end' not in kwargs:
            kwargs['end'] = '... '
        _log_timer = time.time()
    kwargs['flush'] = True
    print(*args, **kwargs)


def log_time():
    global _log_timer
    if _log_timer < 0:
        print('log_time: timer not set!')
    else:
        t = time.time() - _log_timer
        _log_timer = -1
        print(f'{int(t // 60)}m {t % 60:.03f}s')


def validate_dict(data: Dict, expected_structure: Dict[str, Union[type, Tuple[type]]]):
    for k, t in expected_structure.items():
        if k not in data:
            raise KeyError(f'{k} key not found')
        if not isinstance(data[k], t):
            raise ValueError(f'Value of {k} is not {str(t)}')


def yes_no_interact(prompt):
    print(prompt)
    inpt = '-'
    while True:
        inpt = input('[y]/n: ')
        if inpt == '' or inpt in ('y', 'yes'):
            return True
        elif inpt in ('n', 'no'):
            return False
