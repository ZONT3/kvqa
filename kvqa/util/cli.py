from typing import Optional

_prompt_yes = ('y', 't', 'д')
_prompt_no = ('n', 'f', 'н')


def prompt(message, default_yes=True):
    if default_yes:
        message = f'{message} ([y]/n): '
    else:
        message = f'{message} (y/n): '
    ans: Optional[str] = None

    possible_ans = (_prompt_yes + _prompt_no)
    while ans is None or ans[0] not in possible_ans and (not default_yes or len(ans) > 0):
        ans = input(message)

    return len(ans) == 0 and default_yes or ans[0] in _prompt_yes


def log(*args):
    print(*args, flush=True)


def log_rich(s, **kwargs):
    highlight = '=' * len(s)
    print(highlight)
    print(s, **kwargs)
    print(highlight, flush=True)
