"""
Инструмент для конвертации набора данных под формат, принимающий KVQA.
Реализации для конкретных наборов находятся в директории kvqa/task/prepare_ds/
"""

import _pickle as pkl
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Union

from PIL import Image
from tqdm import tqdm


def log(*args, **kwargs):
    file = kwargs.pop('file', sys.stderr)
    print(*args, file=file, **kwargs)


class Question:
    def __init__(self, question: str, answer: Union[List[str], str], image_file: Path):
        self.question: str = question
        self.answer: List[str] = answer if isinstance(answer, list) else [answer]
        self.image_file: Path = image_file


class DatasetConverter:

    def get_questions(self) -> List[Question]:
        raise NotImplementedError()

    def do_convert(self, target_path, l2a_path=None, move=False):
        items = self.get_questions()
        log('Total questions:', len(items), file=sys.stdout)

        target_path = Path(target_path)
        img_dir = target_path / 'img'
        img_dir.mkdir(parents=True, exist_ok=True)
        text_dir = target_path / 'text'
        text_dir.mkdir(parents=True, exist_ok=True)

        if l2a_path:
            log('Loading provided label2ans...')
            with open(l2a_path, 'rb') as fp:
                label2ans = pkl.load(fp)
        else:
            label2ans = list({x for item in tqdm(items, desc='Building labels [1/2]') for x in item.answer})
        ans2label = {v: i for i, v in enumerate(tqdm(label2ans, desc='Building labels [2/2]'))}

        log('Total labels:', len(label2ans), file=sys.stdout)

        id2img = list({x.image_file for x in tqdm(items, desc='Building img ids [1/2]')})
        img2id = {v: i for i, v in enumerate(tqdm(id2img, desc='Building img ids [2/2]'))}

        log('Total images:', len(id2img), file=sys.stdout)

        for img, idx in tqdm(img2id.items(), desc='Copying/moving/converting images'):
            name_split = img.name.split('.')
            ext = name_split[-1].lower()
            new_path = img_dir / f'{idx:06d}.png'

            if ext not in ('png', 'jpg', 'jpeg'):
                log(f'WARN: Unknown img type: {ext}, skipping conversion...')
            elif ext != 'png':
                p_img = Image.open(img)
                p_img.save(new_path)
                if move:
                    os.unlink(img)
                continue

            if move:
                shutil.move(img, new_path)
            else:
                shutil.copy(img, new_path)

        if not l2a_path:
            log('Writing l2a...')
            l2a_path = text_dir / 'label2ans.pkl'
            with open(l2a_path, 'wb') as fp:
                pkl.dump(label2ans, fp)
            log('Writing a2l...')
            with open(text_dir / 'ans2label.pkl', 'wb') as fp:
                pkl.dump(ans2label, fp)

        questions = [{
            'q': q.question,
            'an': [ans2label[a] for a in q.answer if a in ans2label],
            'an_str': q.answer[0] if len(q.answer) > 0 else None,
            'img_id': img2id[q.image_file]
        } for q in tqdm(items, desc='Building questions.json')]

        log('Writing questions.json...')
        with open(text_dir / 'questions.json', 'w') as fp:
            json.dump(questions, fp)
        log('Done')

        return l2a_path
