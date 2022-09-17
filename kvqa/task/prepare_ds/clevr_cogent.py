"""
Реализация DatasetConverter
для набора данных Clevr CoGenT v.1.0
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

from kvqa.util.dataset_converting import DatasetConverter, Question, log


class ClevrConverter(DatasetConverter):
    def __init__(self, path, split='trainA'):
        self.path = Path(path)
        self.split = split

    def get_questions(self):
        q_path = self.path / f'questions/CLEVR_{self.split}_questions.json'
        img_path = self.path / f'images/{self.split}'

        log('Loading JSON...')
        with open(q_path, 'r') as f:
            q_list: list = json.load(f)['questions']

        q_prepared = []
        for q in tqdm(q_list, desc='Preparing questions'):
            img = img_path / q['image_filename']
            question = q['question']

            if not img.is_file():
                log(f'WARN: Cannot find image for q: {question} file: {img}')
            else:
                q_prepared.append(Question(question, q['answer'] if 'answer' in q else [], img))

        return q_prepared


def main():
    if len(sys.argv) < 3:
        raise ValueError('Usage: clevr_cogent COGENT_DIR TARGET_DIR')
    cogent_path = sys.argv[1]
    target_path = Path(sys.argv[2])

    conv = ClevrConverter(cogent_path, 'trainA')
    l2a = conv.do_convert(target_path / 'trainA')

    for split in ('valA', 'valB', 'testA', 'testB'):
        conv = ClevrConverter(cogent_path, split)
        conv.do_convert(target_path / split, l2a_path=l2a)


if __name__ == '__main__':
    main()
