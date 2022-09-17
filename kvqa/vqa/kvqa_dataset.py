import _pickle as pickle
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.utils.data as td
from tqdm import tqdm

from kvqa.modeling.kvqa_text import Tokenizer
from kvqa.modeling.kvqa_vision import ObjectFeatureExtractor
from kvqa.util import log_rich, log, validate_dict, yes_no_interact, log_time

_QUESTION_KEYS = {'q': str, 'an': list, 'img_id': int}


class Dataset(td.Dataset):
    """
    Класс-загрузчик датасета
    """

    l2a: List[str]
    questions: Dict[int, Dict]

    def __init__(self, args, dataset_dir=None, dataset_name=None):
        self.args = args
        self.dataset_dir = Path(dataset_dir if dataset_dir else args.dataset)
        self.img_dir = self.dataset_dir / 'img'
        self.feats_dir = self.dataset_dir / 'feats'
        self.text_dir = self.dataset_dir / 'text'
        self.questions_path = self.text_dir / 'questions.json'
        self.features = None
        self.feats_infos = None
        self.feats_prepared = False
        self.found_img = set()

        if args.load_checkpoint:
            cp_dir = Path(args.load_checkpoint)
            assert cp_dir.is_file()
            cp_dir = cp_dir.parent

            self.l2a_path = cp_dir / 'label2ans.pkl'
            self.a2l_path = cp_dir / 'ans2label.pkl'

            if not self.l2a_path.is_file():
                raise DatasetSetupError('Checkpoint dir must contain label2ans.pkl')
        else:
            self.l2a_path = self.text_dir / 'label2ans.pkl'
            self.a2l_path = self.text_dir / 'ans2label.pkl'

            if not self.l2a_path.is_file():
                raise DatasetSetupError('Text subdir must contain label2ans.pkl')

        if not self.dataset_dir.is_dir():
            raise DatasetSetupError('Dataset is not a directory')
        if not self.text_dir.is_dir():
            raise DatasetSetupError('No text subdir in target dataset dir')
        if not self.feats_dir.is_dir() and not self.img_dir.is_dir():
            raise DatasetSetupError('Both feats and img subdirs are absent. One of these or both must exist')

        if dataset_name:
            log_rich(f'Preparing {dataset_name} dataset')
        else:
            log_rich('Preparing dataset')

        log('Loading label2ans', start_timer=True)
        with open(self.l2a_path, 'rb') as f:
            l2a = pickle.load(f)
        log('Loading questions', start_timer=True)
        with open(self.questions_path) as f:
            questions = json.load(f)

        self.questions = self._init_questions(questions, len(l2a))
        self.l2a, self.a2l = self._init_labels(l2a)
        self.num_labels = len(self.l2a)

        if not self.feats_prepared or self.args.update_features:
            self._prepare_feats()
        else:
            self._load_features()

        log('Prepare tensors', start_timer=True)
        self.tokenizer = Tokenizer(args)
        self.tensors = [self._prepare_tensor(k, v) for k, v in self.questions.items()]
        log('Done')

    def _prepare_tensor(self, idx, q):
        visual_data = self.features[q['img_id']]

        max_image_length = self.args.max_image_length
        if visual_data.shape[0] > max_image_length:
            visual_data = visual_data[:max_image_length, ]

        input_ids, token_type_ids, attention_mask = self.tokenizer.tokenize(q['q'])

        pad_matrix = torch.zeros((max_image_length - visual_data.shape[0], visual_data.shape[1]))
        attention_mask = attention_mask + ([1] * visual_data.shape[0]) + ([0] * pad_matrix.shape[0])
        visual_data = torch.cat((visual_data, pad_matrix), 0)

        label = torch.tensor(np.zeros((self.num_labels,)), dtype=torch.float)
        if 'an' in q and len(q['an']) > 0:
            for an in q['an']:
                label[an] = 1.

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(token_type_ids, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                visual_data,
                label,
                torch.tensor([idx], dtype=torch.long))

    def _init_questions(self, questions, total_labels):
        log('Init questions')

        if self.args.tiny:
            questions = questions[:30]

        valid_questions = {}
        invalid_questions = []
        found_img = set()
        found_feat = set()
        not_found_img = set()

        for i, q in enumerate(tqdm(questions)):
            try:
                validate_dict(q, _QUESTION_KEYS)
            except Exception as e:
                raise DatasetSetupError(f"Questions validation failed on #{i} ({q['q'] if 'q' in q else 'unknown'}): "
                                        f"{repr(e)}")
            for a in q['an']:
                if a >= total_labels:
                    raise DatasetSetupError(f'Unknown answer id {a} for question {q["q"]}')

            q['q_id'] = i

            img_id = q['img_id']
            if self._id_to_feat_file(img_id).is_file():
                valid_questions[i] = q
                found_img.add(img_id)
                found_feat.add(img_id)
            elif self._id_to_img_file(img_id).is_file():
                valid_questions[i] = q
                found_img.add(img_id)
            else:
                invalid_questions.append(q)
                not_found_img.add(img_id)

        if len(valid_questions) == 0:
            raise DatasetSetupError(
                f'No valid questions in dataset. Not found {len(not_found_img)} image(s) of {len(questions)} questions')

        if len(not_found_img) > 0:
            if not self.args.yes:
                if len(not_found_img) > 10:
                    prompt = f'Many of images ({len(not_found_img)}) referred from questions has not found. ' \
                             f'Would you like to continue?'
                else:
                    prompt = f'Not found following images referred from questions: {", ".join(not_found_img)}.\n' \
                             f'Would you like to continue?'
                if not yes_no_interact(prompt):
                    raise DatasetSetupError('Cancelled setup due to absence of some images')
            log(f'Dropping {len(invalid_questions)} questions')

        self.found_img = found_img
        self.feats_prepared = len(found_img) == len(found_feat)

        return valid_questions

    def _init_labels(self, l2a):
        log('Init labels', start_timer=True)
        a2l = None
        total_labels = len(l2a)
        if self.a2l_path.is_file():
            with open(self.a2l_path, 'rb') as f:
                a2l = pickle.load(f)
            max_id = max(*a2l.values())
            expected_max_id = total_labels - 1
            if max_id != expected_max_id:
                log(f'Found incorrect ans2label. (Max label id {max_id} != {expected_max_id})')
                a2l = None
                os.remove(self.a2l_path)
        if a2l is None:
            log('Generating new ans2label', start_timer=True)
            a2l = {}
            for i, s in enumerate(l2a):
                a2l[s] = i
            with open(self.a2l_path, 'wb') as f:
                pickle.dump(a2l, f)
            log_time()
        return l2a, a2l

    def _load_features(self):
        log('Load features')
        tensors = []
        for idx in tqdm(self.found_img, desc='Reading features'):
            tensors.append(torch.load(self._id_to_feat_file(idx)))
        self.features = {k: v for k, v in zip(self.found_img, tensors)}

    def _prepare_feats(self):
        """
        Конвертировать сырые изображения в тензоры для работы с моделью.
        После возврата из этого метода, self.feats_prepared должно быть True,
        иначе выкинуто исключение
        """
        log_rich('Preparing features')
        self.feats_dir.mkdir(parents=True, exist_ok=True)

        extractor = ObjectFeatureExtractor(self.args.vision_model_path, self.args.batch_size, self.args.device)
        features, infos = extractor.extract_features_save(
            list(map(lambda x: self._id_to_img_file(x), self.found_img)),
            self.feats_dir)
        extractor.free()
        del extractor

        self.features = {k: v for k, v in zip(self.found_img, features)}
        self.feats_infos = {k: v for k, v in zip(self.found_img, infos)}
        self.feats_prepared = True

    def _id_to_feat_file(self, idx):
        return self.feats_dir / f'{idx:06d}.png.pt'

    def _id_to_img_file(self, idx):
        return self.img_dir / f'{idx:06d}.png'

    def label_to_answer(self, label):
        return self.l2a[label]

    def id_to_question(self, idx):
        return self.questions[idx]['q']

    def id_to_question_image_file(self, idx):
        return self._id_to_img_file(self.questions[idx]['img_id'])

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def pack_batch(batch):
    """
    @return: text_data, visual_data, labels, question_ids
    """
    text_data = batch[: 3]
    visual_data, labels, question_ids = batch[3:]
    return text_data, visual_data, labels, question_ids


def use_dataset(loader: td.DataLoader, print_delay=60, epochs=1):
    """
    Обертка датасета в :class:`torch.utils.data.DataLoader` и его использование указанное количество эпох.
    Также выводит с указанным интервалом информацию о текущем прогрессе.

    :param loader: Загрузчик датасета
    :param print_delay: Задержка между выводами
    :param epochs: Кол-во эпох
    :return: если epochs > 1: tuple(Optional[any], int, int):
        - batch (None перед каждой эпохой),
        - порядковый номер batch (zero-based, -1 перед каждой эпохой)
        - эпоху (zero-based)
        если epochs == 1: tuple(any, int)
        - batch
        - порядковый номер batch
    """
    num_batches = len(loader)
    total_batches = num_batches * epochs

    for epoch in range(epochs):
        train = epochs > 1

        if train:
            yield None, -1, epoch

        t_prev_batch = time.time()
        t_next_print = 0
        last_batch = -1

        for i, batch in enumerate(loader):
            if i == 0:
                print(f'Preparations took {time.time() - t_prev_batch :.03f} s.')

            if t_next_print <= time.time() and last_batch != i:
                if train:
                    print(f'Epoch {epoch + 1}, Batch {i + 1}/{num_batches}', end='')
                else:
                    print(f'Batch {i + 1}/{num_batches}', end='')

                if i > 0:
                    print(end='; ')

                    t_delta = time.time() - t_prev_batch
                    b_delta = (i - last_batch)
                    batch_per_sec = b_delta / t_delta

                    if b_delta < t_delta:
                        print(f'{t_delta / b_delta:.03f} s/btch', end=' ')
                    else:
                        print(f'{batch_per_sec:.03f} btch/s', end=' ')

                    eta = round(total_batches / batch_per_sec)
                    eta_m = eta // 60
                    print(f'ETA: {eta_m // 60 :d}h{eta_m % 60 :d}m{eta % 60 :d}s', end=' ')
                    eta = round(num_batches / batch_per_sec)
                    eta_m = eta // 60
                    print(f'Epoch ETA: {eta_m // 60 :d}h{eta_m % 60 :d}m{eta % 60 :d}s', end='')

                print(flush=True)

                t_prev_batch = time.time()
                t_next_print = time.time() + print_delay
                last_batch = i

            if train:
                yield batch, i, epoch
            else:
                yield batch, i


class DatasetSetupError(Exception):
    pass
