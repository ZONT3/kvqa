import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.utils.data as td
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from transformers import AdamW, get_linear_schedule_with_warmup

from kvqa.modeling.kvqa_model import KVQAModel
from kvqa.util.cli import prompt, log, log_rich
from kvqa.vqa.kvqa_dataset import Dataset, use_dataset, pack_batch


def prepare_batch(batch, device):
    batch = tuple((x.to(device) if x is not None else None) for x in batch[:-1]) + (batch[-1],)
    return pack_batch(batch)


class VQA:
    """
    Класс для работы с моделью KVQAModel и датасетом.
    Здесь определены методы обучения и обработки результатов
    """

    def __init__(self, args):
        self.args = args
        self.dataset = Dataset(args)
        self.model = KVQAModel(args, self.dataset.num_labels)
        self.model.to(self.model.device)
        self.ckp_dir = None
        self.ckp_index = 1

        if args.val_dataset:
            self.val_dataset = Dataset(args, args.val_dataset, dataset_name='val')
        else:
            self.val_dataset = None

    def prepare_checkpoint_dir(self):
        args = self.args
        ckp_dir = Path(args.checkpoints)
        if ckp_dir.is_dir() and len(os.listdir(ckp_dir)) > 0:
            if not args.yes and not prompt('Checkpoint dir exists and not empty. Continue?'):
                raise InterruptedError()
        elif ckp_dir.is_file():
            raise FileExistsError('Checkpoint dir is a file')
        else:
            ckp_dir.mkdir(parents=True, exist_ok=True)
        self.ckp_dir = ckp_dir

        max_index = -1
        for f in os.listdir(self.ckp_dir):
            m = re.match(r'checkpoint-(\d+)-.+', str(f))
            if not m: continue
            max_index = max(int(m.group(1)), max_index)
        if max_index > 0:
            self.ckp_index = max_index + 1
        log('Current checkpoint index:', self.ckp_index)

        for x in (self.dataset.l2a_path, self.dataset.a2l_path):
            dst = ckp_dir / x.name
            if dst.is_file():
                os.unlink(dst)
            shutil.copy(x, dst)

    def save_checkpoint(self, score, epoch):
        path = self.ckp_dir / f'checkpoint-{self.ckp_index:02d}-{epoch:02d}-{score * 100:.03f}.pt'
        if path.is_file():
            log('WARN: Overwriting existing checkpoint')
            os.unlink(path)
        torch.save(self.model.state_dict(), path)
        log('Saved checkpoint', path)

    def save_history(self, history: List[Dict]):
        path = self.ckp_dir / f'history-{self.ckp_index:02d}.json'
        if path.is_file():
            os.unlink(path)
        with open(path, 'w') as fd:
            json.dump(history, fd)
        log('Saved history', path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self):
        log_rich('Do train')
        self.prepare_checkpoint_dir()

        args = self.args
        model = self.model
        log(f'Epochs: {args.epochs}, '
            f'batch size: {args.batch_size}, '
            f'Do val: {"yes" if self.val_dataset is not None else "no"}')

        loader = td.DataLoader(self.dataset, num_workers=4, sampler=td.RandomSampler(self.dataset),
                               batch_size=self.args.batch_size)
        t_total = len(loader) * args.epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.05
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-05, eps=1e-08)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

        model.zero_grad()
        set_seed(args.seed, args.n_gpu)

        best_score = 0
        train_loss = 0.0
        train_score = 0
        total_norm = 0
        count_norm = 0
        tp, fp, fn = 0, 0, 0

        history = []

        for batch, batch_idx, epoch in use_dataset(loader, args.print_delay, args.epochs):
            if batch_idx < 0 and epoch > 0:
                # noinspection PyTypeChecker
                train_score /= len(loader.dataset)
                log(f'Train loss: {train_loss}, score: {train_score * 100 :.03f}')

                entry = {
                    'loss': train_loss,
                    'accuracy': train_score,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                }

                val_result = self.val(f'epoch {epoch :d}')
                if not val_result:
                    val_score = train_score
                else:
                    val_loss, val_score = val_result
                    entry['val_loss'] = val_loss
                    entry['val_accuracy'] = val_score

                if val_score > best_score:
                    best_score = val_score
                    log('Saving checkpoint...')
                    self.save_checkpoint(val_score, epoch)

                history.append(entry)
                self.save_history(history)

                log(f'*** Epoch {epoch + 1}')

                train_loss = 0.0
                train_score = 0
                total_norm = 0
                count_norm = 0
                tp, fp, fn = 0, 0, 0
                continue

            elif batch_idx < 0 or not batch:
                log('*** Epoch 1')
                continue

            model.train()
            text_data, visual_data, labels, _ = prepare_batch(batch, model.device)

            logits, _ = model(text_data, visual_data)
            loss = bce_loss(logits, labels)

            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            count_norm += 1

            train_loss += loss.mean().item()
            batch_score = torch.sum(compute_score_with_logits(logits, labels), 1)
            train_score += batch_score.sum().item()

            batch_tp, batch_fp, batch_fn = calculate_tp_fp_fn(logits, labels)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn

            optimizer.step()
            scheduler.step()
            model.zero_grad()

    def val(self, info=None):
        if not self.val_dataset:
            return

        args = self.args
        model = self.model

        loader = td.DataLoader(self.val_dataset, num_workers=4, sampler=td.SequentialSampler(self.val_dataset),
                               batch_size=args.batch_size)

        val_loss = 0.0
        score = 0

        if info is None:
            log('** Evaluation')
        else:
            log(f'** Evaluation ({info})')

        for batch in loader:
            model.eval()
            text_data, visual_data, labels, _ = prepare_batch(batch, model.device)

            with torch.no_grad():
                logits, _ = model(text_data, visual_data)
                loss = bce_loss(logits, labels)
                val_loss += loss.mean().item()
                batch_score = torch.sum(compute_score_with_logits(logits, labels), 1)
                score += batch_score.sum().item()

        # noinspection PyTypeChecker
        score /= len(loader.dataset)
        log(f'Val loss: {val_loss}, Score: {score * 100 :.03f}')
        return val_loss, score

    def test(self):
        log_rich('Do test')
        args = self.args
        model = self.model

        path = Path(f'{args.test_results}.json')
        if path.is_file():
            if args.yes or prompt('Test results file already exists. Continue?'):
                path.unlink()
            else:
                raise InterruptedError()
        elif path.exists():
            raise FileExistsError()
        else:
            try:
                with open(path, 'w') as f:
                    json.dump({'test': 'ok'}, f)
                path.unlink()
            except Exception as e:
                raise RuntimeError(f'Cannot create test results file: {repr(e)}')

        loader = td.DataLoader(self.dataset, num_workers=4, sampler=td.SequentialSampler(self.dataset),
                               batch_size=args.batch_size)

        score = 0
        results = []

        for batch, b_idx in use_dataset(loader, args.print_delay):
            model.eval()
            text_data, visual_data, labels, question_ids = prepare_batch(batch, model.device)

            with torch.no_grad():
                logits, _ = model(text_data, visual_data)

                batch_score = torch.sum(compute_score_with_logits(logits, labels), 1)
                score += batch_score.sum().item()

                val, idx = logits.max(1)
                for i in range(idx.size(0)):
                    label = idx[i].item()
                    q_id = question_ids[i].item()
                    results.append({
                        'Question ID': q_id,
                        'Question': self.dataset.id_to_question(q_id),
                        'Image': str(self.dataset.id_to_question_image_file(q_id).name),
                        'Answer label': label,
                        'Answer': self.dataset.label_to_answer(label)
                    })

        if score > 0:
            # noinspection PyTypeChecker
            score /= len(loader.dataset)
            log(f'Score: {score * 100 :.03f}')

        log(f'Saving test results to {path}')
        with open(path, 'w') as f:
            json.dump(results, f)

        return score


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def bce_loss(logits, labels):
    assert logits.dim() == 2
    loss = binary_cross_entropy_with_logits(logits, labels, reduction='mean')
    loss *= labels.size(1)
    return loss


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def calculate_tp_fp_fn(logits, labels):
    logits = torch.max(logits, 1)[1].data
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    tp = torch.sum(one_hots * labels, dim=0).item()
    fp = torch.sum(one_hots * ~labels, dim=0).item()
    fn = torch.sum(~one_hots * labels, dim=0).item()
    return int(tp), int(fp), int(fn)
