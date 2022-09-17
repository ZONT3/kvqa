import torch
from torch import nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler

import kvqa.util.modeling as zm
from kvqa.modeling.oscar_modeling import CaptionBertEncoder


class KVQAModel(zm.Module):
    """
    Модель VQA. Использует кросс-модальность и классифицирует вывод.
    """

    def __init__(self, args, num_labels):
        super().__init__(args)

        bert_config = BertConfig(
            attention_probs_dropout_prob=0.1,
            finetuning_task='vqa_text',
            hidden_act='gelu',
            hidden_dropout_prob=args.hidden_dropout_prob,
            hidden_size=args.hidden_size,
            img_feature_dim=2054,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            num_attention_heads=12,
            num_hidden_layers=12,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
            torchscript=False,
            type_vocab_size=2,
            vocab_size=30522,
        )

        self.xmodal = KVQAXModal(args, bert_config)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)
        self.apply(get_init_weights(bert_config.initializer_range))

    def forward(self, text_data, visual_data):
        output = self.xmodal(text_data, visual_data)
        pooled_out = output[0]
        hidden_states = output[1:]

        pooled_out = self.dropout(pooled_out)
        logits = self.classifier(pooled_out)
        return logits, hidden_states


class KVQAXModal(zm.Module):
    """
    Модуль кросс-модальности.
    Оперирует текстовыми и визуальными данными,
    возвращая их объединение.
    """

    def __init__(self, args, bert_config):
        super().__init__(args)

        self.text_embedding = BertEmbeddings(bert_config)
        self.img_embedding = nn.Linear(bert_config.img_feature_dim, bert_config.hidden_size, bias=True)
        self.encoder = CaptionBertEncoder(bert_config)
        self.pooler = BertPooler(bert_config)
        self.img_dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        self.num_hidden_layers = bert_config.num_hidden_layers

        self.apply(get_init_weights(bert_config.initializer_range))

    def forward(self, text_data, visual_data):
        input_ids, token_type_ids, attention_mask = text_data

        # Оперируем с текстовыми данными, как было представлено в Oscar

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.num_hidden_layers

        # Соединяем текстовые и визуальные данные

        t_embedding_out = self.text_embedding(input_ids, position_ids=None, token_type_ids=token_type_ids)
        i_embedding_out = self.img_dropout(self.img_embedding(visual_data))
        embedding_out = torch.cat((t_embedding_out, i_embedding_out), 1)

        encoder_out = self.encoder(embedding_out, extended_attention_mask, head_mask, None)
        sequence_out = encoder_out[0]
        pooled_out = self.pooler(sequence_out)

        return (pooled_out,) + encoder_out[1:]


def get_init_weights(initializer_range):
    def _fnc(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    return _fnc
