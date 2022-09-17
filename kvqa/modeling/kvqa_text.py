from transformers import BertTokenizer


class Tokenizer:
    """
    Использует предобученный BertTokenizer
    и токенизирует строку под формат, который требует модель
    """

    def __init__(self, args):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
        self.args = args

    def tokenize(self, text):
        args = self.args
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > args.max_text_length - 2:
            tokens = tokens[:(args.max_text_length - 2)]

        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        # Совместимость с архитектурой оскара (лейблы с изображения), пока не реализовано
        token_type_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        pad_len = args.max_text_length - len(input_ids)
        input_ids = input_ids + ([0] * pad_len)
        attention_mask = attention_mask + ([0] * pad_len)
        token_type_ids = token_type_ids + ([0] * pad_len)

        assert all(len(x) == args.max_text_length for x in (input_ids, attention_mask, token_type_ids))
        return input_ids, token_type_ids, attention_mask
