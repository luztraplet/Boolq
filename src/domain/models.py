import os
import string

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, BertForSequenceClassification, \
    BertTokenizer

from src.definitions import get_root_dir


# load models
def load_models():
    models_path = os.path.join(get_root_dir(), "data", "mlmodels")

    # paraphrasing model
    para_model = {
        "model": BertForSequenceClassification.from_pretrained(os.path.join(models_path, 'para')),
        "tokenizer": BertTokenizer.from_pretrained(os.path.join(models_path, 'paratokenizer')),
    }

    # boolq model
    boolq_model = {
        "model": RobertaForSequenceClassification.from_pretrained(os.path.join(models_path, 'boolq')),
        "tokenizer": RobertaTokenizer.from_pretrained(os.path.join(models_path, 'boolqtokenizer')),
    }

    return boolq_model, para_model


# inference paraphrasing model
def is_solution(para_model, passage, solution):
    passage = passage.translate(str.maketrans('', '', string.punctuation))
    input = para_model['tokenizer'](passage, solution, return_tensors='pt')
    logits = para_model['model'](**input).logits
    p = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    return True if p[1] > 0.6 else False


# inference boolq model
def predict(boolq_model, question, passage):
    input = boolq_model['tokenizer'](question, passage, return_tensors='pt')
    logits = boolq_model['model'](**input).logits
    p = torch.argmax(logits, dim=1)
    return "Yes" if p == 1 else "No"
