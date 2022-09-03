import os
import string

import torch

from src.definitions import root_dir

models_path = os.path.join(root_dir, "data", "mlmodels")

para_model = {
    "model_path": os.path.join(models_path, 'para'),
    "token_path": os.path.join(models_path, 'paratokenizer'),
    "model": None,
    "tokenizer": None,
}

boolq_model = {
    "model_path": os.path.join(models_path, 'boolq'),
    "token_path": os.path.join(models_path, 'boolqtokenizer'),
    "model": None,
    "tokenizer": None,
}


def is_solution(passage, solution):
    passage = passage.translate(str.maketrans('', '', string.punctuation))
    input = para_model['tokenizer'](passage, solution, return_tensors='pt')
    logits = para_model['model'](**input).logits
    p = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    return True if p[1] > 0.6 else False


def predict(question, passage):
    input = boolq_model['tokenizer'](question, passage, return_tensors='pt')
    logits = boolq_model['model'](**input).logits
    p = torch.argmax(logits, dim=1)
    return "Yes" if p == 1 else "No"
