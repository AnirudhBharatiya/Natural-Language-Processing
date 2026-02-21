import random
import numpy as np
import torch

def set_seed(seed: int = 123):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_vocab(vocab_file_path):
    vocab = []
    with open(vocab_file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                vocab.append(line)

    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for i, t in enumerate(vocab)}
    return token_to_id, id_to_token

def detokenize(tokens):
    output = []
    for tok in tokens:
        if tok in {"<s>", "</s>"}:
            continue
        if tok.startswith("##"):
            if output:
                output[-1] += tok[2:]
        else:
            output.append(tok)
    return " ".join(output)