import torch
from .q3_utils import detokenize
from question1.preprocessor import TextPreprocessor
from question1.encoder import WordPieceEncoder


def predict_next_token(model, context_ids):
    context = torch.tensor(context_ids).unsqueeze(0)
    logits = model(context)
    return torch.argmax(logits, dim=1).item()


def get_generation_context(ids, context_size, bos_id):
    if len(ids) < context_size:
        pad_len = context_size - len(ids)
        return [bos_id] * pad_len + ids
    else:
        return ids[-context_size:]


def generate_text(model, sentence, token_to_id, id_to_token, context_size, k):
    preproc = TextPreprocessor("", "")
    sentence = preproc.process_line(sentence)
    words = sentence.split()

    encoder = WordPieceEncoder.__new__(WordPieceEncoder)
    encoder.token_to_id = token_to_id

    tokens = ["<s>"]
    for w in words:
        tokens.extend(encoder.get_word_tokens(w))

    ids = [token_to_id.get(t, token_to_id["<UNK>"]) for t in tokens]

    for i in range(k):
        context = get_generation_context(ids, context_size, token_to_id["<s>"])
        next_id = predict_next_token(model, context)
        ids.append(next_id)
        if id_to_token[next_id] == "</s>":
            break

    tokens = [id_to_token[i] for i in ids]
    return detokenize(tokens)


def generate_from_file(input_path, output_path, model, token_to_id, id_to_token, context_size, k):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            generated = generate_text(
                model, line,
                token_to_id, id_to_token,
                context_size, k
            )
            fout.write(generated + "\n")