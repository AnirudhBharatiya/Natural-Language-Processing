import torch
from torch.utils.data import Dataset
from question1.preprocessor import TextPreprocessor
from question1.encoder import WordPieceEncoder
from .q3_utils import load_vocab

def build_q3_datasets(corpus_path, vocab_path, context_size):
    token_to_id, id_to_token = load_vocab(vocab_path)
    unk_id = token_to_id.get("<UNK>", 0)
    bos_id = token_to_id.get("<s>", 1)

    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 90/10 Split
    split_idx = int(0.9 * len(lines))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    preproc = TextPreprocessor("", "")
    encoder = WordPieceEncoder(vocab_path)

    def process_lines(line_list):
        windows = []
        for line in line_list:
            processed = preproc.process_line(line)
            if not processed: continue
            
            # Use Encoder logic manually to get tokens + BOS/EOS
            tokens = ["<s>"]
            for word in processed.split():
                tokens.extend(encoder.get_word_tokens(word))
            tokens.append("</s>")
            
            ids = [token_to_id.get(t, unk_id) for t in tokens]
            
            # Sliding Window
            for i in range(1, len(ids)):
                target = ids[i]
                start = i - context_size
                
                if start < 0:
                    context = [bos_id] * abs(start) + ids[0:i]
                else:
                    context = ids[start:i]
                
                windows.append((context, target))
        return windows

    train_data = process_lines(train_lines)
    val_data = process_lines(val_lines)

    return train_data, val_data, id_to_token

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        ctx, tgt = self.data[idx]
        return torch.tensor(ctx, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)