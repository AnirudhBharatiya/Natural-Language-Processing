import os

class WordPieceDecoder:
    """
    Decodes ID sequences back to text.
    """
    def __init__(self, vocab_file_path):
        self.id_to_token = {}
        self.load_vocab(vocab_file_path)

    def load_vocab(self, vocab_file_path):
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                if token:
                    self.id_to_token[i] = token

    def decode_file(self, input_ids_path, output_text_path): 
        if os.path.dirname(output_text_path):
            os.makedirs(os.path.dirname(output_text_path), exist_ok=True)
            
        with open(input_ids_path, "r", encoding="utf-8") as fin, \
             open(output_text_path, "w", encoding="utf-8") as fout:
            
            for line in fin:
                if not line.strip():
                    fout.write("\n")
                    continue
                
                token_ids = [int(s) for s in line.split()]
                tokens = [self.id_to_token.get(tid, "<UNK>") for tid in token_ids]
                
                decoded_words = []
                for token in tokens:
                    if token in ["<s>", "</s>"]:
                        continue
                    
                    if token.startswith("##"):
                        if decoded_words:
                            decoded_words[-1] += token[2:]
                        else:
                            # Edge case: ## at start of sentence
                            decoded_words.append(token[2:])
                    else:
                        decoded_words.append(token)
                
                fout.write(" ".join(decoded_words) + "\n")