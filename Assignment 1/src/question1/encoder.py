import os

class WordPieceEncoder:
    """
    Encodes text using a learned WordPiece vocabulary.
    """
    def __init__(self, vocab_file_path):
        self.token_to_id = {}
        self.id_to_token = {}
        self.load_vocab(vocab_file_path)

    def load_vocab(self, vocab_file_path):
        with open(vocab_file_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                token = line.strip()
                if token:
                    self.token_to_id[token] = i
                    self.id_to_token[i] = token

    def get_word_tokens(self, word):
        """Greedy Longest-Match-First Strategy."""
        word_tokens = []
        start = 0
        word_len = len(word)
        
        while start < word_len:
            end = word_len
            match_token = None
            
            while end > start:
                sub = word[start:end]
                if start > 0:
                    sub = "##" + sub
                
                if sub in self.token_to_id:
                    match_token = sub
                    break
                end -= 1
            
            if match_token:
                word_tokens.append(match_token)
                start = end
            else:
                return ["<UNK>"]
                
        return word_tokens

    def encode_file(self, preprocessed_input, output_tokens_path, output_ids_path):
        # Create directories if needed
        for path in [output_tokens_path, output_ids_path]:
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
        with open(preprocessed_input, "r", encoding="utf-8") as fin, \
             open(output_tokens_path, "w", encoding="utf-8") as f_tok, \
             open(output_ids_path, "w", encoding="utf-8") as f_ids:
            
            for line in fin:
                line = line.strip()
                if not line:
                    f_tok.write("\n"); f_ids.write("\n")
                    continue
                
                # Add BOS/EOS
                line_tokens = ["<s>"]
                for word in line.split():
                    line_tokens.extend(self.get_word_tokens(word))
                line_tokens.append("</s>")
                
                # Convert to IDs
                line_ids = [self.token_to_id.get(t, self.token_to_id.get("<UNK>", 0)) for t in line_tokens]
                
                f_tok.write(" ".join(line_tokens) + "\n")
                f_ids.write(" ".join(map(str, line_ids)) + "\n")