import os

class WordPieceTrainer:
    """
    Trains a WordPiece vocabulary given a target size.
    """
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.corpus = {}
        self.vocab = {}
        self.token_boundary = {}

    def get_token_counts(self, curr_token_boundary):
        """Aggregates frequency counts for current tokens."""
        token_counts = {}
        
        for tokens_tuple, freq in curr_token_boundary.items():
            for token in tokens_tuple:
                token_counts[token] = token_counts.get(token, 0) + freq
                    
        return token_counts

    def pairwise_freq(self, curr_token_boundary):
        """Calculates frequency of adjacent token pairs."""
        pair_freq = {}
        for element, freq in curr_token_boundary.items():
            for i in range(len(element) - 1):
                pair = (element[i], element[i + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq
        return pair_freq

    def max_likelihood_pair(self, curr_vocab, curr_pairwise_freq):
        """Finds the pair with highest likelihood score to merge."""
        max_l = -1
        max_pair = None
        
        for pair, freq in curr_pairwise_freq.items():
            el1, el2 = pair
            denom = curr_vocab[el1] * curr_vocab[el2]
            likelihood = freq / denom
            
            if likelihood > max_l:
                max_l = likelihood
                max_pair = pair
            # Break ties lexicographically
            elif likelihood == max_l:
                if max_pair is None or pair < max_pair:
                    max_pair = pair
        return max_pair

    def new_token_boundary(self, curr_token_boundary, merge_pair, merged_token):
        """Updates the corpus splits after a merge."""
        el1, el2 = merge_pair
        new_boundary = {}
        
        for element, freq in curr_token_boundary.items():
            new_element = []
            i = 0
            while i < len(element):
                if (i < len(element) - 1) and (element[i] == el1) and (element[i+1] == el2):
                    new_element.append(merged_token)
                    i += 2
                else:
                    new_element.append(element[i])
                    i += 1
            
            new_element = tuple(new_element)
            new_boundary[new_element] = new_boundary.get(new_element, 0) + freq
                
        return new_boundary

    def train(self, input_file_path):        
        # 1. Load Corpus and Initialize Vocab (Characters)
        with open(input_file_path, "r", encoding="utf-8") as fin:
            for line in fin:
                words = line.strip().split()
                if not words: continue
                
                for word in words:
                    self.corpus[word] = self.corpus.get(word, 0) + 1
                    
                    # Initial character vocab
                    for j, char in enumerate(word):
                        token = char if j == 0 else '##' + char
                        self.vocab[token] = self.vocab.get(token, 0) + 1

        # 2. Create Initial Splits
        for word, freq in self.corpus.items():
            tokens = [word[0]] + ['##' + c for c in word[1:]]
            self.token_boundary[tuple(tokens)] = freq

        working_vocab = dict(self.vocab)
        working_boundary = dict(self.token_boundary)

        # 3. Training Loop
        # Reserve 3 spots for <UNK>, <s>, </s>
        while len(working_vocab) < (self.vocab_size - 3):
            
            curr_token_counts = self.get_token_counts(working_boundary)
            working_vocab.update(curr_token_counts)
            
            pairwise_stats = self.pairwise_freq(working_boundary)
            if not pairwise_stats:
                break
                
            best_pair = self.max_likelihood_pair(working_vocab, pairwise_stats)
            if not best_pair:
                break
                
            el1, el2 = best_pair
            # Merge logic: if second part starts with ##, remove it
            merged_token = el1 + el2[2:] if el2.startswith('##') else el1 + el2
                 
            working_vocab[merged_token] = pairwise_stats[best_pair]
            working_boundary = self.new_token_boundary(working_boundary, best_pair, merged_token)
            
        self.vocab = working_vocab

    def save_vocab(self, output_file_path):
        dirname = os.path.dirname(output_file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        special_tokens = ["<UNK>", "<s>", "</s>"]
        sorted_tokens = sorted(list(self.vocab.keys()))
        final_vocab = special_tokens + sorted_tokens
        
        with open(output_file_path, "w", encoding="utf-8") as fout:
            for token in final_vocab:
                fout.write(token + "\n")