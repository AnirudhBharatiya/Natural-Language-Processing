import collections
import pickle
from tqdm import tqdm

class FastTextNGramGenerator:
    def __init__(self, min_n=3, max_n=5):
        self.min_n = min_n
        self.max_n = max_n
        self.ngram_to_idx = {}
        self.idx_to_ngram = []
        self.word_to_idx = {}
        self.idx_to_word = []
        self.word_counts = {}
        
    def get_ngrams(self, word):
        padded = f"<{word}>"
        ngrams = []
        
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(padded) - n + 1):
                ngrams.append(padded[i:i+n])
        
        ngrams.append(padded)
        return ngrams

    def build_vocabulary(self, corpus_path):
        print("Building vocabulary from " + corpus_path)
        
        self.word_counts = collections.Counter()
        total_words = 0
        
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.strip().split()
                self.word_counts.update(words)
                total_words += len(words)
        
        print("Total words: " + format(total_words, ",") + ", Unique words: " + format(len(self.word_counts), ","))
        
        words = self.word_counts.keys()
        
        unique_ngrams = set()
        for word in tqdm(words, desc="Collecting n-grams"):
            unique_ngrams.update(self.get_ngrams(word))
        
        print("Total unique n-grams: " + format(len(unique_ngrams), ","))
        
        # Create mappings
        sorted_ngrams = sorted(unique_ngrams)
        sorted_words = sorted(words)
        
        for i, ng in enumerate(sorted_ngrams):
            self.ngram_to_idx[ng] = i
        self.idx_to_ngram = sorted_ngrams
        for i, w in enumerate(sorted_words):
            self.word_to_idx[w] = i
        self.idx_to_word = sorted_words
        
        print("Vocabulary built: " + format(len(self.word_to_idx), ",") + " words, " + format(len(self.ngram_to_idx), ",") + " n-grams")

    def get_ids_for_word(self, word):
        ngrams = self.get_ngrams(word)
        ids = []
        for ng in ngrams:
            if ng in self.ngram_to_idx:
                ids.append(self.ngram_to_idx[ng])
        return sorted(ids)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)