import os
import math
import random
import pickle
import numpy as np
from tqdm import tqdm

path_to_output_dir = "../data/q2"
if not os.path.exists(path_to_output_dir):
    os.makedirs(path_to_output_dir, exist_ok=True)

class FastTextTrainer:
    def __init__(self, 
                 vector_size=100,
                 learning_rate=0.01,
                 window_size=5,
                 neg_samples=5,
                 sample_threshold=1e-5,
                 table_size=1000000):
        
        self.dim = vector_size
        self.lr = learning_rate
        self.window = window_size
        self.neg_k = neg_samples
        self.sample_threshold = sample_threshold
        self.table_size = table_size
        
        self.W_in = None  # n-gram vectors
        self.W_out = None  # word vectors
        self.neg_sample_table = None
        self.neg_probabilities = None
        self.loss_history = []
        
    def init_model(self, vocab):
        num_ngrams = len(vocab.ngram_to_idx)
        num_words = len(vocab.word_to_idx)
        
        print("Initializing model: " + format(num_ngrams, ",") + " n-grams, " + format(num_words, ",") + " words")
        
        self.W_in = (np.random.rand(num_ngrams, self.dim).astype(np.float32) - 0.5) / self.dim
        self.W_out = np.zeros((num_words, self.dim), dtype=np.float32)
        
        self._build_neg_probabilities(vocab)
        self._build_neg_sample_table()

    def _build_neg_probabilities(self, vocab):
        word_indices = np.arange(len(vocab.idx_to_word), dtype=np.int32)
        frequencies = np.zeros(len(word_indices), dtype=np.float64)
        for idx, word_idx in enumerate(word_indices):
            frequencies[idx] = vocab.word_counts[vocab.idx_to_word[word_idx]]

        pow_freq = np.power(frequencies, 0.75)
        total_pow = np.sum(pow_freq)

        self.neg_probabilities = pow_freq / total_pow
    
    def _build_neg_sample_table(self):
        print("Building negative sampling table (target size=" + format(self.table_size, ",") + ")")
        
        word_indices = np.arange(len(self.neg_probabilities), dtype=np.int32)

        # Build sampling table
        counts = np.floor(self.neg_probabilities * float(self.table_size)).astype(np.int64)
        total_counts = int(np.sum(counts))

        if total_counts == 0:
            top_idx = int(np.argmax(self.neg_probabilities))
            counts[top_idx] = self.table_size
            total_counts = self.table_size

        desc = np.argsort(-self.neg_probabilities)
        i = 0
        while total_counts < self.table_size:
            counts[desc[i % len(counts)]] += 1
            total_counts += 1
            i += 1

        self.neg_sample_table = np.repeat(word_indices, counts.astype(np.int32))
    
    def _sample_negatives(self, target_idx, k):
        if self.neg_sample_table is None or len(self.neg_sample_table) == 0:
            return []

        negatives = []
        while len(negatives) < k:
            batch = np.random.choice(self.neg_sample_table, size=(k - len(negatives)), replace=True)
            for idx in batch:
                if int(idx) != int(target_idx):
                    negatives.append(int(idx))
                    if len(negatives) >= k:
                        break
        return negatives
    
    def _subsampling_probability(self, freq):
        if freq <= 0:
            return 1.0
        keep_prob = math.sqrt(self.sample_threshold / freq)
        return min(1.0, keep_prob)
    
    def _sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))
    
    def train_step(self, input_ngrams, context_word_idx, learning_rate):
        if not input_ngrams:
            return 0.0
        
        h = np.sum(self.W_in[input_ngrams], axis=0).astype(np.float32)
        targets = [(context_word_idx, 1.0)]
        
        neg_indices = self._sample_negatives(context_word_idx, self.neg_k)
        targets.extend([(idx, 0.0) for idx in neg_indices])
        
        loss = 0.0
        input_grad = np.zeros_like(h, dtype=np.float32)
        
        for ctx_idx, label in targets:
            v_ctx = self.W_out[ctx_idx]
            score = np.dot(v_ctx, h)
            pred = self._sigmoid(score)
            
            if label == 1.0:
                loss += -np.log(pred + 1e-9)
            else:
                loss += -np.log(1.0 - pred + 1e-9)

            grad = (pred - label)
            
            self.W_out[ctx_idx] -= (grad * h * learning_rate).astype(np.float32)
            input_grad += grad * v_ctx
        
        for ng_idx in input_ngrams:
            self.W_in[ng_idx] -= (input_grad * learning_rate).astype(np.float32)
        
        return loss
    
    def train(self, corpus_path, vocab, epochs=5):
        if self.W_in is None:
            self.init_model(vocab)
        
        total_word_count = sum(vocab.word_counts.values())
        
        sub_probs = {}
        for word, count in vocab.word_counts.items():
            freq = count / total_word_count
            sub_probs[word] = self._subsampling_probability(freq)
        
        line_count = 0
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1

        total_examples = total_word_count * epochs
        processed_examples = 0
        start_lr = self.lr
        min_lr = 0.0001

        # Training loop
        for epoch in range(epochs):
            print("\n" + "="*50)
            print("Epoch " + str(epoch + 1) + "/" + str(epochs))
            print("="*50)
            
            total_loss = 0.0
            trained_pairs = 0
            
            with open(corpus_path, "r", encoding="utf-8") as f:
                pbar = tqdm(total=line_count, desc="Training epoch " + str(epoch+1))
                
                for line in f:
                    words = line.strip().split()
                    if not words:
                        pbar.update(1)
                        continue

                    processed_examples += len(words)
                    current_lr = self.lr * (1.0 - (processed_examples / float(total_examples)))
                    if current_lr < min_lr:
                        current_lr = min_lr
                    
                    # Apply subsampling
                    kept_words = []
                    for w in words:
                        if w in vocab.word_to_idx:
                            if random.random() < sub_probs[w]:
                                kept_words.append(w)
                    
                    if len(kept_words) < 2:
                        pbar.update(1)
                        continue
                    
                    # Process each word as target
                    for i, center_word in enumerate(kept_words):
                        input_ids = vocab.get_ids_for_word(center_word)
                        if not input_ids:
                            continue
                        
                        actual_window = random.randint(1, self.window)  
                        start = max(0, i - actual_window)
                        end = min(len(kept_words), i + actual_window + 1)
                        
                        for j in range(start, end):
                            if i == j:
                                continue
                            
                            context_word = kept_words[j]
                            
                            ctx_idx = vocab.word_to_idx[context_word]
                            
                            loss = self.train_step(input_ids, ctx_idx, current_lr)
                            total_loss += loss
                            trained_pairs += 1
                    
                    pbar.update(1)
                    if trained_pairs > 0:
                        avg_loss = total_loss / trained_pairs
                        pbar.set_postfix({
                            "loss": "{:.4f}".format(avg_loss),
                            "pairs": "{:,}".format(trained_pairs),
                            "lr": "{:.6f}".format(current_lr)
                        })
                
                pbar.close()
            
            # Epoch statistics
            if trained_pairs > 0:
                avg_loss = total_loss / trained_pairs
                print("\nEpoch " + str(epoch+1) + " complete:")
                print("  Average loss: {:.4f}".format(avg_loss))
                print("  Trained pairs: {:,}".format(trained_pairs))
                print("  Final learning rate: {:.6f}".format(current_lr))
                
                self.loss_history.append(avg_loss)
                
                loss_file = os.path.join(path_to_output_dir, "fasttext_loss.txt")
                with open(loss_file, "a") as f:
                    f.write(str(epoch+1) + "," + str(avg_loss) + "," + str(trained_pairs) + "," + "{:.6f}".format(current_lr) + "\n")
            else:
                print("Warning: Epoch " + str(epoch+1) + " trained 0 pairs!")
        
        print("\n" + "="*50)
        print("Training complete!")
        if self.loss_history:
            print("Final loss: {:.4f}".format(self.loss_history[-1]))
        print("="*50)
    
    def get_word_vector(self, word, vocab):
        ngram_ids = vocab.get_ids_for_word(word)
        if not ngram_ids:
            return np.zeros(self.dim, dtype=np.float32)
        return np.sum(self.W_in[ngram_ids], axis=0).astype(np.float32)
    
    def analyze_morpheme_importance(self, word, vocab):
        full_vector = self.get_word_vector(word, vocab)
        full_norm = np.linalg.norm(full_vector)
        
        if full_norm < 1e-9:
            return []
        
        ngram_ids = vocab.get_ids_for_word(word)
        if not ngram_ids:
            return []
        
        results = []
        for ng_id in ngram_ids:
            ngram = vocab.idx_to_ngram[ng_id]
            other_ids = [id for id in ngram_ids if id != ng_id]
            
            if not other_ids:
                results.append((ngram, 1.0, 0.0))
                continue
            
            partial_vector = np.sum(self.W_in[other_ids], axis=0)
            partial_norm = np.linalg.norm(partial_vector)
            
            if partial_norm < 1e-9 or full_norm < 1e-9:
                cos_sim = 0.0
            else:
                cos_sim = np.dot(full_vector, partial_vector) / (full_norm * partial_norm)
            
            importance = 1.0 - cos_sim
            results.append((ngram, importance, cos_sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def save_model(self, path, vocab=None):
        model_data = {
            'W_in': self.W_in,
            'W_out': self.W_out,
            'dim': self.dim,
            'lr': self.lr,
            'window': self.window,
            'neg_k': self.neg_k,
            'sample_threshold': self.sample_threshold,
            'loss_history': self.loss_history,
            'table_size': self.table_size,
            'has_vocab': vocab is not None,
            'neg_probabilities': self.neg_probabilities
        }
        
        if vocab is not None:
            model_data['vocab'] = vocab
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saved to " + str(path))
    
    @staticmethod
    def load_model(path, load_vocab=False):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        trainer = FastTextTrainer(
            vector_size=model_data['dim'],
            learning_rate=model_data.get('lr', 0.01),
            window_size=model_data['window'],
            neg_samples=model_data['neg_k'],
            sample_threshold=model_data.get('sample_threshold', 1e-5),
            table_size=model_data.get('table_size', 1000000)
        )
        
        trainer.W_in = model_data['W_in']
        trainer.W_out = model_data['W_out']
        trainer.loss_history = model_data.get('loss_history', [])
        trainer.neg_probabilities = model_data.get('neg_probabilities', None)
        
        vocab = None
        if load_vocab and 'vocab' in model_data:
            vocab = model_data['vocab']
        
        return trainer, vocab