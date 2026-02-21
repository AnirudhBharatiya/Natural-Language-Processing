import argparse
import sys
import os
import torch

from question1 import preprocessor, trainer, encoder, decoder

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from question2.ngramgenerator import FastTextNGramGenerator
from question2.trainer import FastTextTrainer

from question3 import q3_train, q3_generate, q3_utils, q3_model

# Question 2 Helper Functions
def q2_generate_bags_file(input_file, output_file, vocab_file):
    
    temp_preprocessed_file = "temp_q2_bags_preprocessed.txt"
    print("Preprocessing input file for Q2 bags: " + str(input_file))
    preprocessor_obj = preprocessor.TextPreprocessor(input_file, temp_preprocessed_file)
    preprocessor_obj.run()
    
    print("="*50)
    print("Generating n-gram bags")
    print("="*50)
    
    try:
        vocab = FastTextNGramGenerator()
        vocab.build_vocabulary(temp_preprocessed_file)
        vocab.save(vocab_file)
        
        line_count = 0
        with open(temp_preprocessed_file, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
        
        # Generate the bag file
        with open(temp_preprocessed_file, "r", encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8") as fout:
            
            pbar = tqdm(total=line_count, desc="Generating bags")
            
            for line in fin:
                words = line.strip().split()
                line_bags = []
                for word in words:
                    ngram_ids = vocab.get_ids_for_word(word)
                    if ngram_ids:
                        line_bags.append(",".join(map(str, ngram_ids)))
                    else:
                        line_bags.append("") 
                fout.write(" ".join(line_bags) + "\n")
                pbar.update(1)
            
            pbar.close()
        
        print("Bags file saved to " + str(output_file))
    finally:
        if os.path.exists(temp_preprocessed_file):
            os.remove(temp_preprocessed_file)

def q2_train_fasttext_model(corpus_file, vocab_file, model_file, 
                         vector_size=100, epochs=5, 
                         include_vocab_in_model=False):
    
    temp_preprocessed_file = "temp_q2_train_preprocessed.txt"
    print("Preprocessing corpus file for Q2 training: " + str(corpus_file))
    preprocessor_obj = preprocessor.TextPreprocessor(corpus_file, temp_preprocessed_file)
    preprocessor_obj.run()

    print("="*50)
    print("Training FastText model")
    print("="*50)
    
    try:
        vocab = FastTextNGramGenerator.load(vocab_file)
        
        trainer = FastTextTrainer(
            vector_size=vector_size,
            learning_rate=0.01,
            window_size=5,
            neg_samples=5,
            sample_threshold=1e-5,
            table_size=1000000
        )
        
        trainer.train(temp_preprocessed_file, vocab, epochs=epochs)
        
        if include_vocab_in_model:
            trainer.save_model(model_file, vocab)
        else:
            trainer.save_model(model_file)
        
        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(trainer.loss_history) + 1), trainer.loss_history, 
                marker='o', linewidth=2)
        plt.title("FastText Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig("fasttext_loss_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("Loss curve saved to fasttext_loss_curve.png")
        print("Model training complete! Saved to " + str(model_file))
        
        # Run analysis experiments immediately after training
        q2_run_experiments(trainer, vocab)
    finally:
        if os.path.exists(temp_preprocessed_file):
            os.remove(temp_preprocessed_file)

def q2_run_experiments(trainer, vocab):
    print("="*50)
    print("Running FastText Experiments")
    print("="*50)
    
    print("Model dimension: " + str(trainer.dim))
    print("Vocabulary: " + format(len(vocab.word_to_idx), ",") + " words")
    
    # OOV Analysis
    print("\n" + "-"*50)
    print("1) OOV Word Representation (Experiment i)")
    print("-"*50)
    
    # Hardcoded example as requested
    oov_words = [("अतिसुंदरतम", "Hindi compound"), ("xyz123abc", "Synthetic")]
    
    for word, desc in oov_words:
        print("\nOOV Word: '" + str(word) + "' (" + str(desc) + ")")
        vector = trainer.get_word_vector(word, vocab)
        norm = np.linalg.norm(vector)
        print("  Vector norm: {:.6f}".format(norm))
        if norm > 0.001:
            print("Generated vector via n-grams!")
            ngram_ids = vocab.get_ids_for_word(word)
            print("  Number of n-grams used: " + str(len(ngram_ids)))
    
    # Morpheme Analysis
    print("\n" + "-"*50)
    print("2) Morpheme Importance Analysis (Experiment ii)")

    print("-"*50)
    
    test_words = [("दुश्मनी", ["दु", "श्म", "नी"]), ("अलास्का", ["अ", "ला", "स", "्का"])]
    
    for word, expected in test_words:
        print("\nWord: '" + str(word) + "'")
        print("Expected morphemes: " + str(expected))
        results = trainer.analyze_morpheme_importance(word, vocab)[:10]
        for i, (ngram, imp, sim) in enumerate(results):
            print("  " + str(i+1) + ". '" + str(ngram) + "' - Importance: {:.4f}, Cosine sim: {:.4f}".format(imp, sim))
    
    print("\n" + "="*50)
    print("Analysis complete.")


def handle_q1(args):
    args_dict = vars(args)
    temp_preprocessed_file = "temp_q1_preprocessed.txt"

    try:
        # Op 1: Preprocess
        if args_dict['1']:
            print("Op 1: Preprocessing " + str(args.input_file))
            preprocessor_obj = preprocessor.TextPreprocessor(args.input_file, args.output)
            preprocessor_obj.run()
            print("Done.")

        # Op 2: Train Vocab
        elif args_dict['2']:
            if not args.size:
                print("Error: --size is required for Op 2.")
                sys.exit(1)
            print("Op 2: Training Vocab (Size: " + str(args.size) + ")")
            preprocessor_obj = preprocessor.TextPreprocessor(args.input_file, temp_preprocessed_file)
            preprocessor_obj.run()
            
            trainer_obj = trainer.WordPieceTrainer(args.size)
            trainer_obj.train(temp_preprocessed_file)
            trainer_obj.save_vocab(args.output)
            print("Done.")

        # Op 3: Encode to Tokens
        elif args_dict['3']:
            if not args.vocab:
                print("Error: --vocab is required for Op 3.")
                sys.exit(1)
            print("Op 3: Encoding to Tokens")
            preprocessor_obj = preprocessor.TextPreprocessor(args.input_file, temp_preprocessed_file)
            preprocessor_obj.run()
            
            dummy_ids = "temp_dummy_ids.txt"
            encoder_obj = encoder.WordPieceEncoder(args.vocab)
            encoder_obj.encode_file(temp_preprocessed_file, args.output, dummy_ids)
            if os.path.exists(dummy_ids): os.remove(dummy_ids)
            print("Done.")

        # Op 4: Encode to IDs
        elif args_dict['4']:
            if not args.vocab:
                print("Error: --vocab is required for Op 4.")
                sys.exit(1)
            print("Op 4: Encoding to IDs")
            preprocessor_obj = preprocessor.TextPreprocessor(args.input_file, temp_preprocessed_file)
            preprocessor_obj.run()
            
            dummy_tokens = "temp_dummy_tokens.txt"
            encoder_obj = encoder.WordPieceEncoder(args.vocab)
            encoder_obj.encode_file(temp_preprocessed_file, dummy_tokens, args.output)
            if os.path.exists(dummy_tokens): os.remove(dummy_tokens)
            print("Done.")

        # Op 5: Decode
        elif args_dict['5']:
            if not args.vocab:
                print("Error: --vocab is required for Op 5.")
                sys.exit(1)
            print("Op 5: Decoding IDs...")
            decoder_obj = decoder.WordPieceDecoder(args.vocab)
            decoder_obj.decode_file(args.input_file, args.output)
            print("Done.")

    except Exception as e:
        print("Error in Q1: " + str(e))
        sys.exit(1)
    finally:
        if os.path.exists(temp_preprocessed_file):
            os.remove(temp_preprocessed_file)


def handle_q2(args):
    try:
        if args.command == 'bags':
            q2_generate_bags_file(args.input_file, args.output_file, args.vocab)
        elif args.command == 'train':
            q2_train_fasttext_model(
                args.corpus_file, args.vocab, args.model, 
                vector_size=args.dim, epochs=args.epochs
            )
    except Exception as e:
        print("Error in Q2: " + str(e))
        sys.exit(1)


def handle_q3(args):
    try:
        # Load vocab first to get size
        print("Loading vocab from " + str(args.vocab) + "...")
        token_to_id, id_to_token = q3_utils.load_vocab(args.vocab)
        vocab_size = len(token_to_id)
        
        if args.command == 'train':
            print("Training NPLM model...")
            print("Config: Context Size=" + str(args.context_size) + ", Emb=" + str(args.embedding_dim) + ", Hidden=" + str(args.hidden_dim) + ", Act=" + str(args.activation))
            
            # Initialize model
            model = q3_model.NPLM(
                vocab_size=vocab_size,
                embedding_dim=args.embedding_dim,
                context_size=args.context_size,
                hidden_dim=args.hidden_dim,
                activation=args.activation
            )
            
            UNK_ID = token_to_id.get("<UNK>", 0)
            
            # Preparing data
            print("Building datasets from " + str(args.input_file))
            train_data, val_data, _ = q3_train.build_q3_datasets(
                args.input_file, args.vocab, args.context_size
            )
            
            # Training
            q3_train.train_nplm(
                model, train_data, val_data, vocab_size, UNK_ID,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                save_path=args.model
            )
            print("Training complete. Model saved to " + str(args.model))
            return

        # For Eval/Generate, load model metadata
        if not os.path.exists(args.model):
            print("Error: Model file " + str(args.model) + " not found.")
            sys.exit(1)
            
        checkpoint = torch.load(args.model, map_location='cpu')
        context_size = checkpoint.get('context_size', 5) # Default 5 if missing
        
        # Load the model
        model = q3_train.load_trained_model(args.model, vocab_size, q3_model.NPLM)
        
        if args.command == 'eval':
            print("Evaluating " + str(args.input_file))
            # Use build_eval_dataset which likely delegates to build_q3_datasets
            eval_data = q3_train.build_eval_dataset(args.input_file, args.vocab, context_size)
            ppl, acc = q3_train.evaluate_model(model, eval_data)
            print("Perplexity: {:.4f}".format(ppl))
            print("Accuracy: {:.2%}".format(acc))
            
        elif args.command == 'generate':
            print("Generating text from " + str(args.seed_file))
            q3_generate.generate_from_file(
                args.seed_file, args.output, model, 
                token_to_id, id_to_token, context_size, args.k
            )
            print("Generated text saved to " + str(args.output))

    except Exception as e:
        print("Error in Q3: " + str(e))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="CSE556 NLP Assignment 1 - Final CLI")
    subparsers = parser.add_subparsers(dest='question', required=True, help="Assignment Question")

    # Question 1 (WordPiece)
    q1_parser = subparsers.add_parser('q1', help="WordPiece Tokenizer")
    q1_main_group = q1_parser.add_mutually_exclusive_group(required=True)
    q1_main_group.add_argument('-1', action='store_true', help="Op 1: Preprocess")
    q1_main_group.add_argument('-2', action='store_true', help="Op 2: Train Vocab")
    q1_main_group.add_argument('-3', action='store_true', help="Op 3: Encode to Tokens")
    q1_main_group.add_argument('-4', action='store_true', help="Op 4: Encode to IDs")
    q1_main_group.add_argument('-5', action='store_true', help="Op 5: Decode IDs")
    
    q1_parser.add_argument('input_file', help="Input file path")
    q1_parser.add_argument('--output', '-o', required=True, help="Output file path")
    q1_parser.add_argument('--size', '-s', type=int, help="Vocab size (Op 2)")
    q1_parser.add_argument('--vocab', '-v', help="Vocab file (Ops 3,4,5)")

    # Question 2 (FastText)
    q2_parser = subparsers.add_parser('q2', help="FastText")
    q2_subs = q2_parser.add_subparsers(dest='command', required=True)

    # Op 1: Bags
    bags = q2_subs.add_parser('bags', help="Generate bags")
    bags.add_argument('input_file')
    bags.add_argument('output_file')
    bags.add_argument('--vocab', '-v', required=True)

    # Op 2: Train
    train = q2_subs.add_parser('train', help="Train FastText")
    train.add_argument('corpus_file')
    train.add_argument('--vocab', '-v', required=True)
    train.add_argument('--model', '-m', required=True)
    train.add_argument('--dim', '-d', type=int, default=100)
    train.add_argument('--epochs', '-e', type=int, default=5)

    # Question 3 (NPLM)
    q3_parser = subparsers.add_parser('q3', help="NPLM")
    q3_subs = q3_parser.add_subparsers(dest='command', required=True)

    # Op 1: Train
    train = q3_subs.add_parser('train', help="Train NPLM")
    train.add_argument('input_file')
    train.add_argument('--vocab', '-v', required=True)
    train.add_argument('--model', '-m', required=True, help="Path to save model")
    # Hyperparameters for ablation
    train.add_argument('--context_size', '-c', type=int, default=3, help="Context window size")
    train.add_argument('--embedding_dim', '-ed', type=int, default=400)
    train.add_argument('--hidden_dim', '-hd', type=int, default=100)
    train.add_argument('--activation', '-act', type=str, default='tanh', choices=['tanh', 'relu'], help="Activation function")
    train.add_argument('--epochs', '-e', type=int, default=5)
    train.add_argument('--batch_size', '-b', type=int, default=256)
    train.add_argument('--lr', type=float, default=0.001)

    # Op 2: Eval
    ev = q3_subs.add_parser('eval', help="Compute PPL and Acc")
    ev.add_argument('input_file')
    ev.add_argument('--model', '-m', required=True)
    ev.add_argument('--vocab', '-v', required=True)

    # Op 2: Generate
    gen = q3_subs.add_parser('generate', help="Generate text")
    gen.add_argument('seed_file')
    gen.add_argument('k', type=int)
    gen.add_argument('--model', '-m', required=True)
    gen.add_argument('--vocab', '-v', required=True)
    gen.add_argument('--output', '-o', required=True)

    args = parser.parse_args()

    if args.question == 'q1':
        handle_q1(args)
    elif args.question == 'q2':
        handle_q2(args)
    elif args.question == 'q3':
        handle_q3(args)

if __name__ == "__main__":
    main()