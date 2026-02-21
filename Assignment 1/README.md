# NLP Assignment 1: Tokenization, FastText, and NPLM

**Author:** Anirudh Bharatiya (2023090)  
**Course:** Natural Language Processing (CSE556)

This repository contains the source code for Assignment 1. The project implements the following:
1.  **WordPiece Tokenizer** (Preprocessing, Vocabulary Training, Encoding/Decoding)
2.  **FastText** (N-gram bag generation and Embedding training)
3.  **Neural Probabilistic Language Model (NPLM)** (Training, Evaluation, and Generation)

---

## 📂 Project Structure

The project follows a modular Python package structure.

```text
.
├── Assignment 1.pdf      # Problem statement
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusion rules
├── data/                 # Data directory
├── models/               # Model artifacts
└── src/                  # Source Code
    ├── main.py           # CLI Entry Point
    ├── question1/        # WordPiece Tokenizer Modules
    │   ├── preprocessor.py
    │   ├── trainer.py
    │   ├── encoder.py
    │   └── decoder.py
    ├── question2/        # FastText Modules
    │   ├── ngramgenerator.py
    │   └── trainer.py
    └── question3/        # NPLM Modules
        ├── data.py
        ├── model.py
        ├── train.py
        ├── generate.py
        └── utils.py
```

---

## ⚙️ Setup & Installation

### 1. Environment
Run this project in a virtual environment.

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Dependencies
Install the required Python packages.
```bash
pip install -r requirements.txt
```

### 3. Directory Initialization
Since `data/` and `models/` are listed in `.gitignore`, you must ensure they exist before running the code.

```bash
mkdir -p data/q1 data/q2 data/q3 models
```

---

## 🚀 Usage Instructions

**Note:** All commands must be executed from the `src` directory. The `main.py` is the main script to execute all the modules.

```bash
cd src
```

### 1. Question 1: WordPiece Tokenizer
Located in `src/question1/`, this module handles text preprocessing and tokenizer operations.

**Usage:** `python main.py q1 <flag> <input_file> --output <output_file> [options]`

| Flag | Operation | Description |
| :--- | :--- | :--- |
| `-1` | **Preprocess** | Normalizes and cleans raw text corpus. |
| `-2` | **Train Vocab** | Trains the WordPiece vocabulary. |
| `-3` | **Encode (Tokens)** | Tokenizes text into subword strings. |
| `-4` | **Encode (IDs)** | Tokenizes text into integer IDs. |
| `-5` | **Decode** | Converts integer IDs back to text. |

**Examples:**
```bash
# 1. Preprocess
python main.py q1 -1 "../data/raw_corpus.txt" --output "../data/q1/preprocessed.txt"

# 2. Train Vocabulary (Size: 10,000)
python main.py q1 -2 "../data/q1/preprocessed.txt" --output "../data/q1/vocab.txt" --size 10000

# 3. Encode to IDs
python main.py q1 -4 "../data/q1/preprocessed.txt" --output "../data/q1/encoded_ids.txt" --vocab "../data/q1/vocab.txt"

# 4. Decode back to text
python main.py q1 -5 "../data/q1/encoded_ids.txt" --output "../data/q1/decoded.txt" --vocab "../data/q1/vocab.txt"
```

---

### 2. Question 2: FastText
Located in `src/question2/`, this module generates n-gram representations and trains embeddings.

**Usage:** `python main.py q2 <command> [arguments]`

#### A. Generate Bags (`bags`)
Creates bag-of-ngrams representations and saves the n-gram vocabulary.
```bash
python main.py q2 bags "../data/raw_corpus.txt" "../data/q2/bags.txt" --vocab "../data/q2/vocab_ngram.pkl"
```

#### B. Train Model (`train`)
Trains the FastText model. The loss curve will be saved as `fasttext_loss_curve.png` in the current directory.
```bash
python main.py q2 train "../data/raw_corpus.txt" \
    --vocab "../data/q2/vocab_ngram.pkl" \
    --model "../models/fasttext_model.pth" \
    --dim 100 \
    --epochs 5
```

---

### 3. Question 3: NPLM
Located in `src/question3/`, this module implements the Neural Probabilistic Language Model.

**Usage:** `python main.py q3 <command> [arguments]`

#### A. Training (`train`)
Trains the NPLM model with a specified context window size.
```bash
python main.py q3 train "../data/q1/preprocessed.txt" \
    -v "../data/vocab_q3.txt" \
    -m "../models/nplm_model.pth" \
    --context_size 5 \
    --embedding_dim 400 \
    --epochs 10
```

#### B. Evaluation (`eval`)
Calculates Perplexity (PPL) and Accuracy on a test file.
```bash
python main.py q3 eval "../data/q1/preprocessed.txt" \
    -m "../models/nplm_model.pth" \
    -v "../data/vocab_q3.txt"
```

#### C. Generation (`generate`)
Generates text continuations based on a file of seed sentences.
```bash
python main.py q3 generate "../data/seeds.txt" 10 \
    -m "../models/nplm_model.pth" \
    -v "../data/vocab_q3.txt" \
    -o "../data/q3/generated_output.txt"
```