"""
Handles text tokenization and vocabulary management.

This is a placeholder. You might want to use a pre-trained tokenizer
(e.g., from Hugging Face transformers) or train your own (e.g., BPE, WordPiece)
on your target text corpus.
"""

import config
from collections import Counter
from typing import List, Dict

class SimpleTokenizer:
    """A very basic tokenizer for demonstration purposes."""
    def __init__(self, vocab: Dict[str, int] = None, vocab_size: int = config.VOCAB_SIZE):
        self.vocab = vocab if vocab else {}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = vocab_size
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Ensures special tokens are in the vocabulary."""
        special_tokens = {
            config.PAD_TOKEN: config.PAD_TOKEN_ID,
            config.START_TOKEN: config.START_TOKEN_ID,
            config.END_TOKEN: config.END_TOKEN_ID,
            config.UNK_TOKEN: config.UNK_TOKEN_ID
        }
        for token, idx in special_tokens.items():
            if token not in self.vocab:
                self.vocab[token] = idx
                self.reverse_vocab[idx] = token

    def train(self, texts: List[str]):
        """Builds a simple vocabulary from a list of texts based on frequency."""
        if self.vocab_size <= len(self.vocab): # Already have enough tokens
            print("Vocabulary already meets or exceeds target size. Not training.")
            return

        print("Building vocabulary...")
        word_counts = Counter()
        for text in texts:
            # Basic whitespace and lowercase tokenization
            word_counts.update(text.lower().split())

        # Keep existing special tokens
        current_vocab_count = len(self.vocab)
        tokens_to_add = self.vocab_size - current_vocab_count

        # Sort words by frequency, exclude already added special tokens
        most_common = [
            word for word, count in word_counts.most_common()
            if word not in self.vocab and word not in [config.PAD_TOKEN, config.START_TOKEN, config.END_TOKEN, config.UNK_TOKEN]
        ]

        added_count = 0
        next_idx = max(self.vocab.values()) + 1
        for word in most_common:
            if added_count >= tokens_to_add:
                break
            self.vocab[word] = next_idx
            self.reverse_vocab[next_idx] = word
            next_idx += 1
            added_count += 1

        self.vocab_size = len(self.vocab)
        print(f"Vocabulary built with {self.vocab_size} tokens.")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes a string into a list of token IDs."""
        tokens = text.lower().split()
        encoded = [self.vocab.get(token, config.UNK_TOKEN_ID) for token in tokens]

        if add_special_tokens:
            encoded = [config.START_TOKEN_ID] + encoded + [config.END_TOKEN_ID]

        return encoded

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes a list of token IDs back into a string."""
        tokens = []
        for idx in token_ids:
            token = self.reverse_vocab.get(idx, config.UNK_TOKEN)
            if skip_special_tokens and token in [config.PAD_TOKEN, config.START_TOKEN, config.END_TOKEN]:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def save_vocab(self, filepath: str):
        """Saves the vocabulary to a file (e.g., JSON)."""
        import json
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            print(f"Vocabulary saved to {filepath}")
        except Exception as e:
            print(f"Error saving vocabulary: {e}")

    @classmethod
    def load_vocab(cls, filepath: str):
        """Loads the vocabulary from a file."""
        import json
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            print(f"Vocabulary loaded from {filepath}")
            # Ensure keys are strings and values are ints if loaded from JSON
            vocab = {str(k): int(v) for k, v in vocab.items()}
            # Determine effective vocab size from loaded vocab
            loaded_vocab_size = len(vocab)
            return cls(vocab=vocab, vocab_size=loaded_vocab_size)
        except FileNotFoundError:
            print(f"Error: Vocabulary file not found at {filepath}. Returning new tokenizer.")
            return cls(vocab_size=config.VOCAB_SIZE) # Return a default instance
        except Exception as e:
            print(f"Error loading vocabulary: {e}. Returning new tokenizer.")
            return cls(vocab_size=config.VOCAB_SIZE)

# --- Global Tokenizer Instance --- #
# Load or create a tokenizer instance. Ideally, train it on your dataset
# and save/load the vocabulary.

# Example: Try loading from a file, otherwise create a new one
VOCAB_FILE = f"{config.OUTPUT_DIR}/vocab.json"
tokenizer_instance = SimpleTokenizer.load_vocab(VOCAB_FILE)

# If the loaded vocab is too small (e.g., file not found), train it
# This requires having access to your text data here, which might be better done in train.py
# Example placeholder training call:
# if tokenizer_instance.get_vocab_size() < config.VOCAB_SIZE:
#     # Load your text data (e.g., from config.CAPTIONS_FILE)
#     # texts = load_texts_from_captions(config.CAPTIONS_FILE)
#     # tokenizer_instance.train(texts)
#     # tokenizer_instance.save_vocab(VOCAB_FILE)
#     pass # Placeholder: Handle training in the main script


def get_tokenizer():
    """Returns the globally accessible tokenizer instance."""
    global tokenizer_instance
    # Ensure the tokenizer is initialized (might be redundant if loaded above)
    if tokenizer_instance is None:
         tokenizer_instance = SimpleTokenizer.load_vocab(VOCAB_FILE)
    return tokenizer_instance

# Example Usage
if __name__ == '__main__':
    # Create a dummy tokenizer or load if exists
    dummy_texts = [
        "this is the first sentence",
        "this is another sentence example",
        "example sentence three"
    ]
    tok = SimpleTokenizer(vocab_size=15)
    tok.train(dummy_texts)

    print("Vocabulary:", tok.vocab)
    print("Vocab Size:", tok.get_vocab_size())

    text = "this is a test sentence with unknown words"
    encoded = tok.encode(text)
    print(f"Encoded '{text}': {encoded}")

    decoded = tok.decode(encoded)
    print(f"Decoded {encoded}: '{decoded}'")

    decoded_no_special = tok.decode(encoded, skip_special_tokens=True)
    print(f"Decoded (no special) {encoded}: '{decoded_no_special}'")

    # Test save/load
    # import os
    # if not os.path.exists(config.OUTPUT_DIR):
    #     os.makedirs(config.OUTPUT_DIR)
    # tok.save_vocab(VOCAB_FILE)
    # loaded_tok = SimpleTokenizer.load_vocab(VOCAB_FILE)
    # print("Loaded Vocab Size:", loaded_tok.get_vocab_size())
    # print(loaded_tok.encode("another test"))

</rewritten_file> 