"""
Handles text tokenization using Byte-Pair Encoding (BPE)
and vocabulary management. Uses the Hugging Face `tokenizers` library.
"""

import config
import os
from typing import List, Iterator
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing # For START/END tokens

# --- Global Tokenizer Instance ---
# Will be loaded or trained and assigned in get_tokenizer()
tokenizer_instance = None

def train_tokenizer(captions_iterator: Iterator[str], vocab_size: int, vocab_path: str, merges_path: str):
    """
    Trains a ByteLevelBPETokenizer on the provided captions.

    Args:
        captions_iterator: An iterator yielding caption strings.
        vocab_size: The desired vocabulary size.
        vocab_path: Path to save the vocabulary JSON file.
        merges_path: Path to save the merges text file.
    """
    print(f"Training BPE tokenizer with vocab size {vocab_size}...")
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train_from_iterator(
        captions_iterator,
        vocab_size=vocab_size,
        min_frequency=2, # Ignore words appearing less than twice
        special_tokens=[
            config.PAD_TOKEN,
            config.UNK_TOKEN,
            config.START_TOKEN,
            config.END_TOKEN,
            # Add other special tokens if needed, e.g., MASK
        ]
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(vocab_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the tokenizer files (vocabulary and merge rules)
    tokenizer.save_model(output_dir) # Saves vocab.json and merges.txt

    # Rename files to match config paths if necessary (save_model uses fixed names)
    saved_vocab = os.path.join(output_dir, "vocab.json")
    saved_merges = os.path.join(output_dir, "merges.txt")
    if saved_vocab != vocab_path:
        os.rename(saved_vocab, vocab_path)
        print(f"Saved vocabulary to {vocab_path}")
    if saved_merges != merges_path:
        os.rename(saved_merges, merges_path)
        print(f"Saved merges to {merges_path}")

    print("Tokenizer training complete.")
    return tokenizer # Return the trained tokenizer


def get_tokenizer():
    """
    Loads the trained BPE tokenizer if files exist, otherwise raises an error
    (training should be handled explicitly before calling this if needed).
    Adds START/END token processing.
    """
    global tokenizer_instance
    if tokenizer_instance is not None:
        return tokenizer_instance

    vocab_path = config.VOCAB_PATH
    merges_path = config.MERGES_PATH

    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError(
            f"Tokenizer vocabulary or merges file not found at "
            f"{vocab_path} or {merges_path}. "
            "Please ensure the tokenizer is trained first (e.g., via train.py)."
        )

    print(f"Loading tokenizer from: {vocab_path}, {merges_path}")
    tokenizer = ByteLevelBPETokenizer(
        vocab=vocab_path,
        merges=merges_path,
    )

    # Configure post-processing to add START and END tokens
    tokenizer._tokenizer.post_processor = BertProcessing(
        sep=(config.END_TOKEN, tokenizer.token_to_id(config.END_TOKEN)),
        cls=(config.START_TOKEN, tokenizer.token_to_id(config.START_TOKEN)),
    )
    # Set padding and truncation (optional but good practice)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id(config.PAD_TOKEN), pad_token=config.PAD_TOKEN, length=config.MAX_SEQ_LEN)
    tokenizer.enable_truncation(max_length=config.MAX_SEQ_LEN)


    print("Tokenizer loaded successfully.")
    tokenizer_instance = tokenizer # Assign to global instance
    return tokenizer_instance

# --- Helper functions to bridge between tokenizer and model needs ---

def encode(text: str, add_special_tokens=True) -> List[int]:
    """Encodes text using the global tokenizer instance."""
    tokenizer = get_tokenizer()
    # Note: add_special_tokens is handled by BertProcessing if configured
    # We pass add_special_tokens=False if BertProcessing handles it.
    # Check how BertProcessing interacts with the flag if issues arise.
    # Current HF tokenizers usually handle this via the post_processor.
    return tokenizer.encode(text).ids

def decode(token_ids: List[int], skip_special_tokens: bool = True) -> str:
    """Decodes token IDs using the global tokenizer instance."""
    tokenizer = get_tokenizer()
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

def get_vocab_size() -> int:
    """Gets the vocabulary size from the global tokenizer instance."""
    tokenizer = get_tokenizer()
    return tokenizer.get_vocab_size()

def token_to_id(token: str) -> int:
    """Gets the ID for a specific token."""
    tokenizer = get_tokenizer()
    return tokenizer.token_to_id(token)

# Example Usage
if __name__ == '__main__':
    print("Testing BPE Tokenizer functions...")

    # Dummy data for testing training
    dummy_captions = [
        "A black cat sat on a mat.",
        "Another cat, this one white, sat on the same mat.",
        "Why do cats like mats so much?",
        "Maybe the mat is comfortable.",
    ]
    dummy_vocab_size = 50 # Small size for testing
    dummy_vocab_path = "./dummy_vocab.json"
    dummy_merges_path = "./dummy_merges.txt"
    config.OUTPUT_DIR = "." # Override config for local testing
    config.VOCAB_PATH = dummy_vocab_path
    config.MERGES_PATH = dummy_merges_path
    config.PAD_TOKEN_ID = 0 # Example IDs, ensure consistency
    config.UNK_TOKEN_ID = 1
    config.START_TOKEN_ID = 2
    config.END_TOKEN_ID = 3
    config.MAX_SEQ_LEN = 20 # For testing truncation/padding


    # --- Test Training ---
    print("\n--- Testing Training ---")
    if os.path.exists(dummy_vocab_path): os.remove(dummy_vocab_path)
    if os.path.exists(dummy_merges_path): os.remove(dummy_merges_path)

    trained_tokenizer = train_tokenizer(
        iter(dummy_captions),
        vocab_size=dummy_vocab_size,
        vocab_path=dummy_vocab_path,
        merges_path=dummy_merges_path
    )
    print(f"Trained vocab size: {trained_tokenizer.get_vocab_size()}")

    # --- Test Loading & Encoding/Decoding ---
    print("\n--- Testing Loading, Encoding, Decoding ---")
    # Reset global instance to force loading
    tokenizer_instance = None
    loaded_tokenizer = get_tokenizer()

    print(f"Loaded vocab size: {get_vocab_size()}")

    text = "A white cat on a comfortable mat."
    encoded_output = loaded_tokenizer.encode(text) # Use the tokenizer's method directly
    encoded_ids = encoded_output.ids
    encoded_tokens = encoded_output.tokens

    print(f"Text: '{text}'")
    print(f"Encoded Tokens: {encoded_tokens}")
    print(f"Encoded IDs: {encoded_ids}")
    print(f"Attention Mask: {encoded_output.attention_mask}")

    decoded_text = decode(encoded_ids, skip_special_tokens=False)
    print(f"Decoded (no skip): '{decoded_text}'")

    decoded_text_skipped = decode(encoded_ids, skip_special_tokens=True)
    print(f"Decoded (skip special): '{decoded_text_skipped}'")

    print(f"PAD token ID: {token_to_id(config.PAD_TOKEN)}")
    print(f"UNK token ID: {token_to_id(config.UNK_TOKEN)}")
    print(f"START token ID: {token_to_id(config.START_TOKEN)}")
    print(f"END token ID: {token_to_id(config.END_TOKEN)}")

    # Clean up dummy files
    if os.path.exists(dummy_vocab_path): os.remove(dummy_vocab_path)
    if os.path.exists(dummy_merges_path): os.remove(dummy_merges_path)
    print("\nCleaned up dummy files.")