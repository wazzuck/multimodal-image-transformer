"""
Handles text tokenization using Byte-Pair Encoding (BPE)
and vocabulary management. Uses the Hugging Face `tokenizers` library.
"""

import config # Project configuration file for paths and special tokens.
import os
from typing import List, Iterator # Type hinting.
from tokenizers import ByteLevelBPETokenizer # The BPE tokenizer implementation.
from tokenizers.processors import BertProcessing # For adding special tokens like [CLS] (START) and [SEP] (END).

# --- Global Tokenizer Instance ---
# This variable will hold the loaded tokenizer instance to avoid reloading it multiple times.
# It is initialized to None and populated by the get_tokenizer() function on its first call.
_tokenizer_instance = None # Renamed to indicate it's a module-level private-like variable

def train_tokenizer(captions_iterator: Iterator[str], vocab_size: int, vocab_path: str, merges_path: str):
    """
    Trains a ByteLevelBPETokenizer from an iterator of caption strings.

    Args:
        captions_iterator: An iterator that yields caption strings for training the tokenizer.
        vocab_size: The target vocabulary size for the tokenizer.
        vocab_path: The file path where the trained vocabulary (vocab.json) will be saved.
        merges_path: The file path where the BPE merge rules (merges.txt) will be saved.
    """
    print(f"Training ByteLevelBPETokenizer with target vocab size {vocab_size}...")
    # Initialize a new ByteLevelBPETokenizer.
    tokenizer = ByteLevelBPETokenizer()

    # Train the tokenizer using the provided captions iterator.
    # `min_frequency=2`: Tokens must appear at least twice to be included.
    # `special_tokens`: Defines a list of special tokens to be included in the vocabulary.
    tokenizer.train_from_iterator(
        captions_iterator,
        vocab_size=vocab_size,
        min_frequency=2, # Helps filter out very rare tokens/typos.
        special_tokens=[
            config.PAD_TOKEN,    # Padding token.
            config.UNK_TOKEN,    # Unknown token.
            config.START_TOKEN,  # Start-of-sequence token.
            config.END_TOKEN,    # End-of-sequence token.
            # Add any other custom special tokens if needed, e.g., config.MASK_TOKEN.
        ]
    )

    # Ensure the output directory for tokenizer files exists.
    output_dir = os.path.dirname(vocab_path) # Get directory from vocab_path.
    os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist.

    # Save the tokenizer model. This typically saves two files: vocab.json and merges.txt
    # in the specified directory (output_dir from vocab_path).
    # The `save_model` method names them `vocab.json` and `merges.txt` by default.
    tokenizer.save_model(output_dir)

    # Rename the saved files to match the specific paths defined in config.py, if they differ.
    # This provides flexibility if config.VOCAB_PATH or config.MERGES_PATH are not simply output_dir/vocab.json.
    default_saved_vocab_path = os.path.join(output_dir, "vocab.json")
    default_saved_merges_path = os.path.join(output_dir, "merges.txt")

    if default_saved_vocab_path != vocab_path:
        os.rename(default_saved_vocab_path, vocab_path)
        print(f"Saved vocabulary to specified path: {vocab_path}")
    else:
        print(f"Vocabulary saved to default path: {default_saved_vocab_path}")
    
    if default_saved_merges_path != merges_path:
        os.rename(default_saved_merges_path, merges_path)
        print(f"Saved merges to specified path: {merges_path}")
    else:
        print(f"Merges saved to default path: {default_saved_merges_path}")

    print("Tokenizer training complete.")
    # Update the global tokenizer instance with the newly trained one.
    global _tokenizer_instance
    _tokenizer_instance = tokenizer
    return tokenizer # Return the trained tokenizer instance.


def get_tokenizer(force_reload: bool = False):
    """
    Loads the trained ByteLevelBPETokenizer from files specified in config.py.
    If already loaded, returns the existing instance unless `force_reload` is True.
    Configures post-processing to add START and END tokens automatically.

    Args:
        force_reload: If True, forces reloading of the tokenizer from files even if already loaded.

    Returns:
        The loaded and configured ByteLevelBPETokenizer instance.

    Raises:
        FileNotFoundError: If the tokenizer vocabulary or merges files are not found.
    """
    global _tokenizer_instance
    # Return the cached instance if available and not forcing reload.
    if _tokenizer_instance is not None and not force_reload:
        return _tokenizer_instance

    vocab_path = config.VOCAB_PATH  # Path to vocab.json from config.
    merges_path = config.MERGES_PATH # Path to merges.txt from config.

    # Check if the required tokenizer files exist.
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError(
            f"Tokenizer vocabulary file ('{vocab_path}') or merges file ('{merges_path}') not found. "
            f"Please ensure the tokenizer is trained first (e.g., by running train.py, which calls train_tokenizer)."
        )

    print(f"Loading tokenizer from vocab: {vocab_path}, merges: {merges_path}")
    # Load the tokenizer from the vocabulary and merge rules files.
    tokenizer = ByteLevelBPETokenizer(
        vocab=vocab_path,
        merges=merges_path,
    )

    # Configure post-processing using BertProcessing.
    # This automatically adds START (like [CLS]) and END (like [SEP]) tokens to sequences.
    # It requires the string representation and token ID for these special tokens.
    try:
        start_token_str = config.START_TOKEN
        start_token_id = tokenizer.token_to_id(start_token_str)
        end_token_str = config.END_TOKEN
        end_token_id = tokenizer.token_to_id(end_token_str)

        if start_token_id is None or end_token_id is None:
            raise ValueError("START_TOKEN or END_TOKEN not found in tokenizer vocabulary after loading.")

        tokenizer._tokenizer.post_processor = BertProcessing(
            sep=(end_token_str, end_token_id),    # Defines the END token and its ID.
            cls=(start_token_str, start_token_id), # Defines the START token and its ID.
        )
    except Exception as e:
        print(f"Warning: Could not set BertProcessing for START/END tokens: {e}. Special tokens might not be added automatically by encode().")

    # Enable padding to `config.MAX_SEQ_LEN` using the PAD_TOKEN_ID.
    # This ensures all sequences in a batch have the same length.
    try:
        pad_token_id = tokenizer.token_to_id(config.PAD_TOKEN)
        if pad_token_id is None:
            raise ValueError(f"PAD_TOKEN '{config.PAD_TOKEN}' not found in tokenizer vocabulary.")
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token=config.PAD_TOKEN, length=config.MAX_SEQ_LEN)
    except Exception as e:
        print(f"Warning: Could not enable padding for tokenizer: {e}. Manual padding might be required.")

    # Enable truncation to `config.MAX_SEQ_LEN`.
    # This ensures sequences longer than max_length are cut.
    try:
        tokenizer.enable_truncation(max_length=config.MAX_SEQ_LEN)
    except Exception as e:
        print(f"Warning: Could not enable truncation for tokenizer: {e}")

    print("Tokenizer loaded and configured (post-processing, padding, truncation).")
    _tokenizer_instance = tokenizer # Cache the loaded instance.
    return tokenizer_instance

# --- Helper functions to use the global tokenizer instance --- 
# These provide a simpler interface and ensure the tokenizer is loaded.

def encode_text(text: str) -> List[int]:
    """Encodes a text string into a list of token IDs using the global tokenizer.
       Special tokens (START/END) are added based on BertProcessing configuration.
    """
    tokenizer = get_tokenizer() # Ensures tokenizer is loaded and configured.
    # The `encode` method of the configured tokenizer handles special tokens, padding, and truncation.
    return tokenizer.encode(text).ids # Return only the token IDs.

def decode_ids(token_ids: List[int], skip_special_tokens: bool = True) -> str:
    """Decodes a list of token IDs back into a text string using the global tokenizer."""
    tokenizer = get_tokenizer()
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

def get_tokenizer_vocab_size() -> int:
    """Gets the vocabulary size from the loaded global tokenizer instance."""
    tokenizer = get_tokenizer()
    return tokenizer.get_vocab_size()

def get_token_id(token: str) -> int:
    """Gets the integer ID for a specific token string from the global tokenizer."""
    tokenizer = get_tokenizer()
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        # Fallback to UNK_TOKEN_ID if the token is not in the vocabulary.
        # This behavior depends on whether an explicit UNK token ID is more desirable than None.
        print(f"Warning: Token '{token}' not found in vocabulary. Returning UNK_TOKEN_ID if available, else None.")
        unk_id = tokenizer.token_to_id(config.UNK_TOKEN)
        return unk_id if unk_id is not None else None # Or raise error
    return token_id

# Example Usage: This block runs if the script is executed directly (e.g., `python tokenizer.py`).
if __name__ == '__main__':
    print("--- Testing Tokenizer Functionality ---")

    # --- Setup for Dummy Test ---
    # Create dummy captions for testing tokenizer training.
    dummy_captions_for_training = [
        "A black cat sat on a mat.",
        "Another cat, this one white, sat on the same mat.",
        "Why do cats like mats so much? It is a mystery.",
        "Maybe the mat is comfortable for the cat.",
        "The cat sleeps on the mat all day."
    ]
    dummy_target_vocab_size = 50 # A small vocab size for quick testing.
    # Define paths for dummy tokenizer files (will be created in the current directory).
    # These override paths from config.py for this local test.
    _tokenizer_instance = None # Reset global instance for testing
    config.OUTPUT_DIR = "." # Save dummy files in the current directory.
    config.VOCAB_PATH = os.path.join(config.OUTPUT_DIR, "test_dummy_vocab.json")
    config.MERGES_PATH = os.path.join(config.OUTPUT_DIR, "test_dummy_merges.txt")
    config.MAX_SEQ_LEN = 20 # Max sequence length for testing padding/truncation.
    # Ensure special token strings are defined in config for the test
    # (they should be already, but good to be mindful of dependencies)

    # --- Test Tokenizer Training ---
    print("\n--- 1. Testing Tokenizer Training ---")
    # Clean up any pre-existing dummy files from previous runs.
    if os.path.exists(config.VOCAB_PATH): os.remove(config.VOCAB_PATH)
    if os.path.exists(config.MERGES_PATH): os.remove(config.MERGES_PATH)

    # Train a new tokenizer with the dummy data.
    trained_tokenizer_instance = train_tokenizer(
        iter(dummy_captions_for_training), # Pass iterator.
        vocab_size=dummy_target_vocab_size,
        vocab_path=config.VOCAB_PATH,
        merges_path=config.MERGES_PATH
    )
    print(f"Actual trained vocab size: {trained_tokenizer_instance.get_vocab_size()} (Target was {dummy_target_vocab_size})")

    # --- Test Loading the Trained Tokenizer ---
    print("\n--- 2. Testing Loading the Trained Tokenizer ---")
    # Force reload by resetting the global instance.
    _tokenizer_instance = None 
    loaded_tokenizer_instance = get_tokenizer() # This should load from the dummy files.

    print(f"Loaded tokenizer effective vocab size: {get_tokenizer_vocab_size()}")
    print(f"PAD token: '{config.PAD_TOKEN}' -> ID: {get_token_id(config.PAD_TOKEN)}")
    print(f"UNK token: '{config.UNK_TOKEN}' -> ID: {get_token_id(config.UNK_TOKEN)}")
    print(f"START token: '{config.START_TOKEN}' -> ID: {get_token_id(config.START_TOKEN)}")
    print(f"END token: '{config.END_TOKEN}' -> ID: {get_token_id(config.END_TOKEN)}")

    # --- Test Encoding and Decoding --- 
    print("\n--- 3. Testing Encoding and Decoding with Loaded Tokenizer ---")
    test_sentence = "A white cat sleeps on a comfortable mat."
    
    # Use the tokenizer's own .encode() method to get the Encoding object
    # This object contains IDs, tokens, attention mask, etc., after post-processing.
    encoding_result = loaded_tokenizer_instance.encode(test_sentence)
    encoded_ids_from_result = encoding_result.ids
    encoded_tokens_from_result = encoding_result.tokens
    attention_mask_from_result = encoding_result.attention_mask

    print(f"Original Text: '{test_sentence}'")
    print(f"Encoded Tokens (from Encoding object): {encoded_tokens_from_result}")
    print(f"Encoded IDs (from Encoding object):    {encoded_ids_from_result}")
    print(f"Attention Mask (from Encoding object): {attention_mask_from_result}")
    print(f"Length of IDs: {len(encoded_ids_from_result)} (should be <= MAX_SEQ_LEN={config.MAX_SEQ_LEN})")

    # Test decoding helper function.
    decoded_text_no_skip = decode_ids(encoded_ids_from_result, skip_special_tokens=False)
    print(f"Decoded Text (special tokens included): '{decoded_text_no_skip}'")

    decoded_text_with_skip = decode_ids(encoded_ids_from_result, skip_special_tokens=True)
    print(f"Decoded Text (special tokens skipped): '{decoded_text_with_skip}'")
    
    # Test a longer sentence to observe truncation
    long_sentence = "This is a very long sentence that is definitely going to be longer than the max sequence length specified for testing truncation."
    long_encoding_result = loaded_tokenizer_instance.encode(long_sentence)
    print(f"\nOriginal Long Text: '{long_sentence[:50]}...'")
    print(f"Encoded Long Tokens: {long_encoding_result.tokens}")
    print(f"Encoded Long IDs:    {long_encoding_result.ids}")
    print(f"Length of Long IDs: {len(long_encoding_result.ids)} (should be MAX_SEQ_LEN={config.MAX_SEQ_LEN})")


    # --- Clean up dummy files created during the test ---
    print("\n--- 4. Cleaning Up Dummy Files ---")
    if os.path.exists(config.VOCAB_PATH): os.remove(config.VOCAB_PATH)
    if os.path.exists(config.MERGES_PATH): os.remove(config.MERGES_PATH)
    print(f"Cleaned up dummy files: {config.VOCAB_PATH}, {config.MERGES_PATH}")
    print("--- Tokenizer Test Finished ---")