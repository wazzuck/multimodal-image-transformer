"""
Configuration settings for the Multimodal Image Transformer project.
"""

import torch

# --- General Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# --- Data Settings ---
# Adjust these paths based on your dataset structure
DATA_DIR = "data/" # Directory containing images and caption file
IMAGE_DIR = f"{DATA_DIR}/images/"
CAPTIONS_FILE = f"{DATA_DIR}/captions.json" # Example: JSON file mapping image filenames to captions
OUTPUT_DIR = "outputs/" # Directory to save checkpoints and logs

# --- Model Settings ---
# Choose the pre-trained image encoder model
# Options: e.g., "google/vit-base-patch16-224-in21k", "openai/clip-vit-base-patch32"
ENCODER_MODEL_NAME = "google/vit-base-patch16-224-in21k"

# Decoder hyperparameters
VOCAB_SIZE = 10000       # Size of the output vocabulary (adjust based on tokenizer)
MAX_SEQ_LEN = 100       # Maximum length of the generated text sequence
DECODER_EMBED_DIM = 512   # Embedding dimension for the decoder
DECODER_LAYERS = 6        # Number of Transformer Decoder layers
DECODER_HEADS = 8         # Number of attention heads in the decoder
DECODER_FF_DIM = 2048     # Dimension of the feed-forward network in the decoder
DECODER_DROPOUT = 0.1     # Dropout rate for the decoder

# --- Training Settings ---
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_VALUE = 5.0 # Optional gradient clipping

# --- Tokenizer Settings ---
# Special tokens (adjust based on your tokenizer)
PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"

PAD_TOKEN_ID = 0 # Ensure these match your tokenizer's vocabulary
START_TOKEN_ID = 1
END_TOKEN_ID = 2
UNK_TOKEN_ID = 3

# --- Inference Settings ---
BEAM_SIZE = 3 # Optional: for beam search decoding 