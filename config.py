"""
Configuration settings for the Multimodal Image Transformer project.
"""

import torch
import os

# --- General Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# --- Data Settings ---
# Adjust these paths based on your dataset structure
DATA_DIR = "../assets/multimodal_image_transformer/" # Root directory for prepared dataset assets
IMAGE_DIR = DATA_DIR + "images"  # Path where prepare_dataset.py saves images
CAPTIONS_FILE = DATA_DIR + "captions.json"  # Path to your captions file (JSON format expected)
OUTPUT_DIR = DATA_DIR  # Directory to save checkpoints, tokenizer, etc.

# --- Model Settings ---
# Choose the pre-trained image encoder model
# Options: e.g., "google/vit-base-patch16-224-in21k", "openai/clip-vit-base-patch32"
ENCODER_MODEL_NAME = "google/vit-base-patch16-224-in21k"
#ENCODER_MODEL_NAME = "openai/clip-vit-base-patch32"

# Decoder hyperparameters
VOCAB_SIZE = 10000       # Size of the output vocabulary (adjust based on tokenizer)
MAX_SEQ_LEN = 100       # Maximum length of the generated text sequence
DECODER_EMBED_DIM = 512   # Embedding dimension for the decoder
DECODER_LAYERS = 6        # Number of Transformer Decoder layers
DECODER_HEADS = 8         # Number of attention heads in the decoder
DECODER_FF_DIM = 2048     # Dimension of the feed-forward network in the decoder
DECODER_DROPOUT = 0.1     # Dropout rate for the decoder
PROJECTION_DIM = 512      # Dimension for projecting encoder features (if used, often same as DECODER_EMBED_DIM)

# --- Training Settings ---
BATCH_SIZE = 32
NUM_EPOCHS = 20 # Renamed from EPOCHS for clarity
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_VALUE = 5.0 # Optional gradient clipping
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.98
ADAM_EPS = 1e-9
WARMUP_STEPS = 0 # Number of warmup steps for scheduler (0 for no warmup)

LOG_INTERVAL = 50 # Log training loss every N batches
VALIDATION_INTERVAL = 1 # Run validation every N epochs
CHECKPOINT_PREFIX = "model_checkpoint" # Prefix for saved checkpoint filenames

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

# Corrected Tokenizer paths relative to the assumed parent directory
VOCAB_PATH = "/workspace/assets/multimodal_image_transformer/vocab.json" # Where the tokenizer vocab is saved/loaded
MERGES_PATH = "/workspace/assets/multimodal_image_transformer/merges.txt" # Where the tokenizer merges are saved/loaded

# --- Wandb Configuration ---
WANDB_PROJECT = "multimodal-image-transformer" # Your project name
WANDB_ENTITY = None # Your wandb username or team (optional, defaults to your default entity)
WANDB_RUN_NAME = None # Optional: A specific name for this run (e.g., "vit-base-lr-1e-4")

# --- Inference Settings ---
BEAM_SIZE = 3 # Optional: for beam search decoding 

# --- Hugging Face Hub Settings ---
HF_REPO_ID = "wazzuck/multimodal_image_transformer" # Repository ID on Hugging Face Hub 
