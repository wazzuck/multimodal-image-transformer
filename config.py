"""
Configuration settings for the Multimodal Image Transformer project.
"""

import torch
import os

# --- General Settings ---
# Determine the computation device: use CUDA (GPU) if available, otherwise use CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Seed for random number generators to ensure reproducibility.
RANDOM_SEED = 42

# --- Data Settings ---
# Base directory for all dataset-related assets.
# Adjust this path to point to where your prepared dataset (images, captions) is or will be stored.
DATA_DIR = "../assets/multimodal_image_transformer/" # Root directory for prepared dataset assets
# Directory where image files are stored. Used by prepare_dataset.py and the data loader.
IMAGE_DIR = DATA_DIR + "images"  # Path where prepare_dataset.py saves images
# Path to the JSON file containing image captions. Expected format: {"image_name.jpg": ["caption1", "caption2"]}.
CAPTIONS_FILE = DATA_DIR + "captions.json"  # Path to your captions file (JSON format expected)
# Directory to save training outputs like model checkpoints and tokenizer files.
# Currently set to DATA_DIR, meaning outputs will be saved alongside dataset assets.
OUTPUT_DIR = DATA_DIR  # Directory to save checkpoints, tokenizer, etc.

# Ratio of the dataset to be used for training (e.g., 0.8 for 80% train, 20% validation).
TRAIN_SPLIT_RATIO = 0.9

# Number of worker processes for data loading. 
# 0 means data will be loaded in the main process.
# More workers can speed up data loading if I/O or CPU preprocessing is a bottleneck.
NUM_WORKERS = 2

# Whether to use pinned memory for DataLoader. Speeds up CPU to GPU data transfer if DEVICE is CUDA.
PIN_MEMORY = True if DEVICE == "cuda" else False

# --- Model Settings ---
# Specifies the pre-trained image encoder model to be used from Hugging Face Transformers.
# Different encoders capture image features differently and may affect performance.
# Examples: "google/vit-base-patch16-224-in21k" (Vision Transformer), "openai/clip-vit-base-patch32" (CLIP's ViT).
#ENCODER_MODEL_NAME = "google/vit-base-patch16-224-in21k"
#ENCODER_MODEL_NAME = "Salesforce/blip-image-captioning-base"
ENCODER_MODEL_NAME = "openai/clip-vit-base-patch32" # Using CLIP ViT model

# Specifies the image processor to be used, corresponding to the encoder model.
#IMAGE_PROCESSOR_NAME = "Salesforce/blip-image-captioning-base"
IMAGE_PROCESSOR_NAME = "openai/clip-vit-base-patch32" # Using CLIP ViT processor

# Specifies how image transformations are applied.
# 'hf_processor': Use the transformations defined by the Hugging Face image processor.
# 'custom': Implies custom transformation pipeline (not fully implemented in this snippet context).
IMG_TRANSFORM_MODE = "hf_processor"

# Decoder hyperparameters - These define the architecture of the Transformer decoder.
# Size of the vocabulary the decoder will predict over. This should match the tokenizer's vocabulary size.
VOCAB_SIZE = 10000       # Size of the output vocabulary (adjust based on tokenizer)
# Maximum length of token sequences the decoder can handle (including special tokens).
MAX_SEQ_LEN = 100       # Maximum length of the generated text sequence
# Dimensionality of the token embeddings and the hidden states within the decoder.
DECODER_EMBED_DIM = 512   # Embedding dimension for the decoder
# Number of stacked decoder layers. More layers can capture more complex patterns but increase model size.
DECODER_LAYERS = 6        # Number of Transformer Decoder layers
# Number of attention heads in the multi-head attention mechanisms of the decoder.
DECODER_HEADS = 8         # Number of attention heads in the decoder
# Dimensionality of the point-wise feed-forward networks within each decoder layer.
DECODER_FF_DIM = 2048     # Dimension of the feed-forward network in the decoder
# Dropout rate applied in the decoder layers to prevent overfitting.
DECODER_DROPOUT = 0.1     # Dropout rate for the decoder
# Dimension of the linear layer that projects image encoder features to match decoder embedding dimension.
# Often set to be the same as DECODER_EMBED_DIM for direct compatibility.
PROJECTION_DIM = 512      # Dimension for projecting encoder features (if used, often same as DECODER_EMBED_DIM)

# --- Training Settings ---
# Number of samples processed in one iteration (forward/backward pass).
BATCH_SIZE = 32
# Total number of times the training loop will iterate over the entire dataset.
NUM_EPOCHS = 20 # Renamed from EPOCHS for clarity
# Initial learning rate for the optimizer.
LEARNING_RATE = 1e-4
# Weight decay (L2 regularization) parameter for the optimizer to prevent overfitting.
WEIGHT_DECAY = 1e-5
# Maximum value for gradient clipping to prevent exploding gradients during training. (0.0 means no clipping).
GRAD_CLIP_VALUE = 5.0 # Optional gradient clipping
# Beta1 parameter for the Adam/AdamW optimizer (exponential decay rate for the first moment estimates).
ADAM_BETA1 = 0.9
# Beta2 parameter for the Adam/AdamW optimizer (exponential decay rate for the second moment estimates).
ADAM_BETA2 = 0.98
# Epsilon parameter for the Adam/AdamW optimizer (term added to the denominator to improve numerical stability).
ADAM_EPS = 1e-9
# Number of initial training steps during which the learning rate is linearly increased from 0 to LEARNING_RATE.
# Set to 0 for no warmup phase.
WARMUP_STEPS = 0 # Number of warmup steps for scheduler (0 for no warmup)

# Frequency (in terms of batches) at which training loss and other metrics are logged.
LOG_INTERVAL = 50 # Log training loss every N batches
# Frequency (in terms of epochs) at which model validation is performed on the validation set.
VALIDATION_INTERVAL = 1 # Run validation every N epochs
# Prefix for filenames when saving model checkpoints during training.
CHECKPOINT_PREFIX = "model_checkpoint" # Prefix for saved checkpoint filenames

# Path to a specific checkpoint file (.pt) to resume training from.
# Set to None or an empty string to start training from scratch.
RESUME_CHECKPOINT_PATH = "../assets/multimodal_image_transformer/model_checkpoint_Salesforce_blip-image-captioning-base_epoch_2_val_loss_2.7088.safetensors"

# --- Tokenizer Settings ---
# Special tokens used by the tokenizer and model.
# These tokens have specific roles (padding, start/end of sequence, unknown words).
PAD_TOKEN = "<PAD>"      # Token for padding sequences to the same length.
START_TOKEN = "<START>"  # Token indicating the beginning of a caption.
END_TOKEN = "<END>"      # Token indicating the end of a caption.
UNK_TOKEN = "<UNK>"      # Token for words not present in the vocabulary.

# Corresponding IDs for the special tokens. These must match the tokenizer's vocabulary.
# It is crucial that these IDs are consistent between tokenizer training and model usage.
PAD_TOKEN_ID = 0
START_TOKEN_ID = 1
END_TOKEN_ID = 2
UNK_TOKEN_ID = 3

# Paths for saving/loading tokenizer files (vocabulary and merges for BPE).
# These are relative to OUTPUT_DIR (which is currently DATA_DIR).
VOCAB_PATH = OUTPUT_DIR + "vocab.json" # Path where the tokenizer vocabulary (token to ID map) is saved/loaded.
MERGES_PATH = OUTPUT_DIR + "merges.txt" # Path where BPE merges are saved/loaded (if using BPE tokenizer).

# --- Wandb Configuration (Weights & Biases for experiment tracking) ---
# Name of the project in Weights & Biases where runs will be logged.
WANDB_PROJECT = "multimodal-image-transformer" # Your project name
# Your Weights & Biases username or team name. Set to None to use default entity.
WANDB_ENTITY = None # Your wandb username or team (optional, defaults to your default entity)
# A specific name for the current training run in Wandb. If None, a random name is generated.
WANDB_RUN_NAME = None # Optional: A specific name for this run (e.g., "vit-base-lr-1e-4")

# --- Inference Settings ---
# Beam size for beam search decoding during inference. A larger beam size can lead to better captions but is slower.
# Only used if 'beam' search is selected as the generation method.
BEAM_SIZE = 3 # Optional: for beam search decoding

# --- Hugging Face Hub Settings ---
# Repository ID on the Hugging Face Model Hub if you plan to upload your trained models.
# Format: "username/repository_name" or "organization_name/repository_name".
HF_REPO_ID = "wazzuck/multimodal_image_transformer" # Repository ID on Hugging Face Hub

# Whether to upload the best performing model checkpoints (based on validation loss) to the Hugging Face Hub.
HF_UPLOAD_BEST_CHECKPOINTS = True
