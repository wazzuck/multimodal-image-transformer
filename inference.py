"""Kenton Kwok
ch
Script for generating text from an image using a trained model checkpoint.
"""

import torch
from PIL import Image
import argparse
import config
from model import ImageToTextModel
from tokenizer import get_tokenizer
import os
from safetensors.torch import load_file # For loading model weights from .safetensors files

DEFAULT_CHECKPOINT_PATH = "../assets/multimodal_image_transformer/model_checkpoint_openai_clip-vit-base-patch32_epoch_10_val_loss_2.4654.safetensors"

def generate_caption(image_path: str, device: str, checkpoint_path: str) -> str:
    """Generates a caption for a single image using a predefined model checkpoint.

    Args:
        image_path: Path to the input image file.
        device: The device to run inference on ('cuda' or 'cpu').
        checkpoint_path: Path to the .safetensors model checkpoint file.

    Returns:
        The generated caption string.
    """
    # Define the fixed path to the model checkpoint.
    # This path is hardcoded for simplicity in this inference script.
    # In a more general setup, this might be a configurable parameter.
    # checkpoint_path = "../assets/multimodal_image_transformer/model_checkpoint_openai_clip-vit-base-patch32_epoch_10_val_loss_2.4654.safetensors"

    # Validate that the image file exists.
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    # Validate that the checkpoint file exists.
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading tokenizer...")
    # Load the tokenizer, which is responsible for converting text to token IDs and vice-versa.
    # The tokenizer should be the same as the one used during training.
    tokenizer = get_tokenizer()
    # Get the effective vocabulary size from the tokenizer.
    effective_vocab_size = tokenizer.get_vocab_size()
    # Get special token IDs from the config, used to control the generation process.
    start_token_id = config.START_TOKEN_ID # Marks the beginning of a sequence.
    end_token_id = config.END_TOKEN_ID   # Marks the end of a sequence.

    print(f"Loading model from {checkpoint_path}...")
    # Initialize the model with the same architecture and hyperparameters as used during training.
    # These parameters are typically stored in a configuration file (config.py here).
    model = ImageToTextModel(
        decoder_vocab_size=effective_vocab_size,    # Size of the vocabulary for the decoder.
        decoder_embed_dim=config.DECODER_EMBED_DIM,  # Dimensionality of token embeddings.
        decoder_heads=config.DECODER_HEADS,          # Number of attention heads in the decoder.
        decoder_layers=config.DECODER_LAYERS,        # Number of decoder layers.
        decoder_ff_dim=config.DECODER_FF_DIM,        # Dimensionality of the feed-forward networks.
        decoder_max_seq_len=config.MAX_SEQ_LEN,    # Maximum sequence length the decoder can handle.
        decoder_dropout=config.DECODER_DROPOUT,    # Dropout rate (usually disabled during eval by model.eval()).
        decoder_pad_idx=config.PAD_TOKEN_ID        # Token ID used for padding sequences.
    ).to(device) # Move the model to the specified device (e.g., 'cuda' or 'cpu').

    # Load the saved model weights (state dictionary) from the .safetensors file.
    # 'device=device' ensures the loaded tensors are on the correct device.
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict) # Apply the loaded weights to the model.
    model.eval() # Set the model to evaluation mode. This disables dropout and batch normalization updates.

    print(f"Loading and preprocessing image: {image_path}...")
    try:
        # Open the image using Pillow (PIL) and convert it to RGB format.
        # The model's image encoder expects RGB images.
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        # Handle potential errors during image loading (e.g., file not found, corrupted image).
        print(f"Error loading image: {e}")
        return "Error: Could not load image."

    print("Generating text...")
    # Generate a sequence of token IDs based on the input image.
    # The model.generate method internally uses its image encoder to process the image
    # and then its decoder to autoregressively generate the token sequence.
    generated_ids = model.generate(
        image,                          # The input image.
        start_token_id=start_token_id,  # The token ID to start generation with.
        end_token_id=end_token_id,      # The token ID that signifies the end of generation.
        max_len=config.MAX_SEQ_LEN,     # The maximum length of the generated sequence.
        method='greedy'                 # Generation method: 'greedy' selects the highest probability token at each step.
                                        # Other methods like 'beam' search could also be used.
    )

    print(f"Generated token IDs: {generated_ids}")

    # --- Process generated IDs ---
    # Post-process the raw generated token IDs to form a clean caption.
    # Find the first occurrence of the END token ID in the generated sequence.
    try:
        first_end_idx = generated_ids.index(end_token_id)
        # Keep only the token IDs up to (but not including) the first END token.
        # This removes any tokens generated after the intended end of the caption.
        processed_ids = generated_ids[:first_end_idx]
    except ValueError:
        # This case occurs if the END token is not found in generated_ids.
        # This should ideally not happen if the generation loop is correctly implemented
        # to always include or stop at an END token within max_len.
        # As a fallback, use the entire generated sequence.
        processed_ids = generated_ids
    
    # Remove the START token if it's present at the beginning of the processed sequence.
    # This is a common cleanup step as the START token is a generation prompt, not part of the actual caption.
    if processed_ids and processed_ids[0] == start_token_id:
        processed_ids = processed_ids[1:]

    # Decode the processed token IDs back into a human-readable text string.
    # skip_special_tokens=False is used initially because we've manually handled START/END tokens.
    # If the tokenizer has other special tokens that should be skipped, this might be True,
    # or further specific cleaning might be needed.
    generated_text = tokenizer.decode(processed_ids, skip_special_tokens=False)

    # Simple post-processing to remove unknown tokens (<UNK>) if they appear.
    # This replaces instances of the UNK_TOKEN (e.g., "<unk>") with an empty string and strips whitespace.
    generated_text = generated_text.replace(config.UNK_TOKEN, "").strip()
    # Optional: Clean up potential multiple spaces that might result from removing <UNK> tokens or other artifacts.
    # This splits the string by spaces and rejoins with single spaces.
    generated_text = ' '.join(generated_text.split())

    return generated_text

# This block executes if the script is run directly (not imported as a module).
if __name__ == "__main__":
    # Set up an argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(description="Generate text for an image using a trained model.")
    # Add an argument for the image path, making it required.
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the .safetensors model checkpoint file."
    )
    args = parser.parse_args() # Parse the command-line arguments.

    # Determine the device for computation from the config file (e.g., 'cuda' if GPU is available, else 'cpu').
    device = config.DEVICE
    print(f"Using device: {device}")

    # Call the main generation function with the provided image path and configured device.
    caption = generate_caption(args.image_path, device, args.checkpoint_path)

    # Print the results in a formatted way.
    print("\n---")
    print(f"Image: {args.image_path}")
    print(f"Generated Text: {caption}")
    print("---") 
