"""
Script for generating text from an image using a trained model checkpoint.
"""

import torch
from PIL import Image
import argparse
import config
from model import ImageToTextModel
from tokenizer import get_tokenizer
import os

def generate_caption(image_path: str, checkpoint_path: str, device: str) -> str:
    """Generates a caption for a single image.

    Args:
        image_path: Path to the input image file.
        checkpoint_path: Path to the trained model checkpoint (.pth file).
        device: The device to run inference on ('cuda' or 'cpu').

    Returns:
        The generated caption string.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading tokenizer...")
    tokenizer = get_tokenizer()
    effective_vocab_size = tokenizer.get_vocab_size()
    start_token_id = config.START_TOKEN_ID
    end_token_id = config.END_TOKEN_ID

    print(f"Loading model from {checkpoint_path}...")
    # Initialize model with the same architecture as during training
    model = ImageToTextModel(
        decoder_vocab_size=effective_vocab_size,
        decoder_embed_dim=config.DECODER_EMBED_DIM,
        decoder_heads=config.DECODER_HEADS,
        decoder_layers=config.DECODER_LAYERS,
        decoder_ff_dim=config.DECODER_FF_DIM,
        decoder_max_seq_len=config.MAX_SEQ_LEN,
        decoder_dropout=config.DECODER_DROPOUT, # Dropout is typically disabled by model.eval()
        decoder_pad_idx=config.PAD_TOKEN_ID
    ).to(device)

    # Load the saved state dictionary
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # Set to evaluation mode

    print(f"Loading and preprocessing image: {image_path}...")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return "Error: Could not load image."

    print("Generating text...")
    # The model.generate method uses the frozen encoder internally via encoder.encode_image
    generated_ids = model.generate(
        image,
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        max_len=config.MAX_SEQ_LEN,
        method='greedy' # Or 'beam'
    )

    print(f"Generated token IDs: {generated_ids}")

    # Decode the generated IDs into text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text for an image using a trained model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pth file).")
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Using device: {device}")

    # Generate the caption
    caption = generate_caption(args.image_path, args.checkpoint, device)

    print("\n---")
    print(f"Image: {args.image_path}")
    print(f"Generated Text: {caption}")
    print("---") 