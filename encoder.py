"""
Handles loading the pre-trained image encoder (ViT/CLIP) and extracting features.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor, CLIPVisionModel, CLIPProcessor
from PIL import Image
import config

# Global variables to cache the model and processor so they aren't reloaded unnecessarily
_encoder_model = None
_processor = None

def _load_model_and_processor():
    """Loads the specified pre-trained vision model and its processor."""
    global _encoder_model, _processor

    if _encoder_model is not None and _processor is not None:
        return _encoder_model, _processor

    model_name = config.ENCODER_MODEL_NAME
    print(f"Loading encoder model: {model_name}...")

    if "clip" in model_name.lower():
        _encoder_model = CLIPVisionModel.from_pretrained(model_name).to(config.DEVICE)
        _processor = CLIPProcessor.from_pretrained(model_name)
    elif "vit" in model_name.lower():
        _encoder_model = ViTModel.from_pretrained(model_name).to(config.DEVICE)
        _processor = ViTImageProcessor.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported encoder model type in name: {model_name}. Use 'vit' or 'clip'.")

    # Freeze the encoder parameters
    for param in _encoder_model.parameters():
        param.requires_grad = False

    _encoder_model.eval() # Set to evaluation mode
    print("Encoder model loaded and frozen.")
    return _encoder_model, _processor

def encode_image(image: Image.Image) -> torch.Tensor:
    """Encodes a single PIL image using the pre-trained vision model.

    Args:
        image: A PIL Image object.

    Returns:
        A tensor representing the image features (e.g., last hidden state).
        Shape typically (1, sequence_length, hidden_size) for ViT/CLIP Vision.
    """
    encoder_model, processor = _load_model_and_processor()

    # Preprocess the image
    # For CLIP, processor expects images=[image], return_tensors="pt"
    # For ViT, processor expects images=image, return_tensors="pt"
    if isinstance(processor, CLIPProcessor):
        inputs = processor(images=[image], return_tensors="pt", padding=True).to(config.DEVICE)
    elif isinstance(processor, ViTImageProcessor):
        inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
    else:
         # Fallback or error for unexpected processor type
         raise TypeError(f"Unsupported processor type: {type(processor)}")

    # Get encoder outputs
    with torch.no_grad(): # Ensure no gradients are calculated for the frozen encoder
        outputs = encoder_model(**inputs)

    # Extract the desired features
    # For both ViTModel and CLIPVisionModel, last_hidden_state is usually what we want.
    # It contains the sequence of hidden states for each patch/token.
    image_features = outputs.last_hidden_state

    return image_features # Shape: (batch_size=1, seq_len, hidden_dim)

def get_encoder_output_dim() -> int:
    """Returns the hidden dimension size of the loaded encoder model."""
    encoder_model, _ = _load_model_and_processor()
    # Access the config of the loaded model to get the hidden size
    return encoder_model.config.hidden_size

# Example Usage (optional, for testing)
if __name__ == '__main__':
    # Create a dummy black image
    dummy_image = Image.new('RGB', (224, 224))

    print(f"Using device: {config.DEVICE}")

    # Encode the image
    features = encode_image(dummy_image)
    print(f"Encoder output shape: {features.shape}") # E.g., (1, 197, 768) for ViT-Base

    # Get output dimension
    hidden_dim = get_encoder_output_dim()
    print(f"Encoder hidden dimension: {hidden_dim}")

    # Try loading CLIP
    original_model_name = config.ENCODER_MODEL_NAME
    config.ENCODER_MODEL_NAME = "openai/clip-vit-base-patch32"
    _encoder_model = None # Reset cache
    _processor = None
    features_clip = encode_image(dummy_image)
    print(f"CLIP Encoder output shape: {features_clip.shape}") # E.g., (1, 50, 768) for CLIP ViT-B/32
    hidden_dim_clip = get_encoder_output_dim()
    print(f"CLIP Encoder hidden dimension: {hidden_dim_clip}")

    # Restore original config if needed
    config.ENCODER_MODEL_NAME = original_model_name 