"""
Handles loading pre-trained image encoder models (e.g., ViT, CLIP)
from Hugging Face Transformers, and extracting image features.
This module provides a consistent interface for different vision models.
"""

import torch
import torch.nn as nn # Not strictly used here, but common for model files.
from transformers import ViTModel, ViTImageProcessor, CLIPVisionModel, CLIPProcessor # Hugging Face models and processors.
from PIL import Image # For handling image objects.
import config # Project configuration file for model names and device settings.

# --- Global Caching for Model and Processor ---
# These global variables are used to cache the loaded encoder model and its corresponding processor.
# This prevents reloading them from disk/Hugging Face Hub every time they are needed,
# which can be time-consuming.
_encoder_model = None
_processor = None

def _load_model_and_processor():
    """Loads the pre-trained vision model and its associated processor specified in `config.ENCODER_MODEL_NAME`.

    The function caches the loaded model and processor in global variables `_encoder_model`
    and `_processor` to avoid redundant loading. It also freezes the model parameters
    and sets the model to evaluation mode, as it's used as a fixed feature extractor.

    Returns:
        A tuple containing the loaded and frozen encoder model and its processor.

    Raises:
        ValueError: If the `config.ENCODER_MODEL_NAME` specifies an unsupported model type.
    """
    global _encoder_model, _processor # Declare intent to modify global variables.

    # If model and processor are already loaded and cached, return them directly.
    if _encoder_model is not None and _processor is not None:
        return _encoder_model, _processor

    model_name = config.ENCODER_MODEL_NAME # Get the specified model name from config.
    print(f"Loading image encoder model: {model_name}...")

    # --- Model and Processor Loading based on Name ---
    # Check if the model name indicates a CLIP model.
    if "clip" in model_name.lower():
        _encoder_model = CLIPVisionModel.from_pretrained(model_name).to(config.DEVICE)
        _processor = CLIPProcessor.from_pretrained(model_name)
    # Check if the model name indicates a ViT model (and not CLIP, as CLIP also contains "vit").
    elif "vit" in model_name.lower():
        _encoder_model = ViTModel.from_pretrained(model_name).to(config.DEVICE)
        _processor = ViTImageProcessor.from_pretrained(model_name)
    else:
        # If the model name doesn't match supported types, raise an error.
        raise ValueError(f"Unsupported encoder model type in name: '{model_name}'. Supported types contain 'vit' or 'clip'.")

    # --- Freeze Encoder Parameters ---
    # Iterate through all parameters of the loaded encoder model and set `requires_grad` to False.
    # This freezes the weights, so they are not updated during the training of the decoder.
    for param in _encoder_model.parameters():
        param.requires_grad = False

    _encoder_model.eval() # Set the encoder model to evaluation mode (disables dropout, etc.).
    print(f"Image encoder model '{model_name}' loaded, frozen, and set to evaluation mode.")
    return _encoder_model, _processor

def encode_image(image: Image.Image) -> torch.Tensor:
    """Encodes a single PIL image into a feature tensor using the loaded pre-trained vision model.

    Args:
        image: A PIL Image object to be encoded.

    Returns:
        A PyTorch tensor representing the extracted image features.
        The shape is typically (1, sequence_length, hidden_size) for models like ViT and CLIP Vision,
        where sequence_length corresponds to the number of patches (plus CLS token if present).

    Raises:
        TypeError: If an unsupported processor type is encountered (should not happen with current logic).
    """
    # Load (or get from cache) the encoder model and its processor.
    encoder_model, processor = _load_model_and_processor()

    # --- Image Preprocessing ---
    # Preprocess the image using the model-specific processor.
    # Processors handle resizing, normalization, and formatting the image into the tensor format
    # expected by the model.
    if isinstance(processor, CLIPProcessor):
        # CLIPProcessor typically expects a list of images.
        inputs = processor(images=[image], return_tensors="pt", padding=True).to(config.DEVICE)
    elif isinstance(processor, ViTImageProcessor):
        # ViTImageProcessor can often take a single image directly.
        inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
    else:
         # Fallback error for an unexpected processor type (defensive coding).
         raise TypeError(f"Unsupported image processor type: {type(processor)}. Expected CLIPProcessor or ViTImageProcessor.")

    # --- Feature Extraction ---
    # Pass the preprocessed inputs to the encoder model.
    # `torch.no_grad()` ensures that no gradients are computed during this forward pass,
    # as the encoder is frozen.
    with torch.no_grad():
        outputs = encoder_model(**inputs) # Use dictionary unpacking for inputs.

    # Extract the desired features from the model's output.
    # For both ViTModel and CLIPVisionModel, `last_hidden_state` is commonly used.
    # This tensor contains the sequence of hidden states from the last layer of the model,
    # corresponding to each input patch (or token, in NLP terms).
    image_features = outputs.last_hidden_state
    # `image_features` shape: (batch_size=1, sequence_length, hidden_dimension)

    return image_features

def get_encoder_output_dim() -> int:
    """Returns the hidden dimension size of the currently loaded encoder model.

    This is useful for configuring downstream layers (e.g., projection layers or the decoder)
    to match the encoder's output feature dimension.

    Returns:
        The integer value of the hidden dimension size.
    """
    # Load (or get from cache) the encoder model.
    encoder_model, _ = _load_model_and_processor()
    # Access the model's configuration object (`.config`) to get its `hidden_size` attribute.
    return encoder_model.config.hidden_size

# --- Example Usage (for testing the module directly) --- #
if __name__ == '__main__':
    print("--- Testing Image Encoder --- ")
    # Create a dummy black PIL Image for testing purposes.
    # The size (224, 224) is a common input size for many ViT models.
    dummy_image = Image.new('RGB', (224, 224), color='black')

    print(f"Using device for encoding: {config.DEVICE}")

    # --- Test with default encoder (from config.py) ---
    print("\n--- Testing with default model specified in config.py ---")
    # Encode the dummy image using the default model specified in config.ENCODER_MODEL_NAME.
    features_default = encode_image(dummy_image)
    print(f"Default encoder output features shape: {features_default.shape}")
    # Example: (1, 197, 768) for ViT-Base-Patch16-224 (196 patches + 1 CLS token, 768 hidden dim).

    # Get the output dimension of the default encoder.
    hidden_dim_default = get_encoder_output_dim()
    print(f"Default encoder hidden dimension: {hidden_dim_default}")

    # --- Test with a CLIP model (temporarily overriding config) ---
    print("\n--- Testing with CLIP model (openai/clip-vit-base-patch32) ---")
    original_model_name_backup = config.ENCODER_MODEL_NAME # Backup original model name.
    config.ENCODER_MODEL_NAME = "openai/clip-vit-base-patch32" # Temporarily set to a CLIP model.
    
    # Reset cached model and processor to force reloading the new CLIP model.
    _encoder_model = None 
    _processor = None
    
    features_clip = encode_image(dummy_image)
    print(f"CLIP encoder output features shape: {features_clip.shape}")
    # Example: (1, 50, 768) for CLIP ViT-B/32 (49 patches + 1 CLS token for 224x224 input, 768 hidden dim).
    hidden_dim_clip = get_encoder_output_dim()
    print(f"CLIP encoder hidden dimension: {hidden_dim_clip}")

    # Restore original config model name after the test.
    config.ENCODER_MODEL_NAME = original_model_name_backup
    _encoder_model = None # Clear cache again so next call uses original config.
    _processor = None
    print(f"\nRestored config.ENCODER_MODEL_NAME to: {config.ENCODER_MODEL_NAME}")
    print("--- Image Encoder Test Finished ---") 