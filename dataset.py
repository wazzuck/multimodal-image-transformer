"""
Dataset loading and preprocessing for image-text pairs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Use transforms from encoder processor later
from PIL import Image
import os
import json
from typing import List, Dict, Any, Tuple
import config
# Import the specific processor type based on config
from transformers import ViTImageProcessor, CLIPProcessor
from tokenizer import get_tokenizer # Use the shared tokenizer instance

# Get the appropriate processor from the encoder module or config
# This avoids direct dependency on encoder internals here
if "clip" in config.ENCODER_MODEL_NAME.lower():
    image_processor = CLIPProcessor.from_pretrained(config.ENCODER_MODEL_NAME)
elif "vit" in config.ENCODER_MODEL_NAME.lower():
    image_processor = ViTImageProcessor.from_pretrained(config.ENCODER_MODEL_NAME)
else:
    raise ValueError(f"Unsupported encoder model type in config: {config.ENCODER_MODEL_NAME}")

class ImageTextDataset(Dataset):
    """PyTorch Dataset for loading image-caption pairs."""
    def __init__(self, image_dir: str, captions_file: str, max_seq_len: int):
        """
        Args:
            image_dir: Directory containing the images.
            captions_file: Path to the JSON file containing captions.
                           Expected format: { "image_filename1": "caption1", ... }
                           or [{ "image_id": "filename1", "caption": "caption1"}, ...]
            max_seq_len: Maximum length for tokenized captions (including special tokens).
        """
        self.image_dir = image_dir
        self.max_seq_len = max_seq_len
        self.tokenizer = get_tokenizer()
        self.image_processor = image_processor # Use the globally loaded processor

        print(f"Loading captions from: {captions_file}")
        try:
            with open(captions_file, 'r', encoding='utf-8') as f:
                captions_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Captions file not found at {captions_file}")
            self.image_paths = []
            self.captions = []
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {captions_file}")
            self.image_paths = []
            self.captions = []
            return

        self.image_paths = []
        self.captions = []

        # Adapt based on the actual JSON structure
        if isinstance(captions_data, dict):
            # Handle { "filename": ["caption1", "caption2", ...] }
            for filename, caption_list in captions_data.items():
                img_path = os.path.join(self.image_dir, filename)
                if os.path.exists(img_path):
                    # Add an entry for each caption in the list
                    for caption in caption_list:
                         if isinstance(caption, str): # Ensure caption is a string
                            self.image_paths.append(img_path)
                            self.captions.append(caption)
                         else:
                            print(f"Warning: Found non-string caption for {filename}: {caption}. Skipping.")
                else:
                    print(f"Warning: Image file not found: {img_path}")

        # Note: The original handling for list format [{ "image_id": "filename", "caption": "caption"}, ...] 
        # is removed as prepare_dataset.py creates the dict format.
        # You could add it back with an elif if needed for other caption file formats.
        # elif isinstance(captions_data, list):
        #     # Handle [{ "image_id": "filename", "caption": "caption"}, ...]
        #     ...

        else:
            print("Error: Captions data is not a dictionary as expected from prepare_dataset.py.")

        if not self.image_paths:
             print("Error: No valid image-caption pairs were loaded. Check paths and caption file format.")

        print(f"Found {len(self.image_paths)} image-caption pairs.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads an image, preprocesses it, tokenizes the caption."""
        img_path = self.image_paths[idx]
        caption = self.captions[idx]

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Skipping.")
            # Return a dummy item or handle differently
            # To make it work with collate_fn, return something with expected keys
            dummy_image = Image.new('RGB', (224, 224)) # Placeholder size
            processed_image = self.image_processor(images=dummy_image, return_tensors="pt")['pixel_values'].squeeze(0)
            dummy_caption_tokens = [config.PAD_TOKEN_ID] * self.max_seq_len
            return {
                "image_path": "error_path",
                "image": processed_image,
                "caption_tokens": torch.tensor(dummy_caption_tokens, dtype=torch.long)
            }

        # Preprocess image using the appropriate processor
        # processor output is a dict, we usually need 'pixel_values'
        # Squeeze(0) removes the batch dimension added by the processor
        processed_image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        # Tokenize caption
        caption_tokens = self.tokenizer.encode(caption, add_special_tokens=True)

        # Pad or truncate caption tokens
        padded_tokens = self._pad_or_truncate(caption_tokens)

        return {
            "image_path": img_path,
            "image": processed_image, # This is the processed tensor for the model
            "caption_tokens": torch.tensor(padded_tokens, dtype=torch.long)
        }

    def _pad_or_truncate(self, tokens: List[int]) -> List[int]:
        """Pads or truncates a token list to max_seq_len."""
        if len(tokens) < self.max_seq_len:
            # Pad with PAD token ID
            padded = tokens + [config.PAD_TOKEN_ID] * (self.max_seq_len - len(tokens))
        else:
            # Truncate, ensuring END token is kept if present
            if tokens[self.max_seq_len - 1] == config.END_TOKEN_ID:
                padded = tokens[:self.max_seq_len]
            else:
                padded = tokens[:self.max_seq_len - 1] + [config.END_TOKEN_ID]
        return padded

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to handle batching of images and padded sequences."""
    image_paths = [item['image_path'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    caption_tokens = torch.stack([item['caption_tokens'] for item in batch])

    # Separate decoder input and target for training
    # Decoder input: <START> token1 token2 ... <END/PAD>
    # Target:       token1 token2 ... <END> <PAD>
    decoder_input_tokens = caption_tokens[:, :-1]
    target_tokens = caption_tokens[:, 1:]

    return {
        "image_paths": image_paths,
        "images": images.to(config.DEVICE), # Move to device here or in training loop
        "decoder_input_tokens": decoder_input_tokens.to(config.DEVICE),
        "target_tokens": target_tokens.to(config.DEVICE)
    }

# Example Usage
if __name__ == '__main__':
    # Create dummy data for testing
    DUMMY_IMG_DIR = "./dummy_images"
    DUMMY_CAPTIONS_FILE = "./dummy_captions.json"
    if not os.path.exists(DUMMY_IMG_DIR):
        os.makedirs(DUMMY_IMG_DIR)

    # Create dummy images and captions
    captions = {}
    for i in range(10):
        img_name = f"img_{i}.jpg"
        img_path = os.path.join(DUMMY_IMG_DIR, img_name)
        try:
            Image.new('RGB', (200 + i*10, 150 + i*5)).save(img_path)
            captions[img_name] = f"This is a caption for image {i}"
        except Exception as e:
            print(f"Error creating dummy image {img_name}: {e}")

    with open(DUMMY_CAPTIONS_FILE, 'w') as f:
        json.dump(captions, f)

    # Initialize Dataset
    dataset = ImageTextDataset(
        image_dir=DUMMY_IMG_DIR,
        captions_file=DUMMY_CAPTIONS_FILE,
        max_seq_len=config.MAX_SEQ_LEN
    )

    # Initialize DataLoader
    if len(dataset) > 0:
        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )

        print(f"\nDataset size: {len(dataset)}")
        print(f"Number of batches: {len(dataloader)}")

        # Get one batch
        batch = next(iter(dataloader))

        print("\nSample batch keys:", batch.keys())
        print("Image paths batch sample:", batch['image_paths'][:2])
        print("Images batch shape:", batch['images'].shape)
        print("Decoder input tokens batch shape:", batch['decoder_input_tokens'].shape)
        print("Target tokens batch shape:", batch['target_tokens'].shape)

        print("\nSample decoder input tokens:", batch['decoder_input_tokens'][0, :15])
        print("Sample target tokens:", batch['target_tokens'][0, :15])

        # Clean up dummy files
        import shutil
        # shutil.rmtree(DUMMY_IMG_DIR)
        # os.remove(DUMMY_CAPTIONS_FILE)
        print("\nDummy files created in current directory (dummy_images/, dummy_captions.json). Remove them manually if needed.")
    else:
        print("Dataset is empty, cannot create DataLoader.") 