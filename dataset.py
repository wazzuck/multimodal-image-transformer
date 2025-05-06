"""
Dataset loading and preprocessing for image-text pairs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Although imported, specific transforms are handled by the HF processor.
from PIL import Image # Python Imaging Library for image manipulation.
import os
import json # For loading caption files in JSON format.
from typing import List, Dict, Any, Tuple # Type hinting.
import config # Project configuration file.
# Import the specific Hugging Face image processor based on the encoder model specified in config.
from transformers import ViTImageProcessor, CLIPProcessor
from tokenizer import get_tokenizer # Function to get the shared tokenizer instance.
from tokenizers import Encoding # Type from the `tokenizers` library for tokenized output.

# --- Global Image Processor Setup ---
# Dynamically select and initialize the appropriate image processor from Hugging Face Transformers
# based on the ENCODER_MODEL_NAME specified in the config.py file.
# This processor will handle image normalization, resizing, and other necessary transformations.
if "clip" in config.ENCODER_MODEL_NAME.lower():
    # If the encoder name contains "clip", use CLIPProcessor.
    image_processor = CLIPProcessor.from_pretrained(config.ENCODER_MODEL_NAME)
elif "vit" in config.ENCODER_MODEL_NAME.lower():
    # If the encoder name contains "vit" (and not "clip"), use ViTImageProcessor.
    image_processor = ViTImageProcessor.from_pretrained(config.ENCODER_MODEL_NAME)
else:
    # If the encoder model type is not supported, raise an error.
    raise ValueError(f"Unsupported encoder model type in config: {config.ENCODER_MODEL_NAME}. Please use 'clip' or 'vit' based models.")

class ImageTextDataset(Dataset):
    """PyTorch Dataset for loading and preprocessing image-caption pairs."""
    def __init__(self, image_dir: str, captions_file: str, max_seq_len: int):
        """
        Initializes the ImageTextDataset.

        Args:
            image_dir: Directory containing the image files.
            captions_file: Path to the JSON file containing captions.
                           Expected format from prepare_dataset.py is a dictionary:
                           { "image_filename1.jpg": ["caption1 for image1", "caption2 for image1", ...],
                             "image_filename2.jpg": ["captionA for image2", ...], ... }
            max_seq_len: Maximum length for tokenized captions (including special tokens like START, END, PAD).
        """
        self.image_dir = image_dir # Store the path to the directory containing images.
        self.max_seq_len = max_seq_len # Store the maximum sequence length for captions.
        self.tokenizer = get_tokenizer() # Get the globally initialized tokenizer.
        self.image_processor = image_processor # Use the globally initialized image processor.

        print(f"Loading captions from: {captions_file}")
        try:
            # Attempt to open and load the JSON captions file.
            with open(captions_file, 'r', encoding='utf-8') as f:
                captions_data = json.load(f)
        except FileNotFoundError:
            # Handle cases where the captions file does not exist.
            print(f"Error: Captions file not found at {captions_file}. Dataset will be empty.")
            self.image_paths = [] # Initialize empty lists for paths and captions.
            self.captions = []
            return
        except json.JSONDecodeError:
            # Handle cases where the captions file is not valid JSON.
            print(f"Error: Could not decode JSON from {captions_file}. Dataset will be empty.")
            self.image_paths = []
            self.captions = []
            return

        self.image_paths = [] # List to store full paths to valid image files.
        self.captions = []    # List to store corresponding caption strings.

        # Process the loaded captions data, assuming the dictionary format produced by `prepare_dataset.py`.
        if isinstance(captions_data, dict):
            # Iterate through each image filename (key) and its list of captions (value).
            for filename, caption_list in captions_data.items():
                img_path = os.path.join(self.image_dir, filename) # Construct the full path to the image file.
                if os.path.exists(img_path): # Check if the image file actually exists.
                    # If the image exists, iterate through its associated captions.
                    for caption in caption_list:
                         if isinstance(caption, str): # Ensure the caption is a string.
                            self.image_paths.append(img_path) # Add the image path.
                            self.captions.append(caption)     # Add the corresponding caption.
                         else:
                            # Warn if a non-string caption is found for an image.
                            print(f"Warning: Found non-string caption for image {filename}: {caption}. Skipping this caption.")
                else:
                    # Warn if an image file listed in the captions JSON is not found in the image directory.
                    print(f"Warning: Image file not found, but listed in captions: {img_path}. Skipping associated captions.")
        else:
            # If the captions data is not in the expected dictionary format.
            print(f"Error: Captions data from {captions_file} is not in the expected dictionary format. Dataset may be empty or incorrect.")

        # Check if any valid image-caption pairs were loaded.
        if not self.image_paths:
             print("Error: No valid image-caption pairs were loaded. Check image_dir, captions_file, and their contents.")

        print(f"Successfully loaded {len(self.image_paths)} image-caption pairs.")

    def __len__(self) -> int:
        """Returns the total number of image-caption pairs in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads, preprocesses an image, and tokenizes its corresponding caption for a given index."""
        img_path = self.image_paths[idx] # Get the image path for the given index.
        caption = self.captions[idx]     # Get the caption for the given index.

        try:
            # Load the image from the path and convert it to RGB format.
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Handle errors during image loading (e.g., corrupted file, file not found despite earlier check).
            print(f"Error loading image {img_path}: {e}. Returning a dummy item.")
            # Create a placeholder/dummy image and caption to prevent DataLoader from crashing.
            # The size (224,224) is a common default for ViT models but might need adjustment based on the actual processor.
            dummy_image_pil = Image.new('RGB', (self.image_processor.size['height'], self.image_processor.size['width'])) 
            # Process the dummy image using the configured image_processor.
            processed_image_tensor = self.image_processor(images=dummy_image_pil, return_tensors="pt")['pixel_values'].squeeze(0)
            # Create dummy caption tokens (all PAD tokens).
            dummy_caption_token_ids = [config.PAD_TOKEN_ID] * self.max_seq_len
            return {
                "image_path": "error_loading_image_path", # Indicate an error.
                "image": processed_image_tensor,          # Processed dummy image tensor.
                "caption_tokens": torch.tensor(dummy_caption_token_ids, dtype=torch.long) # Dummy caption tokens.
            }

        # Preprocess the loaded image using the Hugging Face image processor.
        # `return_tensors="pt"` ensures PyTorch tensors are returned.
        # The processor typically returns a dictionary; `pixel_values` contains the image tensor.
        # `.squeeze(0)` removes the batch dimension (1, C, H, W) -> (C, H, W) as DataLoader will re-add batch dim.
        processed_image_tensor = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        # Tokenize the caption string using the pre-loaded tokenizer.
        # `add_special_tokens=True` (default for many tokenizers) adds START/END tokens if configured in the tokenizer.
        # The result `tokenized_caption` is an `Encoding` object from the `tokenizers` library.
        tokenized_caption = self.tokenizer.encode(caption, add_special_tokens=True)

        # Pad or truncate the tokenized caption to `max_seq_len`.
        padded_token_ids_tensor = self._pad_or_truncate(tokenized_caption)

        return {
            "image_path": img_path,                     # Original path of the image.
            "image": processed_image_tensor,            # Processed image tensor for the model.
            "caption_tokens": padded_token_ids_tensor  # Padded/truncated caption token IDs as a tensor.
        }

    def _pad_or_truncate(self, tokens: Encoding) -> torch.Tensor:
        """Pads or truncates a sequence of token IDs to the configured max_seq_len."""
        token_ids = tokens.ids # Extract the list of integer token IDs from the Encoding object.
        
        # Truncate if longer than max_seq_len.
        # We take up to max_seq_len tokens.
        processed_token_ids = token_ids[:self.max_seq_len]

        # Ensure the sequence ends with an END_TOKEN_ID if it was truncated right before it,
        # or if it naturally fits and is not already the END token.
        # This step is crucial for signaling the end of a caption to the decoder.
        if len(processed_token_ids) == self.max_seq_len and \
           processed_token_ids[self.max_seq_len - 1] != config.END_TOKEN_ID:
             # If the sequence is exactly max_seq_len AND the last token is NOT already END_TOKEN_ID,
             # force the last token to be END_TOKEN_ID. This ensures truncation doesn't cut off generation prematurely.
             processed_token_ids[self.max_seq_len - 1] = config.END_TOKEN_ID
        
        # Pad with PAD_TOKEN_ID if the sequence is shorter than max_seq_len.
        if len(processed_token_ids) < self.max_seq_len:
            padding_needed = self.max_seq_len - len(processed_token_ids)
            processed_token_ids.extend([config.PAD_TOKEN_ID] * padding_needed)

        return torch.tensor(processed_token_ids, dtype=torch.long) # Convert the list of IDs to a PyTorch tensor.

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for the DataLoader to handle batching of processed data.
    
    This function takes a list of dictionaries (each dictionary being an item from __getitem__)
    and stacks the tensors into batches. It also prepares decoder input and target sequences.
    """
    # Extract image paths, image tensors, and caption tokens from the batch.
    image_paths = [item['image_path'] for item in batch] # List of image paths in the batch.
    # Stack individual image tensors (C, H, W) into a batch tensor (Batch Size, C, H, W).
    images = torch.stack([item['image'] for item in batch])
    # Stack individual caption token tensors (Max Seq Len) into a batch tensor (Batch Size, Max Seq Len).
    caption_tokens = torch.stack([item['caption_tokens'] for item in batch])

    # Prepare decoder input and target sequences for training (teacher forcing).
    # Example: if caption_tokens is [<START>, t1, t2, <END>, <PAD>]
    # - decoder_input_tokens will be [<START>, t1, t2, <END>] (input to the decoder)
    # - target_tokens will be [t1, t2, <END>, <PAD>] (what the decoder tries to predict)
    
    # Decoder input tokens: all tokens in caption_tokens except the last one.
    decoder_input_tokens = caption_tokens[:, :-1]
    # Target tokens: all tokens in caption_tokens except the first one (usually <START>).
    target_tokens = caption_tokens[:, 1:]

    # The tensors are returned on CPU by default. They will be moved to the target device
    # (e.g., GPU) within the training loop for efficiency.
    return {
        "image_paths": image_paths,                # Batch of image paths.
        "images": images,                          # Batch of processed image tensors.
        "decoder_input_tokens": decoder_input_tokens, # Batch of decoder input sequences.
        "target_tokens": target_tokens             # Batch of target sequences for loss calculation.
    }

# Example Usage: This block runs if the script is executed directly (e.g., `python dataset.py`).
if __name__ == '__main__':
    print("Running ImageTextDataset example usage...")
    # --- Create dummy data for testing --- 
    # Define paths for dummy images and a dummy captions file.
    DUMMY_IMG_DIR = "./dummy_images_for_dataset_test" # Temporary directory for dummy images.
    DUMMY_CAPTIONS_FILE = "./dummy_captions_for_dataset_test.json" # Temporary dummy captions file.
    
    # Create the dummy image directory if it doesn't exist.
    if not os.path.exists(DUMMY_IMG_DIR):
        os.makedirs(DUMMY_IMG_DIR)

    # Create a few dummy images and a corresponding captions dictionary.
    captions_dict_for_dummy_file = {}
    num_dummy_items = 5 # Number of dummy image-caption pairs to create.
    for i in range(num_dummy_items):
        img_name = f"dummy_img_{i}.jpg"
        img_path = os.path.join(DUMMY_IMG_DIR, img_name)
        try:
            # Create a simple PIL image and save it.
            Image.new('RGB', (60, 30), color = 'red').save(img_path) # Small dummy images.
            # Create a list of captions for each dummy image (as per expected format).
            captions_dict_for_dummy_file[img_name] = [
                f"This is the first caption for dummy image {i}.",
                f"Another caption for dummy image {i}!"
            ]
        except Exception as e:
            print(f"Error creating dummy image {img_name} or its caption: {e}")

    # Save the dummy captions dictionary to a JSON file.
    with open(DUMMY_CAPTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(captions_dict_for_dummy_file, f, indent=2)
    print(f"Created dummy images in {DUMMY_IMG_DIR} and captions in {DUMMY_CAPTIONS_FILE}")

    # --- Initialize Dataset --- 
    # Create an instance of the ImageTextDataset using the dummy data.
    dataset = ImageTextDataset(
        image_dir=DUMMY_IMG_DIR,
        captions_file=DUMMY_CAPTIONS_FILE,
        max_seq_len=config.MAX_SEQ_LEN # Use max_seq_len from project config.
    )

    # --- Initialize DataLoader (if dataset is not empty) --- 
    if len(dataset) > 0:
        dataloader = DataLoader(
            dataset,                            # The initialized ImageTextDataset instance.
            batch_size=min(config.BATCH_SIZE, len(dataset)), # Use configured batch size or dataset size if smaller.
            shuffle=True,                       # Shuffle data for training. 
            collate_fn=collate_fn               # Use the custom collate function.
        )

        print(f"\nDataset size: {len(dataset)} image-caption pairs.")
        print(f"Number of batches in DataLoader: {len(dataloader)}")

        # --- Get and inspect one batch from the DataLoader --- 
        try:
            batch = next(iter(dataloader))

            print("\nSample batch keys:", batch.keys())
            print("Image paths batch sample (first 2):", batch['image_paths'][:2])
            print("Images batch tensor shape:", batch['images'].shape)
            print("Decoder input tokens batch tensor shape:", batch['decoder_input_tokens'].shape)
            print("Target tokens batch tensor shape:", batch['target_tokens'].shape)

            # Print a sample of token IDs from the first item in the batch.
            print("\nSample decoder input tokens (first item, first 15 tokens):", batch['decoder_input_tokens'][0, :15])
            print("Sample target tokens (first item, first 15 tokens):", batch['target_tokens'][0, :15])
        except StopIteration:
            print("Could not retrieve a batch from the DataLoader. This shouldn't happen if dataset is not empty.")
        except Exception as e:
            print(f"Error while inspecting DataLoader batch: {e}")

    else:
        print("Dataset is empty, cannot create or test DataLoader.")

    # --- Clean up dummy files and directory --- 
    # It's good practice to clean up temporary test files.
    # Consider uncommenting these lines if you want automatic cleanup.
    # import shutil
    # if os.path.exists(DUMMY_IMG_DIR):
    #     shutil.rmtree(DUMMY_IMG_DIR)
    # if os.path.exists(DUMMY_CAPTIONS_FILE):
    #     os.remove(DUMMY_CAPTIONS_FILE)
    print("\nDummy files and directory (dummy_images_for_dataset_test/, dummy_captions_for_dataset_test.json) were created for testing.")
    print("Please remove them manually if they are no longer needed.") 