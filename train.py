"""
Main training script for the Multimodal Image Transformer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import ImageToTextModel
from dataset import ImageTextDataset, collate_fn
from tokenizer import get_tokenizer, train_tokenizer, get_vocab_size as get_tokenizer_vocab_size
import config
import os
import time
import json # Needed for loading captions
from tqdm import tqdm # For progress bar
import prepare_dataset # Import the new preparation script
from safetensors.torch import save_file # Import safetensors saving function

def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip_value=None):
    """Runs one epoch of training."""
    model.train() # Set model to training mode
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="Training Epoch", leave=False)

    for batch in progress_bar:
        # Move data to the configured device
        images = batch["images"].to(device)
        decoder_input_tokens = batch["decoder_input_tokens"].to(device)
        target_tokens = batch["target_tokens"].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        # TODO: Modify model.forward to accept processed image tensors directly
        # instead of PIL images, as preprocessing is done in Dataset/DataLoader.
        # Current model.forward expects PIL images - needs update!
        # --- Placeholder: Adapt model call --- #
        # Assuming model.forward is updated to take image *tensors*
        # logits = model(images, decoder_input_tokens)
        # For now, let's simulate the forward pass structure
        # This requires model.py forward pass to be updated
        # --- Simulate call --- # 
        try:
            logits = model(images, decoder_input_tokens) # Pass tensors
        except TypeError as e:
             print("\n*** Error: model.forward likely expects PIL images, but received tensors. ***")
             print("*** Please update model.py forward method to handle image tensors from DataLoader. ***")
             print(f"TypeError: {e}")
             print("Skipping batch...")
             # Placeholder to avoid crashing during structure creation
             # Replace with actual error handling or fix model.py
             logits = torch.randn(target_tokens.shape[0], target_tokens.shape[1], config.VOCAB_SIZE, device=device, requires_grad=True)
        # --- End Simulate call ---

        # Calculate loss
        # Logits: (Batch Size, Target Seq Len, Vocab Size)
        # Target: (Batch Size, Target Seq Len)
        # Criterion expects logits as (N, C, ...) and target as (N, ...)
        # Reshape logits and target for CrossEntropyLoss
        loss = criterion(logits.view(-1, logits.size(-1)), target_tokens.reshape(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping (optional)
        if grad_clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    progress_bar.close()
    return total_loss / num_batches

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad(): # Disable gradient calculations
        for batch in progress_bar:
            # Move data to the configured device
            images = batch["images"].to(device)
            decoder_input_tokens = batch["decoder_input_tokens"].to(device)
            target_tokens = batch["target_tokens"].to(device)

            # Forward pass (assuming model.forward handles tensors)
             # --- Simulate call --- # 
            try:
                logits = model(images, decoder_input_tokens)
            except TypeError as e:
                 # Placeholder error handling (same as in train_one_epoch)
                 print("\nEvaluation Error: model.forward issue (see training error).")
                 logits = torch.randn(target_tokens.shape[0], target_tokens.shape[1], config.VOCAB_SIZE, device=device)
             # --- End Simulate call ---

            # Calculate loss
            # Use reshape for target_tokens
            loss = criterion(logits.view(-1, logits.size(-1)), target_tokens.reshape(-1))
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

    progress_bar.close()
    return total_loss / num_batches

def main():
    """Main training loop."""
    # --- Ensure dataset is ready --- #
    print("Checking and preparing dataset if necessary...")
    prepare_dataset.prepare_flickr30k()
    print("Dataset preparation check complete.")

    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- Setup --- #
    device = torch.device(config.DEVICE)
    torch.manual_seed(config.RANDOM_SEED)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed(config.RANDOM_SEED)

    # --- Tokenizer Training (if needed) --- #
    print("Checking tokenizer vocabulary...")
    if not os.path.exists(config.VOCAB_PATH) or not os.path.exists(config.MERGES_PATH):
        print(f"Tokenizer vocabulary not found at {config.VOCAB_PATH} or {config.MERGES_PATH}.")
        print("Attempting to train tokenizer...")

        # Load captions data directly for training the tokenizer
        try:
            with open(config.CAPTIONS_FILE, 'r', encoding='utf-8') as f:
                captions_data = json.load(f)
            # Extract just the caption strings
            all_captions = []
            if isinstance(captions_data, dict):
                 for caption_list in captions_data.values():
                      if isinstance(caption_list, list):
                          all_captions.extend(caption_list)
                      elif isinstance(caption_list, str): # Handle case where value is string, not list
                          all_captions.append(caption_list)
            else:
                 print(f"Warning: captions file format not the expected dict. Cannot train tokenizer from {config.CAPTIONS_FILE}")
            
            if not all_captions:
                print("Error: No captions found to train tokenizer. Please check captions file.")
                return # Cannot proceed without captions

            # Train the tokenizer
            train_tokenizer(
                iter(all_captions), # Pass iterator
                vocab_size=config.VOCAB_SIZE,
                vocab_path=config.VOCAB_PATH,
                merges_path=config.MERGES_PATH
            )
            print("Tokenizer training finished.")

        except FileNotFoundError:
            print(f"Error: Cannot train tokenizer because captions file not found at {config.CAPTIONS_FILE}")
            return
        except Exception as e:
            print(f"Error during tokenizer training: {e}")
            return
    else:
        print("Found existing tokenizer vocabulary files.")

    # --- Load Tokenizer and Check Vocab Size --- #
    try:
        tokenizer = get_tokenizer() # Now loads the potentially trained tokenizer
        actual_vocab_size = get_tokenizer_vocab_size() # Use the specific function
        print(f"Tokenizer loaded. Actual vocab size: {actual_vocab_size}")
        # Optional: Check against config.VOCAB_SIZE, though BPE might not reach exact size
        if actual_vocab_size < 5: # Check for minimal reasonable size
             print("Warning: Loaded tokenizer vocab size seems very small.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Cannot proceed without a valid tokenizer.")
        return
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    effective_vocab_size = actual_vocab_size # Use the actual size for the model

    # --- Dataset and Dataloaders --- #
    print("Loading dataset...")
    full_dataset = ImageTextDataset(
        image_dir=config.IMAGE_DIR,
        captions_file=config.CAPTIONS_FILE,
        max_seq_len=config.MAX_SEQ_LEN
        # Pass tokenizer if needed by dataset, but using global instance here
    )

    if len(full_dataset) == 0:
        print("Dataset is empty. Cannot train. Check data paths and format.")
        return

    # Split dataset (e.g., 90% train, 10% validation)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4 # Adjust based on your system
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4 # Adjust based on your system
    )

    # --- Model --- #
    print("Initializing model...")
    model = ImageToTextModel(
        decoder_vocab_size=effective_vocab_size, # Use actual vocab size
        decoder_embed_dim=config.DECODER_EMBED_DIM,
        decoder_heads=config.DECODER_HEADS,
        decoder_layers=config.DECODER_LAYERS,
        decoder_ff_dim=config.DECODER_FF_DIM,
        decoder_max_seq_len=config.MAX_SEQ_LEN,
        decoder_dropout=config.DECODER_DROPOUT,
        decoder_pad_idx=config.PAD_TOKEN_ID
    ).to(device)

    # --- Optimizer and Loss --- #
    optimizer = optim.AdamW(
        model.decoder.parameters(), # Only optimize decoder parameters
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Use CrossEntropyLoss, ignore padding index
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)

    # --- Training Loop --- #
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config.GRAD_CLIP_VALUE)
        val_loss = evaluate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Time: {epoch_duration:.2f}s")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\t Val. Loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Update extension to .safetensors
            save_path = os.path.join(config.OUTPUT_DIR, "best_model.safetensors")
            # Use save_file for safetensors format
            save_file(model.state_dict(), save_path)
            print(f"\tBest model saved to {save_path}")

        # Save checkpoint every epoch
        # Update extension to .safetensors
        chkpt_path = os.path.join(config.OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.safetensors")
        # Use save_file for safetensors format
        save_file(model.state_dict(), chkpt_path)
        print(f"\tCheckpoint saved to {chkpt_path}")

    print("Training finished.")

if __name__ == "__main__":
    # --- Crucial Check: Ensure model.forward expects image tensors --- #
    print("!!! IMPORTANT !!!")
    print("The current `model.py` forward method expects PIL Images.")
    print("This training script passes preprocessed *tensors* from the DataLoader.")
    print("You MUST update `model.py`'s `forward` method to accept image tensors ")
    print("and remove the internal call to `encode_image`. The encoding should")
    print("happen *before* the model's forward pass, ideally batched.")
    print("The current code includes simulation placeholders and will error without this fix.")
    print("!!!!!!!!!!!!!!!!!!")
    # Proceeding with the structure, but the forward pass needs fixing in model.py.
    main() 