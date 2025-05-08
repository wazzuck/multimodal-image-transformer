"""
Main training script for the Multimodal Image Transformer.
"""

import torch # PyTorch library, for tensor computations and neural networks.
import torch.nn as nn # Neural network module from PyTorch, for layers and loss functions.
import torch.optim as optim # Optimization algorithms from PyTorch, like Adam or SGD.
from torch.utils.data import DataLoader, random_split # Utilities for loading and splitting datasets.
from model import ImageToTextModel # The main model combining encoder and decoder.
from dataset import ImageTextDataset, collate_fn # Custom dataset and collate function for batching.
from tokenizer import get_tokenizer, train_tokenizer, get_tokenizer_vocab_size # Tokenizer utilities.
import config # Configuration file with all hyperparameters and paths.
import os # Operating system functionalities, like file and directory manipulation.
import time # Time-related functions, for measuring execution time.
import json # For loading captions if needed (e.g., for tokenizer training).
from tqdm import tqdm # For displaying progress bars during training and evaluation.
import prepare_dataset # Script to download and prepare the dataset.
from safetensors.torch import save_file # For saving model checkpoints in .safetensors format.
import wandb # Weights & Biases for experiment tracking and visualization.
from huggingface_hub import HfApi, create_repo # For interacting with the Hugging Face Hub (uploading models).
from transformers import get_linear_schedule_with_warmup # Learning rate scheduler.

def setup_wandb(cfg):
    """Initializes a new Weights & Biases run for experiment tracking."""
    # Consolidate relevant hyperparameters from the config object into a dictionary for logging.
    hyperparameters = {
        'encoder_model': cfg.ENCODER_MODEL_NAME,      # Name of the pre-trained vision model used as encoder
        'decoder_layers': cfg.DECODER_LAYERS,         # Number of transformer layers in the decoder
        'decoder_heads': cfg.DECODER_HEADS,           # Number of attention heads in each decoder layer
        'decoder_ff_dim': cfg.DECODER_FF_DIM,         # Dimension of the feed-forward network in decoder layers
        'embedding_dim': cfg.DECODER_EMBED_DIM,       # Dimension of token embeddings in the decoder
        'max_seq_len': cfg.MAX_SEQ_LEN,               # Maximum sequence length for generated captions
        'dropout': cfg.DECODER_DROPOUT,               # Dropout rate for regularization in decoder
        'learning_rate': cfg.LEARNING_RATE,           # Initial learning rate for the optimizer
        'epochs': cfg.NUM_EPOCHS,                     # Total number of training epochs
        'batch_size': cfg.BATCH_SIZE,                 # Number of samples processed in each training batch
        'vocab_size': cfg.VOCAB_SIZE,                 # Target size of the tokenizer vocabulary
        'warmup_steps': cfg.WARMUP_STEPS,             # Number of steps for learning rate warmup
        'adam_beta1': cfg.ADAM_BETA1,                 # First moment estimate exponential decay rate for Adam
        'adam_beta2': cfg.ADAM_BETA2,                 # Second moment estimate exponential decay rate for Adam
        'adam_eps': cfg.ADAM_EPS,                     # Small constant for numerical stability in Adam
        'weight_decay': cfg.WEIGHT_DECAY,             # L2 regularization coefficient
        'grad_clip': cfg.GRAD_CLIP_VALUE,             # Maximum norm for gradient clipping
        'projection_dim': cfg.PROJECTION_DIM,         # Dimension for projecting image features to decoder space
        'image_dir': cfg.IMAGE_DIR,                   # Directory containing training images
        'captions_file': cfg.CAPTIONS_FILE,           # Path to file containing image captions
        'output_dir': cfg.OUTPUT_DIR,                 # Directory for saving model checkpoints
        'checkpoint_prefix': cfg.CHECKPOINT_PREFIX,   # Prefix for checkpoint filenames
        'log_interval': cfg.LOG_INTERVAL,             # Number of batches between training loss logs
        'validation_interval': cfg.VALIDATION_INTERVAL # Number of epochs between validation runs
    }
    # Initialize the wandb run.
    run = wandb.init(
        project=cfg.WANDB_PROJECT,    # Project name from config.
        entity=cfg.WANDB_ENTITY,      # Wandb entity (username or team) from config (optional).
        config=hyperparameters,       # Logged hyperparameters.
        name=cfg.WANDB_RUN_NAME       # Optional custom run name from config.
    )
    print(f"Wandb run initialized. View at: {run.url}")
    return run # Return the wandb run object.

def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip_value, scheduler, epoch, log_interval, wandb_run):
    """Runs a single epoch of training for the model."""
    model.train() # Set the model to training mode (enables dropout, batch norm updates, etc.).
    total_loss = 0.0 # Accumulator for the total loss over the epoch.
    num_batches = len(dataloader) # Total number of batches in this epoch.

    # Wrap the dataloader with tqdm for a progress bar.
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False)

    for i, batch in enumerate(progress_bar):
        global_step = epoch * num_batches + i # Calculate the global training step number.

        # Move batch data to the configured computation device (CPU/GPU).
        images = batch["images"].to(device) # Preprocessed image tensors.
        decoder_input_tokens = batch["decoder_input_tokens"].to(device) # Input tokens for the decoder (e.g., <START> cap).
        target_tokens = batch["target_tokens"].to(device) # Target tokens for loss calculation (e.g., cap <END>).

        # Zero out any gradients from the previous iteration.
        optimizer.zero_grad()

        # Perform the forward pass: get model predictions (logits).
        logits = model(images, decoder_input_tokens)

        # Calculate the loss.
        # Logits shape: (Batch Size, Target Sequence Length, Vocab Size)
        # Target shape: (Batch Size, Target Sequence Length)
        # CrossEntropyLoss expects logits as (N, C, ...) and target as (N, ...),
        # so we reshape them: flatten the batch and sequence dimensions.
        loss = criterion(logits.view(-1, logits.size(-1)), target_tokens.reshape(-1))

        # Perform the backward pass: compute gradients of the loss w.r.t. model parameters.
        loss.backward()

        # Apply gradient clipping to prevent exploding gradients (optional).
        if grad_clip_value > 0: # Only clip if a non-zero, positive value is provided.
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        # Update model parameters using the optimizer.
        optimizer.step()
        
        # Update the learning rate using the scheduler (if one is configured).
        if scheduler:
             current_lr = scheduler.get_last_lr()[0] # Get the current learning rate from the scheduler.
             scheduler.step() # Advance the scheduler.
        else:
             current_lr = optimizer.param_groups[0]['lr'] # Get LR from optimizer if no scheduler.

        batch_loss = loss.item() # Get the scalar loss value for the current batch.
        total_loss += batch_loss # Accumulate the batch loss.
        # Update the progress bar with the current batch loss and learning rate.
        progress_bar.set_postfix({'loss': batch_loss, 'lr': current_lr})

        # Log metrics to Weights & Biases periodically.
        if wandb_run and (global_step + 1) % log_interval == 0:
            wandb.log({
                "train_batch_loss": batch_loss,
                "learning_rate": current_lr,
                "global_step": global_step + 1 # Log step starting from 1 for clarity.
            })

    progress_bar.close() # Close the tqdm progress bar for this epoch.
    return total_loss / num_batches # Return the average training loss for the epoch.

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model\'s performance on a given dataset (e.g., validation set)."""
    model.eval() # Set the model to evaluation mode (disables dropout, etc.).
    total_loss = 0.0 # Accumulator for the total loss over the evaluation dataset.
    num_batches = len(dataloader) # Total number of batches in the evaluation dataloader.

    # Wrap the dataloader with tqdm for a progress bar.
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad(): # Disable gradient calculations during evaluation.
        for batch in progress_bar:
            # Move batch data to the configured computation device.
            images = batch["images"].to(device)
            decoder_input_tokens = batch["decoder_input_tokens"].to(device)
            target_tokens = batch["target_tokens"].to(device)

            # Perform the forward pass to get model predictions (logits).
            logits = model(images, decoder_input_tokens)

            # Calculate the loss, reshaping logits and targets as in training.
            loss = criterion(logits.view(-1, logits.size(-1)), target_tokens.reshape(-1))
            total_loss += loss.item() # Accumulate the batch loss.
            # Update the progress bar with the current batch loss.
            progress_bar.set_postfix({'loss': loss.item()})

    progress_bar.close() # Close the tqdm progress bar.
    return total_loss / num_batches # Return the average evaluation loss.

def main():
    """Main function to orchestrate the training process."""
    # --- Ensure dataset is ready --- #
    print("Checking and preparing dataset if necessary...")
    # Call the dataset preparation script (e.g., to download, extract, and format Flickr30k).
    prepare_dataset.prepare_flickr30k()
    print("Dataset preparation check complete.")

    # Ensure the output directory (for checkpoints, tokenizer files, etc.) exists.
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- Setup: Device and Random Seeds --- #
    # Set the computation device (CPU or CUDA/GPU) based on availability and config.
    device = torch.device(config.DEVICE)
    # Set random seeds for PyTorch and CUDA (if used) for reproducibility.
    torch.manual_seed(config.RANDOM_SEED)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed(config.RANDOM_SEED)

    # --- Weights & Biases Setup --- #
    # Initialize a Weights & Biases run for experiment tracking.
    wandb_run = setup_wandb(config)

    # --- Hugging Face Hub Setup --- #
    print(f"Attempting to create or use Hugging Face Hub repo: {config.HF_REPO_ID}")
    try:
        hf_api = HfApi() # Initialize the Hugging Face API client.
        # Create the repository on the Hub if it doesn't exist. `exist_ok=True` prevents errors if it already exists.
        create_repo(config.HF_REPO_ID, repo_type="model", exist_ok=True)
        print(f"Hugging Face Hub repository '{config.HF_REPO_ID}' ensured.")
    except Exception as e:
        # If repository creation/access fails, print a warning and disable uploads.
        print(f"Warning: Could not create/access Hugging Face Hub repo '{config.HF_REPO_ID}'. Uploads will be skipped. Error: {e}")
        hf_api = None # Set hf_api to None to indicate that Hub interaction is disabled.

    # --- Tokenizer Training (if vocabulary files are not found) --- #
    print("Checking tokenizer vocabulary...")
    # Check if tokenizer vocabulary and merges files (for BPE) exist at configured paths.
    if not os.path.exists(config.VOCAB_PATH) or not os.path.exists(config.MERGES_PATH):
        print(f"Tokenizer vocabulary not found at {config.VOCAB_PATH} or {config.MERGES_PATH}.")
        print("Attempting to train tokenizer from scratch...")

        try:
            # Load captions data from the JSON file to train the tokenizer.
            with open(config.CAPTIONS_FILE, 'r', encoding='utf-8') as f:
                captions_data = json.load(f) # Assumes captions are in a JSON structure.
            
            all_captions = [] # List to store all caption strings.
            # Extract caption strings from the loaded JSON data.
            # Handles dict format: {"img_name": ["cap1", "cap2"]} or {"img_name": "cap"}
            if isinstance(captions_data, dict):
                 for caption_list_or_str in captions_data.values():
                      if isinstance(caption_list_or_str, list):
                          all_captions.extend(caption_list_or_str) # Add all captions from the list.
                      elif isinstance(caption_list_or_str, str): # Handle case where value is a single string caption.
                          all_captions.append(caption_list_or_str)
            else:
                 # If format is not as expected, warn and skip tokenizer training from this file.
                 print(f"Warning: Captions file format at {config.CAPTIONS_FILE} is not the expected dictionary. Cannot train tokenizer.")
            
            if not all_captions:
                print("Error: No caption strings found to train the tokenizer. Please check your captions file.")
                return # Exit if no captions are available for training.

            # Train the BPE tokenizer using the extracted captions.
            train_tokenizer(
                iter(all_captions),       # Pass an iterator over the caption strings.
                vocab_size=config.VOCAB_SIZE, # Target vocabulary size from config.
                vocab_path=config.VOCAB_PATH, # Path to save the vocabulary file.
                merges_path=config.MERGES_PATH # Path to save the merges file.
            )
            print("Tokenizer training finished and files saved.")

        except FileNotFoundError:
            print(f"Error: Cannot train tokenizer because captions file was not found at {config.CAPTIONS_FILE}")
            return # Exit if captions file is missing.
        except Exception as e:
            print(f"Error occurred during tokenizer training: {e}")
            return # Exit on other tokenizer training errors.
    else:
        print("Found existing tokenizer vocabulary and merges files.")

    # --- Load Tokenizer and Determine Effective Vocabulary Size --- #
    try:
        # Load the (potentially newly trained or pre-existing) tokenizer.
        tokenizer = get_tokenizer() 
        # Get the actual vocabulary size from the loaded tokenizer.
        # This might differ slightly from `config.VOCAB_SIZE` especially for BPE tokenizers.
        actual_vocab_size = get_tokenizer_vocab_size()
        print(f"Tokenizer loaded successfully. Actual vocabulary size: {actual_vocab_size}")
        
        # Optional: Sanity check for a very small vocabulary size.
        if actual_vocab_size < 5: # A vocab size this small is usually problematic.
             print("Warning: Loaded tokenizer vocabulary size seems extremely small. Check tokenizer files and training.")

    except FileNotFoundError as e:
        print(f"Error loading tokenizer files: {e}. Ensure {config.VOCAB_PATH} and {config.MERGES_PATH} exist or can be trained.")
        print("Cannot proceed without a valid tokenizer.")
        return # Exit if tokenizer cannot be loaded.
    except Exception as e:
        print(f"An unexpected error occurred while loading the tokenizer: {e}")
        return # Exit on other tokenizer loading errors.

    # Use the actual vocabulary size obtained from the tokenizer for the model.
    effective_vocab_size = actual_vocab_size

    # --- Dataset and Dataloaders --- #
    print("Loading and preparing dataset...")
    # Initialize the custom ImageTextDataset.
    # This dataset handles loading images and their corresponding tokenized captions.
    full_dataset = ImageTextDataset(
        image_dir=config.IMAGE_DIR,          # Directory containing image files.
        captions_file=config.CAPTIONS_FILE,  # Path to the JSON captions file.
        max_seq_len=config.MAX_SEQ_LEN       # Maximum sequence length for tokenized captions.
        # The tokenizer instance is implicitly used by the dataset via global access or could be passed explicitly.
    )

    if len(full_dataset) == 0:
        print("Error: The dataset is empty. Please check image directory and captions file paths and content.")
        return # Exit if the dataset has no samples.

    # Split the dataset into training and validation sets.
    # Define split sizes (e.g., 90% train, 10% validation).
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Create DataLoaders for training and validation sets.
    # DataLoaders handle batching, shuffling, and parallel data loading.
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, # Shuffle training data each epoch.
        collate_fn=collate_fn # Custom collate function to pad sequences in a batch.
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, # No need to shuffle validation data.
        collate_fn=collate_fn
    )
    print("DataLoaders created.")

    # --- Model Initialization --- #
    print(f"Initializing model with effective vocab size: {effective_vocab_size}...")
    # Initialize the ImageToTextModel with hyperparameters from config and the effective vocabulary size.
    model = ImageToTextModel(
        decoder_vocab_size=effective_vocab_size,
        decoder_embed_dim=config.DECODER_EMBED_DIM,
        decoder_heads=config.DECODER_HEADS,
        decoder_layers=config.DECODER_LAYERS,
        decoder_ff_dim=config.DECODER_FF_DIM,
        decoder_max_seq_len=config.MAX_SEQ_LEN,
        decoder_dropout=config.DECODER_DROPOUT,
        decoder_pad_idx=config.PAD_TOKEN_ID
    ).to(device) # Move the model to the configured device.
    print("Model initialized and moved to device.")

    # --- Optimizer and Loss Criterion --- #
    # Initialize the AdamW optimizer with configured learning rate and weight decay.
    # AdamW is often preferred for training Transformer models.
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        eps=config.ADAM_EPS,
        weight_decay=config.WEIGHT_DECAY
    )
    # Initialize the CrossEntropyLoss criterion.
    # `ignore_index=config.PAD_TOKEN_ID` ensures that padding tokens do not contribute to the loss.
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)
    print("Optimizer and loss criterion initialized.")

    # --- Learning Rate Scheduler (Optional) --- #
    scheduler = None # Initialize scheduler to None.
    if config.WARMUP_STEPS > 0:
        print(f"Setting up learning rate scheduler with {config.WARMUP_STEPS} warmup steps.")
        # Total training steps for scheduler calculation.
        num_training_steps = len(train_dataloader) * config.NUM_EPOCHS
        # Use a linear warmup followed by linear decay scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=num_training_steps
        )
        print("Scheduler initialized.")

    # --- Training Loop --- #
    print(f"Starting training for {config.NUM_EPOCHS} epochs on device '{device}'...")
    best_val_loss = float('inf') # Initialize best validation loss for checkpointing.

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time() # Record epoch start time.

        # Perform one epoch of training.
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, criterion, device, 
            config.GRAD_CLIP_VALUE, scheduler, epoch, config.LOG_INTERVAL, wandb_run
        )
        
        epoch_duration = time.time() - start_time # Calculate epoch duration.

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Duration: {epoch_duration:.2f}s")

        # Log epoch-level training metrics to WandB.
        if wandb_run:
            wandb.log({
                "epoch_train_loss": train_loss,
                "epoch": epoch + 1
            })

        # Perform validation at specified intervals.
        if (epoch + 1) % config.VALIDATION_INTERVAL == 0:
            val_loss = evaluate(model, val_dataloader, criterion, device)
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Validation Loss: {val_loss:.4f}")
            
            # Log validation metrics to WandB.
            if wandb_run:
                wandb.log({"val_loss": val_loss, "epoch": epoch + 1})

            # Save the model checkpoint if validation loss has improved.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_name = f"{config.CHECKPOINT_PREFIX}_epoch_{epoch+1}_val_loss_{val_loss:.4f}.safetensors"
                checkpoint_path = os.path.join(config.OUTPUT_DIR, checkpoint_name)
                # Save model state dictionary using safetensors for safety and efficiency.
                save_file(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path} (Best val_loss: {best_val_loss:.4f})")

                # Upload to Hugging Face Hub if API is available.
                if hf_api:
                    try:
                        print(f"Uploading checkpoint {checkpoint_name} to Hugging Face Hub...")
                        hf_api.upload_file(
                            path_or_fileobj=checkpoint_path,
                            path_in_repo=checkpoint_name, # Filename in the Hub repository.
                            repo_id=config.HF_REPO_ID,
                            repo_type="model"
                        )
                        print(f"Successfully uploaded {checkpoint_name} to {config.HF_REPO_ID}.")
                    except Exception as e:
                        print(f"Error uploading checkpoint to Hugging Face Hub: {e}")
            else:
                # Optionally, save checkpoints even if not the best, e.g., every few epochs.
                pass # Current setup only saves the best based on val_loss.

    print("Training finished.")
    # --- Finalize Wandb Run --- #
    if wandb_run:
        wandb.finish() # Mark the Wandb run as complete.
        print("Wandb run finished.")

# Standard Python entry point: execute main() if the script is run directly.
if __name__ == "__main__":
    main() 