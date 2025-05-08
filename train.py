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
    except Exception as e: # Catch other potential errors during tokenizer loading.
        print(f"An unexpected error occurred while loading the tokenizer: {e}")
        print("Cannot proceed without a valid tokenizer.")
        return

    # Use the actual vocabulary size obtained from the tokenizer for the model.
    effective_vocab_size = actual_vocab_size

    # --- Dataset and DataLoader Setup --- #
    print("Loading and preparing datasets...")
    # Create the full dataset using images and captions.
    # The image processor is sourced from the config (e.g., ViTImageProcessor, CLIPProcessor).
    full_dataset = ImageTextDataset(
        image_dir=config.IMAGE_DIR,
        captions_file=config.CAPTIONS_FILE,
        max_seq_len=config.MAX_SEQ_LEN
        # tokenizer, image_processor_name, and img_transform_mode are not part of ImageTextDataset __init__ signature
        # tokenizer is obtained via get_tokenizer() within the dataset
        # image_processor is globally defined and used within the dataset
        # img_transform_mode is not currently used by ImageTextDataset
    )

    # Split the dataset into training and validation sets.
    train_size = int(config.TRAIN_SPLIT_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Create DataLoaders for training and validation.
    # DataLoaders handle batching, shuffling, and multi-processing for data loading.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, # Shuffle training data each epoch.
        collate_fn=collate_fn, # Use custom collate function directly.
        num_workers=config.NUM_WORKERS, # Number of worker processes for data loading.
        pin_memory=config.PIN_MEMORY   # If True, copies tensors to CUDA pinned memory before returning them (faster GPU transfer).
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # No need to shuffle validation data.
        collate_fn=collate_fn, # Use custom collate function directly.
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    print("DataLoaders created.")

    # --- Model Initialization --- #
    print(f"Initializing model: {config.ENCODER_MODEL_NAME} encoder, with {config.DECODER_LAYERS} decoder layers.")
    # Instantiate the ImageToTextModel.
    model = ImageToTextModel(
        # encoder_model_name is loaded from config inside the model's __init__
        decoder_vocab_size=actual_vocab_size,         # Actual vocabulary size from the tokenizer.
        decoder_embed_dim=config.DECODER_EMBED_DIM,   # Embedding dimension for decoder tokens.
        decoder_heads=config.DECODER_HEADS,           # Number of attention heads in decoder.
        decoder_layers=config.DECODER_LAYERS,         # Number of transformer layers in decoder.
        decoder_ff_dim=config.DECODER_FF_DIM,         # Feed-forward dimension in decoder layers.
        decoder_max_seq_len=config.MAX_SEQ_LEN,       # Maximum sequence length for decoder.
        decoder_dropout=config.DECODER_DROPOUT,       # Dropout rate for regularization in decoder.
        decoder_pad_idx=config.PAD_TOKEN_ID          # Padding token ID from config.
        # projection_dim is determined and handled inside the model's __init__
    ).to(device) # Move the model to the configured device.
    print("Model initialized and moved to device.")

    # --- Optimizer and Loss Function Setup --- #
    # Initialize the AdamW optimizer with configured parameters.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        eps=config.ADAM_EPS,
        weight_decay=config.WEIGHT_DECAY
    )
    # Define the loss function: CrossEntropyLoss, ignoring padding tokens.
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)
    print("Optimizer and loss function initialized.")

    # --- Learning Rate Scheduler Setup (Optional) --- #
    scheduler = None # Initialize scheduler to None.
    if config.WARMUP_STEPS > 0:
        # Calculate total training steps for the scheduler.
        num_training_steps = len(train_dataloader) * config.NUM_EPOCHS
        # Create a linear learning rate scheduler with warmup.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=num_training_steps
        )
        print(f"Learning rate scheduler with {config.WARMUP_STEPS} warmup steps initialized.")

    # --- Checkpoint Loading for Resuming Training --- #
    start_epoch = 0
    best_val_loss = float('inf')

    if hasattr(config, 'RESUME_CHECKPOINT_PATH') and config.RESUME_CHECKPOINT_PATH and os.path.exists(config.RESUME_CHECKPOINT_PATH):
        print(f"Attempting to resume training from checkpoint: {config.RESUME_CHECKPOINT_PATH}")
        try:
            checkpoint = torch.load(config.RESUME_CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf')) # Use get for backward compatibility
            
            print(f"Successfully resumed training. Starting from epoch {start_epoch}.")
            if wandb_run:
                 wandb.log({"status": f"Resumed from epoch {start_epoch-1}", "resumed_checkpoint": config.RESUME_CHECKPOINT_PATH})
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
            # Potentially log this error to wandb as well if needed
            if wandb_run:
                wandb.log({"warning": f"Checkpoint loading failed: {str(e)}. Training from scratch."})
    else:
        if hasattr(config, 'RESUME_CHECKPOINT_PATH') and config.RESUME_CHECKPOINT_PATH:
            print(f"Warning: RESUME_CHECKPOINT_PATH ('{config.RESUME_CHECKPOINT_PATH}') was set, but the file does not exist. Starting training from scratch.")
            if wandb_run:
                 wandb.log({"warning": f"Checkpoint not found at {config.RESUME_CHECKPOINT_PATH}. Training from scratch."})
        else:
            print("No resume checkpoint specified or feature not enabled in config. Starting training from scratch.")


    # --- Training Loop --- #
    print(f"Starting training for {config.NUM_EPOCHS - start_epoch} epochs (from epoch {start_epoch + 1} to {config.NUM_EPOCHS}).")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time() # Record the start time of the epoch.

        # Run one epoch of training.
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, criterion, device,
            config.GRAD_CLIP_VALUE, scheduler, epoch, config.LOG_INTERVAL, wandb_run
        )
        epoch_duration = time.time() - epoch_start_time # Calculate epoch duration.

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Duration: {epoch_duration:.2f}s")
        if wandb_run:
            wandb.log({
                "epoch_train_loss": train_loss,
                "epoch": epoch + 1, # Log epoch starting from 1.
                "epoch_duration_seconds": epoch_duration
            })

        # --- Validation (Periodically) --- #
        if (epoch + 1) % config.VALIDATION_INTERVAL == 0:
            val_start_time = time.time() # Record validation start time.
            val_loss = evaluate(model, val_dataloader, criterion, device)
            val_duration = time.time() - val_start_time # Calculate validation duration.

            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Validation Loss: {val_loss:.4f} | Val Duration: {val_duration:.2f}s")
            if wandb_run:
                wandb.log({
                    "epoch_val_loss": val_loss,
                    "epoch": epoch + 1, # Log epoch starting from 1.
                    "val_duration_seconds": val_duration
                })

            # --- Checkpointing (Save model if validation loss improved) --- #
            if val_loss < best_val_loss:
                best_val_loss = val_loss # Update the best validation loss.
                # Construct checkpoint filename with epoch and validation loss.
                # Sanitize encoder model name for use in filename
                safe_encoder_name = config.ENCODER_MODEL_NAME.replace('/', '_')
                checkpoint_filename = f"{config.CHECKPOINT_PREFIX}_{safe_encoder_name}_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt"
                checkpoint_path = os.path.join(config.OUTPUT_DIR, checkpoint_filename)

                # Prepare checkpoint dictionary
                checkpoint_data = {
                    'epoch': epoch, # Save the completed epoch number (0-indexed)
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': {k: v for k, v in config.__dict__.items() if not k.startswith('__') and not callable(v)} # Save non-private config attributes
                }
                if scheduler:
                    checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                
                try:
                    # Save the model checkpoint using torch.save for the dictionary.
                    torch.save(checkpoint_data, checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path} (Val Loss: {val_loss:.4f})")

                    if wandb_run:
                        # Log an artifact to W&B (optional but good practice).
                        artifact_name = f"{config.WANDB_RUN_NAME if config.WANDB_RUN_NAME else 'model'}-epoch{epoch+1}"
                        model_artifact = wandb.Artifact(artifact_name, type='model', description=f"Model checkpoint at epoch {epoch+1} with val loss {val_loss:.4f}")
                        model_artifact.add_file(checkpoint_path)
                        wandb.log_artifact(model_artifact)
                        print(f"W&B Artifact '{artifact_name}' created and logged.")

                    # --- Hugging Face Hub Model Upload --- #
                    if hf_api and config.HF_UPLOAD_BEST_CHECKPOINTS: # Check if upload is enabled.
                        print(f"Attempting to upload best checkpoint to Hugging Face Hub: {config.HF_REPO_ID}")
                        try:
                            # Upload the checkpoint file to the HF Hub repository.
                            hf_api.upload_file(
                                path_or_fileobj=checkpoint_path,
                                path_in_repo=checkpoint_filename, # Name of the file in the repository.
                                repo_id=config.HF_REPO_ID,
                                repo_type="model"
                            )
                            # Optionally, upload a README or model card if not already present or needs update.
                            # For simplicity, only the checkpoint is uploaded here.
                            print(f"Successfully uploaded checkpoint '{checkpoint_filename}' to {config.HF_REPO_ID} on Hugging Face Hub.")
                        except Exception as e:
                            print(f"Error uploading checkpoint to Hugging Face Hub: {e}")
                            if wandb_run:
                                wandb.log({"warning": f"HF Hub upload failed for {checkpoint_filename}: {str(e)}"})

                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
                    if wandb_run:
                        wandb.log({"error": f"Checkpoint saving failed for {checkpoint_filename}: {str(e)}"})
            else:
                print(f"Validation loss ({val_loss:.4f}) did not improve from best ({best_val_loss:.4f}). Not saving checkpoint.")

    # --- Finalization --- #
    print("Training finished.")
    if wandb_run:
        wandb.finish() # Close the Weights & Biases run.
        print("WandB run finished.")

# Standard Python entry point: execute main() if the script is run directly.
if __name__ == "__main__":
    main() 