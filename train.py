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
import wandb # <-- Add wandb import
from huggingface_hub import HfApi, create_repo # <-- Add Hub imports
from transformers import get_linear_schedule_with_warmup # <-- Add scheduler import

def setup_wandb(cfg):
    """Initializes wandb run."""
    # Combine relevant hyperparameters into a dictionary
    hyperparameters = {
        'encoder_model': cfg.ENCODER_MODEL_NAME,
        'decoder_layers': cfg.DECODER_LAYERS,
        'decoder_heads': cfg.DECODER_HEADS,
        'decoder_ff_dim': cfg.DECODER_FF_DIM,
        'embedding_dim': cfg.DECODER_EMBED_DIM,
        'max_seq_len': cfg.MAX_SEQ_LEN,
        'dropout': cfg.DECODER_DROPOUT,
        'learning_rate': cfg.LEARNING_RATE,
        'epochs': cfg.NUM_EPOCHS,
        'batch_size': cfg.BATCH_SIZE,
        'vocab_size': cfg.VOCAB_SIZE, # Log vocab size after tokenizer setup
        'warmup_steps': cfg.WARMUP_STEPS,
        'adam_beta1': cfg.ADAM_BETA1,
        'adam_beta2': cfg.ADAM_BETA2,
        'adam_eps': cfg.ADAM_EPS,
        'weight_decay': cfg.WEIGHT_DECAY,
        'grad_clip': cfg.GRAD_CLIP_VALUE,
        'projection_dim': cfg.PROJECTION_DIM,
        'image_dir': cfg.IMAGE_DIR,
        'captions_file': cfg.CAPTIONS_FILE,
        'output_dir': cfg.OUTPUT_DIR,
        'checkpoint_prefix': cfg.CHECKPOINT_PREFIX,
        'log_interval': cfg.LOG_INTERVAL,
        'validation_interval': cfg.VALIDATION_INTERVAL
    }
    # Initialize wandb
    run = wandb.init(
        project=cfg.WANDB_PROJECT, # Get project name from config
        entity=cfg.WANDB_ENTITY,   # Get entity name from config (optional)
        config=hyperparameters,
        name=cfg.WANDB_RUN_NAME    # Optional run name from config
    )
    print(f"Wandb run initialized. View at: {run.url}")
    return run

def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip_value, scheduler, epoch, log_interval, wandb_run):
    """Runs one epoch of training."""
    model.train() # Set model to training mode
    total_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False)

    for i, batch in enumerate(progress_bar):
        global_step = epoch * num_batches + i

        # Move data to the configured device
        images = batch["images"].to(device)
        decoder_input_tokens = batch["decoder_input_tokens"].to(device)
        target_tokens = batch["target_tokens"].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        # Model's forward method should now correctly handle image tensors
        logits = model(images, decoder_input_tokens)

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
        if scheduler:
             current_lr = scheduler.get_last_lr()[0] # Get current LR
             scheduler.step() # Step the scheduler
        else:
             current_lr = optimizer.param_groups[0]['lr'] # Get LR from optimizer if no scheduler

        batch_loss = loss.item()
        total_loss += batch_loss
        progress_bar.set_postfix({'loss': batch_loss, 'lr': current_lr})

        # Log metrics to wandb periodically
        if wandb_run and (global_step + 1) % log_interval == 0:
            wandb.log({
                "train_batch_loss": batch_loss,
                "learning_rate": current_lr,
                "global_step": global_step + 1 # Log step starting from 1
            })

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

            # Forward pass
            # Model's forward method handles tensors
            logits = model(images, decoder_input_tokens)

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

    # --- Wandb Setup ---
    wandb_run = setup_wandb(config) # <-- Initialize wandb here

    # --- Hugging Face Hub Setup ---
    print(f"Attempting to create or use repo: {config.HF_REPO_ID}")
    try:
        hf_api = HfApi()
        create_repo(config.HF_REPO_ID, repo_type="model", exist_ok=True)
        print(f"Hugging Face Hub repository '{config.HF_REPO_ID}' ensured.")
    except Exception as e:
        print(f"Warning: Could not create/access Hugging Face Hub repo '{config.HF_REPO_ID}'. Uploads will be skipped. Error: {e}")
        hf_api = None # Set api to None if setup fails

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

    # --- Scheduler (Optional) ---
    if config.WARMUP_STEPS > 0:
        num_training_steps = len(train_loader) * config.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=num_training_steps
        )
        print(f"Using linear warmup scheduler with {config.WARMUP_STEPS} warmup steps.")
    else:
        scheduler = None
        print("No learning rate scheduler used.")

    # --- Training Loop --- #
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        # Pass wandb_run and log_interval to train_one_epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            config.GRAD_CLIP_VALUE, scheduler, epoch, config.LOG_INTERVAL, wandb_run
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Time: {epoch_duration:.2f}s")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\t Val. Loss: {val_loss:.4f}")

        # Log epoch metrics to wandb
        epoch_logs = {
            "train_epoch_loss": train_loss,
            "val_epoch_loss": val_loss,
            "epoch": epoch + 1
            }
        if wandb_run: wandb.log(epoch_logs)

        # --- Save and Upload Best Model --- 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config.OUTPUT_DIR, "best_model_train.safetensors")
            save_file(model.state_dict(), save_path)
            print(f"\tBest model saved to {save_path}")
            # Upload best model checkpoint to Hub
            if hf_api:
                try:
                    hf_api.upload_file(
                        path_or_fileobj=save_path,
                        path_in_repo=os.path.basename(save_path),
                        repo_id=config.HF_REPO_ID,
                        repo_type="model",
                        commit_message=f"Upload best model from epoch {epoch+1} (val_loss: {val_loss:.4f})"
                    )
                    print(f"\tUploaded best model to Hugging Face Hub: {config.HF_REPO_ID}")
                except Exception as e:
                    print(f"\tWarning: Failed to upload best model checkpoint to Hub. Error: {e}")

        # --- Save and Upload Epoch Checkpoint --- 
        chkpt_path = os.path.join(config.OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}_train.safetensors")
        save_file(model.state_dict(), chkpt_path)
        print(f"\tCheckpoint saved to {chkpt_path}")
        # Upload epoch checkpoint to Hub
        if hf_api:
            try:
                hf_api.upload_file(
                    path_or_fileobj=chkpt_path,
                    path_in_repo=os.path.basename(chkpt_path),
                    repo_id=config.HF_REPO_ID,
                    repo_type="model",
                    commit_message=f"Upload checkpoint from epoch {epoch+1}"
                )
                print(f"\tUploaded epoch checkpoint to Hugging Face Hub: {config.HF_REPO_ID}")
            except Exception as e:
                print(f"\tWarning: Failed to upload epoch checkpoint to Hub. Error: {e}")

    print("\nTraining finished.")

    # --- Save and Upload Final Model --- 
    final_save_path = os.path.join(config.OUTPUT_DIR, f"final_model_train.safetensors") # Consistent naming
    save_file(model.state_dict(), final_save_path)
    print(f"Final model state_dict saved at '{final_save_path}'")
    # Note: Optimizer/scheduler state not saved in the safetensors file.
    # Upload final model to Hub
    if hf_api:
        try:
            hf_api.upload_file(
                path_or_fileobj=final_save_path,
                path_in_repo=os.path.basename(final_save_path),
                repo_id=config.HF_REPO_ID,
                repo_type="model",
                commit_message="Upload final model checkpoint"
            )
            print(f"Uploaded final model to Hugging Face Hub: {config.HF_REPO_ID}")
        except Exception as e:
            print(f"Warning: Failed to upload final model checkpoint to Hub. Error: {e}")

    # Optional: Save final model to wandb
    # if wandb_run:
    #     wandb.save(final_save_path) # Save the final safetensors file

    # --- Cleanup --- 
    if wandb_run: wandb.finish() # Ensure wandb finishes cleanly

if __name__ == "__main__":
    # Model forward pass expects tensors, which are provided by the DataLoader.
    # The previous warning is no longer needed.
    main() 