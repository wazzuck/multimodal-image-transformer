# Multimodal Image Transformer

## Overview

This project implements an image-to-text generation model using a multimodal transfer learning approach. It combines a pre-trained, frozen image encoder (like ViT or CLIP) with a custom-trained Transformer decoder to generate textual descriptions or answer questions based on input images.

The core idea is to leverage the powerful image representations learned by large vision models and adapt them to a text generation task by training only the decoder component.

## Architecture

The model follows a standard Encoder-Decoder architecture:

1.  **Encoder (Frozen):**
    *   Uses a pre-trained Vision Transformer (ViT) or CLIP image encoder loaded from the Hugging Face `transformers` library.
    *   Takes an input image.
    *   Processes the image and outputs a sequence of hidden states (image features/embeddings).
    *   The weights of the encoder are **frozen** during training; only its features are used.

2.  **Decoder (Trainable):**
    *   A standard Transformer Decoder architecture built using PyTorch.
    *   Takes the sequence of image features from the encoder as input to its cross-attention layers.
    *   Takes the previously generated text tokens (shifted right) as input to its self-attention layers.
    *   Predicts the next token in the output text sequence.
    *   The decoder weights are **trained from scratch** on the target image-text dataset.

```
+-----------------+      +----------------------+      +-----------------------+
|  Input Image    | ---> | Frozen Image Encoder | ---> |  Image Features       |
|                 |      | (ViT / CLIP)         |      | (Sequence of Vectors)|
+-----------------+      +----------------------+      +-----------------------+
                                                            |
                                                            | (Cross-Attention Input)
                                                            V
+-----------------+      +----------------------+      +-----------------------+
|  <START> Token  | ---> | Trainable Decoder    | ---> | Predicted Next Token  |
|  Generated Text |      | (Transformer)        | ---> |                       |
|  (Shifted Right)| ---> |                      | ---> | <END> Token           |
+-----------------+      +----------------------+      +-----------------------+
  (Self-Attention Input)
```

## Requirements

*   Python 3.8+
*   PyTorch (>= 1.8)
*   Transformers (Hugging Face)
*   Pillow (PIL)
*   NumPy
*   Requests (for dataset download)
*   tqdm
*   wandb (for experiment tracking)
*   (See `requirements.txt` for specific versions)

## Dataset Setup (Flickr30k)

The model is configured to use the Flickr30k dataset.

*   **Automatic Download:** When you run the training script (`train.py`) for the first time, it will automatically check if the dataset exists in the directory specified by `DATA_DIR` in `config.py` (defaulting to `../assets/multimodal_image_transformer/` relative to the project root).
    *   If the dataset (specifically the images directory, e.g., `../assets/multimodal_image_transformer/images/`, and the captions file, e.g., `../assets/multimodal_image_transformer/captions.json`) is not found, the script `prepare_dataset.py` (which reads its paths from `config.py`) will automatically:
        1.  Download the Flickr30k dataset (images and captions CSV) from [awsaf49/flickr-dataset](https://github.com/awsaf49/flickr-dataset) (approx. 4.4 GB compressed) into a temporary subfolder within your `DATA_DIR`.
        2.  Extract the images into the `images/` subdirectory within your `DATA_DIR` (e.g., `../assets/multimodal_image_transformer/images/`).
        3.  Convert the caption data from the original CSV format into `captions.json` within your `DATA_DIR` (e.g., `../assets/multimodal_image_transformer/captions.json`).
        4.  Clean up temporary download files.
    *   This process requires the `requests` library (included in `requirements.txt`) and an internet connection. It may take a significant amount of time depending on your connection speed.
*   **Manual Setup (Optional):** If you prefer to download the data manually or use a different dataset, you will need to:
    1.  Ensure `config.py` has `DATA_DIR` and `IMAGE_DIR` pointing to your desired locations.
    2.  Place your image files in the directory specified by `IMAGE_DIR` (e.g., `../assets/multimodal_image_transformer/images/`).
    3.  Create a captions file (e.g., `captions.json`) in the location specified by `CAPTIONS_FILE` in `config.py` (e.g., `../assets/multimodal_image_transformer/captions.json`). This file should map your image filenames to a list of their corresponding captions. The structure should be: `{"image1.jpg": ["caption A", "caption B"], "image2.jpg": ["caption C"], ...}`.
    4.  You might need to adapt `dataset.py` if your data format or structure differs significantly.

## Usage

1.  **Installation:**
    *   Ensure you have Miniconda or Anaconda installed.
    *   (Optional) Run the setup scripts:
        ```bash
        # Installs Miniconda if needed
        bash 00_setup.sh 
        # Installs dependencies from requirements.txt (now includes wandb)
        bash 01_setup.sh 
        # Make sure conda environment is active or restart shell
        ```
    *   Alternatively, manually create an environment and install requirements:
        ```bash
        # conda create -n multimodal python=3.10 # Example
        # conda activate multimodal
        pip install -r requirements.txt
        ```
    *   **Weights & Biases Login:** Before running training with wandb enabled, log in to your wandb account:
        ```bash
        wandb login
        ```
        You will be prompted to enter your API key, which you can find at [https://wandb.ai/authorize](https://wandb.ai/authorize).
    *   **Hugging Face Hub Login:** To enable automatic model uploads to the Hugging Face Hub during training, log in using the CLI:
        ```bash
        huggingface-cli login
        ```
        You will be prompted for a token with `write` access, which you can generate at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

2.  **Configuration:**4
    *   Modify `config.py` to set hyperparameters, choose the encoder model (`ENCODER_MODEL_NAME`), review file paths, etc.
    *   **Add Wandb Settings:** Ensure you add the following settings to `config.py` for experiment tracking:
        ```python
        # --- Wandb Configuration ---
        WANDB_PROJECT = "multimodal-image-transformer" # Your project name
        WANDB_ENTITY = None # Your wandb username or team (optional, defaults to your default entity)
        WANDB_RUN_NAME = None # Optional: A specific name for this run (e.g., "vit-base-lr-1e-4")
        ```
    *   **Add Hugging Face Hub Settings:** Configure the target repository for model uploads:
        ```python
        # --- Hugging Face Hub Settings ---
        HF_REPO_ID = "wazzuck/multimodal_image_transformer" # Repository ID (e.g., "your-username/your-repo-name")
        ```

3.  **Training:**
    ```bash
    python train.py
    ```
    *   The first time you run this, it will attempt to download and prepare the Flickr30k dataset automatically (see Dataset Setup section).
    *   Checkpoints will be saved in the directory specified by `OUTPUT_DIR` in `config.py` (which defaults to the same path as `DATA_DIR`, e.g., `../assets/multimodal_image_transformer/`).
    *   If you logged into Hugging Face Hub, checkpoints (`.safetensors` files) will be automatically uploaded to the specified `HF_REPO_ID` at the end of each epoch.

4.  **Inference:**
    ```bash
    python inference.py --image_path /path/to/your/image.jpg --checkpoint /path/to/your/checkpoint.pth
    ```
    *   This will load the specified checkpoint and generate text for the given image.

## File Structure

The project structure is organized as follows. Note that paths for data, tokenizer files, and outputs are now primarily managed through `config.py`.

**Important Note on Data Paths:** The default `DATA_DIR` in `config.py` is set to `"../assets/multimodal_image_transformer/"`. This means the `assets` directory is expected to be a **sibling** to your project directory, not inside it. For example, if your project is at `/home/user/multimodal-image-transformer/`, the data will be stored in `/home/user/assets/multimodal_image_transformer/`.

If you clone the repository, you might need to create the `../assets/` directory yourself adjacent to your cloned project folder, or adjust `DATA_DIR` in `config.py` to point to a different location (e.g., `./data/` to keep it within the project).

The structure *within* the `DATA_DIR` (once resolved) is typically:

```
. (Project Root: e.g., /home/user/your_project_name/)
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── config.py               # Configuration settings, hyperparameters, and DATA PATHS
├── encoder.py              # Loads image encoder and extracts features
├── decoder.py              # Implements the Transformer Decoder
├── model.py                # Combines encoder and decoder modules
├── tokenizer.py            # Handles text tokenization and vocabulary
├── dataset.py              # Loads and preprocesses image-text data
├── train.py                # Script for model training
├── inference.py            # Script for generating text from an image
├── prepare_dataset.py      # Script to download and prepare the dataset (uses config.py for paths)
├── utils.py                # Utility functions (e.g., positional encoding)

# Default location for data, models, etc. (sibling to project root by default):
# ../assets/multimodal_image_transformer/
#  ├── images/             
#  ├── captions.json       
#  ├── vocab.json          
#  ├── merges.txt          
#  ├── model_checkpoint_epoch_X.safetensors 
#  └── temp_download/      (Temporary, removed after dataset prep)
```

*The diagram above shows the project files. The data files (images, captions, etc.) will be located relative to the project root as specified by `DATA_DIR` in `config.py` (defaulting to one level up, then into `assets/multimodal_image_transformer/`).*

### Model Checkpointing

The script automatically saves model checkpoints during training:
- Checkpoints are saved to the directory specified by `OUTPUT_DIR` in `config.py`.
- A checkpoint is saved whenever the model achieves a new best (lowest) validation loss.
- The filename includes the epoch number and the validation loss (e.g., `flickr30k_model_epoch_10_val_loss_0.75.pt`).
- These checkpoints contain the model's state dictionary, optimizer state, scheduler state (if used), the completed epoch number, and the best validation loss achieved, allowing for training resumption.

### Resuming Training

To resume training from a saved checkpoint:
1.  **Locate your checkpoint file**: This will be a `.pt` file in your `OUTPUT_DIR` (e.g., `checkpoints/flickr30k_model_epoch_10_val_loss_0.75.pt`).
2.  **Update Configuration**: In your `config.py` file, set the `RESUME_CHECKPOINT_PATH` variable to the full path of this checkpoint file.
    ```python
    # In config.py
    # ... other configurations ...
    RESUME_CHECKPOINT_PATH = "/path/to/your/checkpoints/flickr30k_model_epoch_10_val_loss_0.75.pt" 
    # Set to None or an empty string (or comment out) to train from scratch.
    ```
3.  **Run the Training Script**: Execute `python train.py` as usual.

The script will automatically detect the `RESUME_CHECKPOINT_PATH`, load the model weights, optimizer state, learning rate scheduler state, and the last completed epoch. Training will then continue from where it left off for the remaining epochs specified by `NUM_EPOCHS`.

If `RESUME_CHECKPOINT_PATH` is not set, is `None`, or points to a non-existent file, training will start from scratch.

### Experiment Tracking with Weights & Biases

<!-- ... existing code ... -->
