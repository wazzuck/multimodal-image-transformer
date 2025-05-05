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
*   (See `requirements.txt` for specific versions)

## Dataset Setup (Flickr30k)

The model is configured to use the Flickr30k dataset.

*   **Automatic Download:** When you run the training script (`train.py`) for the first time, it will automatically check if the dataset exists in the `./data` directory.
    *   If the dataset (specifically `./data/images/` and `./data/captions.json`) is not found, the script `prepare_dataset.py` will automatically:
        1.  Download the Flickr30k dataset (images and captions CSV) from [awsaf49/flickr-dataset](https://github.com/awsaf49/flickr-dataset) (approx. 4.4 GB compressed).
        2.  Extract the images into `./data/images/`.
        3.  Convert the caption data from the original CSV format into `./data/captions.json` (mapping image filenames to lists of captions).
        4.  Clean up temporary download files.
    *   This process requires the `requests` library (included in `requirements.txt`) and an internet connection. It may take a significant amount of time depending on your connection speed.
*   **Manual Setup (Optional):** If you prefer to download the data manually or use a different dataset, you will need to:
    1.  Place your image files in `./data/images/`.
    2.  Create a `./data/captions.json` file mapping your image filenames to a list of their corresponding captions. The structure should be: `{"image1.jpg": ["caption A", "caption B"], "image2.jpg": ["caption C"], ...}`.
    3.  You might need to adapt `dataset.py` if your data format or structure differs significantly.

## Usage

1.  **Installation:**
    *   Ensure you have Miniconda or Anaconda installed.
    *   (Optional) Run the setup scripts:
        ```bash
        # Installs Miniconda if needed
        bash 00_setup.sh 
        # Installs dependencies from requirements.txt
        bash 01_setup.sh 
        # Make sure conda environment is active or restart shell
        ```
    *   Alternatively, manually create an environment and install requirements:
        ```bash
        # conda create -n multimodal python=3.10 # Example
        # conda activate multimodal
        pip install -r requirements.txt
        ```

2.  **Configuration:**
    *   Modify `config.py` to set hyperparameters, choose the encoder model (`ENCODER_MODEL_NAME`), review file paths (though defaults should work with automatic download), etc.

3.  **Training:**
    ```bash
    python train.py
    ```
    *   The first time you run this, it will attempt to download and prepare the Flickr30k dataset automatically (see Dataset Setup section).
    *   Checkpoints will be saved in the `./outputs` directory (or as configured in `config.py`).

4.  **Inference:**
    ```bash
    python inference.py --image_path /path/to/your/image.jpg --checkpoint /path/to/your/checkpoint.pth
    ```
    *   This will load the specified checkpoint and generate text for the given image.

## File Structure

```
.
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── config.py               # Configuration settings and hyperparameters
├── encoder.py              # Loads image encoder and extracts features
├── decoder.py              # Implements the Transformer Decoder
├── model.py                # Combines encoder and decoder modules
├── tokenizer.py            # Handles text tokenization and vocabulary
├── dataset.py              # Loads and preprocesses image-text data
├── train.py                # Script for model training
├── inference.py            # Script for generating text from an image
├── utils.py                # Utility functions (e.g., positional encoding)
└── data/                   # Directory for dataset
    ├── images/             # Image files (automatically downloaded)
    └── captions.json       # Caption data (automatically downloaded & converted)
└── outputs/                # Directory for saved models/results
└── temp_download/          # Temporary directory used during dataset download (removed after)
``` 
