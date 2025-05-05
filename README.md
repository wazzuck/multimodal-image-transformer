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
*   (See `requirements.txt` for specific versions)

## Data Format

The model expects a dataset of image-text pairs. A common format would be:

*   A directory containing all image files (e.g., `data/images/`).
*   A corresponding metadata file (e.g., `data/captions.json` or `data/captions.csv`) mapping image filenames to their textual descriptions/captions.

The `dataset.py` script needs to be adapted based on the specific structure of your dataset.

## Usage

1.  **Installation:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    *   Modify `config.py` to set hyperparameters, model choices (ViT/CLIP encoder), data paths, output directories, etc.

3.  **Training:**
    ```bash
    python train.py
    ```
    *   Checkpoints will be saved in the directory specified in `config.py`.

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
└── data/                   # (Example) Directory for dataset
    ├── images/
    └── captions.json
└── outputs/                # (Example) Directory for saved models/results
``` 
