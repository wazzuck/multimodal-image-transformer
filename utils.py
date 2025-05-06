"""
Utility functions for the Multimodal Image Transformer project,
including mask generation and potentially other shared helpers.
"""

import torch
import torch.nn as nn
import math
import config # Access config for device placement, PAD_IDX etc.

def generate_square_subsequent_mask(sz: int, device: str = config.DEVICE) -> torch.Tensor:
    """Generates a square causal mask for preventing attention to future tokens.

    In a Transformer decoder, self-attention layers should only attend to previous
    positions and the current position. This mask ensures that.

    Args:
        sz: The size (sequence length) of the mask to generate (sz x sz).
        device: The device to create the mask on (e.g., 'cpu' or 'cuda').

    Returns:
        A square tensor of shape (sz, sz) where positions with `float('-inf')`
        are masked (not attended to), and positions with `float(0.0)` are allowed.
        The mask is structured such that position `i` can attend to positions `0` to `i`.
    """
    # torch.ones(sz, sz) creates a square matrix of ones.
    # torch.triu(...) keeps the upper triangle of the matrix (including the diagonal).
    # == 1 creates a boolean matrix where True indicates the upper triangle.
    # .transpose(0, 1) flips it so True values are in the lower triangle (and diagonal).
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    
    # Convert the boolean mask to float.
    # Fill positions where mask is False (future positions) with negative infinity.
    # Fill positions where mask is True (current and past positions) with 0.0.
    # This additive mask format is expected by PyTorch's MultiheadAttention.
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask # Return the final mask tensor.

# Note: create_tgt_mask seems redundant as generate_square_subsequent_mask is usually used directly.
# Deprecating or removing it might be clearer unless it serves a specific different purpose.
# def create_tgt_mask(tgt: torch.Tensor) -> torch.Tensor:
#     """Creates a causal mask for the target sequence."""
#     tgt_seq_len = tgt.size(1)
#     tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
#     return tgt_mask

def create_padding_mask(seq: torch.Tensor, pad_idx: int = config.PAD_TOKEN_ID) -> torch.Tensor:
    """Creates a boolean mask for padding tokens in a batch of sequences.

    This mask identifies which elements in the input sequence are padding tokens.
    It's used in attention mechanisms to prevent attending to these padding positions.

    Args:
        seq: Input sequence tensor of shape (Batch Size, Sequence Length).
        pad_idx: The index used to represent padding tokens in the vocabulary.
                 Defaults to `config.PAD_TOKEN_ID`.

    Returns:
        A boolean padding mask tensor of shape (Batch Size, Sequence Length),
        where `True` indicates a padding token position (which should be ignored/masked)
        and `False` indicates a non-padding token position.
    """
    # Compare each element in the sequence tensor `seq` with the `pad_idx`.
    # The result `seq_pad_mask` is a boolean tensor of the same shape as `seq`.
    # `True` where seq[i, j] == pad_idx, `False` otherwise.
    seq_pad_mask = (seq == pad_idx)
    
    # Move the mask to the configured device (e.g., CPU or GPU).
    # While masks are often boolean, moving them ensures compatibility if operations require them on the same device as data.
    return seq_pad_mask.to(config.DEVICE)

# --- Original Positional Encoding (Sequence First) --- #
# Note: This version expects input shape (Sequence Length, Batch Size, Embedding Dimension).
# The `TransformerDecoder` in `decoder.py` now uses `PositionalEncodingBatchFirst`
# which expects (Batch Size, Sequence Length, Embedding Dimension).
# This original class might be useful if working with modules expecting the (S, N, E) format.
class PositionalEncoding(nn.Module):
    """Injects positional information into input embeddings (expects seq_first=True).
    DEPRECATED in favor of PositionalEncodingBatchFirst if using batch_first=True layers.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: Dimension of the embeddings.
            dropout: Dropout probability.
            max_len: Maximum possible sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Same calculation logic as PositionalEncodingBatchFirst, but shapes differ slightly.
        position = torch.arange(max_len).unsqueeze(1) # Shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model) # Shape (max_len, 1, d_model) - note the middle dimension is 1.
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # `pe` buffer shape is (max_len, 1, d_model).
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor expected to be of shape (Sequence Length, Batch Size, Embedding Dimension).

        Returns:
            Tensor with positional encoding added.
        """
        # `x` shape: (S, N, E).
        # `self.pe` shape: (max_len, 1, E).
        # Need to select the first `x.size(0)` positional encodings.
        # `self.pe[:x.size(0)]` has shape (S, 1, E), which broadcasts correctly with `x`.
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Add other utility functions as needed (e.g., specific logging setups, evaluation metrics). 