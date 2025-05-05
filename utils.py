"""
Utility functions for the project.
"""

import torch
import torch.nn as nn
import math
import config

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    Useful for masking future tokens in self-attention.

    Args:
        sz: Size of the sequence.

    Returns:
        A square mask tensor.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(config.DEVICE)

def create_tgt_mask(tgt: torch.Tensor) -> torch.Tensor:
    """Creates a mask for the target sequence.

    Args:
        tgt: Target sequence tensor (Batch Size, Sequence Length).

    Returns:
        A target mask tensor.
    """
    tgt_seq_len = tgt.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    return tgt_mask

def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Creates a mask for padding tokens in a sequence.

    Args:
        seq: Input sequence tensor (Batch Size, Sequence Length).
        pad_idx: Index of the padding token.

    Returns:
        A padding mask tensor (Batch Size, Sequence Length), where True indicates a padding token.
    """
    # Shape: (batch_size, 1, seq_len)
    # seq_pad_mask = (seq == pad_idx).unsqueeze(1)
    # For PyTorch Transformer, expects (Batch Size, Sequence Length)
    # where True indicates locations that should be ignored (padded)
    seq_pad_mask = (seq == pad_idx)
    return seq_pad_mask.to(config.DEVICE)


class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: Dimension of the embeddings.
            dropout: Dropout probability.
            max_len: Maximum possible sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (Sequence Length, Batch Size, Embedding Dimension)
               Note: Pytorch Transformer layers expect (S, N, E) shape.

        Returns:
            Tensor with positional encoding added.
        """
        # x is expected to be of shape (Seq_Len, Batch_Size, Embedding_Dim)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Add other utility functions as needed (e.g., learning rate scheduler, logging setup) 