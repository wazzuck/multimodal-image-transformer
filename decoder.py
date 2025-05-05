"""
Implementation of the Transformer Decoder.
"""

import torch
import torch.nn as nn
from utils import PositionalEncoding, generate_square_subsequent_mask, create_padding_mask
import config

class TransformerDecoder(nn.Module):
    """Standard Transformer Decoder composed of embedding, positional encoding,
       decoder layers, and an output linear layer.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int,
                 ff_dim: int, max_seq_len: int, dropout: float = 0.1, pad_idx: int = 0):
        """
        Args:
            vocab_size: Size of the output vocabulary.
            embed_dim: Dimension of the embeddings.
            num_heads: Number of attention heads.
            num_layers: Number of decoder layers.
            ff_dim: Dimension of the feed-forward network.
            max_seq_len: Maximum sequence length for positional encoding.
            dropout: Dropout probability.
            pad_idx: Index of the padding token in the vocabulary.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len)

        # Stack of Transformer Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True # Expects (N, S, E) input format
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final linear layer to project to vocabulary size
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor,
                memory_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Processes the target sequence tokens and encoder memory to predict the next token logits.

        Args:
            tgt_tokens: Target token sequence (Batch Size, Target Sequence Length).
            memory: Output features from the encoder (Batch Size, Source Sequence Length, Encoder Dimension).
                    Note: This should match the decoder's embed_dim if no projection is used.
            memory_padding_mask: Mask for padding tokens in the encoder output (Batch Size, Source Sequence Length).
                                  True indicates a masked (padded) position.

        Returns:
            Output logits tensor (Batch Size, Target Sequence Length, Vocab Size).
        """
        # 1. Create masks for the target sequence
        # Target mask: prevents attending to future tokens
        tgt_seq_len = tgt_tokens.size(1)
        # generate_square_subsequent_mask expects (T, T) format for nn.TransformerDecoderLayer if batch_first=False
        # but needs (T, T) for nn.TransformerDecoder if batch_first=True ??? Let's check docs.
        # Docs say: If a BoolTensor is provided, positions with True are not allowed to attend.
        # The mask should be shape (T, T)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len) # Shape (T, T)

        # Target padding mask: prevents attending to padding tokens
        # Shape: (N, T), True indicates padding
        tgt_padding_mask = create_padding_mask(tgt_tokens, self.pad_idx)

        # 2. Embed target tokens and add positional encoding
        # Input to embedding: (N, T)
        # Output of embedding: (N, T, E)
        tgt_embedded = self.token_embedding(tgt_tokens) * math.sqrt(self.embed_dim)
        # Input to positional encoding: (N, T, E)
        # Output of positional encoding: (N, T, E)
        # PositionalEncoding class expects (S, N, E) or (N, S, E)?
        # Our implementation expects (S, N, E), but Transformer layers expect (N, S, E) if batch_first=True.
        # Let's adapt PositionalEncoding or transpose here.
        # Adapting PositionalEncoding is cleaner. Let's assume it handles (N, S, E).
        # If PositionalEncoding expects (S, N, E), transpose before and after:
        # tgt_embedded = tgt_embedded.transpose(0, 1)
        # tgt_embedded = self.positional_encoding(tgt_embedded)
        # tgt_embedded = tgt_embedded.transpose(0, 1)

        # Assuming PositionalEncoding is updated for batch_first=True:
        tgt_processed = self.positional_encoding(tgt_embedded)


        # 3. Pass through Transformer Decoder layers
        # Inputs need shape (N, T, E) for target, (N, S, E) for memory
        # Masks: tgt_mask (T, T), tgt_padding_mask (N, T), memory_padding_mask (N, S)
        decoder_output = self.transformer_decoder(
            tgt=tgt_processed,
            memory=memory,
            tgt_mask=tgt_mask,              # Mask for future tokens
            memory_mask=None,               # Mask for memory (optional, maybe from encoder)
            tgt_key_padding_mask=tgt_padding_mask, # Mask for target padding
            memory_key_padding_mask=memory_padding_mask # Mask for memory padding
        )
        # Output shape: (N, T, E)

        # 4. Project to vocabulary size
        # Input: (N, T, E)
        # Output: (N, T, V)
        logits = self.fc_out(decoder_output)

        return logits

# --- Update Positional Encoding for batch_first=True --- #
# Let's redefine it here for clarity, or move to utils and import
import math

class PositionalEncodingBatchFirst(nn.Module):
    """Injects positional information into the input embeddings (batch_first=True)."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Shape: (max_len, d_model / 2)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (max_len, d_model)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (Batch Size, Sequence Length, Embedding Dimension)

        Returns:
            Tensor with positional encoding added.
        """
        # x shape: (N, S, E)
        # self.pe shape: (1, max_len, E)
        # Need self.pe[:, :x.size(1), :] -> shape (1, S, E)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- Replace PositionalEncoding in Decoder class --- #

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int,
                 ff_dim: int, max_seq_len: int, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # Use the batch_first version
        self.positional_encoding = PositionalEncodingBatchFirst(embed_dim, dropout, max_seq_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True # Expects (N, S, E) input format
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor,
                memory_padding_mask: torch.Tensor = None) -> torch.Tensor:
        tgt_seq_len = tgt_tokens.size(1)
        # Generate mask for target sequence (avoids looking ahead)
        # Needs shape (T, T) for nn.TransformerDecoder with batch_first=True
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)

        # Generate padding mask for target sequence
        # Needs shape (N, T), True indicates padding
        tgt_padding_mask = create_padding_mask(tgt_tokens, self.pad_idx)

        # Embed target tokens and add positional encoding
        # Input shape: (N, T), Output shape: (N, T, E)
        tgt_embedded = self.token_embedding(tgt_tokens) * math.sqrt(self.embed_dim)
        tgt_processed = self.positional_encoding(tgt_embedded)

        # Pass through Transformer Decoder layers
        # Inputs: tgt (N, T, E), memory (N, S, E)
        # Masks: tgt_mask (T, T), tgt_key_padding_mask (N, T), memory_key_padding_mask (N, S)
        decoder_output = self.transformer_decoder(
            tgt=tgt_processed,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None, # Optional: if encoder provides its own attention mask
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )
        # Output shape: (N, T, E)

        # Project to vocabulary size
        # Input: (N, T, E), Output: (N, T, V)
        logits = self.fc_out(decoder_output)

        return logits

# Example Usage
if __name__ == '__main__':
    DEVICE = config.DEVICE
    BATCH_SIZE = config.BATCH_SIZE
    MAX_LEN = config.MAX_SEQ_LEN
    VOCAB_SIZE = config.VOCAB_SIZE
    EMBED_DIM = config.DECODER_EMBED_DIM
    NUM_HEADS = config.DECODER_HEADS
    NUM_LAYERS = config.DECODER_LAYERS
    FF_DIM = config.DECODER_FF_DIM
    PAD_IDX = config.PAD_TOKEN_ID

    # Example data
    encoder_output_dim = 768 # Example from ViT-Base
    encoder_seq_len = 197    # Example from ViT-Base (196 patches + 1 CLS)

    # Dummy encoder output (memory)
    # If encoder_output_dim != EMBED_DIM, a projection layer is needed before the decoder
    if encoder_output_dim != EMBED_DIM:
        print(f"Warning: Encoder output dim ({encoder_output_dim}) != Decoder embed dim ({EMBED_DIM}). Requires projection.")
        # Add a projection layer here or in the main model class
        memory_projection = nn.Linear(encoder_output_dim, EMBED_DIM).to(DEVICE)
        memory = torch.randn(BATCH_SIZE, encoder_seq_len, encoder_output_dim).to(DEVICE)
        memory = memory_projection(memory)
    else:
        memory = torch.randn(BATCH_SIZE, encoder_seq_len, EMBED_DIM).to(DEVICE)

    # Dummy target tokens (e.g., shifted captions)
    tgt_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_LEN - 1)).to(DEVICE)
    # Replace some tokens with padding
    tgt_tokens[0, 5:] = PAD_IDX
    tgt_tokens[1, 10:] = PAD_IDX

    # Dummy memory padding mask (optional, assumes no padding here)
    # If encoder output can be padded, create a mask
    memory_padding_mask = torch.zeros(BATCH_SIZE, encoder_seq_len, dtype=torch.bool).to(DEVICE)
    # Example: memory_padding_mask[0, 150:] = True # Mask last tokens of first sample

    # Initialize decoder
    decoder = TransformerDecoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_LEN,
        pad_idx=PAD_IDX
    ).to(DEVICE)

    # Forward pass
    logits = decoder(tgt_tokens, memory, memory_padding_mask)

    print(f"Input target shape: {tgt_tokens.shape}")
    print(f"Input memory shape: {memory.shape}")
    print(f"Output logits shape: {logits.shape}") # Should be (BATCH_SIZE, MAX_LEN - 1, VOCAB_SIZE) 