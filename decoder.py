"""
Implementation of the Transformer Decoder module, a core component for sequence generation.
It uses PyTorch's built-in nn.TransformerDecoderLayer and nn.TransformerDecoder.
"""

import torch
import torch.nn as nn
import math # For math.sqrt in embedding scaling and math.log in positional encoding.
import config # Project configuration for default values if needed for testing.
# Utility functions from utils.py for creating necessary masks.
from utils import generate_square_subsequent_mask, create_padding_mask

# --- Positional Encoding (Batch First) --- #
# This version of PositionalEncoding is designed to work with batch_first=True tensors,
# which is the common convention (Batch Size, Sequence Length, Embedding Dimension).
class PositionalEncodingBatchFirst(nn.Module):
    """Injects positional information into input embeddings, assuming batch_first=True.

    The positional encoding uses sine and cosine functions of different frequencies.
    It is added to the input embeddings to give the model information about the
    relative or absolute position of tokens in the sequence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: The dimensionality of the embeddings (and thus the positional encoding).
            dropout: Dropout probability to apply after adding positional encodings.
            max_len: The maximum sequence length for which positional encodings are pre-calculated.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) # Dropout layer.

        # Create a tensor representing positions (0, 1, ..., max_len-1).
        position = torch.arange(max_len).unsqueeze(1) # Shape: (max_len, 1)
        
        # Calculate the division term for the sine and cosine functions.
        # This term creates varying frequencies for different dimensions of the encoding.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Shape: (d_model / 2)

        # Initialize a positional encoding tensor of zeros.
        pe = torch.zeros(max_len, d_model) # Shape: (max_len, d_model)
        
        # Apply sine to even indices in the PE tensor.
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the PE tensor.
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension to `pe` to make it broadcastable with input `x` (N, S, E).
        # Shape becomes (1, max_len, d_model).
        pe = pe.unsqueeze(0)
        
        # Register `pe` as a buffer. Buffers are model state that should be saved and loaded
        # but are not updated by the optimizer during training (unlike parameters).
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x: Input tensor of shape (Batch Size, Sequence Length, Embedding Dimension).

        Returns:
            The input tensor with positional encoding added, same shape as input.
        """
        # `x` shape: (N, S, E) where N is Batch Size, S is Sequence Length, E is Embedding Dim.
        # `self.pe` shape: (1, max_len, E).
        # We need to select the portion of `pe` corresponding to the input sequence length `x.size(1)`.
        # `self.pe[:, :x.size(1), :]` will have shape (1, S, E), which is broadcastable with `x`.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x) # Apply dropout.

# --- Transformer Decoder --- #
class TransformerDecoder(nn.Module):
    """Standard Transformer Decoder architecture.

    This module consists of:
    1. Token Embedding: Converts input token IDs to dense vectors.
    2. Positional Encoding: Adds positional information to the token embeddings.
    3. A stack of nn.TransformerDecoderLayer instances.
    4. A final Linear layer to project decoder outputs to vocabulary logits.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int,
                 ff_dim: int, max_seq_len: int, dropout: float = 0.1, pad_idx: int = 0):
        """
        Initializes the TransformerDecoder.

        Args:
            vocab_size: Size of the output vocabulary (number of possible tokens).
            embed_dim: Dimensionality of the token embeddings and internal model representations.
            num_heads: Number of attention heads in each multi-head attention mechanism.
            num_layers: Number of stacked TransformerDecoderLayers.
            ff_dim: Dimensionality of the feed-forward network within each decoder layer.
            max_seq_len: Maximum sequence length for which positional encodings are generated.
            dropout: Dropout probability used in embeddings, positional encoding, and decoder layers.
            pad_idx: Index of the padding token in the vocabulary, used by the embedding layer.
        """
        super().__init__()
        self.embed_dim = embed_dim # Store embedding dimension.
        self.pad_idx = pad_idx     # Store padding token index.

        # Token embedding layer: maps vocabulary indices to `embed_dim`-dimensional vectors.
        # `padding_idx` ensures that the embedding for the padding token is zero and not updated.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Positional encoding layer (using the batch_first version defined above).
        self.positional_encoding = PositionalEncodingBatchFirst(embed_dim, dropout, max_seq_len)

        # A single TransformerDecoderLayer. This is the building block.
        # `batch_first=True` means input/output tensors have batch size as the first dimension (N, S, E).
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,       # Dimensionality of the input features (must match embed_dim).
            nhead=num_heads,         # Number of attention heads.
            dim_feedforward=ff_dim,  # Dimension of the feed-forward network.
            dropout=dropout,         # Dropout probability.
            batch_first=True         # Input/output format: (Batch Size, Sequence Length, Embedding Dim).
        )
        # Stack multiple decoder layers to form the complete nn.TransformerDecoder.
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final linear layer to project the decoder output (hidden states) to vocabulary logits.
        # Input dimension: `embed_dim`, Output dimension: `vocab_size`.
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self._init_weights() # Initialize model weights.

    def _init_weights(self):
        """Initializes model weights using Xavier uniform distribution for layers with more than 1 dimension."""
        for p in self.parameters():
            if p.dim() > 1: # Apply only to weight matrices, not biases or 1D parameters.
                nn.init.xavier_uniform_(p)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor,
                memory_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward pass of the TransformerDecoder.

        Args:
            tgt_tokens: Target token sequence tensor of shape (Batch Size, Target Sequence Length).
                        These are the input tokens to the decoder (e.g., <START> followed by caption tokens).
            memory: Output features from the image encoder, serving as context for the decoder.
                    Shape: (Batch Size, Source Sequence Length, Encoder Output Dimension).
                    The Encoder Output Dimension must match `embed_dim` (decoder's internal dimension)
                    or be projected to it before being passed to this decoder.
            memory_padding_mask: Optional mask for padding tokens in the encoder output (`memory`).
                                 Shape: (Batch Size, Source Sequence Length). True indicates a padded position.

        Returns:
            Output logits tensor of shape (Batch Size, Target Sequence Length, Vocab Size).
        """
        # 1. Create Masks for the target sequence.
        tgt_seq_len = tgt_tokens.size(1) # Get the length of the target sequences.
        
        # Target mask (tgt_mask): Prevents attention to future tokens (causal masking).
        # This is a square matrix of shape (Target Sequence Length, Target Sequence Length).
        # `generate_square_subsequent_mask` creates this causal mask where True values are masked.
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=tgt_tokens.device)

        # Target padding mask (tgt_padding_mask): Prevents attention to padding tokens in the target sequence.
        # Shape: (Batch Size, Target Sequence Length). True indicates a padded position.
        tgt_padding_mask = create_padding_mask(tgt_tokens, self.pad_idx)

        # 2. Embed target tokens and add positional encoding.
        # `tgt_tokens` shape: (N, T) where N is Batch Size, T is Target Sequence Length.
        # `tgt_embedded` shape: (N, T, E) where E is Embedding Dimension.
        # Scaling by math.sqrt(self.embed_dim) is a common practice from the original Transformer paper.
        tgt_embedded = self.token_embedding(tgt_tokens) * math.sqrt(self.embed_dim)
        # Add positional encoding. `tgt_processed` shape: (N, T, E).
        tgt_processed = self.positional_encoding(tgt_embedded)

        # 3. Pass the processed target sequence and encoder memory through the Transformer Decoder layers.
        # `tgt`: The embedded and positionally encoded target sequence (N, T, E).
        # `memory`: The encoder output (N, S, E), where S is Source (Encoder) Sequence Length.
        # `tgt_mask`: Causal mask for self-attention in the decoder (T, T).
        # `memory_mask`: Not typically used here if cross-attention handles all memory items (set to None).
        # `tgt_key_padding_mask`: Padding mask for the target sequence (N, T).
        # `memory_key_padding_mask`: Padding mask for the encoder memory (N, S).
        decoder_output = self.transformer_decoder(
            tgt=tgt_processed,
            memory=memory,
            tgt_mask=tgt_mask,                     # Causal mask for target self-attention.
            memory_mask=None,                      # Optional, for masking specific memory elements (not common for global image features).
            tgt_key_padding_mask=tgt_padding_mask, # Padding mask for target tokens.
            memory_key_padding_mask=memory_padding_mask  # Padding mask for memory tokens (e.g., if using patch embeddings).
        )
        # `decoder_output` shape: (N, T, E).

        # 4. Project the decoder output to vocabulary size to get logits.
        # Input shape: (N, T, E), Output shape: (N, T, V) where V is Vocab Size.
        logits = self.fc_out(decoder_output)

        return logits

# Example Usage: This block runs if the script is executed directly (e.g., `python decoder.py`).
if __name__ == '__main__':
    print("--- Testing TransformerDecoder --- ")
    # --- Configuration for Test ---
    # Use some default values from config or define them locally for the test.
    DEVICE = config.DEVICE
    BATCH_SIZE = getattr(config, 'BATCH_SIZE', 4) # Default to 4 if not in config for some reason.
    MAX_LEN = config.MAX_SEQ_LEN
    VOCAB_SIZE = config.VOCAB_SIZE
    EMBED_DIM = config.DECODER_EMBED_DIM
    NUM_HEADS = config.DECODER_HEADS
    NUM_LAYERS = config.DECODER_LAYERS
    FF_DIM = config.DECODER_FF_DIM
    PAD_IDX = config.PAD_TOKEN_ID

    # --- Dummy Data for Testing ---
    # Example: Image encoder output (memory for the decoder).
    # This could be from a ViT (e.g., CLS token or sequence of patch embeddings).
    encoder_output_dim_example = 768 # Example: hidden size of a ViT-Base model.
    # If using only CLS token from encoder, source sequence length is 1.
    # If using all patch embeddings, it would be num_patches + 1 (for CLS).
    encoder_memory_seq_len = 1 # Assuming we use a single vector (e.g., CLS token) as memory from encoder.

    # Create dummy encoder output tensor (`memory`).
    # This memory needs to be projected if its dimension doesn't match the decoder's EMBED_DIM.
    if encoder_output_dim_example != EMBED_DIM:
        print(f"Note: Encoder output dim ({encoder_output_dim_example}) differs from Decoder embed dim ({EMBED_DIM}). Requires projection.")
        # Simulate a projection layer as would be done in the main ImageToTextModel.
        projection_layer = nn.Linear(encoder_output_dim_example, EMBED_DIM).to(DEVICE)
        raw_memory = torch.randn(BATCH_SIZE, encoder_memory_seq_len, encoder_output_dim_example).to(DEVICE)
        memory = projection_layer(raw_memory) # Projected memory, shape: (N, S_mem, E_dec).
    else:
        memory = torch.randn(BATCH_SIZE, encoder_memory_seq_len, EMBED_DIM).to(DEVICE)
    
    print(f"Memory (encoder output) shape: {memory.shape}") # Expected: (Batch, EncMemSeqLen, DecEmbedDim)

    # Dummy memory padding mask (optional, depends on how memory is constructed).
    # If encoder_memory_seq_len is 1 (e.g. CLS token), no padding is typically needed for memory.
    # If using variable length patch sequences, this would be relevant.
    memory_padding_mask = None # Or torch.zeros(BATCH_SIZE, encoder_memory_seq_len, dtype=torch.bool).to(DEVICE)
    if memory_padding_mask is not None:
        print(f"Memory padding mask shape: {memory_padding_mask.shape}")

    # Dummy target token sequence (input to the decoder).
    # Typically starts with START_TOKEN and is padded to MAX_LEN.
    tgt_seq_len_example = MAX_LEN - 5 # Example: shorter than MAX_LEN.
    tgt_tokens = torch.randint(low=PAD_IDX + 1, high=VOCAB_SIZE, size=(BATCH_SIZE, tgt_seq_len_example), device=DEVICE)
    # Add START token at the beginning (if not already part of vocab indices from dataloader)
    # For this test, assume token IDs are direct. In practice, use tokenizer.token_to_id().
    start_token_id_example = getattr(config, 'START_TOKEN_ID', 1) # Default if not in config.
    tgt_tokens[:, 0] = start_token_id_example 
    # Add some padding tokens at the end.
    if tgt_seq_len_example < MAX_LEN:
        padding = torch.full((BATCH_SIZE, MAX_LEN - tgt_seq_len_example), PAD_IDX, device=DEVICE)
        tgt_tokens_padded = torch.cat([tgt_tokens, padding], dim=1)
    else:
        tgt_tokens_padded = tgt_tokens[:, :MAX_LEN]
    
    print(f"Target tokens (padded) shape: {tgt_tokens_padded.shape}") # Expected: (Batch, MaxLen)
    print(f"Sample target tokens (first item, first 10): {tgt_tokens_padded[0, :10]}")

    # --- Initialize Decoder --- 
    decoder = TransformerDecoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_LEN,
        pad_idx=PAD_IDX,
        dropout=0.1
    ).to(DEVICE)
    decoder.eval() # Set to evaluation mode for testing (disables dropout).
    print("\nTransformerDecoder initialized.")

    # --- Perform Forward Pass --- 
    with torch.no_grad(): # Disable gradient calculations for inference test.
        logits = decoder(
            tgt_tokens=tgt_tokens_padded, 
            memory=memory, 
            memory_padding_mask=memory_padding_mask
        )
    
    print(f"\nOutput logits shape: {logits.shape}") # Expected: (Batch, MaxLen, VocabSize)
    print("--- TransformerDecoder Test Finished --- ") 