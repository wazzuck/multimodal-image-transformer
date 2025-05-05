"""
Combines the image encoder and text decoder into a single model.
"""

import torch
import torch.nn as nn
import config
# Remove direct dependency on encode_image, get encoder model directly
# from encoder import encode_image, get_encoder_output_dim
from transformers import AutoModel # Use AutoModel for flexibility
from decoder import TransformerDecoder

class ImageToTextModel(nn.Module):
    """Model integrating the frozen image encoder and the trainable text decoder."""
    def __init__(self, decoder_vocab_size: int, decoder_embed_dim: int, decoder_heads: int,
                 decoder_layers: int, decoder_ff_dim: int, decoder_max_seq_len: int,
                 decoder_dropout: float, decoder_pad_idx: int):
        """
        Initializes the combined model.

        Args:
            decoder_vocab_size: Size of the decoder's output vocabulary.
            decoder_embed_dim: Embedding dimension for the decoder.
            decoder_heads: Number of attention heads in the decoder.
            decoder_layers: Number of layers in the decoder.
            decoder_ff_dim: Feed-forward dimension in the decoder.
            decoder_max_seq_len: Maximum sequence length for the decoder.
            decoder_dropout: Dropout rate for the decoder.
            decoder_pad_idx: Padding index for the decoder vocabulary.
        """
        super().__init__()

        # --- Load and Freeze Encoder --- 
        print(f"Loading encoder model: {config.ENCODER_MODEL_NAME}...")
        self.encoder = AutoModel.from_pretrained(config.ENCODER_MODEL_NAME)
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval() # Set encoder to evaluation mode
        print("Encoder model loaded and frozen.")

        # Get encoder output dimension directly from the loaded model
        self.encoder_output_dim = self.encoder.config.hidden_size
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_pad_idx = decoder_pad_idx

        # Optional: Projection layer if encoder output dim != decoder embed dim
        if self.encoder_output_dim != self.decoder_embed_dim:
            print(f"Adding projection layer: {self.encoder_output_dim} -> {self.decoder_embed_dim}")
            self.projection = nn.Linear(self.encoder_output_dim, self.decoder_embed_dim)
        else:
            self.projection = nn.Identity() # No projection needed

        # Initialize the trainable decoder
        self.decoder = TransformerDecoder(
            vocab_size=decoder_vocab_size,
            embed_dim=decoder_embed_dim,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
            ff_dim=decoder_ff_dim,
            max_seq_len=decoder_max_seq_len,
            dropout=decoder_dropout,
            pad_idx=decoder_pad_idx
        )

    def forward(self, image_tensors: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass for training.

        Args:
            image_tensors: Batch of preprocessed image tensors (Batch Size, C, H, W).
            tgt_tokens: Target token sequences (Batch Size, Target Sequence Length).

        Returns:
            Output logits from the decoder (Batch Size, Target Sequence Length, Vocab Size).
        """
        # 1. Encode image tensors using the frozen encoder
        # Encoder output typically contains last_hidden_state, pooler_output, etc.
        # We usually need the last_hidden_state for sequence-to-sequence tasks
        # Or just the CLS token embedding (first token's hidden state) as memory
        with torch.no_grad(): # Ensure gradients are not computed for the frozen encoder
             encoder_outputs = self.encoder(pixel_values=image_tensors)
             # Use the hidden state of the [CLS] token (index 0)
             # Shape: (Batch Size, Hidden Size)
             image_features = encoder_outputs.last_hidden_state[:, 0, :]

        # 2. Project encoder features (CLS token embedding)
        # Input shape: (Batch Size, Enc Dim) -> Output shape: (Batch Size, Dec Embed Dim)
        memory_cls = self.projection(image_features)

        # 3. Unsqueeze memory_cls to match decoder's expected memory format 
        # Decoder expects memory: (Batch Size, Memory Seq Len, Dec Embed Dim)
        # Here, our memory sequence length is effectively 1 (just the CLS feature)
        memory = memory_cls.unsqueeze(1) 
        # memory shape: (Batch Size, 1, Decoder Embed Dim)

        # 4. Create memory padding mask (no padding needed for single CLS token memory)
        memory_padding_mask = None

        # 5. Decode using the target tokens and projected encoder memory
        logits = self.decoder(
            tgt_tokens=tgt_tokens,
            memory=memory, # Pass the single-item memory sequence
            memory_padding_mask=memory_padding_mask
        )

        return logits

    def generate(self, image_tensor: torch.Tensor, start_token_id: int, end_token_id: int,
                 max_len: int = config.MAX_SEQ_LEN, method: str = 'greedy',
                 beam_size: int = config.BEAM_SIZE) -> list[int]:
        """
        Generates a sequence of token IDs for a given *preprocessed image tensor*.

        Args:
            image_tensor: A single preprocessed image tensor (1, C, H, W).
            start_token_id: The ID of the start token.
            end_token_id: The ID of the end token.
            max_len: Maximum length of the sequence to generate.
            method: Decoding method ('greedy' or 'beam').
            beam_size: Beam width if using beam search.

        Returns:
            A list of generated token IDs.
        """
        self.eval() # Set model to evaluation mode

        with torch.no_grad():
            # 1. Encode the input image tensor using self.encoder
            encoder_outputs = self.encoder(pixel_values=image_tensor.to(config.DEVICE))
            image_features = encoder_outputs.last_hidden_state[:, 0, :]
            # Shape: (1, Enc Dim)

            # 2. Project encoder features
            memory_cls = self.projection(image_features)
            # Shape: (1, Dec Embed Dim)
            memory = memory_cls.unsqueeze(1)
            # Shape: (1, 1, Dec Embed Dim)

            # 3. Handle potential padding (none needed for CLS memory)
            memory_padding_mask = None

            if method == 'greedy':
                # Initialize the sequence with the start token
                # Shape: (1, 1)
                generated_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=config.DEVICE)

                for _ in range(max_len - 1):
                    # Get decoder output logits for the current sequence
                    # Input tgt_tokens shape: (1, current_len)
                    # Input memory shape: (1, 1, Dec Embed Dim)
                    logits = self.decoder(
                        tgt_tokens=generated_ids,
                        memory=memory,
                        memory_padding_mask=memory_padding_mask
                    )
                    # Logits shape: (1, current_len, Vocab Size)

                    # Get the logits for the last predicted token
                    last_logits = logits[:, -1, :] # Shape: (1, Vocab Size)

                    # Find the token with the highest probability (greedy)
                    next_token_id = torch.argmax(last_logits, dim=-1).unsqueeze(1) # Shape: (1, 1)

                    # Append the predicted token ID to the sequence
                    generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                    # Shape: (1, current_len + 1)

                    # Stop if end token is generated
                    if next_token_id.item() == end_token_id:
                        break

                return generated_ids.squeeze(0).tolist()

            elif method == 'beam':
                # Beam search implementation (more complex)
                # TODO: Implement beam search decoding
                print("Beam search not yet implemented. Falling back to greedy.")
                return self.generate(image_tensor, start_token_id, end_token_id, max_len, method='greedy')
            else:
                raise ValueError(f"Unsupported generation method: {method}")

# Example Usage
if __name__ == '__main__':
    from PIL import Image

    # Model parameters from config
    model = ImageToTextModel(
        decoder_vocab_size=config.VOCAB_SIZE,
        decoder_embed_dim=config.DECODER_EMBED_DIM,
        decoder_heads=config.DECODER_HEADS,
        decoder_layers=config.DECODER_LAYERS,
        decoder_ff_dim=config.DECODER_FF_DIM,
        decoder_max_seq_len=config.MAX_SEQ_LEN,
        decoder_dropout=config.DECODER_DROPOUT,
        decoder_pad_idx=config.PAD_TOKEN_ID
    ).to(config.DEVICE)

    # --- Test forward pass (training simulation) ---
    print("--- Testing forward pass ---")
    batch_size = 4
    # Dummy image *tensors* (replace with actual loading/processing)
    dummy_images_tensor = torch.randn(batch_size, 3, 224, 224, device=config.DEVICE)
    # Dummy target tokens (batch size, seq len)
    # Typically start with <START> and end with <END>, padded
    dummy_tgt = torch.randint(1, config.VOCAB_SIZE, (batch_size, config.MAX_SEQ_LEN), device=config.DEVICE)
    dummy_tgt[:, 0] = config.START_TOKEN_ID # Ensure start token
    dummy_tgt[:, 10:] = config.PAD_TOKEN_ID # Add padding
    dummy_tgt[0, 11] = config.END_TOKEN_ID
    dummy_tgt[1, 15] = config.END_TOKEN_ID

    # Shift target tokens for decoder input during training
    # Input: <START> token1 token2 ... <PAD>
    # Target: token1 token2 ... <END> <PAD>
    decoder_input = dummy_tgt[:, :-1] # Exclude last token
    # decoder_target = dummy_tgt[:, 1:] # Exclude first token (<START>)

    # Pass image tensors to the updated forward method
    logits = model(dummy_images_tensor, decoder_input)
    print(f"Input image tensor shape: {dummy_images_tensor.shape}")
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Output logits shape: {logits.shape}") # Should be (batch_size, seq_len-1, vocab_size)

    # --- Test generate method (inference simulation) ---
    print("\n--- Testing generate method (greedy) ---")
    # Dummy preprocessed image tensor
    dummy_image_tensor = torch.randn(1, 3, 224, 224, device=config.DEVICE)
    generated_sequence = model.generate(
        dummy_image_tensor,
        start_token_id=config.START_TOKEN_ID,
        end_token_id=config.END_TOKEN_ID,
        max_len=20 # Generate a short sequence for testing
    )
    print(f"Generated token IDs: {generated_sequence}")
    # Need a tokenizer to convert IDs back to text 