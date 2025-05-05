"""
Combines the image encoder and text decoder into a single model.
"""

import torch
import torch.nn as nn
import config
from encoder import encode_image, get_encoder_output_dim
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

        self.encoder_output_dim = get_encoder_output_dim()
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

    def forward(self, images: list, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass for training.

        Args:
            images: A list or batch of PIL images.
            tgt_tokens: Target token sequences (Batch Size, Target Sequence Length).
                      These should typically be the ground truth captions/text,
                      shifted right and padded.

        Returns:
            Output logits from the decoder (Batch Size, Target Sequence Length, Vocab Size).
        """
        # 1. Encode images (handle batching if necessary)
        # Assuming encode_image processes one image at a time
        # We need to batch the encoding process
        # Note: The encode_image function needs modification to handle batches or we loop here.
        # For simplicity, let's assume encode_image is updated or we handle batching outside.
        # If looping: (Inefficient)
        # image_features_list = [encode_image(img) for img in images]
        # image_features = torch.cat(image_features_list, dim=0)

        # Let's assume encode_image is adapted for batching later or handled in the dataloader
        # Placeholder: assume image_features is already batched (Batch Size, Enc Seq Len, Enc Dim)
        # This needs proper implementation in encoder.py or dataset.py/train.py

        # --- Placeholder for batched image encoding --- #
        # Example if encode_image handles batches (requires modification in encoder.py)
        # image_features = encode_image_batch(images) # Fictional function

        # --- Simulate batched encoding for now --- #
        # This is just for structural correctness, replace with actual batched encoding
        batch_size = tgt_tokens.size(0)
        enc_seq_len = 197 # Example ViT
        image_features = torch.randn(batch_size, enc_seq_len, self.encoder_output_dim).to(config.DEVICE)
        # --- End Placeholder --- #


        # 2. Project encoder features if dimensions don't match
        memory = self.projection(image_features)
        # memory shape: (Batch Size, Enc Seq Len, Decoder Embed Dim)

        # 3. Create memory padding mask (if encoder output can be padded)
        # Assuming no padding from the encoder for now
        memory_padding_mask = None
        # If encoder output is padded, mask should be created, e.g.:
        # memory_padding_mask = torch.zeros(batch_size, enc_seq_len, dtype=torch.bool).to(config.DEVICE)

        # 4. Decode using the target tokens and encoder memory
        logits = self.decoder(
            tgt_tokens=tgt_tokens,
            memory=memory,
            memory_padding_mask=memory_padding_mask
        )

        return logits

    def generate(self, image, start_token_id: int, end_token_id: int,
                 max_len: int = config.MAX_SEQ_LEN, method: str = 'greedy',
                 beam_size: int = config.BEAM_SIZE) -> list[int]:
        """
        Generates a sequence of token IDs for a given image during inference.

        Args:
            image: A single PIL image.
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
            # 1. Encode the input image
            # encode_image returns shape (1, Enc Seq Len, Enc Dim)
            image_features = encode_image(image)
            # Shape: (1, Enc Seq Len, Enc Dim)

            # 2. Project encoder features
            memory = self.projection(image_features)
            # Shape: (1, Enc Seq Len, Decoder Embed Dim)

            # Handle potential padding in encoder output if any (assuming none for now)
            memory_padding_mask = None

            if method == 'greedy':
                # Initialize the sequence with the start token
                # Shape: (1, 1)
                generated_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=config.DEVICE)

                for _ in range(max_len - 1):
                    # Get decoder output logits for the current sequence
                    # Input tgt_tokens shape: (1, current_len)
                    # Input memory shape: (1, Enc Seq Len, Dec Embed Dim)
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
                return self.generate(image, start_token_id, end_token_id, max_len, method='greedy')
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
    # Dummy images (replace with actual loading)
    images = [Image.new('RGB', (224, 224)) for _ in range(batch_size)]
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

    # Note: The loss calculation in train.py will handle the alignment
    # between logits output and the actual target tokens.
    logits = model(images, decoder_input)
    print(f"Input images: {len(images)}")
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Output logits shape: {logits.shape}") # Should be (batch_size, seq_len-1, vocab_size)

    # --- Test generate method (inference simulation) ---
    print("\n--- Testing generate method (greedy) ---")
    dummy_image = Image.new('RGB', (224, 224))
    generated_sequence = model.generate(
        dummy_image,
        start_token_id=config.START_TOKEN_ID,
        end_token_id=config.END_TOKEN_ID,
        max_len=20 # Generate a short sequence for testing
    )
    print(f"Generated token IDs: {generated_sequence}")
    # Need a tokenizer to convert IDs back to text 