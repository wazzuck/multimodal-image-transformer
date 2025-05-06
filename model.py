"""
Combines the image encoder and text decoder into a single model.
"""

import torch
import torch.nn as nn
import config
# Removed direct dependency on old encoder utilities, using Hugging Face AutoModel directly.
from transformers import AutoModel, AutoFeatureExtractor # For loading pre-trained models and their feature extractors.
from decoder import TransformerDecoder # The custom Transformer decoder module.

class ImageToTextModel(nn.Module):
    """Model integrating a frozen pre-trained image encoder and a trainable text decoder."""
    def __init__(self, decoder_vocab_size: int, decoder_embed_dim: int, decoder_heads: int,
                 decoder_layers: int, decoder_ff_dim: int, decoder_max_seq_len: int,
                 decoder_dropout: float, decoder_pad_idx: int):
        """
        Initializes the combined ImageToTextModel.

        Args:
            decoder_vocab_size: Size of the decoder's output vocabulary.
            decoder_embed_dim: Embedding dimension for the decoder tokens and internal representations.
            decoder_heads: Number of attention heads in the decoder's multi-head attention layers.
            decoder_layers: Number of stacked layers in the Transformer decoder.
            decoder_ff_dim: Dimension of the feed-forward network within each decoder layer.
            decoder_max_seq_len: Maximum sequence length the decoder can process or generate.
            decoder_dropout: Dropout rate applied within the decoder layers for regularization.
            decoder_pad_idx: Index of the padding token in the decoder's vocabulary.
        """
        super().__init__()

        # --- Load and Freeze Encoder ---
        print(f"Loading encoder model: {config.ENCODER_MODEL_NAME}...")
        # Load the pre-trained image encoder model (e.g., ViT, CLIP ViT) from Hugging Face Hub.
        self.encoder = AutoModel.from_pretrained(config.ENCODER_MODEL_NAME)
        # Load the corresponding feature extractor for the chosen encoder.
        # This handles image preprocessing (resizing, normalization) specific to the encoder.
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.ENCODER_MODEL_NAME)

        # Freeze all parameters of the encoder to prevent them from being updated during training.
        # The encoder acts as a fixed feature extractor.
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval() # Set the encoder to evaluation mode (disables dropout, etc.).
        print("Encoder model loaded and frozen.")

        # Determine the output dimension of the encoder from its configuration.
        # This is typically the hidden size of the encoder's last layer.
        self.encoder_output_dim = self.encoder.config.hidden_size
        self.decoder_embed_dim = decoder_embed_dim # Store decoder embedding dimension for clarity.
        self.decoder_pad_idx = decoder_pad_idx # Store decoder padding index.

        # Optional: Add a linear projection layer if the encoder's output dimension
        # does not match the decoder's embedding dimension.
        if self.encoder_output_dim != self.decoder_embed_dim:
            print(f"Adding projection layer: {self.encoder_output_dim} -> {self.decoder_embed_dim}")
            self.projection = nn.Linear(self.encoder_output_dim, self.decoder_embed_dim)
        else:
            # If dimensions match, use an identity layer (no-op).
            self.projection = nn.Identity()

        # Initialize the trainable Transformer decoder.
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
        Performs a forward pass for training or evaluation with teacher forcing.

        Args:
            image_tensors: Batch of preprocessed image tensors, typically of shape
                                   (Batch Size, Channels, Height, Width), as output by the feature extractor.
            tgt_tokens: Target token sequences for teacher forcing, typically of shape
                                (Batch Size, Target Sequence Length). These are the ground truth captions,
                                usually shifted (e.g., input is <START> cap, target is cap <END>).

        Returns:
            Output logits from the decoder, shape (Batch Size, Target Sequence Length, Vocab Size).
        """
        # 1. Encode image tensors using the frozen encoder.
        # `torch.no_grad()` ensures that no gradients are computed for the encoder part,
        # reinforcing that it's frozen.
        with torch.no_grad():
             # The encoder (e.g., ViT) processes pixel values.
             # `encoder_outputs` is a model-specific output object (e.g., BaseModelOutputWithPooling for ViT).
             encoder_outputs = self.encoder(pixel_values=image_tensors)
             # For many ViT models, `last_hidden_state` contains the sequence of hidden states for all patches.
             # The hidden state of the first token (index 0) often corresponds to the [CLS] token,
             # which aggregates global image information. We use this as the image representation.
             # Shape: (Batch Size, Encoder Hidden Size)
             image_features = encoder_outputs.last_hidden_state[:, 0, :]

        # 2. Project encoder features (the [CLS] token embedding) to match decoder's embedding dimension.
        # Input shape: (Batch Size, Encoder Output Dim) -> Output shape: (Batch Size, Decoder Embed Dim)
        memory_cls = self.projection(image_features)

        # 3. Unsqueeze `memory_cls` to create a "memory sequence" for the decoder.
        # The decoder's cross-attention mechanism expects memory in the shape:
        # (Batch Size, Memory Sequence Length, Decoder Embed Dim).
        # Here, our effective memory sequence length is 1, representing the global image feature.
        memory = memory_cls.unsqueeze(1)
        # `memory` shape: (Batch Size, 1, Decoder Embed Dim)

        # 4. Create memory padding mask. Since our memory sequence length is 1 (single CLS token),
        # there's no padding within the memory itself, so no mask is needed here.
        # If we were using all patch embeddings from ViT as memory, a mask might be relevant
        # if they were padded.
        memory_padding_mask = None

        # 5. Pass the target tokens and the prepared image memory to the decoder.
        # The decoder will use `tgt_tokens` for self-attention (causally masked)
        # and `memory` for cross-attention to incorporate image information.
        logits = self.decoder(
            tgt_tokens=tgt_tokens,          # Input tokens for the decoder.
            memory=memory,                  # Image features (from [CLS] token) as memory.
            memory_padding_mask=memory_padding_mask # No padding in memory in this setup.
        )

        return logits # Output logits for each token in the target sequence.

    def generate(self, image, start_token_id, end_token_id, max_len=100, method='greedy', beam_size=3):
        """
        Generates a text caption for a given PIL image using autoregressive decoding.

        Args:
            image: A PIL Image object (single image).
            start_token_id: The integer ID of the START token from the tokenizer.
            end_token_id: The integer ID of the END token from the tokenizer.
            max_len: Maximum length of the generated token sequence (including START/END).
            method: Decoding strategy. Currently supports 'greedy'. 'beam' is a placeholder.
            beam_size: Beam width if `method` is 'beam' (not fully implemented).

        Returns:
            A list of generated token IDs, including START and potentially END tokens.
        """
        self.eval() # Ensure the model is in evaluation mode (disables dropout, etc.).
        device = next(self.parameters()).device # Get the device the model is currently on.

        # Preprocess the input PIL image using the feature extractor associated with the encoder.
        # This typically involves resizing, normalization, and conversion to PyTorch tensors.
        # `return_tensors="pt"` ensures PyTorch tensors are returned.
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs['pixel_values'] # Extract the processed pixel values tensor.
                                            # Shape is typically (1, Channels, Height, Width).

        # Encode the preprocessed image to get image features.
        # This is done within `torch.no_grad()` as the encoder is frozen and we are doing inference.
        with torch.no_grad():
            encoder_outputs = self.encoder(pixel_values=pixel_values)
        # Extract the [CLS] token's hidden state as the global image representation.
        encoder_features = encoder_outputs.last_hidden_state[:, 0, :] # Shape: (1, Encoder Hidden Size)

        # Project encoder features to match the decoder's embedding dimension.
        # Input shape: (1, Encoder Output Dim) -> Output shape: (1, Decoder Embed Dim)
        memory_cls = self.projection(encoder_features)

        # Unsqueeze `memory_cls` to create the memory sequence for the decoder.
        # Shape: (1, 1, Decoder Embed Dim), where Memory Sequence Length is 1.
        memory = memory_cls.unsqueeze(1)

        # Memory padding mask is None as our memory sequence length is 1 (no padding).
        memory_padding_mask = None

        # Initialize the decoder input sequence with the START token ID.
        # Shape: (1, 1) representing (Batch Size, Sequence Length).
        decoder_input_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)

        # --- Greedy Search Decoding ---
        if method == 'greedy':
            # Autoregressively generate tokens one by one.
            for _ in range(max_len - 1): # Loop up to max_len-1 because START token is already included.
                # Get output logits from the decoder for the current input sequence.
                # The decoder internally applies causal masking for self-attention.
                output_logits = self.decoder(
                    tgt_tokens=decoder_input_ids, # Current sequence of generated tokens.
                    memory=memory,                # Image features as memory.
                    memory_padding_mask=memory_padding_mask
                ) # Output shape: (1, Current Sequence Length, Vocab Size)

                # Get the logits for the next token prediction (logits of the last token in the current sequence).
                next_token_logits = output_logits[:, -1, :] # Shape: (1, Vocab Size)
                # Select the token with the highest probability (argmax) as the next token.
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0) # Shape: (1, 1)

                # Append the predicted token ID to the current decoder input sequence.
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)

                # Stop generation if the END token is predicted.
                if next_token_id.item() == end_token_id:
                    break
            # Return the generated sequence of token IDs as a Python list.
            return decoder_input_ids[0].cpu().tolist()

        # --- Beam Search Decoding (Placeholder) ---
        elif method == 'beam':
             # A proper beam search implementation is more complex.
             # It involves maintaining `beam_size` candidate sequences (beams) at each step,
             # expanding them, and selecting the top `beam_size` based on accumulated scores (log probabilities).
             # Libraries like Hugging Face Transformers provide robust `generate` methods with beam search.
             print("Beam search not fully implemented in this example. Falling back to greedy.")
             # Fallback to greedy search for this placeholder.
             return self.generate(image, start_token_id, end_token_id, max_len=max_len, method='greedy')

        else:
            raise ValueError(f"Unsupported generation method: {method}. Choose 'greedy' or 'beam'.")

# Example Usage: This block runs if the script is executed directly.
if __name__ == '__main__':
    from PIL import Image # For creating dummy image objects if needed for testing.

    # Initialize the model using parameters from the config file.
    model = ImageToTextModel(
        decoder_vocab_size=config.VOCAB_SIZE,
        decoder_embed_dim=config.DECODER_EMBED_DIM,
        decoder_heads=config.DECODER_HEADS,
        decoder_layers=config.DECODER_LAYERS,
        decoder_ff_dim=config.DECODER_FF_DIM,
        decoder_max_seq_len=config.MAX_SEQ_LEN,
        decoder_dropout=config.DECODER_DROPOUT,
        decoder_pad_idx=config.PAD_TOKEN_ID
    ).to(config.DEVICE) # Move the model to the configured device (CPU/GPU).

    # --- Test forward pass (simulating a training step) ---
    print("--- Testing forward pass ---")
    batch_size = 4 # Define a sample batch size for testing.
    # Create dummy image tensors. In a real scenario, these would come from a DataLoader
    # and would be preprocessed by the feature extractor.
    # Shape: (Batch Size, Channels, Height, Width)
    dummy_images_tensor = torch.randn(batch_size, 3, 224, 224, device=config.DEVICE)
   
    # Create dummy target token sequences.
    # Shape: (Batch Size, Max Sequence Length)
    # These would be ground truth captions, tokenized and padded.
    dummy_tgt = torch.randint(1, config.VOCAB_SIZE, (batch_size, config.MAX_SEQ_LEN), device=config.DEVICE)
    dummy_tgt[:, 0] = config.START_TOKEN_ID # Ensure each sequence starts with START_TOKEN_ID.
    dummy_tgt[:, 10:] = config.PAD_TOKEN_ID # Add some padding to simulate variable length sequences.
    # Simulate some sequences ending before max length.
    dummy_tgt[0, 11] = config.END_TOKEN_ID
    dummy_tgt[1, 15] = config.END_TOKEN_ID

    # For training a Transformer decoder, the input to the decoder is typically the target sequence
    # shifted right (e.g., excluding the last token), and the labels for loss calculation are
    # the target sequence shifted left (e.g., excluding the first token, like <START>).
    # Here, `decoder_input` is what's fed into the decoder during teacher forcing.
    decoder_input = dummy_tgt[:, :-1] # Example: (<START>, t1, t2, ..., tn-1)
    # The `model.forward` method takes care of producing logits that will be compared against
    # the actual target tokens (e.g., t1, t2, ..., <END>) for loss calculation.

    # Perform a forward pass.
    logits = model(dummy_images_tensor, decoder_input)
    print(f"Input image tensor shape: {dummy_images_tensor.shape}")
    print(f"Decoder input shape: {decoder_input.shape}")
    # Expected output logits shape: (Batch Size, Sequence Length of decoder_input, Vocab Size)
    print(f"Output logits shape: {logits.shape}")

    # --- Test generate method (simulating inference) ---
    print("\n--- Testing generate method (greedy) ---")
    # Create a dummy single image tensor for inference (already "preprocessed").
    # In real inference, you'd start with a PIL image, then use feature_extractor.
    dummy_single_image_pil = Image.new('RGB', (224, 224), color = 'red') # Example PIL image

    # Generate a sequence using the model's generate method.
    generated_sequence = model.generate(
        image=dummy_single_image_pil, # Pass the PIL image.
        start_token_id=config.START_TOKEN_ID,
        end_token_id=config.END_TOKEN_ID,
        max_len=20, # Generate a short sequence for this test.
        method='greedy'
    )
    print(f"Generated token IDs: {generated_sequence}")
    # To see the actual text, you would need a tokenizer instance:
    # tokenizer = get_tokenizer() # Assuming get_tokenizer() is available
    # print(f"Generated text: {tokenizer.decode(generated_sequence)}")