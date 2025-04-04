# Sesame CSM Architecture and Training/Inference Process

## Architecture Overview

### Two-Stage Autoregressive Design with Different Token Concepts

- **Stage 1: Backbone (Inter-frame Processing)**
  - Processes the **entire sequence of frames**
  - Each **frame** is treated as a single "token" in the sequence
  - For each frame, all codebook embeddings are summed into a single vector
  - Performs context modeling across frames (horizontal/temporal processing)
  - Handles long-range dependencies between utterances, sentences, etc.
  
- **Stage 2: Decoder (Intra-frame Processing)**
  - Processes **within a single frame**
  - Each **codebook** is treated as a "token" in the sequence
  - First token is the projected backbone hidden state (frame context)
  - Second token is the projected codebook 0 embedding (semantic content)
  - Subsequently generates codebooks 1-31 one by one, adding each to sequence
  - Performs fine-grained acoustic modeling (vertical/spectral processing)

### Model Components
- **Backbone Transformer**: 
  - Large Llama-based model (1B parameters)
  - Processes interleaved text and audio tokens
  - Predicts the zeroth (semantic) codebook through a single classification head
  - Input shape: (batch_size, seq_len, hidden_dim)
  - Output shape: (batch_size, seq_len, hidden_dim)

- **Decoder Transformer**:
  - Smaller Llama-based model (100M parameters)
  - Generates the remaining 31 acoustic codebooks
  - Input shape: (batch_size, seq_len, decoder_dim)
  - Output shape: (batch_size, seq_len, decoder_dim)
  - **Contains 31 separate classification heads (one per codebook)**
    - Each head projects decoder output to vocabulary size for its specific codebook
    - Implemented as a parameter tensor of shape [audio_num_codebooks-1, decoder_dim, audio_vocab_size]

## Tokenization and Data Representation
- Uses Mimi tokenizer (similar to Moshi)
- Produces 32 codebooks per frame
  - 1st codebook: Semantic/linguistic information
  - Remaining 31: Acoustic/audio details
- Frame rate: 12.5 Hz (80ms per frame)
- Input shape: (batch_size, seq_len, audio_num_codebooks+1)
  - Last dimension: text token
  - First N dimensions: audio codebooks

## Training Process

### Data Preparation
- Interleave text and audio tokens
- Encode with speaker markers
- Structured as alternating text and audio sequences
- For each frame:
  ```python
  # Input shape: (batch_size, seq_len, audio_num_codebooks+1)
  
  # Embed text tokens (last position in each frame)
  text_embeds = text_embeddings(tokens[:, :, -1]).unsqueeze(-2)  # [B, S, 1, H]
  
  # Embed audio tokens (first N positions in each frame)
  audio_tokens = tokens[:, :, :-1] + (
      audio_vocab_size * 
      torch.arange(audio_num_codebooks, device=tokens.device)
  )  # [B, S, N]
  
  # Reshape and embed
  audio_embeds = audio_embeddings(audio_tokens.view(-1)).reshape(
      tokens.size(0), tokens.size(1), audio_num_codebooks, -1
  )  # [B, S, N, H]
  
  # Combine audio and text embeddings
  embeds = torch.cat([audio_embeds, text_embeds], dim=-2)  # [B, S, N+1, H]
  
  # Apply attention mask to embeddings
  masked_embeds = embeds * attention_mask.unsqueeze(-1)  # [B, S, N+1, H]
  
  # Sum masked embeddings for backbone input
  h = masked_embeds.sum(dim=2)  # [B, S, H]
  ```

### Optimization: Compute Amortization
- Single forward pass through entire sequence
- Compute amortization trick to reduce memory usage:
  ```python
  # Train zeroth codebook on all frames
  # Train remaining codebooks on only 1/16 of frames
  ```
- Allows full sequence training without massive memory overhead
- No perceivable quality loss from frame sampling

## Inference Process

### Frame-by-Frame Generation

The model uses a unique two-level autoregressive approach:

#### Stage 1: Semantic Content (Backbone - Inter-frame)
```python
# Backbone treats each FRAME as a token in the sequence
# The entire history of frames is processed together

# Embed text and audio tokens separately
embeds = _embed_tokens(context_tokens)  # [B, S(frames), N+1, H]

# Apply attention mask
masked_embeds = embeds * attention_mask.unsqueeze(-1)  # [B, S(frames), N+1, H]

# Sum audio codebook embeddings to create ONE vector per frame
h = masked_embeds.sum(dim=2)  # [B, S(frames), H]

# Process through backbone - each position is ONE FRAME
last_hidden_state = backbone(h, input_pos=input_pos)  # [B, S(frames), H]
last_h = last_hidden_state[:, -1, :]  # [B, H] - representation of final frame

# Predict zeroth (semantic) codebook for next frame
c0_logits = codebook0_head(last_h)  # [B, audio_vocab_size]
c0_sample = sample_topk(c0_logits)  # [B, 1]
c0_embed = audio_embeddings(c0_sample + 0 * audio_vocab_size)  # [B, 1, H]
```

#### Stage 2: Acoustic Details (Decoder - Intra-frame)
```python
# Decoder treats each CODEBOOK as a token in the sequence
# Processing happens WITHIN a single frame

# Reset decoder caches for new frame
decoder.reset_caches()

# Initialize decoder sequence with two tokens:
# 1. Projected backbone context (from processing all previous frames)
# 2. Projected codebook 0 (semantic content for current frame)
projected_h = torch.cat([
    projection(last_h.unsqueeze(1)),  # [B, 1, decoder_dim] - frame context 
    projection(c0_embed)              # [B, 1, decoder_dim] - semantic content
], dim=1)  # [B, 2, decoder_dim]

# Initialize current sample with codebook 0
curr_sample = c0_sample.clone()  # [B, 1]
curr_pos = torch.zeros(B, 2, device=device)  # [B, 2]

# Generate remaining 31 codebooks sequentially
# Each iteration adds one more token to the decoder sequence
for i in range(1, 32):
    # Process entire codebook sequence so far
    decoder_h = decoder(projected_h, input_pos=curr_pos)  # [B, num_codebooks_so_far, decoder_dim]
    last_decoder_h = decoder_h[:, -1, :]  # [B, decoder_dim]
    
    # Generate logits for current codebook using its dedicated classification head
    # Each codebook (1-31) has its own classification head stored in audio_head
    ci_logits = torch.mm(last_decoder_h, audio_head[i-1])  # [B, audio_vocab_size]
    ci_sample = sample_topk(ci_logits)  # [B, 1]
    
    # Embed the new codebook token
    ci_embed = audio_embeddings(ci_sample + i * audio_vocab_size)  # [B, 1, H]
    
    # Add new codebook token to decoder sequence
    projected_h = torch.cat([projected_h, ci_embed], dim=1)  # [B, num_codebooks_so_far+1, H]
    
    # Update position indices for growing codebook sequence
    curr_pos = torch.cat([curr_pos, curr_pos[:, -1:] + 1], dim=1)  # [B, num_codebooks_so_far+1]
    
    # Accumulate samples
    curr_sample = torch.cat([curr_sample, ci_sample], dim=1)  # [B, i+1]
```

After generating all codebooks for the frame, the process repeats for the next frame:

1. The generated codebooks (`curr_sample`) are added to the context tokens
2. When processing the updated context in the next iteration:
   - These codebooks are embedded using `_embed_tokens`
   - The embeddings for all codebooks within each frame are summed together
   - This summing is crucial to transform the multi-codebook representation [B, S, N+1, H] into the backbone's expected input shape [B, S, H]
   - The backbone then processes this condensed representation to predict the next frame

This embedding and summing process ensures that all previously generated audio information (across all codebooks) is properly incorporated into the backbone's context while maintaining the required tensor shapes for processing.

## Performance Considerations

### Inference Bottlenecks
- Sequential generation of 32 codebooks per frame
- Each codebook generation requires embedding lookup
- Limited GPU parallelism within frame generation
- Processing time: ~130ms per 80ms frame (1.6x realtime)

### Optimization Opportunities
- CUDA graphs for compute kernels
- Kernel fusion for embedding operations
- Precision reduction (FP16/INT8)
- Inference engine conversion
- Parallel codebook generation where possible

## Design Tradeoffs
- **Context Understanding**: Backbone's KV cache handles long-range dependencies
- **Semantic-Acoustic Split**: Clean separation between content and delivery
- **High-Quality Audio**: Multiple codebooks capture fine acoustic details
- **Computational Efficiency**: Two-stage design with separate optimizations for each component