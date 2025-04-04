"""
Defines the CSMModel class for a two-stage multimodal architecture:
(1) Backbone (Llama) that processes frames at a coarse level to predict
    the first (semantic) codebook.
(2) Decoder (smaller Llama) that refines each frame's representation,
    predicting the remaining acoustic codebooks one by one.

This is done by embedding text tokens (for textual context) and
audio codebooks (32 of them) together. Each frame is a combination
of (up to) 32 audio codebooks plus a single text token.

Key points:
- The backbone sees each frame as one "token," formed by summing all
  audio codebook embeddings (and optionally a text embedding).
- The decoder then operates within that single frame to produce
  all remaining codebooks [1..N-1], conditioned on codebook0.
- This file also contains a specialized generate_frame method for
  inference, in which we generate codebook0 from the backbone and
  then decode codebooks [1..N-1] sequentially.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class CSMOutput(ModelOutput):
    """
    The output structure for CSMModel, carrying:
    - last_hidden_state:   For convenience, the final hidden state from the backbone
    - logits:              The final frame's codebook0 logits (i.e., the backbone's
                           predicted distribution for codebook0 at the last position)
    - past_key_values:     Optional KV cache for inference
    - samples:             If a generation call is made (generate_frame), the resulting
                           codebook tokens
    - loss:                The total training loss
    - backbone_loss:       The portion of the loss from predicting codebook0
    - decoder_loss:        The portion of the loss from predicting codebooks [1..N-1]
    """
    last_hidden_state: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    samples: Optional[torch.Tensor] = None
    loss: Optional[torch.FloatTensor] = None
    backbone_loss: Optional[torch.FloatTensor] = None
    decoder_loss: Optional[torch.FloatTensor] = None


class CSMConfig(PretrainedConfig):
    """
    Configuration holding two "sub-configs":
      - backbone_config: the LlamaConfig for the main backbone
      - decoder_config:  the LlamaConfig for the smaller decoder
    Also stores text_vocab_size, audio_vocab_size, and audio_num_codebooks.
    """
    model_type = "csm"

    def __init__(
        self,
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
        max_seq_len=2048,
        backbone_config=LlamaConfig(
            vocab_size=128256,
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=16,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=2048,
            rms_norm_eps=1e-5,
            attention_dropout=0.0,
            rope_theta=500000,
            rope_scaling={
                "type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
            architectures=["LlamaForCausalLM"],
            hidden_act="silu",
        ),
        decoder_config=LlamaConfig(
            vocab_size=128256,
            hidden_size=1024,
            intermediate_size=8192,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=32,
            rms_norm_eps=1e-5,
            attention_dropout=0.0,
            rope_theta=500000,
            rope_scaling={
                "type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
            architectures=["LlamaForCausalLM"],
            hidden_act="silu",
        ),
        **kwargs,
    ):
        # Set parameters
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_num_codebooks = audio_num_codebooks
        self.max_seq_len = max_seq_len
        
        # Create new config objects to avoid modifying the defaults
        # For backbone config
        if isinstance(backbone_config, dict):
            # If it's a dict (from JSON), create a LlamaConfig from it
            self.backbone_config = LlamaConfig(**backbone_config)
        else:
            # If it's already a LlamaConfig, make a copy to avoid modifying the original
            self.backbone_config = LlamaConfig(**backbone_config.to_dict())
        
        # Update the new config
        self.backbone_config.vocab_size = text_vocab_size
        self.backbone_config.max_position_embeddings = max_seq_len
        
        # For decoder config
        if isinstance(decoder_config, dict):
            # If it's a dict (from JSON), create a LlamaConfig from it
            self.decoder_config = LlamaConfig(**decoder_config)
        else:
            # If it's already a LlamaConfig, make a copy to avoid modifying the original
            self.decoder_config = LlamaConfig(**decoder_config.to_dict())
        
        # Update the new config
        self.decoder_config.vocab_size = text_vocab_size
        self.decoder_config.max_position_embeddings = audio_num_codebooks

        super().__init__(**kwargs)


def topk_multinomial_sampling(logits: torch.Tensor, topk: int, temperature: float):
    """
    Simple top-k sampling: keep only the top-k logits, scale by temperature,
    draw a single sample from the resulting distribution.
    """
    # (Optionally apply temperature)
    logits = logits / temperature
    # Filter to top-k
    topvals, topidx = torch.topk(logits, k=topk, dim=-1)
    probs = torch.nn.functional.softmax(topvals, dim=-1)
    # Draw from the distribution
    sample_rel = torch.multinomial(probs, num_samples=1)  # index in [0..topk-1]
    # Map back to absolute token IDs
    sample_abs = torch.gather(topidx, -1, sample_rel)
    return sample_abs


def create_llama_backbone(config):
    """
    Builds the Llama backbone. We set embed_tokens to nn.Identity() because
    we will manually embed text+audio outside of the LlamaModel.
    """
    backbone = LlamaModel(config.backbone_config)
    backbone.embed_tokens = nn.Identity()
    return backbone, config.backbone_config.hidden_size


def create_llama_decoder(config):
    """
    Builds the Llama decoder. We set max_seq_len = audio_num_codebooks because
    the decoder processes codebooks [0..N-1] within a single frame, typically up to 32 codebooks.
    """
    decoder = LlamaModel(config.decoder_config)
    decoder.embed_tokens = nn.Identity()
    return decoder, config.decoder_config.hidden_size


def _multinomial_sample_one_no_sync(probs):
    """
    Multinomial sampling that avoids certain CUDA sync overhead.
    This technique is often used for efficient sampling in a loop.
    """
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    """
    Top-k sampling with temperature for codebook generation, commonly used
    to limit the distribution to the top k tokens, scaled by temperature.
    """
    logits = logits / temperature
    filter_value: float = -float("Inf")
    kth_val = torch.topk(logits, topk)[0][..., -1, None]
    indices_to_remove = logits < kth_val
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)

    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)
    return _multinomial_sample_one_no_sync(probs)


class CSMModel(PreTrainedModel):
    """
    Two-stage CSM model:

    1) Backbone (a LlamaModel) processes frames. Each frame is formed by
       summing the embeddings of up to 32 codebooks + 1 text embedding.

       It produces 'h' across all frames in the sequence, and predicts codebook0
       (the "semantic codebook").

    2) Decoder (a smaller LlamaModel) processes each frame's codebooks
       to produce codebooks [1..N-1], using separate classification heads
       (self.audio_head).

    The shape [batch_size, seq_len, audio_num_codebooks+1] is used for input:
      - columns [0..(N-1)] are the N audio codebooks
      - column [N] is the text token
    """

    config_class = CSMConfig
    base_model_prefix = "csm"

    def __init__(self, config):
        super().__init__(config)

        # Build backbone & decoder
        self.backbone, backbone_dim = create_llama_backbone(config)
        self.decoder, decoder_dim = create_llama_decoder(config)

        # Separate embedding tables for text and audio
        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )

        # Linear projection from backbone_dim -> decoder_dim
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)

        # Codebook0 classification head
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)

        # Classification heads for codebooks [1..N-1], each a slice in self.audio_head
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size
            )
        )

        # Standard HF post_init (e.g. weight initialization)
        self.post_init()

        # To track if we use HF's caching
        self._using_kv_cache = False

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """
        Converts a single codebook index + tokens into embeddings.
        The offset ensures codebook i uses [i * audio_vocab_size .. (i+1)*audio_vocab_size-1].
        """
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Takes shape [B, S, N+1], where the last column is text, and the
        first N columns are the audio codebooks. We offset each audio codebook
        so they map into separate slices of self.audio_embeddings, then combine
        them into shape [B, S, N+1, H].
        """
        # text embedding
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)  # [B, S, 1, H]

        # audio embedding
        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )  # [B, S, N]
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )  # [B, S, N, H]

        # Concatenate audio codebooks + text in the last dimension
        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def setup_caches(self, max_batch_size: int):
        """
        Optionally enable cached inference. Not strictly used in training loops.
        """
        self._using_kv_cache = True

    def reset_caches(self):
        """
        Reset any existing caches. The actual HF logic typically
        handles this automatically by passing past_key_values=None.
        """
        pass

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        temperature=1.0,
        topk=50,
        generate_frame=False,
        labels=None,
    ):
        """
        Forward pass:
          1) Embeds tokens (text + audio codebooks).
          2) Sums across codebooks+text to create a single vector per frame [B, S, H].
          3) Passes these frames through the backbone to get codebook0 logits.
          4) If labels are provided, computes:
             - codebook0 cross-entropy with a causal shift
             - codebooks [1..N-1] cross-entropy by re-embedding the valid frames,
               passing them to the decoder.
          5) Returns a CSMOutput with optional loss.

        'labels' should match shape [B, S, N+1], with -100 in positions we ignore.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self._using_kv_cache

        # 1) embed
        embeds = self._embed_tokens(input_ids)  # [B, S, N+1, H]
        masked_embeds = embeds * attention_mask.unsqueeze(-1) if attention_mask is not None else embeds # zero out absent tokens
        # sum across the N+1 dimension => [B, S, H]
        h = masked_embeds.sum(dim=2)

        # Convert to a HF-friendly (1 for present, 0 for masked) attention
        if attention_mask is not None:
            hf_attention_mask = (attention_mask.sum(dim=-1) > 0).float()
        else:
            hf_attention_mask = torch.ones(input_ids.size()[:2], device=input_ids.device)

        # 2) backbone forward => codebook0 logits
        backbone_outputs = self.backbone(
            inputs_embeds=h,
            attention_mask=hf_attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        h = backbone_outputs.last_hidden_state  # [B, S, hidden_size]
        backbone_past_key_values = backbone_outputs.past_key_values if use_cache else None

        # codebook0 logits => [B, S, audio_vocab_size]
        c0_all_logits = self.codebook0_head(h)

        # for convenience we also return the final frame's hidden/logits
        last_h = h[:, -1, :]
        c0_logits = c0_all_logits[:, -1, :]

        # 3) compute losses if 'labels' is given
        loss = None
        backbone_loss = None
        decoder_loss = None

        if labels is not None:
            # codebook0 cross-entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            c0_labels = labels[:, :, 0]  # shape [B, S]

            # typical shift: we predict label[t+1] from hidden[t]
            logits_for_loss = c0_all_logits[:, :-1, :].contiguous()
            shift_labels = c0_labels[:, 1:].contiguous()

            logits_for_loss = logits_for_loss.view(-1, self.config.audio_vocab_size)
            shift_labels = shift_labels.view(-1).to(logits_for_loss.device)

            backbone_loss = loss_fct(logits_for_loss.float(), shift_labels)
            loss = backbone_loss

            # codebooks [1..N-1] cross-entropy
            audio_tokens = input_ids[:, :, : self.config.audio_num_codebooks]  # [B, S, N]
            audio_labels = labels[:, :, : self.config.audio_num_codebooks]      # [B, S, N]

            # frames that have no codebook= -100 => skip
            valid_frames_mask = (audio_labels != -100).all(dim=2)
            valid_frame_indices = valid_frames_mask.nonzero(as_tuple=False)

            if valid_frame_indices.numel() > 0:
                num_valid_frames = valid_frame_indices.shape[0]

                # gather hidden states, codebooks, labels for these frames
                valid_frame_hidden = h[valid_frame_indices[:, 0], valid_frame_indices[:, 1]]
                valid_frame_codebooks = audio_tokens[valid_frame_indices[:, 0], valid_frame_indices[:, 1]]
                valid_frame_labels = audio_labels[valid_frame_indices[:, 0], valid_frame_indices[:, 1]]

                # project backbone hidden to decoder dimension
                projected_frame_hidden = self.projection(valid_frame_hidden)

                # offset each codebook
                codebook_offsets = (
                    torch.arange(self.config.audio_num_codebooks, device=valid_frame_codebooks.device)
                    * self.config.audio_vocab_size
                )
                codebook_indices = valid_frame_codebooks + codebook_offsets

                # embed + project each codebook
                codebook_embeds = self.audio_embeddings(codebook_indices.view(-1))
                codebook_embeds = codebook_embeds.view(
                    num_valid_frames, self.config.audio_num_codebooks, -1
                )
                projected_codebook_embeds = self.projection(codebook_embeds)

                # build decoder input: position 0 => frame_hidden, positions 1.. => codebooks
                decoder_inputs = torch.cat(
                    [
                        projected_frame_hidden.unsqueeze(1),
                        projected_codebook_embeds,
                    ],
                    dim=1,
                )
                decoder_outputs = self.decoder(inputs_embeds=decoder_inputs, return_dict=True)
                decoder_hidden = decoder_outputs.last_hidden_state  # [f, 1+N, dec_dim]

                # for predicting codebooks [1..N-1], we skip the first hidden pos
                codebook_hidden = decoder_hidden[:, 1 : self.config.audio_num_codebooks, :]

                # compute logits for each codebook dimension
                codebook_logits = torch.einsum("fcd,cdv->fcv", codebook_hidden, self.audio_head)
                # shape => [f, N-1, audio_vocab_size]

                codebook_targets = valid_frame_labels[:, 1:]
                flat_logits = codebook_logits.reshape(-1, self.config.audio_vocab_size)
                flat_targets = codebook_targets.reshape(-1)

                decoder_loss = nn.CrossEntropyLoss(ignore_index=-100)(flat_logits, flat_targets)
            else:
                decoder_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)

            loss = backbone_loss + decoder_loss

        if not return_dict:
            outputs = (last_h, c0_logits)
            if loss is not None:
                outputs = (loss,) + outputs
            if use_cache:
                outputs = outputs + (backbone_past_key_values,)
            return outputs

        return CSMOutput(
            last_hidden_state=last_h,
            logits=c0_logits,
            past_key_values=backbone_past_key_values,
            loss=loss,
            backbone_loss=backbone_loss,
            decoder_loss=decoder_loss,
        )

    def generate_frame(
        self,
        input_ids,
        attention_mask,
        position_ids=None,
        temperature=1.0,
        topk=50,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Generates a single audio frame by:
          1) Running backbone forward pass to get codebook0
          2) Iteratively decoding codebooks [1..N-1] using the smaller decoder,
             refeeding the newly generated codebook each time.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self._using_kv_cache

        # forward pass for codebook0
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_h = outputs.last_hidden_state
        c0_logits = outputs.logits
        backbone_past_key_values = outputs.past_key_values if use_cache else None

        batch_size = last_h.size(0)
        generated_tokens = torch.zeros(
            batch_size,
            self.config.audio_num_codebooks,
            dtype=torch.long,
            device=last_h.device,
        )

        # sample codebook0
        c0_sample = sample_topk(c0_logits, topk, temperature)
        generated_tokens[:, 0] = c0_sample.squeeze()

        # embed codebook0 and combine with the backbone hidden state
        c0_embed = self._embed_audio(0, c0_sample)
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(batch_size, 1)
        )
        projected_h = self.projection(curr_h)

        # run the decoder once
        decoder_outputs = self.decoder(
            inputs_embeds=projected_h,
            position_ids=curr_pos,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # decode codebooks [1..N-1] sequentially
        for i in range(1, self.config.audio_num_codebooks):
            hidden_states = decoder_outputs.last_hidden_state
            ci_logits = torch.matmul(hidden_states[:, -1, :], self.audio_head[i - 1])

            ci_sample = sample_topk(ci_logits, topk, temperature)
            generated_tokens[:, i] = ci_sample.squeeze()

            # skip further decoding if we just generated the last codebook
            if i < self.config.audio_num_codebooks - 1:
                ci_embed = self._embed_audio(i, ci_sample)
                projected_ci_embed = self.projection(ci_embed)
                next_pos = torch.full((batch_size, 1), i + 1, device=ci_sample.device)

                decoder_outputs = self.decoder(
                    inputs_embeds=projected_ci_embed,
                    position_ids=next_pos,
                    past_key_values=decoder_outputs.past_key_values,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )

        if not return_dict:
            return generated_tokens

        return CSMOutput(
            last_hidden_state=last_h,
            logits=c0_logits,
            past_key_values=backbone_past_key_values,
            samples=generated_tokens,
            loss=None,
            backbone_loss=None,
            decoder_loss=None,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_frames: int = 100,
        temperature: float = 1.0,
        topk: int = 50,
        use_cache: bool = True,
        stop_on_all_zeros: bool = True,
    ):
        r"""
        Autoregressively generate multiple audio frames (each consisting of 32 codebook tokens),
        using the internal `generate_frame(...)` method at every step.

        * **KV Cache**: We pass `past_key_values` forward at each iteration so the
          backbone does not re-encode all previous frames from scratch.
        * **Context**: The first call uses your full `input_ids` / `attention_mask` as context.
          Subsequent calls only feed the newly generated frame to the backbone,
          while the old context is implicitly in `past_key_values`.
        * **Stopping**: If `stop_on_all_zeros=True`, we break if the model
          generates an all-zero frame (often used as a naive "end of audio" condition).

        **Args**:
          - **input_ids**: shape `[B, T, 33]` (if 32 codebooks + 1 text column)  
            The initial context frames.  If you have no prior frames and only text,
            you can supply something like `[B, text_length, 33]` with codebooks=0 and
            text tokens in the last column.
          - **attention_mask**: same shape, marking which tokens are valid (1) vs. padded (0).
          - **max_new_frames**: how many new frames to generate.
          - **temperature**: sampling temperature for codebook generation.
          - **topk**: top-k sampling cutoff.
          - **use_cache**: whether to pass the key/value cache forward.
          - **stop_on_all_zeros**: if `True`, stop generation as soon as a frame of all zeros
            is produced.
        
        **Returns**:
          - A tensor of shape `[B, n_frames, 32]` containing all newly generated frames.
            If none are generated, returns an empty tensor `[B, 0, 32]`.
        """

        device = input_ids.device
        batch_size = input_ids.size(0)

        # We will collect new frames (each is [B, 32]) in a list.
        generated_frames = []

        # Keep track of the backbone's KV cache from step to step.
        backbone_past_key_values = None

        # For each new frame:
        running_input_ids = input_ids
        running_attention_mask = attention_mask

        # We also need to keep track of the "current position IDs":
        # By default if you pass None, HF will generate them. But for
        # incremental decoding, it is typical to increment them manually.
        # If your model's rope setup handles it automatically, you can set `position_ids=None`.
        position_ids = None

        for frame_idx in range(max_new_frames):
            # Run 'generate_frame(...)' to produce the *next* frame's 32 codebook tokens
            # from the current context (which might be the full conversation the first time,
            # and a single frame + KV cache thereafter).
            out = self.generate_frame(
                input_ids=running_input_ids,
                attention_mask=attention_mask if frame_idx == 0 else None,
                temperature=temperature,
                topk=topk,
                past_key_values=backbone_past_key_values,
                use_cache=use_cache,
                return_dict=True,
            )
            new_frame = out.samples  # shape [B, 32]
            # Update the backbone KV cache
            backbone_past_key_values = out.past_key_values

            # If the model generated "all zero" frames, treat that as an end-of-audio signal.
            if stop_on_all_zeros and torch.all(new_frame == 0):
                break

            # Store the newly generated frame
            generated_frames.append(new_frame)

            # Prepare for the next iteration.  In HF incremental decoding, you typically feed only
            # the newly generated token(s) for the next step, while letting `past_key_values`
            # carry the prior context. Here, one "frame" is effectively the new "token."
            #
            # So we build [B, 1, 33] by concatenating the new 32 codebook tokens plus a
            # "dummy" text column (often 0).  Then we set the attention_mask to 1's for audio
            # codebooks and 0 for the text column, matching the manual loop approach.
            zero_text = torch.zeros((batch_size, 1), dtype=new_frame.dtype, device=device)
            next_row = torch.cat([new_frame, zero_text], dim=1).unsqueeze(1)  # [B, 1, 33]
            
            # Create mask with 1's for audio codebooks and 0 for text column
            # This matches the original manual loop's masking logic where text is masked out
            next_mask = torch.zeros((batch_size, 1, 33), dtype=running_attention_mask.dtype, device=device)
            next_mask[:, :, :32] = 1  # Set 1's for the 32 audio codebooks
            
            running_input_ids = next_row
            running_attention_mask = next_mask

        # Combine all newly generated frames into a single tensor:
        if len(generated_frames) > 0:
            # shape => [B, num_frames, 32]
            generated_frames = torch.stack(generated_frames, dim=1)
        else:
            # If no frames were generated, return an empty placeholder
            generated_frames = torch.zeros((batch_size, 0, 32), dtype=torch.long, device=device)

        return generated_frames
