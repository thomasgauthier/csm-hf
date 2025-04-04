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
      - backbone_config: the Llama parameters for the main backbone
      - decoder_config:  the Llama parameters for the smaller decoder
    Also stores text_vocab_size, audio_vocab_size, and audio_num_codebooks.
    """
    model_type = "csm"

    def __init__(
        self,
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
        max_seq_len=2048,
        **kwargs,
    ):
        self.backbone_flavor = backbone_flavor
        self.decoder_flavor = decoder_flavor
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_num_codebooks = audio_num_codebooks
        self.max_seq_len = max_seq_len

        # Depending on the chosen flavor (e.g. "llama-1B"), define hidden sizes, etc.
        if backbone_flavor == "llama-1B":
            self.backbone_config = {
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 16,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-5,
            }
        elif backbone_flavor == "llama-100M":
            self.backbone_config = {
                "hidden_size": 1024,
                "intermediate_size": 8192,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
            }
        else:
            raise ValueError(f"Unknown backbone flavor: {backbone_flavor}")

        if decoder_flavor == "llama-1B":
            self.decoder_config = {
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 16,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-5,
            }
        elif decoder_flavor == "llama-100M":
            self.decoder_config = {
                "hidden_size": 1024,
                "intermediate_size": 8192,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
            }
        else:
            raise ValueError(f"Unknown decoder flavor: {decoder_flavor}")

        super().__init__(**kwargs)


def _create_llama_config(model_params, max_seq_len):
    """
    Creates a LlamaConfig from a dict of model_params (like hidden_size, etc.)
    and a maximum sequence length. This includes settings for rope scaling.
    """
    return LlamaConfig(
        vocab_size=128256,  # usually matches text_vocab_size
        hidden_size=model_params["hidden_size"],
        num_hidden_layers=model_params["num_hidden_layers"],
        num_attention_heads=model_params["num_attention_heads"],
        num_key_value_heads=model_params["num_key_value_heads"],
        intermediate_size=model_params["intermediate_size"],
        max_position_embeddings=max_seq_len,
        rms_norm_eps=model_params["rms_norm_eps"],
        attention_dropout=0.0,
        rope_theta=500000,  # a large value
        rope_scaling={
            "type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
    )


def create_llama_backbone(config):
    """
    Builds the Llama backbone. We set embed_tokens to nn.Identity() because
    we will manually embed text+audio outside of the LlamaModel.
    """
    llama_config = _create_llama_config(config.backbone_config, config.max_seq_len)
    backbone = LlamaModel(llama_config)
    backbone.embed_tokens = nn.Identity()
    return backbone, config.backbone_config["hidden_size"]


def create_llama_decoder(config):
    """
    Builds the Llama decoder. We set max_seq_len = audio_num_codebooks because
    the decoder processes codebooks [0..N-1] within a single frame, typically up to 32 codebooks.
    """
    llama_config = _create_llama_config(config.decoder_config, config.audio_num_codebooks)
    decoder = LlamaModel(llama_config)
    decoder.embed_tokens = nn.Identity()
    return decoder, config.decoder_config["hidden_size"]


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

        # Create position IDs if missing
        if position_ids is None:
            batch_size, seq_len = input_ids.size()[:2]
            position_ids = (
                torch.arange(seq_len, device=input_ids.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )

        # 1) embed
        embeds = self._embed_tokens(input_ids)  # [B, S, N+1, H]
        masked_embeds = embeds * attention_mask.unsqueeze(-1)  # zero out absent tokens
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
            position_ids=position_ids,
            past_key_values=past_key_values,
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
