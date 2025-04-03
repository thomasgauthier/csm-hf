from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class CSMOutput(ModelOutput):
    """
    Output type for CSM model.
    """

    last_hidden_state: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    samples: Optional[torch.Tensor] = None
    loss: Optional[torch.FloatTensor] = None
    backbone_loss: Optional[torch.FloatTensor] = None
    decoder_loss: Optional[torch.FloatTensor] = None


class CSMConfig(PretrainedConfig):
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

        # Add LLaMA configuration parameters based on flavor
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

        # Add decoder configuration
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
    """Create a LlamaConfig from model parameters"""
    return LlamaConfig(
        vocab_size=128256,  # Match the text_vocab_size from ModelArgs
        hidden_size=model_params["hidden_size"],
        num_hidden_layers=model_params["num_hidden_layers"],
        num_attention_heads=model_params["num_attention_heads"],
        num_key_value_heads=model_params["num_key_value_heads"],
        intermediate_size=model_params["intermediate_size"],
        max_position_embeddings=max_seq_len,
        rms_norm_eps=model_params["rms_norm_eps"],
        attention_dropout=0.0,
        rope_theta=500000,  # rope_base in original
        rope_scaling={
            "type": "llama3",
            "factor": 32.0,  # scale_factor in original
            "low_freq_factor": 1.0,  # from torchtune implementation
            "high_freq_factor": 4.0,  # from torchtune implementation
            "original_max_position_embeddings": 8192,  # old_context_len
        },
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
    )


def create_llama_backbone(config):
    """Create a LlamaModel backbone from a CSMConfig"""
    llama_config = _create_llama_config(config.backbone_config, config.max_seq_len)
    backbone = LlamaModel(llama_config)
    backbone.embed_tokens = nn.Identity()
    return backbone, config.backbone_config["hidden_size"]


def create_llama_decoder(config):
    """Create a LlamaModel decoder from a CSMConfig"""
    # For decoder, use audio_num_codebooks as max_seq_len since that's the maximum
    # number of codebook tokens we'll process in one go - matches the original implementation
    llama_config = _create_llama_config(
        config.decoder_config, config.audio_num_codebooks
    )
    decoder = LlamaModel(llama_config)
    decoder.embed_tokens = nn.Identity()
    return decoder, config.decoder_config["hidden_size"]


def _index_causal_mask(attention_mask, position_ids=None):
    """
    Convert attention mask to the mask expected by the LlamaModel.

    Args:
        attention_mask: [batch_size, seq_len, audio_num_codebooks+1]
        position_ids: [batch_size, seq_len]

    Returns:
        [batch_size, seq_len, seq_len]: Causal attention mask
    """
    batch_size, seq_len = attention_mask.shape[:2]
    device = attention_mask.device

    # Create a causal mask [seq_len, seq_len]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

    # Expand to [batch_size, seq_len, seq_len]
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    return causal_mask


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _multinomial_sample_one_no_sync(probs):
    """Multinomial sampling without cuda synchronization"""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    """Sample from top-k logits with temperature"""
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


class CSMModel(PreTrainedModel):
    config_class = CSMConfig
    base_model_prefix = "csm"

    def __init__(self, config):
        super().__init__(config)

        # Create backbone using HF Llama
        self.backbone, backbone_dim = create_llama_backbone(config)

        # Create decoder using HF Llama
        self.decoder, decoder_dim = create_llama_decoder(config)

        # Embeddings
        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )

        # Projection layer between backbone and decoder
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)

        # Output head for 0th codebook
        self.codebook0_head = nn.Linear(
            backbone_dim, config.audio_vocab_size, bias=False
        )

        # Output heads for 1-31 codebooks
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size
            )
        )

        # Initialize weights
        self.post_init()

        # State to track KV cache usage
        self._using_kv_cache = False

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        """Embed audio tokens for specific codebook index"""
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed the token frames with audio and text embeddings

        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
                   Last dimension is text token, first N dimensions are audio codebooks

        Returns:
            (batch_size, seq_len, audio_num_codebooks+1, hidden_dim)
        """
        # Embed text tokens (last position in each frame)
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(
            -2
        )  # [B, S, 1, H]

        # Embed audio tokens (first N positions in each frame)
        # Add codebook offset to each position
        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size
            * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )

        # Reshape and embed
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )

        # Combine audio and text embeddings
        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def setup_caches(self, max_batch_size: int):
        """Enable KV caching for generation"""
        # Set a flag to use HF's caching
        self._using_kv_cache = True

        # No need to explicitly set up caches in HF - they're created automatically
        # The original implementation set decoder_max_seq_len=audio_num_codebooks

    def reset_caches(self):
        """Reset KV caches"""
        # We'll rely on HF's implementation which handles this automatically
        # when past_key_values is None in forward call
        pass

    def forward(
        self,
        input_ids=None,  # [batch_size, seq_len, audio_num_codebooks+1]
        attention_mask=None,  # [batch_size, seq_len, audio_num_codebooks+1]
        position_ids=None,  # [batch_size, seq_len]
        past_key_values=None,  # For HF KV caching
        use_cache=None,  # For HF KV caching
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        temperature=1.0,  # For sampling
        topk=50,  # For sampling
        generate_frame=False,  # Whether to generate audio frame
        labels=None,  # Added labels parameter
    ):
        """
        Forward pass using HF Llama model as backbone, with correct masking.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self._using_kv_cache

        # Create position IDs if not provided
        if position_ids is None:
            batch_size, seq_len = input_ids.size()[:2]
            position_ids = (
                torch.arange(seq_len, device=input_ids.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )

        # Embed tokens
        embeds = self._embed_tokens(input_ids)  # [B, S, audio_num_codebooks+1, H]
        masked_embeds = embeds * attention_mask.unsqueeze(-1)  # Apply mask
        h = masked_embeds.sum(dim=2)  # Sum across token dimension to get [B, S, H]

        # Create attention mask for HF Llama
        # HF attention masks work differently:
        # - 1 means "use this position"
        # - 0 means "mask/ignore this position"
        if attention_mask is not None:
            # Determine which positions have at least one token
            hf_attention_mask = (
                attention_mask.sum(dim=-1) > 0
            ).float()  # [batch_size, seq_len]
        else:
            # If no mask provided, attend to all positions
            hf_attention_mask = torch.ones(
                input_ids.size()[:2], device=input_ids.device
            )

        # Forward through backbone
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

        # Get hidden state and compute all logits at once
        h = backbone_outputs.last_hidden_state
        backbone_past_key_values = (
            backbone_outputs.past_key_values if use_cache else None
        )

        # Compute logits for all positions (used for both loss and prediction)
        c0_all_logits = self.codebook0_head(
            h
        )  # [batch_size, seq_len, audio_vocab_size]

        # Get last hidden state and last position logits
        last_h = h[:, -1, :]  # [B, H]
        c0_logits = c0_all_logits[:, -1, :]  # [B, audio_vocab_size]

        # Calculate loss if labels are provided
        loss = None
        backbone_loss = None
        decoder_loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            # Calculate backbone loss (codebook 0)
            c0_labels = labels[:, :, 0]  # [batch_size, seq_len]

            # Shift so that tokens < n predict n (causal LM style)
            logits_for_loss = c0_all_logits[
                :, :-1, :
            ].contiguous()  # remove last position
            shift_labels = c0_labels[:, 1:].contiguous()  # remove first position

            # Flatten the tokens
            logits_for_loss = (
                logits_for_loss.float()
            )  # upcast to float as in the provided function
            logits_for_loss = logits_for_loss.view(-1, self.config.audio_vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(logits_for_loss.device)

            # Calculate backbone loss
            backbone_loss = loss_fct(logits_for_loss, shift_labels)
            loss = backbone_loss

            # Calculate decoder loss for codebooks 1 to N-1
            # Get audio tokens (codebooks 1 to N-1)
            audio_tokens = input_ids[:, :, : self.config.audio_num_codebooks]
            # Shape: [batch_size, backbone_seq_len, audio_num_codebooks]
            # Note: backbone_seq_len represents frames in temporal sequence

            # Get corresponding labels
            audio_labels = labels[:, :, : self.config.audio_num_codebooks]
            # Shape: [batch_size, backbone_seq_len, audio_num_codebooks]

            # Create a mask identifying valid frames (where no codebook is -100)
            valid_frames_mask = (audio_labels != -100).all(dim=2)
            # Shape: [batch_size, backbone_seq_len]

            # Get indices of valid frames
            valid_frame_indices = valid_frames_mask.nonzero(
                as_tuple=False
            )  # [num_valid_frames, 2]

            if valid_frame_indices.numel() > 0:
                num_valid_frames = valid_frame_indices.shape[0]

                # Gather data for valid frames only
                valid_frame_hidden = h[
                    valid_frame_indices[:, 0], valid_frame_indices[:, 1]
                ]  # [num_valid_frames, hidden_dim]
                valid_frame_codebooks = audio_tokens[
                    valid_frame_indices[:, 0], valid_frame_indices[:, 1]
                ]  # [num_valid_frames, audio_num_codebooks]
                valid_frame_labels = audio_labels[
                    valid_frame_indices[:, 0], valid_frame_indices[:, 1]
                ]  # [num_valid_frames, audio_num_codebooks]

                # Project backbone hidden states for decoder input
                projected_frame_hidden = self.projection(
                    valid_frame_hidden
                )  # [num_valid_frames, decoder_dim]

                # Prepare codebook embeddings
                codebook_offsets = (
                    torch.arange(
                        self.config.audio_num_codebooks,
                        device=valid_frame_codebooks.device,
                    )
                    * self.config.audio_vocab_size
                )  # [audio_num_codebooks]

                # Add offsets to handle multiple codebook vocabularies
                codebook_indices = (
                    valid_frame_codebooks + codebook_offsets
                )  # [num_valid_frames, audio_num_codebooks]

                # Get embeddings for all codebooks
                codebook_embeds = self.audio_embeddings(
                    codebook_indices.view(-1)
                )  # [num_valid_frames*audio_num_codebooks, hidden_dim]
                codebook_embeds = codebook_embeds.view(
                    num_valid_frames, self.config.audio_num_codebooks, -1
                )  # [num_valid_frames, audio_num_codebooks, hidden_dim]
                projected_codebook_embeds = self.projection(
                    codebook_embeds
                )  # [num_valid_frames, audio_num_codebooks, decoder_dim]

                # Build decoder input sequence: [frame_context, codebook0, codebook1, ..., codebook_N-1]
                decoder_inputs = torch.cat(
                    [
                        projected_frame_hidden.unsqueeze(
                            1
                        ),  # Frame context from backbone
                        projected_codebook_embeds,  # Embedded codebooks
                    ],
                    dim=1,
                )  # [num_valid_frames, 1+audio_num_codebooks, decoder_dim]

                # Forward through decoder
                decoder_outputs = self.decoder(
                    inputs_embeds=decoder_inputs, return_dict=True
                )
                decoder_hidden = (
                    decoder_outputs.last_hidden_state
                )  # [num_valid_frames, 1+audio_num_codebooks, decoder_dim]

                # Get hidden states for predicting next codebook
                # Use positions 1 to N-1 to predict codebooks 1 to N-1
                # Position 0 is the projected backbone output, which we don't use for prediction
                # We want to predict codebooks 1 to N-1, so we use positions 1 to N-1
                codebook_hidden = decoder_hidden[
                    :, 1 : self.config.audio_num_codebooks, :
                ]  # [num_valid_frames, audio_num_codebooks-1, decoder_dim]

                # Compute logits for each codebook using dedicated classification heads
                # audio_head shape: [audio_num_codebooks-1, decoder_dim, audio_vocab_size]
                codebook_logits = torch.einsum(
                    "fcd,cdv->fcv", codebook_hidden, self.audio_head
                )
                # [num_valid_frames, audio_num_codebooks-1, audio_vocab_size]

                # Get targets (codebooks 1 to N-1)
                codebook_targets = valid_frame_labels[
                    :, 1:
                ]  # [num_valid_frames, audio_num_codebooks-1]

                # Reshape for cross entropy
                flat_logits = codebook_logits.reshape(
                    -1, self.config.audio_vocab_size
                )  # [num_valid_frames*(audio_num_codebooks-1), audio_vocab_size]
                flat_targets = codebook_targets.reshape(
                    -1
                )  # [num_valid_frames*(audio_num_codebooks-1)]

                # Compute loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                decoder_loss = loss_fct(flat_logits, flat_targets)
            else:
                decoder_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)

            # Combine backbone and decoder losses
            if loss is None:
                loss = decoder_loss
            else:
                loss = backbone_loss + decoder_loss  # Add the losses

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
        Specialized method for generating an audio frame

        Args:
            input_ids: [batch_size, seq_len, audio_num_codebooks+1]
            attention_mask: [batch_size, seq_len, audio_num_codebooks+1]
            position_ids: [batch_size, seq_len]
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            past_key_values: For HF KV caching
            use_cache: For HF KV caching
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a ModelOutput object

        Returns:
            Tensor of shape [batch_size, audio_num_codebooks] with sampled tokens
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self._using_kv_cache

        # First, get the backbone outputs
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

        # Get last hidden state and last position logits
        last_h = outputs.last_hidden_state  # [B, H]
        c0_logits = outputs.logits  # [B, audio_vocab_size]
        backbone_past_key_values = outputs.past_key_values if use_cache else None

        # Initialize output tensor to store all codebook tokens
        batch_size = last_h.size(0)
        generated_tokens = torch.zeros(
            batch_size,
            self.config.audio_num_codebooks,
            dtype=torch.long,
            device=last_h.device,
        )

        # Sample and store codebook 0
        c0_sample = sample_topk(c0_logits, topk, temperature)
        generated_tokens[:, 0] = c0_sample.squeeze()

        # Embed the first token (codebook 0)
        c0_embed = self._embed_audio(0, c0_sample)

        # Initialize input for first decoder step: backbone output + codebook 0
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_pos = (
            torch.arange(0, curr_h.size(1), device=curr_h.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        projected_h = self.projection(curr_h)

        # Run first decoder step
        decoder_outputs = self.decoder(
            inputs_embeds=projected_h,
            position_ids=curr_pos,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Generate the rest of the codebooks
        for i in range(1, self.config.audio_num_codebooks):
            # Get decoder hidden states
            hidden_states = decoder_outputs.last_hidden_state

            # Get logits for current codebook from last position
            ci_logits = torch.matmul(hidden_states[:, -1, :], self.audio_head[i - 1])

            # Sample token for current codebook
            ci_sample = sample_topk(ci_logits, topk, temperature)

            # Store sampled token in output
            generated_tokens[:, i] = ci_sample.squeeze()

            # Don't run the decoder for the last step
            if i < self.config.audio_num_codebooks - 1:
                # Embed the new token
                ci_embed = self._embed_audio(i, ci_sample)

                # Project the new token embedding
                projected_ci_embed = self.projection(ci_embed)

                # Prepare position ID for next token
                next_pos = torch.full((batch_size, 1), i + 1, device=ci_sample.device)

                # Run decoder with just the new token and past_key_values
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
