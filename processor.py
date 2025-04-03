import random
from typing import Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer, ProcessorMixin


class CSMProcessor(ProcessorMixin):
    def __init__(self, tokenizer: PreTrainedTokenizer, audio_tokenizer):
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.sample_rate = getattr(self.audio_tokenizer, "sample_rate", 16000)

    def __call__(
        self,
        messages=None,
        text=None,
        audios=None,
        speaker_id=None,
        return_tensors="pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 2048,
        amortize_decoder_training: bool = True,
        amortization_ratio: int = 16,
        messages_training_mask: Optional[
            Union[List[int], List[bool], List[List[int]], List[List[bool]]]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process inputs into model-ready format.

        Args:
            messages: A list of message dicts with keys 'role' and 'content',
                      or a list of lists for batched processing
            text: Text input when not using messages
            audios: List of audio tensors for content items with type "audio",
                    or a list of lists for batched processing
            speaker_id: Speaker ID when not using messages
            return_tensors: Return format ('pt' for PyTorch)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            amortize_decoder_training: Whether to apply compute amortization for decoder training
            amortization_ratio: Fraction denominator for frame sampling (e.g., 16 means 1/16th of frames)
            messages_training_mask: an optional list (or list of lists, if batched) of
                booleans/integers indicating which messages should remain trainable (True/1)
                vs. masked out entirely (False/0). If a message is masked, all of its frames
                will have labels set to -100.
        """
        if messages is not None:
            # Check if we have a batch of conversations or a single conversation
            is_batched = isinstance(messages[0], list) if messages else False

            if not is_batched:
                # Single conversation case
                messages = [messages]
                audios = [audios] if audios is not None else [None]
            elif audios is not None and not isinstance(audios[0], list):
                # If we have a batch of messages but a single audios list
                audios = [audios]

            # Similarly handle messages_training_mask to match batch dimensions
            if messages_training_mask is not None:
                if not is_batched:
                    # Single conversation => wrap in a list
                    if isinstance(messages_training_mask[0], list):
                        raise ValueError(
                            "`messages_training_mask` looks nested (multiple lists), "
                            "but `messages` is a single conversation. Shapes must match."
                        )
                    messages_training_mask = [messages_training_mask]
                else:
                    # batched => each conversation must have its own mask list
                    pass

            # Process each conversation separately
            batch_outputs = []
            for i, convo_messages in enumerate(messages):
                convo_audios = audios[i] if i < len(audios) else None
                # if there's a mask provided, fetch it for this conversation
                convo_mask = None
                if messages_training_mask is not None:
                    if i >= len(messages_training_mask):
                        raise ValueError(
                            f"messages_training_mask was provided but has fewer entries ({len(messages_training_mask)}) "
                            f"than the number of conversations ({len(messages)})"
                        )
                    convo_mask = messages_training_mask[i]

                batch_outputs.append(
                    self._process_messages(
                        convo_messages,
                        convo_audios,
                        return_tensors,
                        padding,
                        truncation,
                        max_length,
                        amortize_decoder_training,
                        amortization_ratio,
                        convo_mask,
                    )
                )

            # Combine into batched format
            if return_tensors == "pt":
                if batch_outputs:
                    # Find the maximum sequence length across the batch
                    max_seq_len = max(
                        output["input_ids"].size(0) for output in batch_outputs
                    )

                    padded_inputs = []
                    padded_masks = []
                    padded_labels = []

                    for output in batch_outputs:
                        seq_len = output["input_ids"].size(0)
                        if seq_len < max_seq_len and padding:
                            # Left pad to max_seq_len
                            padded_input = torch.zeros(max_seq_len, 33).long()
                            padded_mask = torch.zeros(max_seq_len, 33)
                            padded_label = torch.full((max_seq_len, 33), -100).long()

                            # Place content at the end for left padding
                            padded_input[max_seq_len - seq_len :] = output["input_ids"]
                            padded_mask[max_seq_len - seq_len :] = output[
                                "attention_mask"
                            ]
                            padded_label[max_seq_len - seq_len :] = output["labels"]
                        else:
                            # Already at or above max_seq_len (though typically won't exceed it)
                            padded_input = output["input_ids"]
                            padded_mask = output["attention_mask"]
                            padded_label = output["labels"]

                        padded_inputs.append(padded_input.unsqueeze(0))
                        padded_masks.append(padded_mask.unsqueeze(0))
                        padded_labels.append(padded_label.unsqueeze(0))

                    return {
                        "input_ids": torch.cat(padded_inputs, dim=0),
                        "attention_mask": torch.cat(padded_masks, dim=0),
                        "labels": torch.cat(padded_labels, dim=0),
                    }
                else:
                    return {
                        "input_ids": torch.zeros(0, 0, 33),
                        "attention_mask": torch.zeros(0, 0, 33),
                        "labels": torch.zeros(0, 0, 33, dtype=torch.long),
                    }
            else:
                raise ValueError(f"Unsupported return format: {return_tensors}")

        elif text is not None and speaker_id is not None:
            # Process individual components
            message = {
                "role": f"speaker_{speaker_id}",
                "content": [{"type": "text", "text": text}],
            }
            # Wrap in a batch
            return self.__call__(
                [message],
                audios,
                return_tensors,
                padding,
                truncation,
                max_length,
                amortize_decoder_training,
                amortization_ratio,
            )
        else:
            raise ValueError(
                "Must provide either 'messages' or 'text' and 'speaker_id'"
            )

    def _process_messages(
        self,
        messages: List[Dict],
        audios: List[torch.Tensor],
        return_tensors: str,
        padding: bool,
        truncation: bool,
        max_length: int,
        amortize_decoder_training: bool = True,
        amortization_ratio: int = 16,
        messages_training_mask: Optional[Union[List[int], List[bool]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process an individual conversation (list of messages) into model inputs,
        optionally applying a message-level mask that sets labels to -100 for
        messages marked as inactive in messages_training_mask.
        """

        device = next(self.audio_tokenizer.parameters()).device

        all_tokens = []
        all_masks = []

        audio_index = 0  # Track which audio we're using
        message_boundaries = []  # (start_idx, end_idx, keep_message)

        for msg_idx, message in enumerate(messages):
            # Derive speaker ID from role
            speaker_id = int(message["role"].split("_")[-1])

            # Determine if we're keeping this message's labels
            keep_message = (
                True
                if (messages_training_mask is None)
                else bool(messages_training_mask[msg_idx])
            )

            text_content = []
            has_audio_content = False

            for item in message["content"]:
                if item["type"] == "text" and item.get("text", ""):
                    text_content.append(item["text"])
                elif item["type"] == "audio":
                    has_audio_content = True

            text = " ".join(text_content)

            start_idx = sum(chunk.size(0) for chunk in all_tokens)

            # Tokenize text if present
            if text:
                text_tokens = self.tokenizer.encode(f"[{speaker_id}]{text}")
                text_frame = torch.zeros(len(text_tokens), 33).long()
                text_frame_mask = torch.zeros(len(text_tokens), 33, dtype=torch.int)

                text_frame[:, -1] = torch.tensor(text_tokens)
                text_frame_mask[:, -1] = 1

                all_tokens.append(text_frame)
                all_masks.append(text_frame_mask)

            # Handle audio content
            if (
                has_audio_content
                and audios
                and audio_index < len(audios)
                and audios[audio_index] is not None
            ):
                audio_tensor = audios[audio_index]
                audio_index += 1

                if not isinstance(audio_tensor, torch.Tensor):
                    raise ValueError(
                        f"Audio must be a torch.Tensor, got {type(audio_tensor)}"
                    )

                with torch.no_grad():
                    audio_tokens = self.audio_tokenizer.encode(
                        audio_tensor.unsqueeze(0).unsqueeze(0).to(device)
                    )[0]

                # Add EOS frame
                eos_frame = torch.zeros(audio_tokens.size(0), 1, device=device)
                audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

                audio_frame = torch.zeros(audio_tokens.size(1), 33).long()
                audio_frame_mask = torch.zeros(
                    audio_tokens.size(1), 33, dtype=torch.int
                )
                audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
                audio_frame_mask[:, :-1] = True

                all_tokens.append(audio_frame)
                all_masks.append(audio_frame_mask)
            elif has_audio_content:
                # If we have an audio content type but no audio tensor provided
                message_id = message.get("role", "unknown")
                print(
                    f"Warning: Audio content type present but no audio tensor provided for message with {message_id}"
                )

            end_idx = sum(chunk.size(0) for chunk in all_tokens)
            message_boundaries.append((start_idx, end_idx, keep_message))

        if audios and audio_index < len(audios):
            print(f"Warning: {len(audios) - audio_index} audio tensors were not used")

        # Combine tokens
        if all_tokens:
            tokens = torch.cat(all_tokens, dim=0)  # shape: [S, 33]
            tokens_mask = torch.cat(all_masks, dim=0)  # shape: [S, 33]

            # Truncate if needed: keep the last `max_length` frames for left padding
            if truncation and tokens.size(0) > max_length:
                tokens = tokens[-max_length:]
                tokens_mask = tokens_mask[-max_length:]
        else:
            tokens = torch.zeros(0, 33).long()
            tokens_mask = torch.zeros(0, 33)

        # Create labels
        labels = tokens.clone()
        labels = labels.masked_fill(tokens_mask == 0, -100)
        labels[:, -1] = -100

        # Apply message-level masking
        for start_idx, end_idx, keep_msg in message_boundaries:
            if start_idx >= labels.size(0):
                break
            if end_idx > labels.size(0):
                end_idx = labels.size(0)
            if not keep_msg:
                labels[start_idx:end_idx, :] = -100

        # Amortization logic
        if amortize_decoder_training:
            seq_len = labels.shape[0]
            valid_frames = torch.any(labels[:, :-1] != -100, dim=-1)
            valid_indices = torch.where(valid_frames)[0]

            if len(valid_indices) > 0:
                num_to_select = max(1, len(valid_indices) // amortization_ratio)
                selected_indices = random.sample(valid_indices.tolist(), num_to_select)
                frame_mask = torch.zeros(seq_len, dtype=torch.bool)
                frame_mask[selected_indices] = True
            else:
                frame_mask = torch.zeros(seq_len, dtype=torch.bool)

            codebook_mask = torch.zeros_like(labels, dtype=torch.bool)
            codebook_mask[:, -1] = True  # always keep text position
            # codebook 0 (position 0) for all valid frames
            valid_frames_mask = torch.any(labels != -100, dim=-1, keepdim=True).expand(
                -1, 1
            )
            codebook_mask[:, 0:1] = valid_frames_mask

            for s in range(seq_len):
                if frame_mask[s]:
                    codebook_mask[s, 1:-1] = True

            new_labels = labels.clone()
            new_labels = torch.where(
                (labels != -100) & ~codebook_mask, torch.full_like(labels, -100), labels
            )
            labels = new_labels

        if return_tensors == "pt":
            return {
                "input_ids": tokens,
                "attention_mask": tokens_mask,
                "labels": labels,
            }
        else:
            raise ValueError(f"Unsupported return format: {return_tensors}")
