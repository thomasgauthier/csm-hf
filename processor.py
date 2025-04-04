"""
Defines the CSMProcessor, which converts a set of "messages" + audio waveforms
into model-ready token tensors [S, 33] (if we have 32 audio codebooks + 1 text token).

Flow:
1) Each message has 'role' (like speaker_0) and a content list with "text" or "audio".
2) For text, we tokenize it into the last column (index 32).
3) For audio, we pass waveforms through an audio tokenizer (like Mimi) that
   outputs discrete codebook tokens, which we place in columns [0..31].
4) We produce 'labels' where text positions are set to -100 to skip text prediction.

We also implement optional "decoder amortization," randomizing which frames
fully train codebooks [1..N-1].
"""

import random
from typing import Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer, ProcessorMixin


class CSMProcessor(ProcessorMixin):
    def __init__(self, tokenizer: PreTrainedTokenizer, audio_tokenizer):
        """
        :param tokenizer:      a text tokenizer (e.g. Llama)
        :param audio_tokenizer: a multi-codebook audio tokenizer (e.g. Mimi)
        """
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
        Main entry point to process either:
         - a single conversation or a batch of multiple conversations
         - or a single text snippet with speaker_id

        Returns a dict with {input_ids, attention_mask, labels} shaped [B, S, 33],
        where 33 = 32 codebook columns + 1 text column.

        If 'messages' is not None, we parse them:
          - If there's 'audio', we pass waveforms to self.audio_tokenizer to get codebook tokens.
          - If there's 'text', we pass it to self.tokenizer for the last column.
          - We build a single conversation's frames, optionally skip entire messages
            via messages_training_mask, and optionally do "amortization" on codebooks [1..N-1].

        Finally, we might left-pad these sequences in the calling environment if
        multiple conversations have different lengths.
        """
        if messages is not None:
            # Determine if we have a batch of conversations or a single conversation
            is_batched = isinstance(messages[0], list) if messages else False

            if not is_batched:
                # Single conversation => wrap in a list
                messages = [messages]
                audios = [audios] if audios is not None else [None]
            elif audios is not None and not isinstance(audios[0], list):
                # If we do have multiple messages but audios is not similarly nested, wrap it
                audios = [audios]

            if messages_training_mask is not None:
                if not is_batched:
                    if isinstance(messages_training_mask[0], list):
                        raise ValueError(
                            "`messages_training_mask` looks nested (multiple lists), "
                            "but we have a single conversation. Must match shapes."
                        )
                    messages_training_mask = [messages_training_mask]

            # Process each conversation in the batch
            batch_outputs = []
            for i, convo_messages in enumerate(messages):
                convo_audios = audios[i] if i < len(audios) else None
                convo_mask   = None
                if messages_training_mask is not None:
                    if i >= len(messages_training_mask):
                        raise ValueError(
                            f"messages_training_mask has fewer entries "
                            f"({len(messages_training_mask)}) than conversations ({len(messages)})"
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

            # Combine results into a single batch dimension
            if return_tensors == "pt":
                if batch_outputs:
                    max_seq_len = max(
                        output["input_ids"].size(0) for output in batch_outputs
                    )

                    padded_inputs = []
                    padded_masks  = []
                    padded_labels = []

                    for output in batch_outputs:
                        seq_len = output["input_ids"].size(0)
                        if seq_len < max_seq_len and padding:
                            # Left pad
                            padded_input = torch.zeros(max_seq_len, 33).long()
                            padded_mask  = torch.zeros(max_seq_len, 33)
                            padded_label = torch.full((max_seq_len, 33), -100).long()

                            padded_input[max_seq_len - seq_len :] = output["input_ids"]
                            padded_mask [max_seq_len - seq_len :] = output["attention_mask"]
                            padded_label[max_seq_len - seq_len :] = output["labels"]
                        else:
                            padded_input = output["input_ids"]
                            padded_mask  = output["attention_mask"]
                            padded_label = output["labels"]

                        padded_inputs.append(padded_input.unsqueeze(0))
                        padded_masks .append(padded_mask .unsqueeze(0))
                        padded_labels.append(padded_label.unsqueeze(0))

                    return {
                        "input_ids":      torch.cat(padded_inputs, dim=0),
                        "attention_mask": torch.cat(padded_masks,  dim=0),
                        "labels":         torch.cat(padded_labels, dim=0),
                    }
                else:
                    # No data
                    return {
                        "input_ids":      torch.zeros(0, 0, 33),
                        "attention_mask": torch.zeros(0, 0, 33),
                        "labels":         torch.zeros(0, 0, 33, dtype=torch.long),
                    }
            else:
                raise ValueError(f"Unsupported return format: {return_tensors}")

        elif text is not None and speaker_id is not None:
            # Single text snippet usage
            message = {
                "role": f"speaker_{speaker_id}",
                "content": [{"type": "text", "text": text}],
            }
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
                "Must provide either 'messages' or 'text' plus 'speaker_id'"
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
        Converts an individual conversation into a dictionary:
          {"input_ids": [S, 33], "attention_mask": [S, 33], "labels": [S, 33]}

        Each message can contain text and/or audio. We place text tokens in
        column 32, audio codebooks in columns [0..31]. The shape index 0, i.e. S,
        can vary as we accumulate frames.

        We also:
         - Possibly mark entire messages as non-trainable (label = -100).
         - Possibly apply "amortization" so codebooks [1..N-1] are only trainable
           on a random subset of frames (reducing memory usage).
        """
        device = next(self.audio_tokenizer.parameters()).device

        all_tokens = []
        all_masks  = []
        audio_index = 0
        message_boundaries = []

        for msg_idx, message in enumerate(messages):
            speaker_id = int(message["role"].split("_")[-1])
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

            # embed text
            if text:
                text_tokens = self.tokenizer.encode(f"[{speaker_id}]{text}")
                text_frame  = torch.zeros(len(text_tokens), 33).long()
                text_frame_mask = torch.zeros(len(text_tokens), 33, dtype=torch.int)

                text_frame[:, -1] = torch.tensor(text_tokens)
                text_frame_mask[:, -1] = 1

                all_tokens.append(text_frame)
                all_masks .append(text_frame_mask)

            # embed audio
            if (
                has_audio_content
                and audios
                and audio_index < len(audios)
                and audios[audio_index] is not None
            ):
                audio_tensor = audios[audio_index]
                audio_index += 1

                if not isinstance(audio_tensor, torch.Tensor):
                    raise ValueError(f"Audio must be torch.Tensor, got {type(audio_tensor)}")

                with torch.no_grad():
                    audio_tokens = self.audio_tokenizer.encode(audio_tensor.unsqueeze(0).unsqueeze(0).to(device))[0]

                # add an extra column for EOS
                eos_frame = torch.zeros(audio_tokens.size(0), 1, device=device)
                audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

                audio_frame = torch.zeros(audio_tokens.size(1), 33).long()
                audio_frame_mask = torch.zeros(audio_tokens.size(1), 33, dtype=torch.int)
                audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
                audio_frame_mask[:, :-1] = True

                all_tokens.append(audio_frame)
                all_masks .append(audio_frame_mask)
            elif has_audio_content:
                # Audio content declared but no actual audio tensor
                message_id = message.get("role", "unknown")
                print(
                    f"Warning: Audio content type present but no audio tensor provided for message with {message_id}"
                )

            end_idx = sum(chunk.size(0) for chunk in all_tokens)
            message_boundaries.append((start_idx, end_idx, keep_message))

        if audios and audio_index < len(audios):
            print(f"Warning: {len(audios) - audio_index} audio tensors were not used")

        # Combine into a single [S, 33] matrix
        if all_tokens:
            tokens      = torch.cat(all_tokens, dim=0)
            tokens_mask = torch.cat(all_masks,  dim=0)
            if truncation and tokens.size(0) > max_length:
                tokens      = tokens     [-max_length:]
                tokens_mask = tokens_mask[-max_length:]
        else:
            tokens      = torch.zeros(0, 33).long()
            tokens_mask = torch.zeros(0, 33)

        # Build labels (we do not train on text => col 32 => -100)
        labels = tokens.clone()
        labels = labels.masked_fill(tokens_mask == 0, -100)
        labels[:, -1] = -100

        # Mark entire messages as -100 if keep_message=False
        for start_idx, end_idx, keep_msg in message_boundaries:
            if start_idx >= labels.size(0):
                break
            if end_idx > labels.size(0):
                end_idx = labels.size(0)
            if not keep_msg:
                labels[start_idx:end_idx, :] = -100

        # Optionally amortize the decoder by training codebooks [1..N-1]
        # on only 1/(amortization_ratio) frames
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

            # codebook_mask is a boolean mask of shape [S, 33]
            # we always keep codebook0 (col 0) and text col (col 32) for valid frames
            codebook_mask = torch.zeros_like(labels, dtype=torch.bool)
            codebook_mask[:, -1] = True
            valid_frames_mask = torch.any(labels != -100, dim=-1, keepdim=True).expand(-1, 1)
            codebook_mask[:, 0:1] = valid_frames_mask

            # for the randomly selected frames, keep codebooks [1..N-1]
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
