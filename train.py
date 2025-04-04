"""
This file orchestrates training for a CSM model using a standard
HuggingFace Trainer. We define:

- CSMAudioTextDataset: loads conversation JSON lines, fetches audio from
  disk, applies the CSMProcessor, and returns a single example.
- CSMDataCollator: left-pads sequences from different examples to the
  same length. 
- We parse CLI arguments to pick train_file, model path, etc.
- Then we construct a CSMTrainer that logs the separate backbone vs.
  decoder losses, and run trainer.train().
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from modeling_csm import CSMConfig, CSMModel
from processor import CSMProcessor

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CSMAudioTextDataset(Dataset):
    """
    A dataset that reads a JSONL file, each line describing a "conversation".
    For each conversation, we load waveforms from disk (if any),
    and pass them to the CSMProcessor along with the message text.
    The final output is a dictionary with input_ids, attention_mask, labels.
    """

    def __init__(self, data_path, audio_cache_dir=None, processor=None):
        """
        :param data_path: Path to a JSON lines file. Each line is a dict with "messages"
                          possibly containing "text" or "audio" elements.
        :param audio_cache_dir: optional directory to store or read local copies of audio
        :param processor: the CSMProcessor instance
        """
        self.data_path = data_path
        self.audio_cache_dir = audio_cache_dir
        self.processor = processor

        # Create cache directory if needed
        if audio_cache_dir and not os.path.exists(audio_cache_dir):
            os.makedirs(audio_cache_dir)

        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f]

        logger.info(f"Loaded {len(self.data)} conversations from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        For an item, we go through its messages, load any audio from disk,
        possibly resample it, and feed the text+audio to the CSMProcessor.
        The result is a single conversation item with shape [S, 33],
        which we then squeeze out the batch dimension for the trainer.
        """
        item = self.data[idx]
        messages = item["messages"]
        training_mask = item.get("training_mask", None)

        audio_tensors = []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "audio" and "url" in content:
                    audio_path = content["url"]

                    if self.audio_cache_dir:
                        cache_path = os.path.join(self.audio_cache_dir, os.path.basename(audio_path))
                        if os.path.exists(cache_path):
                            audio_path = cache_path

                    try:
                        waveform, sample_rate = torchaudio.load(audio_path)
                        # Convert to mono
                        if waveform.size(0) > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)

                        # Resample if needed
                        if sample_rate != self.processor.sample_rate:
                            resampler = torchaudio.transforms.Resample(sample_rate, self.processor.sample_rate)
                            waveform = resampler(waveform)

                        audio_tensors.append(waveform.squeeze(0))
                    except Exception as e:
                        logger.warning(f"Error loading audio {audio_path}: {e}")
                        audio_tensors.append(None)

        # Use the processor to build input tensors
        processed = self.processor(
            messages=messages,
            audios=audio_tensors,
            messages_training_mask=training_mask,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            amortize_decoder_training=True,
        )

        # The processor returns shape [1, S, 33], so we remove the batch dimension
        return {
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
            "labels": processed["labels"].squeeze(0),
        }


@dataclass
class CSMDataCollator:
    """
    A collator that left-pads sequences so that the final frames
    line up at the bottom. The model sees them in a temporal order
    from top to bottom if we consider a strictly causal backbone.

    For each feature in the batch:
      - If dimension is [S, 33], and S < max_seq_len in the batch,
        we create a [pad_rows, 33] region at the top.

    text_pad_token_id: used for any "text" column filler, ensures consistency.
    """
    text_pad_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not features:
            return {}

        max_seq_len = max(f["input_ids"].size(0) for f in features)
        keys = features[0].keys()
        padded_dict = {}

        for key in keys:
            padded_tensors = []
            for f in features:
                tensor = f[key]
                seq_len, width = tensor.size()
                if seq_len < max_seq_len:
                    pad_rows = max_seq_len - seq_len
                    if key == "labels":
                        pad_tensor = torch.full((pad_rows, width), -100, dtype=tensor.dtype, device=tensor.device)
                    elif key == "attention_mask":
                        pad_tensor = torch.zeros((pad_rows, width), dtype=tensor.dtype, device=tensor.device)
                    else:
                        pad_tensor = torch.zeros((pad_rows, width), dtype=tensor.dtype, device=tensor.device)
                        pad_tensor[:, -1] = self.text_pad_token_id

                    padded_tensor = torch.cat([pad_tensor, tensor], dim=0)
                else:
                    padded_tensor = tensor

                padded_tensors.append(padded_tensor.unsqueeze(0))
            padded_dict[key] = torch.cat(padded_tensors, dim=0)

        return padded_dict


@dataclass
class DataTrainingArguments:
    """
    CLI arguments for data, e.g. training file path, optional eval file, cache directory.
    """
    train_file: str = field(metadata={"help": "Path to training data"})
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "Path to evaluation data"}
    )
    audio_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to cache audio files"}
    )


@dataclass
class ModelArguments:
    """
    CLI arguments specifying the model to load or create from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Extends the HuggingFace TrainingArguments to include additional fields if desired,
    such as num_workers for the dataloader.
    """
    dataloader_num_workers: int = field(
        default=0, metadata={"help": "Number of workers for dataloader"}
    )


def load_llama3_tokenizer():
    """
    Loads a Llama3-based tokenizer from a huggingface model,
    with an added post-processor for BOS/EOS. This must match
    what the CSM expects for text tokens.
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id),
            (f"{eos}", tokenizer.eos_token_id),
        ],
    )
    return tokenizer


class CSMTrainer(Trainer):
    """
    Custom Trainer that extracts backbone_loss and decoder_loss from the
    model's outputs and logs them separately. The rest is standard HF Trainer.
    """

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        outputs = model(**inputs)
        loss = outputs.loss

        # If the model returned separate backbone/decoder losses, we can log them
        backbone_loss = outputs.backbone_loss
        decoder_loss = outputs.decoder_loss

        # Log to wandb or other trackers if present
        if backbone_loss is not None and "wandb" in self.args.report_to:
            self.log({"train/backbone_loss": backbone_loss.detach().float().item()})
        if decoder_loss is not None and "wandb" in self.args.report_to:
            self.log({"train/decoder_loss": decoder_loss.detach().float().item()})

        return (loss, outputs) if return_outputs else loss


def main():
    """
    Main function for training a CSM model:
     1) parse arguments
     2) set random seed
     3) load text + audio tokenizers
     4) create or load a CSMModel
     5) create datasets + data collator
     6) run HF Trainer
     7) save final model
    """
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(
            f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
        )
        logger.info(
            f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
        )
    else:
        logger.warning("CUDA is not available. Using CPU for training.")

    # 1) load text tokenizer
    logger.info("Loading tokenizers...")
    text_tokenizer = load_llama3_tokenizer()

    # 2) load audio tokenizer from HF Hub
    logger.info("Loading mimi audio tokenizer")
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    audio_tokenizer = loaders.get_mimi(mimi_weight, device=device)
    audio_tokenizer.set_num_codebooks(32)

    # 3) combine into a single processor
    processor = CSMProcessor(text_tokenizer, audio_tokenizer)

    # 4) create or load model
    if model_args.model_name_or_path:
        logger.info(f"Loading model from {model_args.model_name_or_path}")
        model = CSMModel.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=(
                torch.bfloat16
                if training_args.bf16
                else (torch.float16 if training_args.fp16 else torch.float32)
            ),
        )
    else:
        logger.info("Creating new model from config")
        config = CSMConfig()
        model = CSMModel(config)

    model.to(device)
    print("model dtype", model.dtype)

    # 5) create datasets
    train_dataset = CSMAudioTextDataset(
        data_args.train_file,
        audio_cache_dir=data_args.audio_cache_dir,
        processor=processor,
    )

    eval_dataset = None
    if data_args.eval_file:
        eval_dataset = CSMAudioTextDataset(
            data_args.eval_file,
            audio_cache_dir=data_args.audio_cache_dir,
            processor=processor,
        )

    # 6) data collator (left pad)
    data_collator = CSMDataCollator(text_pad_token_id=text_tokenizer.eos_token_id)

    # 7) Trainer
    trainer = CSMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Log effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    )
    if torch.cuda.device_count() > 1:
        effective_batch_size *= torch.cuda.device_count()

    logger.info(
        f"Effective batch size: {effective_batch_size} (per_device_batch={training_args.per_device_train_batch_size} "
        f"× grad_accum={training_args.gradient_accumulation_steps} × num_gpus={torch.cuda.device_count()})"
    )

    # memory hints
    memory_opts = []
    if training_args.gradient_accumulation_steps > 1:
        memory_opts.append(f"gradient accumulation (steps={training_args.gradient_accumulation_steps})")
    if training_args.fp16:
        memory_opts.append("fp16 mixed precision")
    elif training_args.bf16:
        memory_opts.append("bf16 mixed precision")

    if memory_opts:
        logger.info(f"Memory optimization: {', '.join(memory_opts)}")

    # 8) Train
    logger.info("Starting training...")
    trainer.train()

    # 9) Save final model
    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()