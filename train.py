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
    def __init__(self, data_path, audio_cache_dir=None, processor=None):
        """
        Dataset for CSM model training.

        Args:
            data_path: Path to dataset file with conversation data
            audio_cache_dir: Directory to cache audio files
            processor: CSMProcessor instance
        """
        self.data_path = data_path
        self.audio_cache_dir = audio_cache_dir
        self.processor = processor

        # Create cache dir if it doesn't exist
        if audio_cache_dir and not os.path.exists(audio_cache_dir):
            os.makedirs(audio_cache_dir)

        # Load dataset
        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f]

        logger.info(f"Loaded {len(self.data)} conversations from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single conversation with its audio data"""
        item = self.data[idx]
        messages = item["messages"]

        # Load audio for each audio content item
        audio_tensors = []

        training_mask = item.get("training_mask", None)

        for message in messages:
            for content in message["content"]:
                if content["type"] == "audio" and "url" in content:
                    audio_path = content["url"]

                    # Check if we should use cached version
                    if self.audio_cache_dir:
                        cache_path = os.path.join(
                            self.audio_cache_dir, os.path.basename(audio_path)
                        )
                        if os.path.exists(cache_path):
                            audio_path = cache_path

                    # Load audio
                    try:
                        waveform, sample_rate = torchaudio.load(audio_path)
                        # Convert to mono if needed
                        if waveform.size(0) > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)

                        # Resample if needed to match processor's expected rate
                        if sample_rate != self.processor.sample_rate:
                            resampler = torchaudio.transforms.Resample(
                                sample_rate, self.processor.sample_rate
                            )
                            waveform = resampler(waveform)

                        audio_tensors.append(waveform.squeeze(0))
                    except Exception as e:
                        logger.warning(f"Error loading audio {audio_path}: {e}")
                        # Add None to maintain index alignment
                        audio_tensors.append(None)

        # Process with CSMProcessor, including the training mask
        processed = self.processor(
            messages=messages,
            audios=audio_tensors,
            messages_training_mask=training_mask,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,  # Adjust based on model's maximum context length
            amortize_decoder_training=True,
        )

        # Remove batch dimension since the DataLoader will add it back
        return {
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
            "labels": processed["labels"].squeeze(0),
        }


@dataclass
class CSMDataCollator:
    """
    Data collator for CSM model that handles batching of pre-processed inputs
    with left padding along dim=1. For each tensor (except labels and attention_mask),
    padding is 0 for all columns except the last one, which is set to the text tokenizer pad token.
    For attention_mask, the entire padded frame is set to 0.
    """

    text_pad_token_id: int

    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if not features:
            return {}

        # Determine the maximum sequence length across the batch (along dim=0)
        max_seq_len = max(f["input_ids"].size(0) for f in features)
        keys = features[0].keys()
        padded_dict = {}

        for key in keys:
            padded_tensors = []
            for f in features:
                tensor = f[key]  # Expected shape: [seq_len, width]
                seq_len, width = tensor.size()
                if seq_len < max_seq_len:
                    pad_rows = max_seq_len - seq_len
                    if key == "labels":
                        pad_tensor = torch.full(
                            (pad_rows, width),
                            -100,
                            dtype=tensor.dtype,
                            device=tensor.device,
                        )
                    elif key == "attention_mask":
                        pad_tensor = torch.zeros(
                            (pad_rows, width), dtype=tensor.dtype, device=tensor.device
                        )
                    else:
                        pad_tensor = torch.zeros(
                            (pad_rows, width), dtype=tensor.dtype, device=tensor.device
                        )
                        pad_tensor[:, -1] = self.text_pad_token_id
                    # Left pad: concatenate pad_tensor before the actual content
                    padded_tensor = torch.cat([pad_tensor, tensor], dim=0)
                else:
                    padded_tensor = tensor
                padded_tensors.append(padded_tensor.unsqueeze(0))
            padded_dict[key] = torch.cat(padded_tensors, dim=0)

        return padded_dict


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to the data used for training and evaluation.
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
    Arguments pertaining to which model/config we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Custom training arguments extending HuggingFace's TrainingArguments.
    """

    dataloader_num_workers: int = field(
        default=0, metadata={"help": "Number of workers for dataloader"}
    )


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
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
    Custom Trainer for CSM that logs individual loss components to wandb.
    """

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        """
        Override compute_loss to extract and log individual loss components.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        # Extract individual loss components
        backbone_loss = outputs.backbone_loss
        decoder_loss = outputs.decoder_loss

        # Log to wandb if available
        if backbone_loss is not None:
            self.log({"train/backbone_loss": backbone_loss.detach().float().item()})
        if decoder_loss is not None:
            self.log({"train/decoder_loss": decoder_loss.detach().float().item()})

        return (loss, outputs) if return_outputs else loss


def main():
    """Main training function"""
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
            f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
        )
    else:
        logger.warning("CUDA is not available. Using CPU for training.")

    # Load tokenizers
    logger.info("Loading tokenizers...")
    text_tokenizer = load_llama3_tokenizer()

    # Load audio tokenizer
    logger.info("Loading mimi audio tokenizer")
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    audio_tokenizer = loaders.get_mimi(mimi_weight, device=device)
    audio_tokenizer.set_num_codebooks(32)

    # Create processor
    processor = CSMProcessor(text_tokenizer, audio_tokenizer)

    # Load or create model
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

    # Create datasets
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

    # Create data collator
    data_collator = CSMDataCollator(text_pad_token_id=text_tokenizer.eos_token_id)

    # Create trainer
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
        f"Effective batch size: {effective_batch_size} (per_device_batch={training_args.per_device_train_batch_size} × grad_accum={training_args.gradient_accumulation_steps} × num_gpus={torch.cuda.device_count()})"
    )

    # Log memory optimization techniques
    memory_opts = []
    if training_args.gradient_accumulation_steps > 1:
        memory_opts.append(
            f"gradient accumulation (steps={training_args.gradient_accumulation_steps})"
        )
    if training_args.fp16:
        memory_opts.append("fp16 mixed precision")
    elif training_args.bf16:
        memory_opts.append("bf16 mixed precision")

    if memory_opts:
        logger.info(f"Memory optimization: {', '.join(memory_opts)}")

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
