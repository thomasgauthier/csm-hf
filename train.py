"""
CSM Training Script

Training for a two-stage CSM (Conversational Speech Model) via Hugging Face Trainer.

This script:
- Creates a custom dataset (`CSMAudioTextDataset`) that reads JSON lines describing multi-turn conversations
  with text and audio references. Audio is loaded, resampled, and tokenized into discrete codebooks,
  while text is tokenized into the last column of each frame.
- Applies a data collator (`CSMDataCollator`) that left-pads frames to align sequences of different lengths.
- Introduces a custom trainer (`CSMTrainer`) for logging separate backbone vs. decoder losses.
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

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CSMAudioTextDataset(Dataset):
    """
    Reads a JSONL file where each line describes a conversation containing text
    and optional audio references. Loads audio waveforms, applies tokenization,
    and outputs tensors of shape [S, 33] for each conversation.

    The 'processor' is responsible for converting messages + waveforms into
    (input_ids, attention_mask, labels). The resulting item has shape [S, 33].
    """

    def __init__(
        self, data_path, audio_cache_dir=None, processor=None, num_train_epochs=10
    ):
        """
        Args:
            data_path: Path to a JSONL file with "messages" describing text/audio content.
            audio_cache_dir: Optional local directory for storing fetched audio files.
            processor: CSMProcessor instance handling text/audio tokenization.
            num_train_epochs: Number of times to repeat the dataset to ensure decoder amortization
                              randomizes over different frames in each epoch.
        """
        self.data_path = data_path
        self.audio_cache_dir = audio_cache_dir
        self.processor = processor
        self.num_train_epochs = num_train_epochs

        if audio_cache_dir and not os.path.exists(audio_cache_dir):
            os.makedirs(audio_cache_dir)

        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f]

        logger.info(f"Loaded {len(self.data)} conversations from {data_path}")
        
        self.amortization_ratio = getattr(processor, "amortization_ratio", 16)

    def __len__(self):
        """
        Returns a length that is deliberately larger to repeat data items
        multiple times. This helps with large-batch training or sampling,
        and ensures decoder amortization uses different frames each epoch.
        """
        return len(self.data) * self.num_train_epochs

    def __getitem__(self, idx):
        """
        Retrieves the conversation at index `idx % len(self.data)`, loads
        audio from disk, resamples if needed, and processes text/audio via CSMProcessor.

        Returns:
            A dict with:
              "input_ids": [S, 33]
              "attention_mask": [S, 33]
              "labels": [S, 33]
        """
        idx = idx % len(self.data)
        item = self.data[idx]
        messages = item["messages"]
        training_mask = item.get("training_mask", None)

        audio_tensors = []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "audio" and "url" in content:
                    audio_path = content["url"]
                    if self.audio_cache_dir:
                        cache_path = os.path.join(
                            self.audio_cache_dir, os.path.basename(audio_path)
                        )
                        if os.path.exists(cache_path):
                            audio_path = cache_path

                    try:
                        waveform, sample_rate = torchaudio.load(audio_path)
                        if waveform.size(0) > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        if sample_rate != self.processor.sample_rate:
                            resampler = torchaudio.transforms.Resample(
                                sample_rate, self.processor.sample_rate
                            )
                            waveform = resampler(waveform)
                        audio_tensors.append(waveform.squeeze(0))
                    except Exception as e:
                        logger.warning(f"Error loading audio {audio_path}: {e}")
                        audio_tensors.append(None)

        processed = self.processor(
            messages=messages,
            audios=audio_tensors,
            messages_training_mask=training_mask,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            amortize_decoder_training=True,
            amortization_ratio=self.amortization_ratio,
        )
        return {
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
            "labels": processed["labels"].squeeze(0),
        }


@dataclass
class CSMDataCollator:
    """
    Data collator for CSM training. Left-pads sequences so all items in a batch
    share the same temporal dimension. The model processes frames bottom-to-top
    in a causal manner.

    text_pad_token_id: Token ID used to fill empty text columns.
    """

    text_pad_token_id: int

    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Gathers a list of features (each containing ["input_ids", "attention_mask", "labels"])
        and pads them to the same length (max_seq_len) along dimension 0.

        Returns a dict with padded "input_ids", "attention_mask", "labels".
        """
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

                    padded_tensor = torch.cat([pad_tensor, tensor], dim=0)
                else:
                    padded_tensor = tensor

                padded_tensors.append(padded_tensor.unsqueeze(0))
            padded_dict[key] = torch.cat(padded_tensors, dim=0)

        return padded_dict


@dataclass
class DataTrainingArguments:
    """
    Arguments specifying paths for training and evaluation data, plus optional audio caching.
    """

    train_file: str = field(
        metadata={"help": "Path to the JSONL file containing training data."}
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a JSONL file containing evaluation data."},
    )
    audio_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Local directory to store/read cached audio files."},
    )
    amortization_ratio: int = field(
        default=16,
        metadata={"help": "Amortization ratio for decoder training. Higher values mean fewer frames used for decoder training."},
    )


@dataclass
class ModelArguments:
    """
    Arguments for loading or creating a CSM model from a given path or config.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained CSM model, or identifier on HF Hub."},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Extension of HF TrainingArguments with extra fields like dataloader_num_workers.
    """

    dataloader_num_workers: int = field(
        default=0, metadata={"help": "Number of workers for data loading."}
    )
    logging_steps: int = field(
        default=1, metadata={"help": "Log training metrics every X update steps."}
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "Checkpoint save strategy (e.g., 'epoch', 'steps')."},
    )
    save_total_limit: int = field(
        default=3, metadata={"help": "Maximum number of checkpoints to retain."}
    )
    learning_rate: float = field(
        default=5e-6, metadata={"help": "Initial learning rate for training."}
    )
    num_train_epochs: float = field(
        default=3, metadata={"help": "Number of training epochs."}
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per device during training."}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )


def load_llama3_tokenizer():
    """
    Loads a Llama3-based tokenizer and sets a post-processor for BOS/EOS tokens.
    This should align with the text token format expected by the CSM model.
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (bos, tokenizer.bos_token_id),
            (eos, tokenizer.eos_token_id),
        ],
    )
    return tokenizer


class CSMTrainer(Trainer):
    """
    Custom trainer for the two-stage CSM model. Logs separate backbone vs. decoder losses for monitoring during training.
    """

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        """
        Overrides Trainer's compute_loss to extract backbone_loss and decoder_loss
        from the model outputs and logs them separately.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        backbone_loss = outputs.backbone_loss
        decoder_loss = outputs.decoder_loss

        if backbone_loss is not None:
            self.log({"train/backbone_loss": backbone_loss.detach().float().item()})
        if decoder_loss is not None:
            self.log({"train/decoder_loss": decoder_loss.detach().float().item()})

        return (loss, outputs) if return_outputs else loss


def main():
    """
    Main entry for CSM training:
      - Parse command-line arguments (model, data, training config)
      - Initialize random seed
      - Load text and audio tokenizers, then combine into a CSMProcessor
      - Create or load a CSMModel (two-stage: backbone + decoder)
      - Build train/eval datasets and data collator
      - Instantiate a custom CSMTrainer and run training
      - Save final model weights
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
        logger.warning("CUDA is not available, proceeding on CPU.")

    logger.info("Loading text tokenizer...")
    text_tokenizer = load_llama3_tokenizer()

    logger.info("Loading multi-codebook audio tokenizer (Mimi)...")
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    audio_tokenizer = loaders.get_mimi(mimi_weight, device=device)
    audio_tokenizer.set_num_codebooks(32)

    processor = CSMProcessor(text_tokenizer, audio_tokenizer)

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
        logger.info("Creating a new model from default CSMConfig")
        config = CSMConfig()
        model = CSMModel(config)

    model.to(device)
    logger.info(f"Model dtype: {model.dtype}")

    # Store original num_train_epochs for dataset repetition
    original_num_train_epochs = training_args.num_train_epochs
    
    # Set amortization ratio on processor
    processor.amortization_ratio = data_args.amortization_ratio
    logger.info(f"Using decoder amortization ratio: {data_args.amortization_ratio}")

    train_dataset = CSMAudioTextDataset(
        data_args.train_file,
        audio_cache_dir=data_args.audio_cache_dir,
        processor=processor,
        num_train_epochs=int(original_num_train_epochs),
    )

    eval_dataset = None
    if data_args.eval_file:
        eval_dataset = CSMAudioTextDataset(
            data_args.eval_file,
            audio_cache_dir=data_args.audio_cache_dir,
            processor=processor,
            num_train_epochs=1,  # For eval we don't need to repeat
        )

    data_collator = CSMDataCollator(text_pad_token_id=text_tokenizer.eos_token_id)

    # Calculate steps_per_old_epoch for proper save/eval scheduling
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * world_size
    )

    # Use original dataset length (not the repeated one)
    original_data_size = len(train_dataset.data)
    steps_per_old_epoch = (
        original_data_size + effective_batch_size - 1
    ) // effective_batch_size  # Ceiling division

    # Override training args to use steps-based scheduling
    if training_args.save_strategy == "epoch":
        training_args.save_strategy = "steps"
        training_args.save_steps = steps_per_old_epoch
        logger.info(
            f"Changed save_strategy to 'steps' with save_steps={steps_per_old_epoch}"
        )
    elif training_args.save_strategy == "steps" and hasattr(
        training_args, "save_steps"
    ):
        # If already step-based, adjust the steps to account for our dataset repeat trick
        original_save_steps = training_args.save_steps
        steps_per_original_epoch = steps_per_old_epoch
        training_args.save_steps = original_save_steps * steps_per_original_epoch
        logger.info(
            f"Adjusted save_steps from {original_save_steps} to {training_args.save_steps}"
        )

    # Also adjust evaluation strategy if needed
    if training_args.evaluation_strategy == "epoch":
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = steps_per_old_epoch
        logger.info(
            f"Changed evaluation_strategy to 'steps' with eval_steps={steps_per_old_epoch}"
        )
    elif training_args.evaluation_strategy == "steps" and hasattr(
        training_args, "eval_steps"
    ):
        original_eval_steps = training_args.eval_steps
        steps_per_original_epoch = steps_per_old_epoch
        training_args.eval_steps = original_eval_steps * steps_per_original_epoch
        logger.info(
            f"Adjusted eval_steps from {original_eval_steps} to {training_args.eval_steps}"
        )

    # Set num_train_epochs to 1 since we're handling epochs via dataset repetition
    training_args.num_train_epochs = 1
    logger.info(f"Set num_train_epochs=1 (original was {original_num_train_epochs})")
    logger.info(
        f"Dataset will be repeated {original_num_train_epochs} times internally"
    )

    logger.info(f"Original dataset size: {original_data_size}")
    logger.info(f"Expanded dataset size: {len(train_dataset)}")
    logger.info(f"Decoder amortization ratio: 1/{data_args.amortization_ratio} of frames")

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

    logger.info("Starting training loop...")
    trainer = CSMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info(
        f"Effective batch size: {effective_batch_size} "
        f"(per_device_batch={training_args.per_device_train_batch_size} "
        f"× grad_accum={training_args.gradient_accumulation_steps} "
        f"× num_gpus={world_size})"
    )
    logger.info(f"Steps per original epoch: {steps_per_old_epoch}")
    logger.info(f"Original dataset size: {original_data_size}")
    logger.info(f"Expanded dataset size: {len(train_dataset)}")

    trainer.train()

    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved at {training_args.output_dir}")


if __name__ == "__main__":
    main()
