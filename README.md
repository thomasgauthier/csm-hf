# CSM-HF

## Overview

CSM-HF is a Hugging Face implementation of [Sesame's Conversational Speech Model (CSM)](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice). CSM-HF is a complete rewrite of the [pytorch code provided by Sesame](https://github.com/SesameAILabs/csm). This codebase is designed to be fully compatible with Hugging Face `transformers`, from inference to training.

## Changes from Sesame's implementation

- created a `CSMModel` class
- replaced backbone and decoder torchtune models with HF transformers `LllamaModel`
- added a processor class to prepare inputs for the model
- added labels support and [decoder training amortization](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#:~:text=The%20audio%20decoder%20is%20trained%20on%20only%20a%20random%201/16%20subset%20of%20the%20audio%20frames%2C%20while%20the%20zeroth%20codebook%20is%20trained%20on%20every%20frame.)
- added `generate_frame` and `generate` methods to the model class for generating audio
- full support for HuggingFace `Trainer`

## Architecture

Model architecture is discussed in [ARCHITECTURE.md](ARCHITECTURE.md) (written by O1)

## Training

### Data Format

CSM-HF expects training data in a JSONL format, where each line is a JSON object containing a conversation. Each conversation consists of:

- `messages`: An array of message objects, each with:
  - `role`: Speaker identifier (e.g., "speaker_0", "speaker_1")
  - `content`: Array of content objects, which can be:
    - Text: `{"type": "text", "text": "The message text"}`
    - Audio: `{"type": "audio", "url": "path/to/audio/file.wav"}`
- `training_mask`: Boolean array indicating which messages should be used for training (true) or context (false)

Example data format:

```json
{
  "messages": [
    {
      "role": "speaker_0",
      "content": [
        {"type": "text", "text": "We have a chance for a new life here."},
        {"type": "audio", "url": "clips/example_audio.wav"}
      ]
    },
    {
      "role": "speaker_1",
      "content": [
        {"type": "text", "text": "Uncle?"},
        {"type": "audio", "url": "clips/response_audio.wav"}
      ]
    }
  ],
  "training_mask": [false, true]
}
```

### Training Process

The model uses a two-stage autoregressive architecture:

1. **Backbone (Inter-frame Processing)**:
   - Processes the entire sequence of frames
   - Each frame represents a combined embedding of all codebooks
   - Handles long-range dependencies between utterances

2. **Decoder (Intra-frame Processing)**:
   - Processes a single frame at a time
   - Generates 32 codebooks sequentially (1 semantic + 31 acoustic)
   - Each codebook is treated as a token in the sequence

Training leverages compute amortization techniques:
- The zeroth (semantic) codebook is trained on all frames
- The remaining codebooks (1-31) are trained on only `amortization_ratio` of the frames
- This significantly reduces memory usage while maintaining quality

To train the model:

```bash
python train.py \
  --train_file path/to/training_data.jsonl \
  --output_dir ./output \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6
```

## Generation

You can use the model to generate audio from text input. Here's an example for voice cloning:

```python
import torch
from modeling_csm import CSMModel
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from moshi.models import loaders
from processor import CSMProcessor
import torchaudio

device = 'cuda'

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
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer

text_tokenizer = load_llama3_tokenizer()

mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
audio_tokenizer = loaders.get_mimi(mimi_weight, device=device)
audio_tokenizer.set_num_codebooks(32)

processor = CSMProcessor(text_tokenizer, audio_tokenizer)


def load_audio(path, target_sr):
    audio, sr = torchaudio.load(path)
    audio = audio.squeeze(0)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
    return audio


model = CSMModel.from_pretrained("thomasgauthier/csm-1b-hf", torch_dtype=torch.bfloat16)
model.to('cuda')


inputs = processor(
    messages=[
        {
        "role": "speaker_0",
        "content": [
            {"type": "text", "text": "<AUDIO_CLIP_TRANSCRIPT>"},
            {"type": "audio"} # This placeholder is required for audio tokenization (it maps to the first element in the `audios` list passed to the processor)
        ]
    },
            {
        "role": "speaker_0",
        "content": [
            {"type": "text", "text": "Hello, this is voice cloning speaking"},
            # does not include audio as the model will generate it
        ]
    }
        ], 
    audios=[load_audio('AUDIO_CLIP_FOR_VOICE_CLONING.wav', audio_tokenizer.sample_rate)],
    return_tensors="pt"
)

import torch

with torch.inference_mode():
    # Generate up to 50 new frames
    gen_frames = model.generate(
        input_ids=inputs['input_ids'].cuda(),
        attention_mask=inputs['attention_mask'].cuda(),
        max_new_frames=50,
        topk=50,
        temperature=1.0,
        use_cache=True,
        stop_on_all_zeros=True,

    )

decoded_audio = audio_tokenizer.decode(gen_frames.permute(0, 2, 1)).squeeze(0).squeeze(0)

audio_array = (decoded_audio * 32768).to(torch.int16).cpu().numpy()

# Audio can be played with the following code:
# from IPython.display import Audio
# Audio(audio_array, rate=audio_tokenizer.sample_rate)
```

## TODO

- [x] Two-stage autoregressive architecture implementation
- [x] Multi-codebook audio tokenization
- [x] Compute amortization for efficient training
- [x] Dataset preparation with interleaved text/audio
- [x] Custom training loop with separate backbone/decoder losses
- [x] Proper handling of epoch repetition for decoder amortization
- [x] Memory optimization techniques (mixed precision, gradient accumulation)
- [ ] LoRA support for efficient fine-tuning
- [ ] Faster inference with `torch.compile`
- [ ] Coice cloning with prompt tuning / prefix optimization
- [ ] Support for DPO
- [ ] Support for RL (GRPO, RLOO, etc.)

## Acknowledgements

Special thanks to:
- **Sesame Labs** for the original architecture design and implementation
- **Hugging Face** for the Transformers library and training infrastructure
- **Claude** and **ChatGPT** for assistance with documentation and code development

This project builds upon research and tools from the open-source community. I am grateful for the collaborative spirit that makes projects like this possible.