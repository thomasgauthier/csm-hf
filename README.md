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