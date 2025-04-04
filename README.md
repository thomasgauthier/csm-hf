Sharing something I've been working on: a full rewrite of [Sesame's CSM modeling code](https://github.com/SesameAILabs/csm) for Hugging Face Transformers. It has support for training with HF `Trainer` (with [decoder training amortization](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#:~:text=The%20audio%20decoder%20is%20trained%20on%20only%20a%20random%201/16%20subset%20of%20the%20audio%20frames%2C%20while%20the%20zeroth%20codebook%20is%20trained%20on%20every%20frame.)) as well as generation.  

Finetuning is possible with 24GB ram (2048 frames seq_len, batch size 1, but gradient accumulation is supported for larger effective batch sizes).  

For now, generation seems to be slower than realtime (tested with NVIDIA RTX A5000), but I'm hopeful the model can be optimized. In any case this code can always be used for training only, with possibility of using finetuned weights with different inference code.  

Would love to add LoRA/PEFT support, let me know if that is something that would benefit your use case.