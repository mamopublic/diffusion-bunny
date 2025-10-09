# Stage 5: Fine-Tuning with LoRA

This stage fine-tunes a Stable Diffusion model on your character dataset using **LoRA (Low-Rank Adaptation)** via the HuggingFace **PEFT** library.

## Overview

- **Method**: DreamBooth LoRA with PEFT
- **Dataset**: 65 frames from Stage 4 LLM analysis
- **Training Platform**: Google Colab (free GPU) or local
- **Expected Output**: LoRA adapter weights (~10-100MB)
- **Training Time**: 1-2 hours on Colab, 1.5-2 days on CPU

## Quick Start

### Step 1: Prepare Data

Convert Stage 4 LLM analysis to PEFT-compatible format:

```bash
python -m src.stage5_finetuning.prepare_data \
  --llm-analysis outputs/run_20250902_165141/llm_analysis/llm_analysis.json \
  --output-dir training_data \
  --min-confidence 0.5
```

This creates:
```
training_data/
├── instance_images/
│   ├── frame_000702.jpg
│   ├── frame_000706.jpg
│   ├── ...
│   └── metadata.jsonl  # Captions for each image
└── data_summary.json
```

### Step 2: Set Up PEFT

Clone the PEFT repository (contains DreamBooth training script):

```bash
git clone https://github.com/huggingface/peft
cd peft/examples/lora_dreambooth
pip install -r requirements.txt
```

### Step 3: Configure Training

Set environment variables:

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"  # or "Linaqruf/anything-v3.0" for anime
export INSTANCE_DIR="path/to/training_data/instance_images"
export CLASS_DIR="path/to/regularization_images"  # Optional: prior preservation
export OUTPUT_DIR="outputs/lora_weights"
```

### Step 4: Train

#### Option A: Google Colab (Recommended - Free GPU)

1. Upload `training_data/` to Google Drive
2. Open new Colab notebook
3. Mount Drive and clone PEFT:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   !git clone https://github.com/huggingface/peft
   %cd peft/examples/lora_dreambooth
   !pip install -r requirements.txt
   ```

4. Run training:
   ```bash
   !accelerate launch train_dreambooth.py \
     --pretrained_model_name_or_path="Linaqruf/anything-v3.0" \
     --instance_data_dir="/content/drive/MyDrive/training_data/instance_images" \
     --output_dir="/content/drive/MyDrive/lora_weights" \
     --instance_prompt="anime character" \
     --resolution=512 \
     --train_batch_size=1 \
     --gradient_accumulation_steps=1 \
     --learning_rate=1e-4 \
     --lr_scheduler="constant" \
     --lr_warmup_steps=0 \
     --max_train_steps=1000 \
     --use_lora \
     --lora_r=16 \
     --lora_alpha=27 \
     --lora_text_encoder_r=16 \
     --lora_text_encoder_alpha=17 \
     --gradient_checkpointing \
     --mixed_precision="fp16"
   ```

#### Option B: Local Training

For local CPU training (very slow):

```bash
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="nota-ai/sd-v1-5-tiny" \
  --instance_data_dir="training_data/instance_images" \
  --output_dir="outputs/lora_weights" \
  --instance_prompt="anime character" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --learning_rate=5e-5 \
  --max_train_steps=500 \
  --use_lora \
  --lora_r=16 \
  --lora_alpha=16 \
  --gradient_checkpointing
```

### Step 5: Inference

Use your trained LoRA adapter:

```python
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "Linaqruf/anything-v3.0",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA weights
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    "outputs/lora_weights/unet"
)
pipe.text_encoder = PeftModel.from_pretrained(
    pipe.text_encoder,
    "outputs/lora_weights/text_encoder"
)

# Generate!
prompt = "ellie, anime character, smiling in forest"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
image.save("output.png")
```

## Training Configuration

### Recommended Settings for Our Dataset (65 frames)

#### For Google Colab/Kaggle (with GPU):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `base_model` | `Linaqruf/anything-v3.0` | Anime-optimized SD model |
| `lora_r` | 16-32 | LoRA rank (higher = more capacity) |
| `lora_alpha` | 16-32 | LoRA scaling factor |
| `learning_rate` | 1e-4 | Training learning rate |
| `max_train_steps` | 1000 | Total training steps |
| `batch_size` | 1-2 | Batch size (GPU dependent) |
| `gradient_accumulation` | 2-4 | Effective batch size multiplier |
| `mixed_precision` | fp16 | Use half precision |
| `gradient_checkpointing` | true | Save memory |

#### For Local CPU:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `base_model` | `nota-ai/sd-v1-5-tiny` | Smaller 1.6GB model |
| `lora_r` | 8-16 | Lower rank for faster training |
| `learning_rate` | 5e-5 | Lower learning rate |
| `max_train_steps` | 500 | Fewer steps |
| `batch_size` | 1 | CPU limitation |
| `mixed_precision` | no | CPU doesn't support |

### Prior Preservation (Optional)

To prevent overfitting with small dataset, add:

```bash
--with_prior_preservation \
--prior_loss_weight=1.0 \
--class_prompt="anime character" \
--num_class_images=200 \
--class_data_dir="path/to/regularization_images"
```

This requires ~200 generic anime character images for regularization.

## File Structure

```
src/stage5_finetuning/
├── __init__.py
├── README.md                 # This file
├── prepare_data.py          # Data preparation script
└── inference_example.py     # Example inference script (to be created)

training_data/               # Created by prepare_data.py
├── instance_images/
│   ├── frame_*.jpg
│   └── metadata.jsonl
└── data_summary.json

outputs/
└── lora_weights/            # Training output
    ├── unet/
    │   ├── adapter_config.json
    │   └── adapter_model.bin
    └── text_encoder/
        ├── adapter_config.json
        └── adapter_model.bin
```

## Troubleshooting

### Out of Memory (OOM) Errors

If training fails with OOM:

1. **Reduce batch size**: Set `--train_batch_size=1`
2. **Enable gradient checkpointing**: Add `--gradient_checkpointing`
3. **Use smaller LoRA rank**: Set `--lora_r=8`
4. **Use mixed precision**: Add `--mixed_precision="fp16"`

### Poor Quality Results

If generated images are poor quality:

1. **Increase training steps**: Try `--max_train_steps=2000`
2. **Increase LoRA rank**: Try `--lora_r=32`
3. **Use prior preservation**: Add regularization images
4. **Try different base model**: Use anime-specific model

### Colab Disconnects

Colab free tier disconnects after ~12 hours idle:

1. **Save checkpoints**: Training script auto-saves every 500 steps
2. **Download weights periodically**: Sync to Google Drive
3. **Monitor training**: Check logs regularly

## Cost Analysis

| Platform | GPU | Time | Cost |
|----------|-----|------|------|
| **Google Colab (Free)** | T4 | 1-2 hours | $0 |
| **Kaggle** | P100 | 1-2 hours | $0 |
| **Local CPU** | None | 1.5-2 days | $0 (electricity) |
| **RunPod** | RTX 4090 | 1 hour | ~$0.40 |
| **AWS** | g4dn.xlarge | 2 hours | ~$1.00 |

**Recommended**: Google Colab (free, fast, sufficient for our dataset)

## Next Steps

After training completes:

1. **Test inference** with various prompts
2. **Evaluate character consistency**
3. **Adjust hyperparameters** if needed
4. **Move to Stage 6** for inference pipeline integration

## Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [DreamBooth LoRA Guide](https://huggingface.co/docs/peft/main/en/task_guides/dreambooth_lora)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
