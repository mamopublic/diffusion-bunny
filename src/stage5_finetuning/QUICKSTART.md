# Stage 5: LoRA Training - Quick Start Guide

**Pure Python CPU Training for Diffusion Bunny Pipeline**

## Prerequisites

✅ **Stage 4 LLM Analysis completed** (`outputs/*/llm_analysis/llm_analysis.json` exists)  
✅ **32GB RAM available**  
✅ **~48 hours of CPU time**  

**Note:** Training data preparation is now automatic! You don't need to manually run `prepare_data.py`.

## Step-by-Step Workflow

### Step 1: Verify Configuration

Check `config.yaml` finetuning section:

```yaml
finetuning:
  base_model: "nota-ai/sd-v1-5-tiny"  # 1.6GB CPU-friendly model
  device: "cpu"
  data_dir: "training_data/instance_images"
  
  lora:
    rank: 8  # Lower rank for faster CPU training
    alpha: 16
  
  training:
    learning_rate: 5.0e-5
    max_train_steps: 500
    batch_size: 1
    gradient_accumulation_steps: 8
```

### Step 2: Start Training (Self-Contained)

**One command does everything!**

```bash
python -m src.stage5_finetuning.train
```

This will:
1. ✅ Auto-detect if training data needs preparation
2. ✅ If needed, automatically prepare data from Stage 4 LLM analysis
3. ✅ Train LoRA weights
4. ✅ Save checkpoints every 100 steps

**Or programmatically:**

```python
from src.stage5_finetuning.train import LoRATrainer

# Self-contained - handles data prep automatically
trainer = LoRATrainer(config_path="config.yaml")
trainer.train()
```

**What happens automatically:**
- Searches for most recent `llm_analysis.json` in `outputs/`
- Prepares training data if `training_data/instance_images/metadata.jsonl` doesn't exist
- Loads SD-Tiny model and applies LoRA
- Trains for configured steps
- Saves final weights to `outputs/lora_weights/final/`

### Step 4: Monitor Progress

Training will display:

```
==================================================
LoRA Trainer Initialized
==================================================
Device: cpu
Base Model: nota-ai/sd-v1-5-tiny
LoRA Rank: 8
Learning Rate: 5e-05
Max Steps: 500
==================================================
Loading models...
✅ Models loaded successfully
Applying LoRA adapters...
✅ LoRA applied successfully
UNet trainable parameters: 294,912
Text Encoder trainable parameters: 147,456
Creating dataloader...
✅ Dataloader created: 65 batches
Setting up optimizer...
✅ Optimizer created: 2 parameter groups

==================================================
STARTING TRAINING
==================================================
Training: 100%|████████| 500/500 [48:23:15<00:00, loss=0.0234]
💾 Checkpoint saved: outputs/lora_weights/checkpoint-100
💾 Checkpoint saved: outputs/lora_weights/checkpoint-200
💾 Checkpoint saved: outputs/lora_weights/checkpoint-300
💾 Checkpoint saved: outputs/lora_weights/checkpoint-400
💾 Checkpoint saved: outputs/lora_weights/checkpoint-500

==================================================
TRAINING COMPLETE
==================================================
✅ Final weights saved: outputs/lora_weights/final
==================================================
```

### Step 5: Test Inference

Generate images with trained LoRA:

```bash
python -m src.stage5_finetuning.inference_example \
  --lora-weights outputs/lora_weights/final \
  --prompt "ellie, anime character, smiling in forest, detailed" \
  --output test_output.png
```

**Or programmatically:**

```python
from src.stage5_finetuning.inference_example import load_lora_pipeline, generate_image

# Load pipeline
pipe = load_lora_pipeline(
    base_model="nota-ai/sd-v1-5-tiny",
    lora_weights_dir="outputs/lora_weights/final",
    device="cpu"
)

# Generate
image = generate_image(
    pipe,
    prompt="ellie, anime character, smiling in forest",
    seed=42
)
image.save("output.png")
```

## Expected Timeline (CPU)

| Step | Time | Output Size |
|------|------|-------------|
| **Data Preparation** | 2-5 min | ~300MB (65 images) |
| **Model Download** | 5-10 min | 1.6GB (first time only) |
| **Training** | 36-48 hours | ~50MB (LoRA weights) |
| **Inference** | 5-10 min/image | ~500KB/image |

## Directory Structure After Training

```
diffusion-bunny/
├── training_data/
│   ├── instance_images/
│   │   ├── frame_*.jpg (65 images)
│   │   └── metadata.jsonl
│   └── data_summary.json
│
├── outputs/
│   └── lora_weights/
│       ├── checkpoint-100/
│       ├── checkpoint-200/
│       ├── checkpoint-300/
│       ├── checkpoint-400/
│       ├── checkpoint-500/
│       └── final/  ← Use this for inference
│           ├── unet/
│           │   ├── adapter_config.json
│           │   └── adapter_model.bin
│           └── text_encoder/
│               ├── adapter_config.json
│               └── adapter_model.bin
```

## Troubleshooting

### Out of Memory

If training crashes with OOM:

1. **Reduce batch size** (already at minimum: 1)
2. **Reduce LoRA rank** in config.yaml:
   ```yaml
   lora:
     rank: 4  # Instead of 8
   ```
3. **Close other applications**

### Training Too Slow

Expected: **~6 minutes per step** on CPU

To speed up:
- Reduce `max_train_steps` to 300
- Skip checkpoints (increase `save_steps`)
- Ensure no other CPU-intensive tasks running

### Poor Image Quality

If generated images are poor:

1. **Train longer**: Increase `max_train_steps` to 800
2. **Use better checkpoint**: Try checkpoint-300 or checkpoint-400
3. **Adjust prompts**: Be more specific with character names

## Integration with Pipeline

Add to your main pipeline:

```python
# Stage 5: Fine-tuning
if 'finetuning' in config['pipeline']['stages']:
    from src.stage5_finetuning.train import LoRATrainer
    
    trainer = LoRATrainer()
    trainer.train()
```

## Next Steps

After training completes:

1. ✅ Test inference with various prompts
2. ✅ Evaluate character consistency
3. ✅ Move trained weights to production
4. ✅ Proceed to Stage 6: Inference Pipeline

## FAQ

**Q: Can I stop and resume training?**  
A: Yes! Training saves checkpoints every 100 steps. You can resume from checkpoints (manual implementation required).

**Q: Can I use GPU instead?**  
A: Yes! Change `device: "cuda"` in config.yaml. Training will be much faster (1-2 hours vs 36-48 hours).

**Q: How much does this cost?**  
A: $0 for CPU training (just electricity). You already have the hardware.

**Q: Can I train on multiple characters?**  
A: Yes! The model learns all characters in your dataset. Use character names in prompts during inference.
