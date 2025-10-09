"""
Inference Example for DreamBooth LoRA
Demonstrates how to use trained LoRA weights for image generation
"""

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from pathlib import Path
import argparse
import yaml


def load_lora_pipeline(
    lora_weights_dir: str,
    device: str = None,
    dtype = None
):
    """
    Load Stable Diffusion pipeline with LoRA weights
    Automatically loads base model from run directory config
    
    Args:
        lora_weights_dir: Path to trained LoRA weights directory (e.g., "outputs/run_XXX/lora_weights/final")
        device: Device to run on ("cuda" or "cpu", auto-detects if None)
        dtype: Model precision (auto-selects based on device if None)
    
    Returns:
        Configured pipeline ready for inference
    """
    lora_weights_dir = Path(lora_weights_dir)
    
    # Derive run directory from lora_weights path
    # e.g., outputs/run_20250902_165141/lora_weights/final -> outputs/run_20250902_165141
    run_dir = lora_weights_dir.parent
    if lora_weights_dir.name in ["final", "checkpoint-100", "checkpoint-200", "checkpoint-300", "checkpoint-400", "checkpoint-500"]:
        run_dir = lora_weights_dir.parent.parent
    
    # Load config from run directory
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found in run directory: {config_path}\n"
            f"This run may have been created before config copying was implemented.\n"
            f"Please manually specify base model or re-run training."
        )
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    base_model = config['finetuning']['base_model']
    print("="*50)
    print(f"✅ Loaded config from: {config_path}")
    print(f"📥 LOADING BASE MODEL: {base_model}")
    print("="*50)
    
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-select dtype based on device
    if dtype is None:
        if device == "cpu":
            dtype = torch.float32  # CPU requires float32
        else:
            dtype = torch.float16  # GPU can use float16
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Downloading/Loading model components...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype
    ).to(device)
    
    # Load UNet LoRA weights
    print("\n" + "="*50)
    print("LOADING LORA ADAPTERS")
    print("="*50)
    unet_lora_path = lora_weights_dir / "unet"
    if unet_lora_path.exists():
        print(f"✅ Found UNet LoRA at: {unet_lora_path}")
        print(f"   Loading UNet LoRA adapter...")
        pipe.unet = PeftModel.from_pretrained(
            pipe.unet,
            str(unet_lora_path)
        )
        print(f"   ✅ UNet LoRA adapter loaded successfully")
    else:
        print(f"❌ UNet LoRA not found at {unet_lora_path}")
        print(f"   WARNING: Running inference WITHOUT LoRA fine-tuning!")
    
    # Load Text Encoder LoRA weights (optional)
    text_encoder_lora_path = lora_weights_dir / "text_encoder"
    if text_encoder_lora_path.exists():
        print(f"✅ Found Text Encoder LoRA at: {text_encoder_lora_path}")
        print(f"   Loading Text Encoder LoRA adapter...")
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder,
            str(text_encoder_lora_path)
        )
        print(f"   ✅ Text Encoder LoRA adapter loaded successfully")
    else:
        print(f"ℹ️  Text Encoder LoRA not found (expected for UNet-only training)")
    print("="*50)
    
    # Apply half precision only on GPU
    if device != "cpu" and dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()
    
    # CPU-specific optimizations
    if device == "cpu":
        print("⚠️  CPU inference will be slow (5-10 minutes per image)")
        pipe.enable_attention_slicing()  # Reduce memory usage
    
    print("✅ Pipeline loaded successfully!")
    return pipe


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str = "low quality, blurry, unfinished",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = None
):
    """
    Generate an image using the LoRA-enhanced pipeline
    
    Args:
        pipe: Configured pipeline
        prompt: Text prompt describing the desired image
        negative_prompt: What to avoid in the image
        num_inference_steps: Number of denoising steps (more = better quality, slower)
        guidance_scale: How strongly to follow the prompt (7-15 typical)
        width: Image width in pixels
        height: Image height in pixels
        seed: Random seed for reproducibility (None for random)
    
    Returns:
        Generated PIL Image
    """
    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    print(f"\n📝 Prompt: {prompt}")
    print(f"⛔ Negative: {negative_prompt}")
    print(f"🎲 Seed: {seed if seed else 'random'}")
    print("🎨 Generating...")
    
    # Generate
    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator
    )
    
    image = result.images[0]
    print("✅ Generation complete!")
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Generate images with trained LoRA")
    parser.add_argument(
        '--run-dir',
        type=str,
        default=None,
        help='Run directory (e.g., outputs/run_20250902_165141). If provided, --lora-weights defaults to "final"'
    )
    parser.add_argument(
        '--lora-weights',
        type=str,
        default=None,
        help='Path to LoRA weights. Can be absolute or relative to --run-dir (default: "final" when --run-dir is provided)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Text prompt for generation'
    )
    parser.add_argument(
        '--negative-prompt',
        type=str,
        default="low quality, blurry, unfinished",
        help='Negative prompt'
    )
    parser.add_argument(
        '--output',
        type=str,
        default="generated_image.png",
        help='Output image path'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Number of inference steps'
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=7.5,
        help='Guidance scale (7-15 typical)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to run on (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Determine lora_weights path
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise ValueError(f"Run directory does not exist: {run_dir}")
        
        # If lora_weights not specified, default to "final"
        if args.lora_weights is None:
            lora_weights_path = run_dir / "lora_weights" / "final"
        else:
            # If lora_weights is relative (e.g., "checkpoint-300"), use run_dir
            lora_weights = Path(args.lora_weights)
            if not lora_weights.is_absolute():
                lora_weights_path = run_dir / "lora_weights" / args.lora_weights
            else:
                lora_weights_path = lora_weights
    elif args.lora_weights:
        lora_weights_path = Path(args.lora_weights)
    else:
        raise ValueError("Either --run-dir or --lora-weights must be specified")
    
    if not lora_weights_path.exists():
        raise ValueError(f"LoRA weights not found: {lora_weights_path}")
    
    print(f"Using LoRA weights: {lora_weights_path}")
    
    # Load pipeline (base model loaded automatically from run config)
    pipe = load_lora_pipeline(
        lora_weights_dir=str(lora_weights_path),
        device=args.device
    )
    
    # Generate image
    image = generate_image(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )
    
    # Save
    image.save(args.output)
    print(f"\n💾 Saved to: {args.output}")


if __name__ == "__main__":
    # Example usage without command line:
    # Uncomment and modify as needed
    
    # # Load pipeline (base model auto-loaded from run config)
    # pipe = load_lora_pipeline(
    #     lora_weights_dir="outputs/run_20250902_165141/lora_weights/final",
    #     device="cpu"  # or "cuda"
    # )
    # 
    # # Test with different characters from training data
    # prompts = [
    #     "ellie, anime character, smiling in forest, detailed, high quality",
    #     "phil, anime character, casual clothes, outdoor scene",
    #     "rex, anime character, portrait, detailed",
    #     "victoria, anime character, elegant dress"
    # ]
    # 
    # for i, prompt in enumerate(prompts):
    #     image = generate_image(pipe, prompt, seed=42, num_inference_steps=30)
    #     image.save(f"test_output_{i}.png")
    
    main()
