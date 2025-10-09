"""
Pure Python LoRA Training Script
CPU-optimized DreamBooth LoRA fine-tuning using PEFT

Self-contained stage: Automatically prepares data if needed, then trains
"""

import torch
from torch.optim import AdamW
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime
import glob
import shutil

from .training_config import load_config, FinetuningConfig
from .data_loader import create_dataloader
from .prepare_data import DreamBoothDataPreparer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRATrainer:
    """LoRA fine-tuning trainer for Stable Diffusion"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize trainer
        
        Args:
            config_path: Path to config.yaml
        """
        self.config = load_config(config_path)
        self.device = torch.device(self.config.device)
        
        # Set random seed
        torch.manual_seed(self.config.training.seed)
        
        # Initialize components
        self.tokenizer = None
        self.text_encoder = None
        self.unet = None
        self.noise_scheduler = None
        self.optimizer = None
        self.train_dataloader = None
        
        logger.info("="*50)
        logger.info("LoRA Trainer Initialized")
        logger.info("="*50)
        logger.info(f"Device: {self.device}")
        logger.info(f"Base Model: {self.config.base_model}")
        logger.info(f"LoRA Rank: {self.config.lora.rank}")
        logger.info(f"Learning Rate: {self.config.training.learning_rate}")
        logger.info(f"Max Steps: {self.config.training.max_train_steps}")
        logger.info("="*50)
    
    def load_models(self):
        """Load Stable Diffusion models"""
        logger.info("="*50)
        logger.info(f"📥 LOADING BASE MODEL: {self.config.base_model}")
        logger.info("="*50)
        
        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.base_model,
            subfolder="tokenizer"
        )
        
        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model,
            subfolder="text_encoder"
        )
        
        # Load VAE (for encoding images to latent space)
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model,
            subfolder="vae"
        )
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model,
            subfolder="unet"
        )
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.base_model,
            subfolder="scheduler"
        )
        
        # Move to device and freeze VAE
        self.vae.to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        self.text_encoder.to(self.device)
        self.unet.to(self.device)
        
        logger.info("✅ Models loaded successfully")
    
    def apply_lora(self):
        """Apply LoRA adapter to UNet only (text encoder frozen)"""
        logger.info("\n" + "="*50)
        logger.info("APPLYING LORA ADAPTERS FOR TRAINING")
        logger.info("="*50)
        logger.info(f"LoRA Configuration:")
        logger.info(f"  Rank: {self.config.lora.rank}")
        logger.info(f"  Alpha: {self.config.lora.alpha}")
        logger.info(f"  Dropout: {self.config.lora.dropout}")
        logger.info(f"  Target modules: {self.config.lora.target_modules}")
        
        # UNet LoRA configuration
        unet_lora_config = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            init_lora_weights="gaussian",
        )
        
        # Apply LoRA to UNet only
        logger.info("\n✅ Applying LoRA to UNet...")
        self.unet = get_peft_model(self.unet, unet_lora_config)
        logger.info("   ✅ UNet LoRA adapter applied successfully")
        
        # Freeze text encoder (no training)
        logger.info("ℹ️  Freezing Text Encoder (no LoRA, no training)...")
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        logger.info("   ✅ Text Encoder frozen")
        
        # Print trainable parameters
        unet_trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        text_encoder_trainable = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        
        logger.info("\n" + "="*50)
        logger.info("TRAINABLE PARAMETERS SUMMARY")
        logger.info("="*50)
        logger.info(f"UNet trainable parameters: {unet_trainable:,}")
        logger.info(f"Text Encoder trainable parameters: {text_encoder_trainable:,} (frozen)")
        logger.info("="*50)
    
    def _check_training_data(self) -> bool:
        """Check if training data exists"""
        metadata_file = self.config.data_dir / "metadata.jsonl"
        return metadata_file.exists()
    
    def _find_llm_analysis(self) -> Path:
        """Find the most recent LLM analysis JSON file"""
        # Look for llm_analysis.json in outputs directory
        outputs_dir = Path("outputs")
        
        if not outputs_dir.exists():
            raise FileNotFoundError(
                "No outputs directory found. Please run Stage 4 (LLM Analysis) first."
            )
        
        # Search for llm_analysis.json files
        llm_analysis_files = list(outputs_dir.glob("*/llm_analysis/llm_analysis.json"))
        
        if not llm_analysis_files:
            raise FileNotFoundError(
                "No LLM analysis files found in outputs/. "
                "Please run Stage 4 (LLM Analysis) first."
            )
        
        # Get the most recent one
        most_recent = max(llm_analysis_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found LLM analysis: {most_recent}")
        
        return most_recent
    
    def _prepare_data(self, config_path: str = "config.yaml"):
        """Automatically prepare training data from LLM analysis"""
        logger.info("\n" + "="*50)
        logger.info("PREPARING TRAINING DATA")
        logger.info("="*50)
        
        try:
            # Find LLM analysis file
            llm_analysis_json = self._find_llm_analysis()
            
            # Prepare data (output_dir will be auto-derived from LLM analysis path)
            preparer = DreamBoothDataPreparer(
                llm_analysis_json=llm_analysis_json,
                output_dir=None,  # Auto-derive: run_dir/training_data
                min_confidence=0.5
            )
            
            summary = preparer.prepare()
            
            # Update config paths to use the run-based directory
            run_dir = llm_analysis_json.parent.parent
            self.config.data_dir = run_dir / "training_data" / "instance_images"
            self.config.output_dir = run_dir / "lora_weights"
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy config to run directory for reproducibility
            config_dest = run_dir / "config.yaml"
            if Path(config_path).exists():
                shutil.copy2(config_path, config_dest)
                logger.info(f"📋 Config copied to: {config_dest}")
            
            logger.info(f"Training data: {self.config.data_dir}")
            logger.info(f"LoRA weights will be saved to: {self.config.output_dir}")
            
            logger.info("="*50)
            logger.info("DATA PREPARATION COMPLETE")
            logger.info("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def create_dataloader(self):
        """Create training dataloader"""
        logger.info("Creating dataloader...")
        
        self.train_dataloader = create_dataloader(
            data_dir=self.config.data_dir,
            batch_size=self.config.training.batch_size,
            resolution=self.config.resolution,
            center_crop=self.config.center_crop,
            random_flip=self.config.random_flip,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=True
        )
        
        logger.info(f"✅ Dataloader created: {len(self.train_dataloader)} batches")
    
    def setup_optimizer(self):
        """Setup optimizer"""
        logger.info("Setting up optimizer...")
        
        # Get trainable parameters
        params_to_optimize = (
            list(self.unet.parameters()) +
            list(self.text_encoder.parameters())
        )
        
        # Filter for trainable parameters only
        params_to_optimize = [p for p in params_to_optimize if p.requires_grad]
        
        # Create optimizer
        self.optimizer = AdamW(
            params_to_optimize,
            lr=self.config.training.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        logger.info(f"✅ Optimizer created: {len(params_to_optimize)} parameter groups")
    
    def encode_prompt(self, prompts):
        """Encode text prompts to embeddings"""
        # Tokenize
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        
        return text_embeddings
    
    def train_step(self, batch):
        """Single training step"""
        # Get batch data
        pixel_values = batch['pixel_values'].to(self.device)
        captions = batch['captions']
        
        # Encode images to latent space using VAE
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215  # Scaling factor for SD 1.5
        
        # Encode prompts
        encoder_hidden_states = self.encode_prompt(captions)
        
        # Sample noise in latent space
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            0, 
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device
        ).long()
        
        # Add noise to latents (forward diffusion)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise in latent space
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states
        ).sample
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
        
        return loss
    
    def save_checkpoint(self, step: int):
        """Save UNet LoRA weights checkpoint (text encoder not trained)"""
        checkpoint_dir = self.config.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save UNet LoRA only
        unet_dir = checkpoint_dir / "unet"
        self.unet.save_pretrained(unet_dir)
        
        # Save training state
        state = {
            'step': step,
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config.to_dict()
        }
        torch.save(state, checkpoint_dir / "training_state.pt")
        
        logger.info(f"💾 Checkpoint saved (UNet only): {checkpoint_dir}")
    
    def train(self):
        """Main training loop - Self-contained stage"""
        logger.info("\n" + "="*50)
        logger.info("STAGE 5: LORA FINE-TUNING")
        logger.info("="*50)
        
        # Auto-prepare data if needed
        if not self._check_training_data():
            logger.info("⚠️  Training data not found. Running data preparation...")
            self._prepare_data()
        else:
            logger.info("✅ Training data already prepared. Skipping data preparation.")
            
            # Derive run directory from existing training data path
            # e.g., outputs/run_20250902_165141/training_data/instance_images
            # -> outputs/run_20250902_165141
            if "outputs" in str(self.config.data_dir):
                # Extract run directory from path
                parts = self.config.data_dir.parts
                if "outputs" in parts:
                    outputs_idx = parts.index("outputs")
                    if outputs_idx + 1 < len(parts):
                        run_dir = Path(*parts[:outputs_idx+2])
                        self.config.output_dir = run_dir / "lora_weights"
                        self.config.output_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Using run directory: {run_dir}")
                        logger.info(f"LoRA weights will be saved to: {self.config.output_dir}")
                        
                        # Copy config if not already there
                        config_dest = run_dir / "config.yaml"
                        if not config_dest.exists() and Path("config.yaml").exists():
                            shutil.copy2("config.yaml", config_dest)
                            logger.info(f"📋 Config copied to: {config_dest}")
        
        # Setup
        logger.info("\n" + "="*50)
        logger.info("STARTING TRAINING")
        logger.info("="*50)
        
        self.load_models()
        self.apply_lora()
        self.create_dataloader()
        self.setup_optimizer()
        
        # Training state
        global_step = 0
        train_loss = 0.0
        
        # Enable gradient checkpointing for UNet only
        if self.config.training.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        
        # Set UNet to training mode, keep text encoder frozen
        self.unet.train()
        self.text_encoder.eval()  # Keep frozen
        
        # Training loop
        progress_bar = tqdm(
            total=self.config.training.max_train_steps,
            desc="Training"
        )
        
        while global_step < self.config.training.max_train_steps:
            for batch in self.train_dataloader:
                # Forward pass
                loss = self.train_step(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Track loss
                train_loss += loss.detach().item()
                
                # Logging
                if (global_step + 1) % self.config.training.logging_steps == 0:
                    avg_loss = train_loss / self.config.training.logging_steps
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    train_loss = 0.0
                
                # Save checkpoint
                if (global_step + 1) % self.config.training.save_steps == 0:
                    self.save_checkpoint(global_step + 1)
                
                # Update progress
                global_step += 1
                progress_bar.update(1)
                
                # Check if done
                if global_step >= self.config.training.max_train_steps:
                    break
        
        progress_bar.close()
        
        # Final save
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETE")
        logger.info("="*50)
        self.save_checkpoint(global_step)
        
        # Save final UNet LoRA weights only
        final_dir = self.config.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        unet_final = final_dir / "unet"
        self.unet.save_pretrained(unet_final)
        
        logger.info(f"✅ Final UNet LoRA weights saved: {final_dir}/unet")
        logger.info("="*50)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LoRA for Stable Diffusion")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        default=None,
        help='Specific run directory to use (e.g., outputs/run_20250902_165141)'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = LoRATrainer(config_path=args.config)
    
    # Override paths if run-dir specified
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise ValueError(f"Run directory does not exist: {run_dir}")
        
        # Set paths based on run directory
        trainer.config.data_dir = run_dir / "training_data" / "instance_images"
        trainer.config.output_dir = run_dir / "lora_weights"
        trainer.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using specified run directory: {run_dir}")
        logger.info(f"Training data: {trainer.config.data_dir}")
        logger.info(f"LoRA weights: {trainer.config.output_dir}")
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
