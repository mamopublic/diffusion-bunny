"""
Training configuration for LoRA fine-tuning
Loads settings from config.yaml
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import yaml


@dataclass
class LoRAConfig:
    """LoRA adapter configuration"""
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["to_k", "to_q", "to_v", "to_out.0"])
    train_text_encoder: bool = False  # Only train UNet by default


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    learning_rate: float = 5e-5
    batch_size: int = 1
    max_train_steps: int = 500
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    mixed_precision: str = "no"
    seed: int = 42
    save_steps: int = 100
    logging_steps: int = 10
    validation_prompt: str = "ellie, anime character, smiling"
    num_validation_images: int = 2


@dataclass
class PriorPreservationConfig:
    """Prior preservation settings"""
    enabled: bool = False
    num_class_images: int = 200
    class_prompt: str = "anime character"
    class_data_dir: str = "regularization_images"
    prior_loss_weight: float = 1.0


@dataclass
class FinetuningConfig:
    """Complete fine-tuning configuration"""
    # Model settings
    base_model: str = "nota-ai/sd-v1-5-tiny"
    method: str = "lora"
    output_dir: str = "outputs/lora_weights"
    device: str = "cpu"
    
    # Data settings
    data_dir: str = "training_data/instance_images"
    instance_prompt: str = "anime character"
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = False
    
    # Sub-configs
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prior_preservation: PriorPreservationConfig = field(default_factory=PriorPreservationConfig)
    
    # Performance settings
    num_workers: int = 0
    pin_memory: bool = False
    use_8bit_adam: bool = False
    
    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "FinetuningConfig":
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        finetuning_data = config_data.get('finetuning', {})
        
        # Extract nested configs
        lora_data = finetuning_data.get('lora', {})
        training_data = finetuning_data.get('training', {})
        prior_data = finetuning_data.get('prior_preservation', {})
        
        # Create config objects
        lora_config = LoRAConfig(**lora_data)
        training_config = TrainingConfig(**training_data)
        prior_config = PriorPreservationConfig(**prior_data)
        
        # Remove nested dicts to avoid duplication
        finetuning_data_clean = {
            k: v for k, v in finetuning_data.items()
            if k not in ['lora', 'training', 'prior_preservation']
        }
        
        return cls(
            **finetuning_data_clean,
            lora=lora_config,
            training=training_config,
            prior_preservation=prior_config
        )
    
    def __post_init__(self):
        """Validate configuration"""
        # Ensure paths are Path objects
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        
        # Validate device
        if self.device not in ['cpu', 'cuda']:
            raise ValueError(f"Invalid device: {self.device}. Must be 'cpu' or 'cuda'")
        
        # Validate mixed precision for CPU
        if self.device == 'cpu' and self.training.mixed_precision != 'no':
            raise ValueError("CPU training does not support mixed precision. Set to 'no'")
        
        # Validate batch size
        if self.training.batch_size < 1:
            raise ValueError("Batch size must be >= 1")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            'base_model': self.base_model,
            'method': self.method,
            'output_dir': str(self.output_dir),
            'device': self.device,
            'data_dir': str(self.data_dir),
            'instance_prompt': self.instance_prompt,
            'resolution': self.resolution,
            'lora_rank': self.lora.rank,
            'lora_alpha': self.lora.alpha,
            'learning_rate': self.training.learning_rate,
            'batch_size': self.training.batch_size,
            'max_train_steps': self.training.max_train_steps,
            'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
            'mixed_precision': self.training.mixed_precision,
            'prior_preservation_enabled': self.prior_preservation.enabled,
        }


def load_config(config_path: str = "config.yaml") -> FinetuningConfig:
    """
    Convenience function to load fine-tuning configuration
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        FinetuningConfig instance
    """
    return FinetuningConfig.from_yaml(config_path)
