"""
Siamese Network Trainer
Main training orchestrator for the Siamese network stage.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import logging
from datetime import datetime

from .siamese_network import create_siamese_network, create_loss_function
from .data_generator import create_data_module
from .embedding_extractor import create_embedding_extractor, create_character_database
from .utils import save_training_config, save_training_log, calculate_accuracy


class SiameseTrainer:
    """
    Main trainer class for Siamese network training.
    
    Handles the complete training pipeline including data loading,
    model training, validation, and saving results.
    """
    
    def __init__(self, config: Dict, project_root: Path):
        """
        Initialize the Siamese trainer.
        
        Args:
            config: Configuration dictionary
            project_root: Root directory of the project (e.g., pipeline_data/sprite)
        """
        self.config = config
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.siamese_config = config.get('siamese', {})
        self.training_config = self.siamese_config.get('training', {})
        
        # Training parameters
        self.epochs = self.training_config.get('epochs', 50)
        self.learning_rate = self.training_config.get('learning_rate', 0.001)
        self.batch_size = self.training_config.get('batch_size', 32)
        self.device = self._get_device()
        
        # Paths
        self.characters_dir = project_root / config['input']['character_references_dir']
        self.output_dir = project_root / "siamese"
        self.output_dir.mkdir(exist_ok=True)
        
        # Model and training components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.data_module = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        self.logger.info(f"Initialized Siamese trainer for project: {project_root}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Characters directory: {self.characters_dir}")
    
    def _get_device(self) -> str:
        """Determine the device to use for training."""
        device_config = self.training_config.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info("CUDA available, using GPU for training")
            else:
                device = 'cpu'
                self.logger.info("CUDA not available, using CPU for training")
        else:
            device = device_config
            self.logger.info(f"Using specified device: {device}")
        
        return device
    
    def _validate_characters_directory(self) -> bool:
        """Validate that character reference images exist."""
        if not self.characters_dir.exists():
            self.logger.error(f"Characters directory not found: {self.characters_dir}")
            return False
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        character_files = []
        
        for ext in supported_formats:
            character_files.extend(self.characters_dir.glob(f"*{ext}"))
            character_files.extend(self.characters_dir.glob(f"*{ext.upper()}"))
        
        if len(character_files) < 2:
            self.logger.error(f"Need at least 2 character reference images, found {len(character_files)}")
            return False
        
        # Count unique characters
        character_names = set(f.stem for f in character_files)
        if len(character_names) < 2:
            self.logger.error(f"Need at least 2 different characters, found {len(character_names)}")
            return False
        
        self.logger.info(f"Found {len(character_files)} images for {len(character_names)} characters")
        return True
    
    def _initialize_components(self):
        """Initialize model, loss function, optimizer, and data module."""
        self.logger.info("Initializing training components...")
        
        # Create model
        self.model = create_siamese_network(self.config)
        self.model.to(self.device)
        
        # Create loss function
        self.criterion = create_loss_function(self.config)
        
        # Create optimizer
        optimizer_type = self.training_config.get('optimizer', 'adam')
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.training_config.get('weight_decay', 1e-4)
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.training_config.get('momentum', 0.9),
                weight_decay=self.training_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Create learning rate scheduler
        scheduler_type = self.training_config.get('scheduler', 'step')
        if scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.training_config.get('scheduler_step_size', 20),
                gamma=self.training_config.get('scheduler_gamma', 0.5)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.training_config.get('scheduler_gamma', 0.5),
                patience=self.training_config.get('scheduler_patience', 10),
                verbose=True
            )
        
        # Create data module
        self.data_module = create_data_module(self.characters_dir, self.config)
        
        self.logger.info("Training components initialized successfully")
    
    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        train_loader = self.data_module.train_dataloader()
        
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            # Move to device
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            embedding1, embedding2 = self.model(img1, img2)
            
            # Compute loss
            loss = self.criterion(embedding1, embedding2, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                batch_accuracy = calculate_accuracy(embedding1, embedding2, labels)
                correct_predictions += batch_accuracy * len(labels)
                total_predictions += len(labels)
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                # Move to device
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embedding1, embedding2 = self.model(img1, img2)
                
                # Compute loss
                loss = self.criterion(embedding1, embedding2, labels)
                
                # Statistics
                total_loss += loss.item()
                
                # Calculate accuracy
                batch_accuracy = calculate_accuracy(embedding1, embedding2, labels)
                correct_predictions += batch_accuracy * len(labels)
                total_predictions += len(labels)
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "model_weights.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to: {best_path}")
    
    def _load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load model checkpoint for resuming training."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.training_history = checkpoint['training_history']
            
            self.logger.info(f"Resumed training from epoch {self.current_epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def train(self, resume: bool = False) -> Dict:
        """
        Run the complete training process.
        
        Args:
            resume: Whether to resume from existing checkpoint
            
        Returns:
            Training results dictionary
        """
        try:
            self.logger.info("Starting Siamese network training...")
            
            # Validate inputs
            if not self._validate_characters_directory():
                return {"success": False, "error": "Invalid characters directory"}
            
            # Initialize components
            self._initialize_components()
            
            # Resume from checkpoint if requested
            if resume:
                checkpoint_path = self.output_dir / "checkpoint_latest.pth"
                if checkpoint_path.exists():
                    self._load_checkpoint(checkpoint_path)
                else:
                    self.logger.warning("Resume requested but no checkpoint found")
            
            # Save training configuration
            config_path = self.output_dir / "training_config.json"
            save_training_config(self.config, config_path)
            
            # Training loop
            start_time = time.time()
            
            for epoch in range(self.current_epoch, self.epochs):
                epoch_start_time = time.time()
                
                # Regenerate data pairs for this epoch
                self.data_module.regenerate_data()
                
                # Train
                train_loss, train_accuracy = self._train_epoch()
                
                # Validate
                val_loss, val_accuracy = self._validate_epoch()
                
                # Update learning rate
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                
                # Record history
                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['train_accuracy'].append(train_accuracy)
                self.training_history['val_accuracy'].append(val_accuracy)
                self.training_history['learning_rate'].append(current_lr)
                
                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                # Save checkpoint
                self._save_checkpoint(epoch + 1, is_best)
                
                # Log progress
                epoch_time = time.time() - epoch_start_time
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                    f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s"
                )
                
                self.current_epoch = epoch + 1
            
            # Training completed
            total_time = time.time() - start_time
            self.logger.info(f"Training completed in {total_time:.2f} seconds")
            
            # Save final training log
            log_path = self.output_dir / "training_log.json"
            save_training_log(self.training_history, total_time, log_path)
            
            # Create character embedding database
            self._create_embedding_database()
            
            result = {
                "success": True,
                "epochs_trained": self.epochs,
                "best_val_loss": self.best_val_loss,
                "final_train_accuracy": self.training_history['train_accuracy'][-1],
                "final_val_accuracy": self.training_history['val_accuracy'][-1],
                "training_time": total_time,
                "model_path": str(self.output_dir / "model_weights.pth"),
                "embeddings_path": str(self.output_dir / "character_embeddings.pkl"),
                "output_directory": str(self.output_dir)
            }
            
            self.logger.info("Siamese network training completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "output_directory": str(self.output_dir) if hasattr(self, 'output_dir') else None
            }
    
    def _create_embedding_database(self):
        """Create character embedding database after training."""
        try:
            self.logger.info("Creating character embedding database...")
            
            # Load the best model
            model_path = self.output_dir / "model_weights.pth"
            extractor = create_embedding_extractor(model_path, self.config, self.device)
            
            # Create database
            database = create_character_database(self.characters_dir, extractor)
            
            # Save database
            embeddings_path = self.output_dir / "character_embeddings.pkl"
            database.save_database(embeddings_path)
            
            self.logger.info(f"Character embedding database saved to: {embeddings_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create embedding database: {e}")
            raise


def main():
    """Main entry point for Siamese training stage."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Train Siamese network for character recognition")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--project-root", required=True, help="Path to project root (e.g., pipeline_data/sprite)")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        return 1
    except yaml.YAMLError as e:
        print(f"Error: Error parsing configuration file: {e}")
        return 1
    
    # Run training
    try:
        project_root = Path(args.project_root)
        trainer = SiameseTrainer(config, project_root)
        result = trainer.train(resume=args.resume)
        
        if result["success"]:
            print("Siamese network training completed successfully!")
            print(f"Best validation loss: {result['best_val_loss']:.4f}")
            print(f"Final validation accuracy: {result['final_val_accuracy']:.4f}")
            print(f"Training time: {result['training_time']:.2f} seconds")
            print(f"Model saved to: {result['model_path']}")
            print(f"Embeddings saved to: {result['embeddings_path']}")
            return 0
        else:
            print(f"Error: Training failed: {result['error']}")
            if result.get('output_directory'):
                print(f"Output directory: {result['output_directory']}")
            return 1
            
    except Exception as e:
        print(f"Error: Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
