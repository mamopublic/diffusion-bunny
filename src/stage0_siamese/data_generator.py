"""
Siamese Data Generator
Generates training pairs (positive and negative) from character reference images.
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging


class SiameseDataGenerator(Dataset):
    """
    Dataset class for generating Siamese network training pairs.
    
    Creates positive pairs (same character) and negative pairs (different characters)
    with data augmentation to increase training data diversity.
    """
    
    def __init__(self, characters_dir: Path, config: dict, split: str = 'train'):
        """
        Initialize the Siamese data generator.
        
        Args:
            characters_dir: Directory containing character reference images
            config: Configuration dictionary
            split: Dataset split ('train' or 'val')
        """
        self.characters_dir = characters_dir
        self.config = config
        self.split = split
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        siamese_config = config.get('siamese', {})
        training_config = siamese_config.get('training', {})
        augmentation_config = training_config.get('augmentation', {})
        
        self.image_size = training_config.get('image_size', 224)
        self.pairs_per_epoch = training_config.get('pairs_per_epoch', 1000)
        self.positive_ratio = training_config.get('positive_ratio', 0.5)
        
        # Load character images
        self.character_images = self._load_character_images()
        self.character_names = list(self.character_images.keys())
        
        if len(self.character_names) < 2:
            raise ValueError("Need at least 2 characters for Siamese training")
        
        # Create data transforms
        self.transform = self._create_transforms(augmentation_config)
        self.base_transform = self._create_base_transforms()
        
        # Generate pairs for this epoch
        self.pairs = self._generate_pairs()
        
        self.logger.info(f"Initialized Siamese dataset with {len(self.character_names)} characters, "
                        f"{len(self.pairs)} pairs for {split} split")
    
    def _load_character_images(self) -> Dict[str, List[str]]:
        """Load character reference images from directory."""
        character_images = {}
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for image_file in self.characters_dir.iterdir():
            if image_file.suffix.lower() in supported_formats:
                character_name = image_file.stem
                
                if character_name not in character_images:
                    character_images[character_name] = []
                
                character_images[character_name].append(str(image_file))
        
        # Filter out characters with no images
        character_images = {name: images for name, images in character_images.items() if images}
        
        self.logger.info(f"Loaded {len(character_images)} characters: {list(character_images.keys())}")
        for name, images in character_images.items():
            self.logger.info(f"  {name}: {len(images)} images")
        
        return character_images
    
    def _create_transforms(self, augmentation_config: dict) -> transforms.Compose:
        """Create data augmentation transforms."""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        
        # Add augmentations for training
        if self.split == 'train':
            rotation_range = augmentation_config.get('rotation_range', 15)
            brightness_range = augmentation_config.get('brightness_range', 0.2)
            contrast_range = augmentation_config.get('contrast_range', 0.2)
            
            transform_list.extend([
                transforms.RandomRotation(rotation_range),
                transforms.ColorJitter(
                    brightness=brightness_range,
                    contrast=contrast_range,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5
                ),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    
    def _create_base_transforms(self) -> transforms.Compose:
        """Create basic transforms without augmentation."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _generate_pairs(self) -> List[Tuple[str, str, int]]:
        """Generate training pairs for this epoch."""
        pairs = []
        num_positive = int(self.pairs_per_epoch * self.positive_ratio)
        num_negative = self.pairs_per_epoch - num_positive
        
        # Generate positive pairs (same character)
        for _ in range(num_positive):
            character = random.choice(self.character_names)
            images = self.character_images[character]
            
            if len(images) >= 2:
                # Pick two different images of the same character
                img1, img2 = random.sample(images, 2)
            else:
                # Use the same image twice (will be augmented differently)
                img1 = img2 = random.choice(images)
            
            pairs.append((img1, img2, 1))  # 1 = positive pair
        
        # Generate negative pairs (different characters)
        for _ in range(num_negative):
            char1, char2 = random.sample(self.character_names, 2)
            img1 = random.choice(self.character_images[char1])
            img2 = random.choice(self.character_images[char2])
            
            pairs.append((img1, img2, 0))  # 0 = negative pair
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        return pairs
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB PIL Image."""
        try:
            # Try PIL first
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception:
            # Fallback to OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
    
    def __len__(self) -> int:
        """Return the number of pairs in the dataset."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training pair.
        
        Args:
            idx: Index of the pair
            
        Returns:
            Tuple of (image1, image2, label)
        """
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        # Apply transforms
        img1_tensor = self.transform(img1)
        img2_tensor = self.transform(img2)
        
        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return img1_tensor, img2_tensor, label_tensor
    
    def regenerate_pairs(self):
        """Regenerate pairs for a new epoch."""
        self.pairs = self._generate_pairs()
        self.logger.debug(f"Regenerated {len(self.pairs)} pairs for {self.split} split")


class SiameseDataModule:
    """
    Data module for managing Siamese training and validation datasets.
    """
    
    def __init__(self, characters_dir: Path, config: dict):
        """
        Initialize the data module.
        
        Args:
            characters_dir: Directory containing character reference images
            config: Configuration dictionary
        """
        self.characters_dir = characters_dir
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        siamese_config = config.get('siamese', {})
        training_config = siamese_config.get('training', {})
        
        self.batch_size = training_config.get('batch_size', 32)
        self.num_workers = training_config.get('num_workers', 4)
        self.val_split = training_config.get('val_split', 0.2)
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self):
        """Setup train and validation datasets."""
        # Create training dataset
        train_config = self.config.copy()
        train_config['siamese']['training']['pairs_per_epoch'] = \
            int(self.config['siamese']['training'].get('pairs_per_epoch', 1000) * (1 - self.val_split))
        
        self.train_dataset = SiameseDataGenerator(
            self.characters_dir,
            train_config,
            split='train'
        )
        
        # Create validation dataset
        val_config = self.config.copy()
        val_config['siamese']['training']['pairs_per_epoch'] = \
            int(self.config['siamese']['training'].get('pairs_per_epoch', 1000) * self.val_split)
        
        self.val_dataset = SiameseDataGenerator(
            self.characters_dir,
            val_config,
            split='val'
        )
        
        self.logger.info(f"Setup datasets: {len(self.train_dataset)} train pairs, "
                        f"{len(self.val_dataset)} validation pairs")
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def regenerate_data(self):
        """Regenerate training and validation pairs for new epoch."""
        if self.train_dataset:
            self.train_dataset.regenerate_pairs()
        if self.val_dataset:
            self.val_dataset.regenerate_pairs()


def create_data_module(characters_dir: Path, config: dict) -> SiameseDataModule:
    """
    Factory function to create a Siamese data module.
    
    Args:
        characters_dir: Directory containing character reference images
        config: Configuration dictionary
        
    Returns:
        Configured SiameseDataModule instance
    """
    data_module = SiameseDataModule(characters_dir, config)
    data_module.setup()
    return data_module


def visualize_pairs(data_module: SiameseDataModule, num_pairs: int = 8, save_path: Optional[Path] = None):
    """
    Visualize training pairs for debugging.
    
    Args:
        data_module: Siamese data module
        num_pairs: Number of pairs to visualize
        save_path: Optional path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.getLogger(__name__).warning("Matplotlib not available for visualization")
        return
    
    # Get some training pairs
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    img1_batch, img2_batch, labels_batch = batch
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    fig, axes = plt.subplots(num_pairs, 2, figsize=(8, 2 * num_pairs))
    
    for i in range(min(num_pairs, len(labels_batch))):
        # Denormalize and convert to numpy
        img1 = img1_batch[i] * std + mean
        img2 = img2_batch[i] * std + mean
        
        img1 = torch.clamp(img1, 0, 1).permute(1, 2, 0).numpy()
        img2 = torch.clamp(img2, 0, 1).permute(1, 2, 0).numpy()
        
        label = labels_batch[i].item()
        
        # Plot images
        axes[i, 0].imshow(img1)
        axes[i, 0].set_title(f"Image 1 (Label: {'Same' if label == 1 else 'Different'})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img2)
        axes[i, 1].set_title("Image 2")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.getLogger(__name__).info(f"Saved pair visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()
