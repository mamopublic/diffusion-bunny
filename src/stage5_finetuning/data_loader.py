"""
Data loader for LoRA fine-tuning
Loads images and captions from metadata.jsonl
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class DreamBoothDataset(Dataset):
    """
    Dataset for DreamBooth LoRA training
    Loads images and captions from metadata.jsonl
    """
    
    def __init__(
        self,
        data_dir: Path,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing images and metadata.jsonl
            resolution: Target image resolution
            center_crop: Whether to center crop images
            random_flip: Whether to randomly flip images horizontally
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        
        # Load metadata
        self.metadata_path = self.data_dir / "metadata.jsonl"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata.jsonl not found in {self.data_dir}")
        
        self.data = self._load_metadata()
        
        # Create image transforms
        self.transforms = self._create_transforms()
        
        logger.info(f"Loaded {len(self.data)} training examples from {self.data_dir}")
    
    def _load_metadata(self) -> List[Dict]:
        """Load metadata from JSONL file"""
        data = []
        with open(self.metadata_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                
                # Validate entry
                if 'file_name' not in entry or 'text' not in entry:
                    logger.warning(f"Skipping invalid entry: {entry}")
                    continue
                
                # Check if image file exists
                image_path = self.data_dir / entry['file_name']
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                data.append({
                    'image_path': image_path,
                    'caption': entry['text']
                })
        
        if not data:
            raise ValueError("No valid data found in metadata.jsonl")
        
        return data
    
    def _create_transforms(self):
        """Create image transformation pipeline"""
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR))
        
        # Center crop if requested
        if self.center_crop:
            transform_list.append(transforms.CenterCrop(self.resolution))
        else:
            transform_list.append(transforms.RandomCrop(self.resolution))
        
        # Random flip if requested
        if self.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize to [-1, 1] (standard for Stable Diffusion)
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single training example
        
        Returns:
            dict with keys:
                - pixel_values: Image tensor [3, resolution, resolution]
                - caption: Text caption string
        """
        item = self.data[idx]
        
        # Load and transform image
        image = Image.open(item['image_path']).convert('RGB')
        pixel_values = self.transforms(image)
        
        return {
            'pixel_values': pixel_values,
            'caption': item['caption']
        }


def collate_fn(examples: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader
    Batches multiple examples together
    
    Args:
        examples: List of examples from __getitem__
        
    Returns:
        Batched dictionary
    """
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    captions = [example['caption'] for example in examples]
    
    return {
        'pixel_values': pixel_values,
        'captions': captions
    }


def create_dataloader(
    data_dir: Path,
    batch_size: int = 1,
    resolution: int = 512,
    center_crop: bool = True,
    random_flip: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for training
    
    Args:
        data_dir: Directory containing images and metadata.jsonl
        batch_size: Batch size
        resolution: Target image resolution
        center_crop: Whether to center crop images
        random_flip: Whether to randomly flip images
        num_workers: Number of data loading workers (0 for Windows)
        pin_memory: Whether to pin memory (False for CPU)
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    dataset = DreamBoothDataset(
        data_dir=data_dir,
        resolution=resolution,
        center_crop=center_crop,
        random_flip=random_flip
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader
