"""
Embedding Extractor
Extracts embeddings from trained Siamese network for character recognition.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from torchvision import transforms
import pickle
import json
import logging

from .siamese_network import SiameseNetwork


class EmbeddingExtractor:
    """
    Extracts embeddings from images using a trained Siamese network.
    
    This class handles loading the trained model, preprocessing images,
    and extracting normalized embeddings for character recognition.
    """
    
    def __init__(self, model_path: Path, config: dict, device: str = 'cpu'):
        """
        Initialize the embedding extractor.
        
        Args:
            model_path: Path to the trained Siamese network weights
            config: Configuration dictionary
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        siamese_config = config.get('siamese', {})
        training_config = siamese_config.get('training', {})
        
        self.image_size = training_config.get('image_size', 224)
        self.embedding_dim = siamese_config.get('embedding_dim', 128)
        self.backbone = siamese_config.get('backbone', 'mobilenetv2')
        
        # Initialize model
        self.model = None
        self.transform = None
        
        self._load_model()
        self._create_transform()
        
        self.logger.info(f"Initialized embedding extractor with {self.backbone} backbone")
    
    def _load_model(self):
        """Load the trained Siamese network."""
        try:
            # Create model architecture
            self.model = SiameseNetwork(
                embedding_dim=self.embedding_dim,
                backbone=self.backbone,
                pretrained=False  # We're loading trained weights
            )
            
            # Load trained weights
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.logger.info(f"Loaded trained model from: {self.model_path}")
            else:
                self.logger.warning(f"Model weights not found at: {self.model_path}")
                raise FileNotFoundError(f"Model weights not found: {self.model_path}")
            
            # Set to evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _create_transform(self):
        """Create image preprocessing transform."""
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for embedding extraction.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized embedding as numpy array
        """
        try:
            # Preprocess image
            tensor = self._preprocess_image(image)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.get_embedding(tensor)
            
            # Convert to numpy and remove batch dimension
            embedding_np = embedding.cpu().numpy().squeeze()
            
            return embedding_np
            
        except Exception as e:
            self.logger.error(f"Failed to extract embedding: {e}")
            raise
    
    def extract_embeddings_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from a batch of images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            Batch of normalized embeddings as numpy array
        """
        try:
            # Preprocess all images
            tensors = []
            for image in images:
                tensor = self._preprocess_image(image)
                tensors.append(tensor.squeeze(0))  # Remove batch dim for stacking
            
            # Stack into batch
            batch_tensor = torch.stack(tensors).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = self.model.forward_one(batch_tensor)
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            
            return embeddings_np
            
        except Exception as e:
            self.logger.error(f"Failed to extract batch embeddings: {e}")
            raise
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                          metric: str = 'cosine') -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)
        
        elif metric == 'euclidean':
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            similarity = 1.0 / (1.0 + distance)
            return float(similarity)
        
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def find_best_match(self, query_embedding: np.ndarray, 
                       reference_embeddings: Dict[str, np.ndarray],
                       threshold: float = 0.8) -> Tuple[Optional[str], float]:
        """
        Find the best matching character for a query embedding.
        
        Args:
            query_embedding: Query embedding to match
            reference_embeddings: Dictionary of character name -> embedding
            threshold: Minimum similarity threshold for a match
            
        Returns:
            Tuple of (character_name, similarity_score) or (None, 0.0) if no match
        """
        best_character = None
        best_similarity = 0.0
        
        for character_name, ref_embedding in reference_embeddings.items():
            similarity = self.compute_similarity(query_embedding, ref_embedding)
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_character = character_name
        
        return best_character, best_similarity


class CharacterEmbeddingDatabase:
    """
    Manages character reference embeddings for fast lookup during inference.
    """
    
    def __init__(self, characters_dir: Path, extractor: EmbeddingExtractor):
        """
        Initialize the character embedding database.
        
        Args:
            characters_dir: Directory containing character reference images
            extractor: Embedding extractor instance
        """
        self.characters_dir = characters_dir
        self.extractor = extractor
        self.logger = logging.getLogger(__name__)
        
        self.character_embeddings = {}
        self.character_metadata = {}
        
    def build_database(self) -> Dict[str, np.ndarray]:
        """
        Build the character embedding database from reference images.
        
        Returns:
            Dictionary mapping character names to their embeddings
        """
        self.logger.info("Building character embedding database...")
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        character_images = {}
        
        # Group images by character
        for image_file in self.characters_dir.iterdir():
            if image_file.suffix.lower() in supported_formats:
                character_name = image_file.stem
                
                if character_name not in character_images:
                    character_images[character_name] = []
                
                character_images[character_name].append(image_file)
        
        # Extract embeddings for each character
        for character_name, image_files in character_images.items():
            self.logger.info(f"Processing character: {character_name} ({len(image_files)} images)")
            
            embeddings = []
            
            for image_file in image_files:
                try:
                    # Load image
                    image = cv2.imread(str(image_file))
                    if image is None:
                        self.logger.warning(f"Could not load image: {image_file}")
                        continue
                    
                    # Extract embedding
                    embedding = self.extractor.extract_embedding(image)
                    embeddings.append(embedding)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process {image_file}: {e}")
                    continue
            
            if embeddings:
                # Average embeddings if multiple images
                if len(embeddings) > 1:
                    avg_embedding = np.mean(embeddings, axis=0)
                    # Re-normalize after averaging
                    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                else:
                    avg_embedding = embeddings[0]
                
                self.character_embeddings[character_name] = avg_embedding
                self.character_metadata[character_name] = {
                    'num_images': len(embeddings),
                    'image_files': [str(f) for f in image_files]
                }
                
                self.logger.info(f"Created embedding for {character_name} from {len(embeddings)} images")
            else:
                self.logger.warning(f"No valid embeddings for character: {character_name}")
        
        self.logger.info(f"Built database with {len(self.character_embeddings)} characters")
        return self.character_embeddings
    
    def save_database(self, save_path: Path):
        """
        Save the character embedding database to disk.
        
        Args:
            save_path: Path to save the database
        """
        database_data = {
            'character_embeddings': self.character_embeddings,
            'character_metadata': self.character_metadata,
            'config': {
                'embedding_dim': self.extractor.embedding_dim,
                'backbone': self.extractor.backbone,
                'image_size': self.extractor.image_size
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(database_data, f)
        
        self.logger.info(f"Saved character embedding database to: {save_path}")
    
    def load_database(self, load_path: Path) -> bool:
        """
        Load the character embedding database from disk.
        
        Args:
            load_path: Path to load the database from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(load_path, 'rb') as f:
                database_data = pickle.load(f)
            
            self.character_embeddings = database_data['character_embeddings']
            self.character_metadata = database_data['character_metadata']
            
            # Verify compatibility
            config = database_data.get('config', {})
            if (config.get('embedding_dim') != self.extractor.embedding_dim or
                config.get('backbone') != self.extractor.backbone):
                self.logger.warning("Database config doesn't match current extractor config")
                return False
            
            self.logger.info(f"Loaded character embedding database with {len(self.character_embeddings)} characters")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load character embedding database: {e}")
            return False
    
    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Get the character embeddings dictionary."""
        return self.character_embeddings
    
    def get_metadata(self) -> Dict[str, dict]:
        """Get the character metadata dictionary."""
        return self.character_metadata


def create_embedding_extractor(model_path: Path, config: dict, device: str = 'cpu') -> EmbeddingExtractor:
    """
    Factory function to create an embedding extractor.
    
    Args:
        model_path: Path to the trained model weights
        config: Configuration dictionary
        device: Device to run on
        
    Returns:
        Configured EmbeddingExtractor instance
    """
    return EmbeddingExtractor(model_path, config, device)


def create_character_database(characters_dir: Path, extractor: EmbeddingExtractor) -> CharacterEmbeddingDatabase:
    """
    Factory function to create a character embedding database.
    
    Args:
        characters_dir: Directory containing character reference images
        extractor: Embedding extractor instance
        
    Returns:
        Configured CharacterEmbeddingDatabase instance
    """
    database = CharacterEmbeddingDatabase(characters_dir, extractor)
    database.build_database()
    return database
