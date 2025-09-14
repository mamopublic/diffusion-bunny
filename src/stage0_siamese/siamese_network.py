"""
Siamese Network Architecture
Implements a Siamese network with MobileNetV2 backbone for character recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional
import logging


class SiameseNetwork(nn.Module):
    """
    Siamese network for character recognition using MobileNetV2 backbone.
    
    The network takes two images as input and outputs embeddings that can be
    compared using cosine similarity or Euclidean distance.
    """
    
    def __init__(self, embedding_dim: int = 128, backbone: str = "mobilenetv2", pretrained: bool = True):
        """
        Initialize the Siamese network.
        
        Args:
            embedding_dim: Dimension of the output embeddings
            backbone: Backbone architecture ('mobilenetv2' or 'efficientnet-b0')
            pretrained: Whether to use pretrained weights
        """
        super(SiameseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone
        self.logger = logging.getLogger(__name__)
        
        # Create backbone network
        self.backbone = self._create_backbone(backbone, pretrained)
        
        # Get the number of features from backbone
        backbone_features = self._get_backbone_features(backbone)
        
        # Embedding layers
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger.info(f"Initialized Siamese network with {backbone} backbone, embedding_dim={embedding_dim}")
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Create the backbone network."""
        if backbone == "mobilenetv2":
            model = models.mobilenet_v2(pretrained=pretrained)
            # Remove the classifier layer
            backbone = nn.Sequential(*list(model.children())[:-1])
        elif backbone == "efficientnet-b0":
            try:
                model = models.efficientnet_b0(pretrained=pretrained)
                # Remove the classifier layer
                backbone = nn.Sequential(*list(model.children())[:-1])
            except AttributeError:
                self.logger.warning("EfficientNet not available, falling back to MobileNetV2")
                model = models.mobilenet_v2(pretrained=pretrained)
                backbone = nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return backbone
    
    def _get_backbone_features(self, backbone: str) -> int:
        """Get the number of output features from the backbone."""
        if backbone == "mobilenetv2":
            return 1280
        elif backbone == "efficientnet-b0":
            return 1280  # EfficientNet-B0 also outputs 1280 features
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def _initialize_weights(self):
        """Initialize the weights of the embedding head."""
        for m in self.embedding_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one image.
        
        Args:
            x: Input image tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        
        # Generate embedding
        embedding = self.embedding_head(features)
        
        # L2 normalize the embedding
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Siamese network.
        
        Args:
            x1: First image tensor of shape (batch_size, 3, height, width)
            x2: Second image tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Tuple of (embedding1, embedding2)
        """
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        
        return embedding1, embedding2
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for a single image (inference mode).
        
        Args:
            x: Input image tensor
            
        Returns:
            Normalized embedding tensor
        """
        with torch.no_grad():
            return self.forward_one(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.
    
    The loss encourages similar pairs to have small distances and
    dissimilar pairs to have large distances (at least margin).
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            label: Binary labels (1 for same character, 0 for different)
            
        Returns:
            Contrastive loss value
        """
        # Compute Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2)
        
        # Contrastive loss
        loss_positive = label * torch.pow(distance, 2)
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(loss_positive + loss_negative)
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for Siamese networks.
    
    Alternative to contrastive loss that uses anchor, positive, and negative samples.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin between positive and negative distances
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embedding
            positive: Positive embedding (same character as anchor)
            negative: Negative embedding (different character)
            
        Returns:
            Triplet loss value
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        
        loss = torch.mean(torch.clamp(distance_positive - distance_negative + self.margin, min=0.0))
        
        return loss


def create_siamese_network(config: dict) -> SiameseNetwork:
    """
    Factory function to create a Siamese network from configuration.
    
    Args:
        config: Configuration dictionary containing siamese settings
        
    Returns:
        Configured SiameseNetwork instance
    """
    siamese_config = config.get('siamese', {})
    
    embedding_dim = siamese_config.get('embedding_dim', 128)
    backbone = siamese_config.get('backbone', 'mobilenetv2')
    
    return SiameseNetwork(
        embedding_dim=embedding_dim,
        backbone=backbone,
        pretrained=True
    )


def create_loss_function(config: dict) -> nn.Module:
    """
    Factory function to create loss function from configuration.
    
    Args:
        config: Configuration dictionary containing siamese settings
        
    Returns:
        Configured loss function
    """
    siamese_config = config.get('siamese', {})
    training_config = siamese_config.get('training', {})
    
    loss_type = training_config.get('loss_type', 'contrastive')
    margin = training_config.get('margin', 1.0)
    
    if loss_type == 'contrastive':
        return ContrastiveLoss(margin=margin)
    elif loss_type == 'triplet':
        return TripletLoss(margin=margin)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
