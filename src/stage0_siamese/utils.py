"""
Siamese Network Utilities
Helper functions for training, evaluation, and data management.
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime


def save_training_config(config: Dict, save_path: Path):
    """
    Save training configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    # Create a copy to avoid modifying original
    config_copy = config.copy()
    
    # Add metadata
    config_copy['training_metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'siamese_version': '1.0.0'
    }
    
    with open(save_path, 'w') as f:
        json.dump(config_copy, f, indent=2)
    
    logging.getLogger(__name__).info(f"Saved training configuration to: {save_path}")


def save_training_log(training_history: Dict, total_time: float, save_path: Path):
    """
    Save training history and metrics to JSON file.
    
    Args:
        training_history: Dictionary containing training metrics
        total_time: Total training time in seconds
        save_path: Path to save the log
    """
    log_data = {
        'training_history': training_history,
        'training_metadata': {
            'total_time_seconds': total_time,
            'total_epochs': len(training_history.get('train_loss', [])),
            'best_train_loss': min(training_history.get('train_loss', [float('inf')])),
            'best_val_loss': min(training_history.get('val_loss', [float('inf')])),
            'best_train_accuracy': max(training_history.get('train_accuracy', [0.0])),
            'best_val_accuracy': max(training_history.get('val_accuracy', [0.0])),
            'final_learning_rate': training_history.get('learning_rate', [0.0])[-1] if training_history.get('learning_rate') else 0.0,
            'completed_at': datetime.now().isoformat()
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    logging.getLogger(__name__).info(f"Saved training log to: {save_path}")


def calculate_accuracy(embedding1: torch.Tensor, embedding2: torch.Tensor, 
                      labels: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Calculate accuracy for Siamese network predictions.
    
    Args:
        embedding1: First set of embeddings
        embedding2: Second set of embeddings
        labels: Ground truth labels (1 for same, 0 for different)
        threshold: Distance threshold for classification
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    with torch.no_grad():
        # Calculate Euclidean distances
        distances = F.pairwise_distance(embedding1, embedding2)
        
        # Convert distances to predictions (1 if distance < threshold, 0 otherwise)
        predictions = (distances < threshold).float()
        
        # Calculate accuracy
        correct = (predictions == labels).float()
        accuracy = correct.mean().item()
        
        return accuracy


def calculate_cosine_accuracy(embedding1: torch.Tensor, embedding2: torch.Tensor,
                             labels: torch.Tensor, threshold: float = 0.8) -> float:
    """
    Calculate accuracy using cosine similarity.
    
    Args:
        embedding1: First set of embeddings
        embedding2: Second set of embeddings
        labels: Ground truth labels (1 for same, 0 for different)
        threshold: Cosine similarity threshold for classification
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    with torch.no_grad():
        # Calculate cosine similarity
        cosine_sim = F.cosine_similarity(embedding1, embedding2)
        
        # Convert similarities to predictions (1 if similarity > threshold, 0 otherwise)
        predictions = (cosine_sim > threshold).float()
        
        # Calculate accuracy
        correct = (predictions == labels).float()
        accuracy = correct.mean().item()
        
        return accuracy


def evaluate_embeddings(embeddings: torch.Tensor, labels: torch.Tensor, 
                       metric: str = 'euclidean') -> Dict[str, float]:
    """
    Evaluate embedding quality using various metrics.
    
    Args:
        embeddings: Tensor of embeddings
        labels: Corresponding labels
        metric: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        Dictionary of evaluation metrics
    """
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Calculate intra-class and inter-class distances
    intra_distances = []
    inter_distances = []
    
    unique_labels = np.unique(labels_np)
    
    for label in unique_labels:
        # Get embeddings for this class
        class_embeddings = embeddings_np[labels_np == label]
        
        # Calculate intra-class distances
        if len(class_embeddings) > 1:
            for i in range(len(class_embeddings)):
                for j in range(i + 1, len(class_embeddings)):
                    if metric == 'euclidean':
                        dist = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
                    else:  # cosine
                        dist = 1 - np.dot(class_embeddings[i], class_embeddings[j]) / (
                            np.linalg.norm(class_embeddings[i]) * np.linalg.norm(class_embeddings[j])
                        )
                    intra_distances.append(dist)
        
        # Calculate inter-class distances
        other_embeddings = embeddings_np[labels_np != label]
        for class_emb in class_embeddings:
            for other_emb in other_embeddings:
                if metric == 'euclidean':
                    dist = np.linalg.norm(class_emb - other_emb)
                else:  # cosine
                    dist = 1 - np.dot(class_emb, other_emb) / (
                        np.linalg.norm(class_emb) * np.linalg.norm(other_emb)
                    )
                inter_distances.append(dist)
    
    # Calculate metrics
    metrics = {
        'mean_intra_distance': np.mean(intra_distances) if intra_distances else 0.0,
        'mean_inter_distance': np.mean(inter_distances) if inter_distances else 0.0,
        'std_intra_distance': np.std(intra_distances) if intra_distances else 0.0,
        'std_inter_distance': np.std(inter_distances) if inter_distances else 0.0,
    }
    
    # Calculate separation ratio (higher is better)
    if metrics['mean_intra_distance'] > 0:
        metrics['separation_ratio'] = metrics['mean_inter_distance'] / metrics['mean_intra_distance']
    else:
        metrics['separation_ratio'] = float('inf')
    
    return metrics


def create_embedding_visualization(embeddings: np.ndarray, labels: List[str], 
                                 save_path: Optional[Path] = None) -> Optional[Path]:
    """
    Create 2D visualization of embeddings using t-SNE or PCA.
    
    Args:
        embeddings: Array of embeddings
        labels: List of character labels
        save_path: Optional path to save the plot
        
    Returns:
        Path to saved plot or None if matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import seaborn as sns
    except ImportError:
        logging.getLogger(__name__).warning("Visualization dependencies not available")
        return None
    
    # Use t-SNE for dimensionality reduction
    if embeddings.shape[0] > 50:
        # Use PCA first if too many samples
        pca = PCA(n_components=50)
        embeddings_reduced = pca.fit_transform(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        embeddings_2d = tsne.fit_transform(embeddings_reduced)
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Get unique labels and colors
    unique_labels = list(set(labels))
    colors = sns.color_palette("husl", len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7, s=50)
    
    plt.title("Character Embedding Visualization (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.getLogger(__name__).info(f"Saved embedding visualization to: {save_path}")
        plt.close()
        return save_path
    else:
        plt.show()
        plt.close()
        return None


def validate_model_compatibility(model_path: Path, config: Dict) -> bool:
    """
    Validate that a saved model is compatible with the current configuration.
    
    Args:
        model_path: Path to the saved model
        config: Current configuration
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check if config is stored in checkpoint
        if 'config' not in checkpoint:
            logging.getLogger(__name__).warning("No config found in checkpoint")
            return False
        
        saved_config = checkpoint['config']
        current_siamese = config.get('siamese', {})
        saved_siamese = saved_config.get('siamese', {})
        
        # Check critical parameters
        critical_params = ['embedding_dim', 'backbone']
        for param in critical_params:
            if current_siamese.get(param) != saved_siamese.get(param):
                logging.getLogger(__name__).error(
                    f"Model incompatibility: {param} mismatch "
                    f"(current: {current_siamese.get(param)}, "
                    f"saved: {saved_siamese.get(param)})"
                )
                return False
        
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to validate model compatibility: {e}")
        return False


def get_model_info(model_path: Path) -> Optional[Dict]:
    """
    Extract information from a saved model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Dictionary with model information or None if failed
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
            'model_size_mb': model_path.stat().st_size / (1024 * 1024),
        }
        
        # Extract config info if available
        if 'config' in checkpoint:
            siamese_config = checkpoint['config'].get('siamese', {})
            info.update({
                'embedding_dim': siamese_config.get('embedding_dim', 'unknown'),
                'backbone': siamese_config.get('backbone', 'unknown'),
            })
        
        # Extract training history if available
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            if history.get('val_accuracy'):
                info['final_val_accuracy'] = history['val_accuracy'][-1]
            if history.get('train_accuracy'):
                info['final_train_accuracy'] = history['train_accuracy'][-1]
        
        return info
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to get model info: {e}")
        return None


def cleanup_training_files(output_dir: Path, keep_best: bool = True, keep_latest: bool = False):
    """
    Clean up training files to save disk space.
    
    Args:
        output_dir: Training output directory
        keep_best: Whether to keep the best model weights
        keep_latest: Whether to keep the latest checkpoint
    """
    logger = logging.getLogger(__name__)
    
    files_to_remove = []
    
    # Checkpoint files
    if not keep_latest:
        checkpoint_latest = output_dir / "checkpoint_latest.pth"
        if checkpoint_latest.exists():
            files_to_remove.append(checkpoint_latest)
    
    # Remove intermediate checkpoints (if any)
    for checkpoint_file in output_dir.glob("checkpoint_epoch_*.pth"):
        files_to_remove.append(checkpoint_file)
    
    # Remove best model if requested
    if not keep_best:
        model_weights = output_dir / "model_weights.pth"
        if model_weights.exists():
            files_to_remove.append(model_weights)
    
    # Remove files
    total_size_freed = 0
    for file_path in files_to_remove:
        try:
            size = file_path.stat().st_size
            file_path.unlink()
            total_size_freed += size
            logger.info(f"Removed: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove {file_path}: {e}")
    
    if total_size_freed > 0:
        logger.info(f"Freed {total_size_freed / (1024 * 1024):.2f} MB of disk space")


def format_training_summary(result: Dict) -> str:
    """
    Format training results into a readable summary.
    
    Args:
        result: Training result dictionary
        
    Returns:
        Formatted summary string
    """
    if not result.get('success'):
        return f"Training failed: {result.get('error', 'Unknown error')}"
    
    summary = [
        "=== Siamese Network Training Summary ===",
        f"Status: {'SUCCESS' if result['success'] else 'FAILED'}",
        f"Epochs trained: {result.get('epochs_trained', 'N/A')}",
        f"Best validation loss: {result.get('best_val_loss', 'N/A'):.4f}",
        f"Final train accuracy: {result.get('final_train_accuracy', 'N/A'):.4f}",
        f"Final validation accuracy: {result.get('final_val_accuracy', 'N/A'):.4f}",
        f"Training time: {result.get('training_time', 'N/A'):.2f} seconds",
        f"Model saved to: {result.get('model_path', 'N/A')}",
        f"Embeddings saved to: {result.get('embeddings_path', 'N/A')}",
        "=" * 45
    ]
    
    return "\n".join(summary)


def estimate_training_time(num_characters: int, num_images: int, epochs: int, 
                          batch_size: int, device: str = 'cpu') -> float:
    """
    Estimate training time based on dataset size and hardware.
    
    Args:
        num_characters: Number of characters
        num_images: Total number of images
        epochs: Number of training epochs
        batch_size: Batch size
        device: Training device ('cpu' or 'cuda')
        
    Returns:
        Estimated training time in seconds
    """
    # Base time per batch (rough estimates)
    if device == 'cuda':
        base_time_per_batch = 0.1  # seconds
    else:
        base_time_per_batch = 0.5  # seconds
    
    # Calculate number of batches per epoch
    pairs_per_epoch = max(1000, num_images * 2)  # Rough estimate
    batches_per_epoch = pairs_per_epoch // batch_size
    
    # Total time estimate
    total_batches = batches_per_epoch * epochs
    estimated_time = total_batches * base_time_per_batch
    
    # Add overhead for validation, checkpointing, etc.
    estimated_time *= 1.2
    
    return estimated_time
