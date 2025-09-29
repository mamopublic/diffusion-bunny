"""
Stage 2: Quality Filtering
Filters extracted frames based on blur, brightness, and contrast quality metrics.
"""

import os
import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import argparse
import yaml
from datetime import datetime
from collections import defaultdict

try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

@dataclass
class QualityMetrics:
    """Quality metrics for a frame"""
    blur_score: float
    brightness_score: float
    contrast_score: float
    
    def passes_filters(self, config: Dict) -> bool:
        """Check if frame passes all quality filters"""
        # Blur filter (higher is better)
        if self.blur_score < config['filtering']['blur_threshold']:
            return False
            
        # Brightness filter (within range)
        brightness_min, brightness_max = config['filtering']['brightness_threshold']
        if not (brightness_min <= self.brightness_score <= brightness_max):
            return False
            
        # Contrast filter (higher is better)
        if self.contrast_score < config['filtering']['contrast_threshold']:
            return False
            
        return True

@dataclass
class FilteredFrame:
    """Represents a frame that passed filtering"""
    id: str
    path: str

class FrameFilter:
    """Filters frames based on quality metrics"""
    
    def __init__(self, config: Dict, resume_run_dir: Optional[Path] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.project_root = Path(config['input']['project_root'])
        
        # Filtering settings
        self.blur_threshold = config['filtering']['blur_threshold']
        self.batch_size = config['filtering']['batch_size']
        self.keep_best_per_batch = config['filtering']['keep_best_per_batch']
        self.min_batch_quality = config['filtering']['min_batch_quality']
        self.brightness_threshold = config['filtering']['brightness_threshold']
        self.contrast_threshold = config['filtering']['contrast_threshold']
        
        # Deduplication settings
        dedup_config = config['filtering'].get('deduplication', {})
        self.deduplication_enabled = dedup_config.get('enabled', False)
        self.dedup_method = dedup_config.get('method', 'perceptual_hash')
        self.similarity_threshold = dedup_config.get('similarity_threshold', 0.95)
        self.hash_size = dedup_config.get('hash_size', 8)
        self.keep_best_quality = dedup_config.get('keep_best_quality', True)
        self.min_group_size = dedup_config.get('min_group_size', 2)
        
        # Check if imagehash is available for deduplication
        if self.deduplication_enabled and not IMAGEHASH_AVAILABLE:
            self.logger.warning("imagehash library not available. Install with: pip install imagehash")
            self.logger.warning("Deduplication will be disabled")
            self.deduplication_enabled = False
        
        # Handle run directory
        if resume_run_dir:
            self.run_dir = resume_run_dir
            self.logger.info(f"Using existing run directory: {self.run_dir}")
        else:
            self.run_dir = self._create_run_directory()
            
        self.output_dir = self.run_dir / "filtering"
        self.output_dir.mkdir(exist_ok=True)
        
    def _create_run_directory(self) -> Path:
        """Create a new run directory with timestamp"""
        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = Path("outputs") / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config to run directory
        config_path = run_dir / "config.yaml"
        with open("config.yaml", 'r') as src, open(config_path, 'w') as dst:
            dst.write(src.read())
            
        self.logger.info(f"Created run directory: {run_dir}")
        return run_dir
        
    def load_frame_metadata(self) -> List[Dict]:
        """Load frame metadata from Stage 1"""
        metadata_path = self.project_root / "frames_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Frame metadata not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        frames = metadata.get('frames', [])
        self.logger.info(f"Loaded metadata for {len(frames)} frames")
        return frames
        
    def calculate_blur_score(self, image_path: str) -> float:
        """Calculate Laplacian variance blur score (your original method)"""
        image = cv2.imread(image_path)
        if image is None:
            self.logger.warning(f"Could not load image: {image_path}")
            return 0.0
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
        
    def calculate_brightness_score(self, image_path: str) -> float:
        """Calculate mean brightness score"""
        image = cv2.imread(image_path)
        if image is None:
            return 0.0
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
        
    def calculate_contrast_score(self, image_path: str) -> float:
        """Calculate contrast score using standard deviation"""
        image = cv2.imread(image_path)
        if image is None:
            return 0.0
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray) / 255.0  # Normalize to 0-1
        
    def calculate_quality_metrics(self, image_path: str) -> QualityMetrics:
        """Calculate all quality metrics for a frame"""
        return QualityMetrics(
            blur_score=self.calculate_blur_score(image_path),
            brightness_score=self.calculate_brightness_score(image_path),
            contrast_score=self.calculate_contrast_score(image_path)
        )
        
    def is_blurry(self, image_path: str, threshold: float) -> Tuple[bool, float]:
        """Check if image is blurry using your original method"""
        score = self.calculate_blur_score(image_path)
        return score < threshold, score
        
    def process_batch(self, batch_frames: List[Dict]) -> List[FilteredFrame]:
        """Process a batch of frames and return filtered results"""
        batch_results = []
        
        for frame_data in batch_frames:
            # Construct full path to frame
            frame_path = self.project_root / frame_data['file_path']
                
            if not frame_path.exists():
                self.logger.warning(f"Frame file not found: {frame_path}")
                continue
                
            # Calculate quality metrics
            metrics = self.calculate_quality_metrics(str(frame_path))
            
            # Check if frame passes all filters
            if metrics.passes_filters(self.config):
                filtered_frame = FilteredFrame(
                    id=frame_data['frame_id'],
                    path=frame_data['file_path']
                )
                batch_results.append((filtered_frame, metrics))
                
        # If keep_best_per_batch is enabled, select only the best frame
        if self.keep_best_per_batch and batch_results:
            # Sort by blur score (higher is better) as primary metric
            batch_results.sort(key=lambda x: x[1].blur_score, reverse=True)
            
            # Check if best frame meets minimum quality threshold
            best_frame, best_metrics = batch_results[0]
            if best_metrics.blur_score >= self.min_batch_quality:
                return [best_frame]
            else:
                self.logger.debug(f"Best frame in batch below minimum quality: {best_metrics.blur_score}")
                return []
        else:
            # Return all frames that passed filters
            return [frame for frame, _ in batch_results]
            
    def calculate_perceptual_hash(self, image_path: str) -> Optional[str]:
        """Calculate perceptual hash for an image"""
        if not IMAGEHASH_AVAILABLE:
            return None
            
        try:
            # Load image using PIL
            pil_image = Image.open(image_path)
            
            # Calculate perceptual hash
            if self.hash_size == 16:
                hash_value = imagehash.phash(pil_image, hash_size=16)
            else:
                hash_value = imagehash.phash(pil_image, hash_size=8)
                
            return str(hash_value)
        except Exception as e:
            self.logger.warning(f"Failed to calculate hash for {image_path}: {e}")
            return None
    
    def calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two perceptual hashes"""
        if not hash1 or not hash2:
            return 0.0
            
        try:
            # Convert string hashes back to imagehash objects
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            
            # Calculate Hamming distance
            hamming_distance = h1 - h2
            
            # Convert to similarity (0.0 = identical, 1.0 = completely different)
            max_distance = len(hash1) * 4  # Each hex char represents 4 bits
            similarity = 1.0 - (hamming_distance / max_distance)
            
            return similarity
        except Exception as e:
            self.logger.warning(f"Failed to calculate hash similarity: {e}")
            return 0.0
    
    def deduplicate_frames(self, filtered_frames: List[FilteredFrame]) -> List[FilteredFrame]:
        """Remove near-duplicate frames using perceptual hashing"""
        if not self.deduplication_enabled or len(filtered_frames) < self.min_group_size:
            return filtered_frames
            
        self.logger.info(f"Starting deduplication of {len(filtered_frames)} frames...")
        
        # Calculate hashes for all frames
        frame_hashes = {}
        frames_with_quality = {}
        
        for frame in filtered_frames:
            frame_path = self.project_root / frame.path
            
            # Calculate hash
            hash_value = self.calculate_perceptual_hash(str(frame_path))
            if hash_value:
                frame_hashes[frame.id] = hash_value
                
                # Calculate quality metrics for ranking
                if self.keep_best_quality:
                    quality_metrics = self.calculate_quality_metrics(str(frame_path))
                    frames_with_quality[frame.id] = quality_metrics
        
        # Group similar frames
        similarity_groups = []
        processed_frames = set()
        
        for frame_id, hash_value in frame_hashes.items():
            if frame_id in processed_frames:
                continue
                
            # Find all similar frames
            similar_group = [frame_id]
            processed_frames.add(frame_id)
            
            for other_frame_id, other_hash in frame_hashes.items():
                if other_frame_id in processed_frames:
                    continue
                    
                similarity = self.calculate_hash_similarity(hash_value, other_hash)
                
                if similarity >= self.similarity_threshold:
                    similar_group.append(other_frame_id)
                    processed_frames.add(other_frame_id)
            
            similarity_groups.append(similar_group)
        
        # Select best frame from each group
        deduplicated_frames = []
        frames_dict = {frame.id: frame for frame in filtered_frames}
        
        removed_count = 0
        
        for group in similarity_groups:
            if len(group) >= self.min_group_size:
                # Multiple similar frames - select the best one
                if self.keep_best_quality and all(fid in frames_with_quality for fid in group):
                    # Sort by quality (blur score as primary metric)
                    best_frame_id = max(group, key=lambda fid: frames_with_quality[fid].blur_score)
                else:
                    # Just keep the first one
                    best_frame_id = group[0]
                
                deduplicated_frames.append(frames_dict[best_frame_id])
                removed_count += len(group) - 1
                
                self.logger.debug(f"Group of {len(group)} similar frames, kept: {best_frame_id}")
            else:
                # Single frame or small group - keep all
                for frame_id in group:
                    deduplicated_frames.append(frames_dict[frame_id])
        
        self.logger.info(f"Deduplication complete: removed {removed_count} duplicate frames, kept {len(deduplicated_frames)} unique frames")
        return deduplicated_frames
    
    def filter_frames(self) -> List[FilteredFrame]:
        """Filter all frames and return results"""
        # Load frame metadata
        frames_metadata = self.load_frame_metadata()
        
        if not frames_metadata:
            self.logger.warning("No frames to filter")
            return []
            
        # Process frames in batches
        filtered_frames = []
        total_batches = (len(frames_metadata) + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"Processing {len(frames_metadata)} frames in {total_batches} batches of {self.batch_size}")
        
        for i in range(0, len(frames_metadata), self.batch_size):
            batch = frames_metadata[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            self.logger.debug(f"Processing batch {batch_num}/{total_batches}")
            
            batch_results = self.process_batch(batch)
            filtered_frames.extend(batch_results)
            
            if batch_num % 10 == 0:
                self.logger.info(f"Processed {batch_num}/{total_batches} batches, kept {len(filtered_frames)} frames so far")
                
        self.logger.info(f"Filtering complete: kept {len(filtered_frames)} out of {len(frames_metadata)} frames ({len(filtered_frames)/len(frames_metadata)*100:.1f}%)")
        
        # Apply deduplication if enabled
        if self.deduplication_enabled:
            filtered_frames = self.deduplicate_frames(filtered_frames)
        
        return filtered_frames
        
    def save_filtered_results(self, filtered_frames: List[FilteredFrame]) -> Path:
        """Save filtered frame results to JSON file"""
        output_path = self.output_dir / "filtered.json"
        
        # Convert to the required format
        output_data = [
            {
                "id": frame.id,
                "path": frame.path
            }
            for frame in filtered_frames
        ]
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        self.logger.info(f"Filtered results saved to: {output_path}")
        return output_path
        
    def copy_filtered_frames(self, filtered_frames: List[FilteredFrame]) -> Path:
        """Copy filtered frame files to run directory for visual inspection"""
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Copying {len(filtered_frames)} filtered frames to: {frames_dir}")
        
        copied_count = 0
        for frame in filtered_frames:
            # Source path
            source_path = self.project_root / frame.path
            
            # Destination path (maintain original filename)
            dest_path = frames_dir / Path(frame.path).name
            
            try:
                if source_path.exists():
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                else:
                    self.logger.warning(f"Source frame not found: {source_path}")
            except Exception as e:
                self.logger.error(f"Failed to copy frame {frame.id}: {str(e)}")
                
        self.logger.info(f"Successfully copied {copied_count}/{len(filtered_frames)} frames to {frames_dir}")
        return frames_dir
        
    def run(self) -> Dict:
        """Run the filtering stage"""
        try:
            self.logger.info("Starting frame filtering...")
            
            # Filter frames
            filtered_frames = self.filter_frames()
            
            if not filtered_frames:
                return {
                    "success": False, 
                    "error": "No frames passed filtering",
                    "run_directory": str(self.run_dir)
                }
                
            # Save results
            output_path = self.save_filtered_results(filtered_frames)
            
            # Copy filtered frames for visual inspection
            frames_dir = self.copy_filtered_frames(filtered_frames)
            
            result = {
                "success": True,
                "frames_kept": len(filtered_frames),
                "output_path": str(output_path),
                "frames_directory": str(frames_dir),
                "run_directory": str(self.run_dir),
                "filtering_directory": str(self.output_dir)
            }
            
            self.logger.info(f"Frame filtering completed: {len(filtered_frames)} frames kept")
            return result
            
        except Exception as e:
            self.logger.error(f"Frame filtering failed: {str(e)}")
            return {
                "success": False, 
                "error": str(e),
                "run_directory": str(self.run_dir) if hasattr(self, 'run_dir') else None
            }

def main():
    """Main entry point for filtering stage"""
    parser = argparse.ArgumentParser(description="Filter extracted frames based on quality metrics")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
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
        print(f"Configuration file not found: {args.config}")
        return 1
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return 1
        
    # Run filtering
    filter_stage = FrameFilter(config)
    result = filter_stage.run()
    
    if result["success"]:
        print(f"Filtering completed successfully!")
        print(f"Frames kept: {result['frames_kept']}")
        print(f"Output: {result['output_path']}")
        print(f"Frames directory: {result['frames_directory']}")
        print(f"Run directory: {result['run_directory']}")
        return 0
    else:
        print(f"Filtering failed: {result['error']}")
        if result.get('run_directory'):
            print(f"Run directory: {result['run_directory']}")
        return 1

if __name__ == "__main__":
    exit(main())
