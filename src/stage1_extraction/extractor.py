"""
Stage 1: Frame Extraction
Extracts frames from video files using OpenCV with keyframe detection or interval sampling.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class FrameMetadata:
    """Metadata for an extracted frame"""
    frame_id: str
    timestamp: float
    frame_number: int
    source_video: str
    file_path: str
    width: int
    height: int
    extraction_method: str

class FrameExtractor:
    """Extracts frames from video files"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.project_root = Path(config['input']['project_root'])
        self.movie_dir = config['input']['movie_dir']
        self.frames_dir = config['input']['frames_dir']
        
        # Extraction settings
        self.method = config['extraction']['method']
        self.target_frames = config['extraction']['target_frames']
        self.interval_seconds = config['extraction']['interval_seconds']
        self.keyframe_threshold = config['extraction']['keyframe_threshold']
        self.output_format = config['extraction']['output_format']
        self.quality = config['extraction']['quality']
        self.max_frames = config['extraction']['max_frames']
        self.resize_width = config['extraction']['resize_width']
        self.resize_height = config['extraction']['resize_height']
        
        # Supported formats
        self.supported_formats = config['input']['supported_video_formats']
        
    def find_video_file(self) -> Optional[Path]:
        """Find the first video file in the movie directory"""
        movie_path = self.project_root / self.movie_dir
        
        if not movie_path.exists():
            self.logger.error(f"Movie directory not found: {movie_path}")
            return None
            
        for file_path in movie_path.iterdir():
            if file_path.suffix.lower() in self.supported_formats:
                self.logger.info(f"Found video file: {file_path}")
                return file_path
                
        self.logger.error(f"No supported video files found in {movie_path}")
        return None
        
    def setup_output_directory(self) -> Path:
        """Create and return the frames output directory"""
        frames_path = self.project_root / self.frames_dir
        frames_path.mkdir(exist_ok=True)
        self.logger.info(f"Frames will be saved to: {frames_path}")
        return frames_path
        
    def detect_keyframes(self, video_path: Path) -> List[int]:
        """Detect keyframes using frame difference analysis"""
        cap = cv2.VideoCapture(str(video_path))
        keyframes = []
        prev_frame = None
        frame_count = 0
        
        self.logger.info("Detecting keyframes...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                diff_score = np.mean(diff) / 255.0
                
                # If difference exceeds threshold, it's a keyframe
                if diff_score > self.keyframe_threshold:
                    keyframes.append(frame_count)
                    self.logger.debug(f"Keyframe detected at frame {frame_count}, diff: {diff_score:.3f}")
                    
            prev_frame = gray
            frame_count += 1
            
            # Limit frames if specified
            if self.max_frames and len(keyframes) >= self.max_frames:
                break
                
        cap.release()
        self.logger.info(f"Detected {len(keyframes)} keyframes from {frame_count} total frames")
        return keyframes
        
    def get_interval_frames(self, video_path: Path) -> List[int]:
        """Get frames at regular intervals"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps <= 0:
            self.logger.warning("Could not determine FPS, using default of 30")
            fps = 30
            
        interval_frames = int(fps * self.interval_seconds)
        frame_numbers = list(range(0, total_frames, interval_frames))
        
        # Limit frames if specified
        if self.max_frames:
            frame_numbers = frame_numbers[:self.max_frames]
            
        self.logger.info(f"Selected {len(frame_numbers)} frames at {self.interval_seconds}s intervals (FPS: {fps})")
        return frame_numbers
        
    def get_uniform_frames(self, video_path: Path) -> List[int]:
        """Get frames uniformly distributed across the video"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps <= 0:
            fps = 30  # Default fallback
            
        # Calculate how many frames to actually extract
        target_frames = self.target_frames
        if self.max_frames and self.max_frames < target_frames:
            target_frames = self.max_frames
            
        if target_frames >= total_frames:
            # If we want more frames than available, take all frames
            frame_numbers = list(range(total_frames))
            self.logger.warning(f"Requested {target_frames} frames but video only has {total_frames}, taking all")
        else:
            # Calculate uniform spacing
            step = total_frames / target_frames
            frame_numbers = [int(i * step) for i in range(target_frames)]
            
        duration = total_frames / fps
        self.logger.info(f"Selected {len(frame_numbers)} frames uniformly from {total_frames} total frames ({duration:.1f}s video)")
        return frame_numbers
        
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if dimensions are specified"""
        if self.resize_width is None and self.resize_height is None:
            return frame
            
        height, width = frame.shape[:2]
        
        if self.resize_width and self.resize_height:
            new_size = (self.resize_width, self.resize_height)
        elif self.resize_width:
            ratio = self.resize_width / width
            new_size = (self.resize_width, int(height * ratio))
        else:  # resize_height only
            ratio = self.resize_height / height
            new_size = (int(width * ratio), self.resize_height)
            
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        
    def extract_frames(self) -> List[FrameMetadata]:
        """Extract frames from video and save with metadata"""
        # Find video file
        video_path = self.find_video_file()
        if not video_path:
            raise FileNotFoundError("No video file found")
            
        # Setup output directory
        frames_path = self.setup_output_directory()
        
        # Get frame numbers to extract
        if self.method == "keyframe":
            frame_numbers = self.detect_keyframes(video_path)
        elif self.method == "interval":
            frame_numbers = self.get_interval_frames(video_path)
        elif self.method == "uniform":
            frame_numbers = self.get_uniform_frames(video_path)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
            
        if not frame_numbers:
            self.logger.warning("No frames selected for extraction")
            return []
            
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default fallback
            
        metadata_list = []
        extracted_count = 0
        
        self.logger.info(f"Extracting {len(frame_numbers)} frames...")
        
        for i, frame_number in enumerate(frame_numbers):
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                self.logger.warning(f"Could not read frame {frame_number}")
                continue
                
            # Resize if needed
            frame = self.resize_frame(frame)
            
            # Generate frame ID and filename
            frame_id = f"frame_{frame_number:06d}"
            filename = f"{frame_id}.{self.output_format}"
            frame_path = frames_path / filename
            
            # Save frame
            if self.output_format.lower() == 'jpg':
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            else:
                cv2.imwrite(str(frame_path), frame)
                
            # Create metadata
            timestamp = frame_number / fps
            height, width = frame.shape[:2]
            
            metadata = FrameMetadata(
                frame_id=frame_id,
                timestamp=timestamp,
                frame_number=frame_number,
                source_video=str(video_path.name),
                file_path=str(frame_path.relative_to(self.project_root)),
                width=width,
                height=height,
                extraction_method=self.method
            )
            
            metadata_list.append(metadata)
            extracted_count += 1
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"Extracted {i + 1}/{len(frame_numbers)} frames")
                
        cap.release()
        
        self.logger.info(f"Successfully extracted {extracted_count} frames")
        return metadata_list
        
    def save_metadata(self, metadata_list: List[FrameMetadata]) -> Path:
        """Save frame metadata to JSON file"""
        metadata_path = self.project_root / "frames_metadata.json"
        
        # Convert to serializable format
        metadata_dict = {
            "extraction_info": {
                "method": self.method,
                "total_frames": len(metadata_list),
                "source_video": metadata_list[0].source_video if metadata_list else None,
                "extraction_settings": {
                    "target_frames": self.target_frames,
                    "interval_seconds": self.interval_seconds,
                    "keyframe_threshold": self.keyframe_threshold,
                    "output_format": self.output_format,
                    "quality": self.quality,
                    "max_frames": self.max_frames,
                    "resize_width": self.resize_width,
                    "resize_height": self.resize_height
                }
            },
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp,
                    "frame_number": frame.frame_number,
                    "source_video": frame.source_video,
                    "file_path": frame.file_path,
                    "width": frame.width,
                    "height": frame.height,
                    "extraction_method": frame.extraction_method
                }
                for frame in metadata_list
            ]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
            
        self.logger.info(f"Metadata saved to: {metadata_path}")
        return metadata_path
        
    def run(self) -> Dict:
        """Run the frame extraction stage"""
        try:
            self.logger.info("Starting frame extraction...")
            
            # Extract frames
            metadata_list = self.extract_frames()
            
            if not metadata_list:
                return {"success": False, "error": "No frames extracted"}
                
            # Save metadata
            metadata_path = self.save_metadata(metadata_list)
            
            result = {
                "success": True,
                "frames_extracted": len(metadata_list),
                "metadata_path": str(metadata_path),
                "frames_directory": str(self.project_root / self.frames_dir)
            }
            
            self.logger.info(f"Frame extraction completed: {len(metadata_list)} frames")
            return result
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {str(e)}")
            return {"success": False, "error": str(e)}

def main():
    """Test the frame extractor"""
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run extraction
    extractor = FrameExtractor(config)
    result = extractor.run()
    
    print(f"Extraction result: {result}")

if __name__ == "__main__":
    main()
