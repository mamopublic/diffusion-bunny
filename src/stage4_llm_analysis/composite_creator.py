"""
Composite Image Creator for LLM Analysis
Creates composite images with character reference strip and current frame for LLM analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass

@dataclass
class CompositeConfig:
    """Configuration for composite image creation"""
    strip_width: int = 150
    character_image_size: int = 150
    strip_background_color: Tuple[int, int, int] = (240, 240, 240)  # Light gray BGR
    label_font_scale: float = 0.5
    label_font_thickness: int = 1
    label_color: Tuple[int, int, int] = (0, 0, 0)  # Black BGR
    label_padding: int = 5

class CompositeImageCreator:
    """Creates composite images for LLM analysis"""
    
    def __init__(self, strip_data_dir: Path, config: CompositeConfig = None):
        self.strip_data_dir = Path(strip_data_dir)
        self.config = config or CompositeConfig()
        self.logger = logging.getLogger(__name__)
        
        # Load character reference images
        self.character_images = self._load_character_images()
        
        if not self.character_images:
            raise FileNotFoundError(f"No character images found in {strip_data_dir}")
        
        self.logger.info(f"Loaded {len(self.character_images)} character reference images")
    
    def _load_character_images(self) -> Dict[str, np.ndarray]:
        """Load character reference images from strip_data directory"""
        character_images = {}
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for image_file in self.strip_data_dir.iterdir():
            if image_file.suffix.lower() in supported_formats:
                character_name = image_file.stem
                image = cv2.imread(str(image_file))
                
                if image is not None:
                    # Resize to standard size
                    resized_image = cv2.resize(
                        image, 
                        (self.config.character_image_size, self.config.character_image_size),
                        interpolation=cv2.INTER_AREA
                    )
                    character_images[character_name] = resized_image
                    self.logger.debug(f"Loaded character image: {character_name}")
                else:
                    self.logger.warning(f"Could not load character image: {image_file}")
        
        return character_images
    
    def _create_character_strip(self, frame_height: int) -> np.ndarray:
        """Create vertical character reference strip"""
        # Calculate strip height to match frame height
        strip_height = frame_height
        
        # Create background strip
        strip = np.full(
            (strip_height, self.config.strip_width, 3),
            self.config.strip_background_color,
            dtype=np.uint8
        )
        
        # Calculate positions for character images
        num_characters = len(self.character_images)
        if num_characters == 0:
            return strip
        
        # Calculate spacing
        available_height = strip_height - (2 * self.config.label_padding)
        image_height = self.config.character_image_size
        label_height = 20  # Approximate height for text
        
        total_item_height = image_height + label_height + self.config.label_padding
        spacing = max(10, (available_height - (num_characters * total_item_height)) // (num_characters + 1))
        
        # Position character images
        y_offset = spacing
        
        for char_name, char_image in self.character_images.items():
            # Calculate position to center image horizontally
            x_offset = (self.config.strip_width - self.config.character_image_size) // 2
            
            # Check if we have enough space
            if y_offset + image_height + label_height > strip_height - self.config.label_padding:
                self.logger.warning(f"Not enough space for character {char_name} in strip")
                break
            
            # Place character image
            strip[y_offset:y_offset + image_height, x_offset:x_offset + self.config.character_image_size] = char_image
            
            # Add character name label
            label_y = y_offset + image_height + 15
            label_x = self.config.label_padding
            
            # Get text size to center it
            text_size = cv2.getTextSize(
                char_name, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                self.config.label_font_scale, 
                self.config.label_font_thickness
            )[0]
            
            # Center text horizontally
            label_x = (self.config.strip_width - text_size[0]) // 2
            
            cv2.putText(
                strip,
                char_name,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.label_font_scale,
                self.config.label_color,
                self.config.label_font_thickness
            )
            
            y_offset += total_item_height + spacing
        
        return strip
    
    def _draw_face_boxes(self, frame: np.ndarray, face_detections: List[Dict]) -> np.ndarray:
        """Draw colored bounding boxes around detected faces"""
        annotated_frame = frame.copy()
        
        # Define colors for face boxes (BGR format)
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for i, face in enumerate(face_detections):
            bbox = face.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                color = colors[i % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add face number label
                label = f"Face {i+1}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                label_y = max(y1 - 10, label_size[1] + 5)
                
                # Draw background for label
                cv2.rectangle(
                    annotated_frame,
                    (x1, label_y - label_size[1] - 5),
                    (x1 + label_size[0] + 10, label_y + 5),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # White text
                    2
                )
        
        return annotated_frame
    
    def create_composite(self, frame_path: Path, face_detections: List[Dict] = None) -> Optional[np.ndarray]:
        """Create composite image with character strip and annotated frame"""
        try:
            # Load the frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                self.logger.error(f"Could not load frame: {frame_path}")
                return None
            
            # Annotate frame with face boxes if detections provided
            if face_detections:
                frame = self._draw_face_boxes(frame, face_detections)
            
            # Create character strip matching frame height
            character_strip = self._create_character_strip(frame.shape[0])
            
            # Combine strip and frame horizontally
            composite = np.hstack([character_strip, frame])
            
            return composite
            
        except Exception as e:
            self.logger.error(f"Failed to create composite for {frame_path}: {e}")
            return None
    
    def save_composite(self, composite: np.ndarray, output_path: Path) -> bool:
        """Save composite image to file"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(output_path), composite)
            
            if success:
                self.logger.debug(f"Saved composite image: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save composite image: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving composite to {output_path}: {e}")
            return False
    
    def create_and_save_composite(self, frame_path: Path, output_path: Path, 
                                face_detections: List[Dict] = None) -> bool:
        """Create and save composite image in one step"""
        composite = self.create_composite(frame_path, face_detections)
        
        if composite is not None:
            return self.save_composite(composite, output_path)
        
        return False
    
    def get_character_names(self) -> List[str]:
        """Get list of available character names"""
        return list(self.character_images.keys())

def main():
    """Test the composite creator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test composite image creation")
    parser.add_argument("--strip-dir", required=True, help="Path to character strip images")
    parser.add_argument("--frame", required=True, help="Path to test frame")
    parser.add_argument("--output", required=True, help="Output path for composite")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create composite
    creator = CompositeImageCreator(Path(args.strip_dir))
    
    # Test face detections (dummy data)
    test_detections = [
        {"bbox": [100, 100, 200, 200]},
        {"bbox": [300, 150, 400, 250]}
    ]
    
    success = creator.create_and_save_composite(
        Path(args.frame),
        Path(args.output),
        test_detections
    )
    
    if success:
        print(f"Composite created successfully: {args.output}")
    else:
        print("Failed to create composite")

if __name__ == "__main__":
    main()
