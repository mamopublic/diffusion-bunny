"""
Test script for composite image creation
Tests the composite creator with existing detection results.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage4_llm_analysis.composite_creator import CompositeImageCreator, CompositeConfig

def test_composite_creation():
    """Test composite image creation with sample data"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Paths
    strip_data_dir = project_root / "pipeline_data" / "sprite" / "strip_data"
    test_output_dir = project_root / "test_outputs"
    test_output_dir.mkdir(exist_ok=True)
    
    # Check if strip data exists
    if not strip_data_dir.exists():
        logger.error(f"Strip data directory not found: {strip_data_dir}")
        return False
    
    # Find a test frame from detection results
    detection_results_path = project_root / "outputs" / "run_20250902_165141" / "detection" / "detected.json"
    
    if not detection_results_path.exists():
        logger.error(f"Detection results not found: {detection_results_path}")
        return False
    
    import json
    with open(detection_results_path, 'r') as f:
        detection_results = json.load(f)
    
    # Find a frame with faces
    test_frame = None
    for frame_data in detection_results:
        if frame_data.get('faces'):
            test_frame = frame_data
            break
    
    if not test_frame:
        logger.error("No frames with faces found in detection results")
        return False
    
    logger.info(f"Using test frame: {test_frame['frame_id']}")
    
    try:
        # Create composite creator
        config = CompositeConfig(
            strip_width=150,
            character_image_size=150,
            strip_background_color=(240, 240, 240)
        )
        
        creator = CompositeImageCreator(strip_data_dir, config)
        logger.info(f"Available characters: {creator.get_character_names()}")
        
        # Create composite
        frame_path = Path(test_frame['frame_path'])
        if not frame_path.is_absolute():
            frame_path = project_root / frame_path
        
        output_path = test_output_dir / f"test_composite_{test_frame['frame_id']}.jpg"
        
        success = creator.create_and_save_composite(
            frame_path=frame_path,
            output_path=output_path,
            face_detections=test_frame['faces']
        )
        
        if success:
            logger.info(f"✅ Composite created successfully: {output_path}")
            return True
        else:
            logger.error("❌ Failed to create composite")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_composite_creation()
    sys.exit(0 if success else 1)
