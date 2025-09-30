#!/usr/bin/env python3
"""
Dual Detection Runner
Runs both Haar Cascade and YOLOv8 detection methods for comparison
"""

import sys
import yaml
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_detection.detector import CharacterDetector

def run_dual_detection(config_path: str, resume_from: str):
    """Run both Haar and YOLOv8 detection methods"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return False
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        return False
    
    results = {}
    
    # Run Haar Cascade Detection
    logger.info("=" * 60)
    logger.info("RUNNING HAAR CASCADE DETECTION")
    logger.info("=" * 60)
    
    try:
        # Configure for Haar detection
        haar_config = config.copy()
        haar_config['detection']['face_detection_method'] = 'haar'
        haar_config['detection']['save_detected_faces'] = True
        haar_config['detection']['save_detected_frames'] = True
        
        # Run Haar detection
        haar_detector = CharacterDetector(haar_config, resume_from, method_suffix="_haar")
        haar_result = haar_detector.run()
        results['haar'] = haar_result
        
        if haar_result['success']:
            logger.info(f"✅ Haar detection completed: {haar_result['faces_detected']} faces detected")
        else:
            logger.error(f"❌ Haar detection failed: {haar_result['error']}")
            
    except Exception as e:
        logger.error(f"❌ Haar detection failed with exception: {e}")
        results['haar'] = {'success': False, 'error': str(e)}
    
    # Run YOLOv8 Detection
    logger.info("=" * 60)
    logger.info("RUNNING YOLOV8 DETECTION")
    logger.info("=" * 60)
    
    try:
        # Configure for YOLOv8 detection
        yolo_config = config.copy()
        yolo_config['detection']['face_detection_method'] = 'yolo_anime'
        yolo_config['detection']['save_detected_faces'] = True
        yolo_config['detection']['save_detected_frames'] = True
        
        # Run YOLOv8 detection
        yolo_detector = CharacterDetector(yolo_config, resume_from, method_suffix="_yolo")
        yolo_result = yolo_detector.run()
        results['yolo'] = yolo_result
        
        if yolo_result['success']:
            logger.info(f"✅ YOLOv8 detection completed: {yolo_result['faces_detected']} faces detected")
        else:
            logger.error(f"❌ YOLOv8 detection failed: {yolo_result['error']}")
            
    except Exception as e:
        logger.error(f"❌ YOLOv8 detection failed with exception: {e}")
        results['yolo'] = {'success': False, 'error': str(e)}
    
    # Print summary
    logger.info("=" * 60)
    logger.info("DETECTION COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    if results.get('haar', {}).get('success'):
        haar_faces = results['haar']['faces_detected']
        haar_matches = results['haar']['character_matches']
        logger.info(f"📊 Haar Cascade: {haar_faces} faces, {haar_matches} character matches")
    else:
        logger.info("📊 Haar Cascade: FAILED")
    
    if results.get('yolo', {}).get('success'):
        yolo_faces = results['yolo']['faces_detected']
        yolo_matches = results['yolo']['character_matches']
        logger.info(f"📊 YOLOv8:       {yolo_faces} faces, {yolo_matches} character matches")
    else:
        logger.info("📊 YOLOv8:       FAILED")
    
    # Print directory structure
    if any(results.get(method, {}).get('success') for method in ['haar', 'yolo']):
        logger.info("\n📁 Output Directory Structure:")
        detection_dir = Path(resume_from) / "detection"
        
        if results.get('haar', {}).get('success'):
            logger.info("   ├── 1_detected_faces_haar/     # Haar face crops")
            logger.info("   ├── 2_detected_frames_haar/    # Haar frames with bounding boxes")
            logger.info("   ├── detected_haar.json         # Haar detection results")
        
        if results.get('yolo', {}).get('success'):
            logger.info("   ├── 1_detected_faces_yolo/     # YOLOv8 face crops")
            logger.info("   ├── 2_detected_frames_yolo/    # YOLOv8 frames with bounding boxes")
            logger.info("   ├── detected_yolo.json         # YOLOv8 detection results")
        
        logger.info("   └── character_database.pkl      # Shared character database")
    
    # Return success status
    return any(results.get(method, {}).get('success') for method in ['haar', 'yolo'])

def main():
    """Main entry point"""
    if len(sys.argv) != 3:
        print("Usage: python run_dual_detection.py <config_path> <resume_from_dir>")
        print("Example: python run_dual_detection.py config.yaml outputs/run_20250902_165141")
        return 1
    
    config_path = sys.argv[1]
    resume_from = sys.argv[2]
    
    success = run_dual_detection(config_path, resume_from)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
