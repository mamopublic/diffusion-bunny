"""
Stage 3: Character Detection
Detects faces and recognizes characters using Haar Cascade and feature matching.
"""

import os
import json
import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import argparse
import yaml
from datetime import datetime

try:
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

@dataclass
class DetectedFace:
    """Represents a detected face in a frame"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    face_image: np.ndarray

@dataclass
class CharacterMatch:
    """Represents a character recognition match"""
    character_name: str
    confidence: float
    match_count: int  # Number of feature matches

@dataclass
class FrameDetection:
    """Represents detection results for a single frame"""
    frame_id: str
    frame_path: str
    faces: List[DetectedFace]
    character_matches: List[CharacterMatch]

class CharacterDatabase:
    """Manages character reference images and feature extraction"""
    
    def __init__(self, characters_dir: Path, feature_method: str = "orb"):
        self.characters_dir = characters_dir
        self.feature_method = feature_method.lower()
        self.characters = {}
        self.feature_detector = self._create_feature_detector()
        self.logger = logging.getLogger(__name__)
        
    def _create_feature_detector(self):
        """Create feature detector based on method"""
        if self.feature_method == "orb":
            return cv2.ORB_create(nfeatures=1000)
        elif self.feature_method == "sift":
            return cv2.SIFT_create()
        elif self.feature_method == "surf":
            return cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError(f"Unsupported feature method: {self.feature_method}")
    
    def load_characters(self) -> Dict[str, Any]:
        """Load character reference images and extract features"""
        self.logger.info(f"Loading character database from: {self.characters_dir}")
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        character_files = []
        
        for ext in supported_formats:
            character_files.extend(self.characters_dir.glob(f"*{ext}"))
            character_files.extend(self.characters_dir.glob(f"*{ext.upper()}"))
        
        if not character_files:
            raise FileNotFoundError(f"No character reference images found in {self.characters_dir}")
        
        for char_file in character_files:
            character_name = char_file.stem
            self.logger.info(f"Processing character: {character_name}")
            
            # Load image
            image = cv2.imread(str(char_file))
            if image is None:
                self.logger.warning(f"Could not load character image: {char_file}")
                continue
            
            # Convert to grayscale for feature extraction
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract features
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            if descriptors is not None:
                self.characters[character_name] = {
                    'image_path': str(char_file),
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'image_shape': gray.shape
                }
                self.logger.info(f"Extracted {len(keypoints)} features for {character_name}")
            else:
                self.logger.warning(f"No features found for character: {character_name}")
        
        self.logger.info(f"Loaded {len(self.characters)} characters into database")
        return self.characters
    
    def save_cache(self, cache_path: Path):
        """Save character database to cache file"""
        cache_data = {}
        for name, data in self.characters.items():
            # Convert keypoints to serializable format
            kp_data = [(kp.pt, kp.angle, kp.size, kp.octave, kp.class_id) for kp in data['keypoints']]
            cache_data[name] = {
                'image_path': data['image_path'],
                'keypoints_data': kp_data,
                'descriptors': data['descriptors'],
                'image_shape': data['image_shape']
            }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        self.logger.info(f"Character database cached to: {cache_path}")
    
    def load_cache(self, cache_path: Path) -> bool:
        """Load character database from cache file"""
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.characters = {}
            for name, data in cache_data.items():
                # Reconstruct keypoints from serialized data
                keypoints = []
                for pt, angle, size, octave, class_id in data['keypoints_data']:
                    kp = cv2.KeyPoint(pt[0], pt[1], size, angle, 0, octave, class_id)
                    keypoints.append(kp)
                
                self.characters[name] = {
                    'image_path': data['image_path'],
                    'keypoints': keypoints,
                    'descriptors': data['descriptors'],
                    'image_shape': data['image_shape']
                }
            
            self.logger.info(f"Loaded character database from cache: {len(self.characters)} characters")
            return True
        except Exception as e:
            self.logger.warning(f"Could not load character database cache: {e}")
            return False

class HaarFaceDetector:
    """Haar Cascade-based face detector"""
    
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.logger = logging.getLogger(__name__)
        
        # Load Haar cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
        
        self.logger.info("Initialized Haar cascade face detector")
    
    def detect_faces(self, image: np.ndarray, min_face_size: int = 50) -> List[DetectedFace]:
        """Detect faces in image using Haar cascades"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use the larger of min_face_size and configured min_size
        effective_min_size = (
            max(min_face_size, self.min_size[0]),
            max(min_face_size, self.min_size[1])
        )
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=effective_min_size
        )
        
        detected_faces = []
        for (x, y, w, h) in faces:
            # Convert (x, y, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Extract face region
            face_image = image[y1:y2, x1:x2]
            
            # Haar doesn't provide confidence, so use a fixed value
            confidence = 0.8
            
            detected_face = DetectedFace(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                face_image=face_image
            )
            detected_faces.append(detected_face)
        
        return detected_faces

class YOLOv8AnimeFaceDetector:
    """YOLOv8-based anime face detector using Hugging Face model"""
    
    def __init__(self, model_repo: str = "Fuyucch1/yolov8_animeface", 
                 model_file: str = "yolov8x6_animeface.pt",
                 confidence_threshold: float = 0.5,
                 device: str = "auto",
                 cache_model: bool = True):
        self.model_repo = model_repo
        self.model_file = model_file
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.cache_model = cache_model
        self.logger = logging.getLogger(__name__)
        self.model = None
        
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 dependencies not available. Install with: pip install ultralytics huggingface_hub torch")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the YOLOv8 model"""
        try:
            self.logger.info(f"Downloading YOLOv8 anime face model from {self.model_repo}")
            
            # Download model from Hugging Face Hub
            model_path = hf_hub_download(
                repo_id=self.model_repo,
                filename=self.model_file,
                cache_dir=".cache/huggingface" if self.cache_model else None
            )
            
            self.logger.info(f"Model downloaded to: {model_path}")
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                    self.logger.info("CUDA available, using GPU for inference")
                else:
                    device = "cpu"
                    self.logger.info("CUDA not available, using CPU for inference")
            else:
                device = self.device
                self.logger.info(f"Using specified device: {device}")
            
            # Load the model
            self.model = YOLO(model_path)
            self.model.to(device)
            
            self.logger.info("YOLOv8 anime face detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLOv8 model: {e}")
            raise RuntimeError(f"YOLOv8 model initialization failed: {e}")
    
    def detect_faces(self, image: np.ndarray, min_face_size: int = 50) -> List[DetectedFace]:
        """Detect faces in image using YOLOv8"""
        if self.model is None:
            raise RuntimeError("YOLOv8 model not initialized")
        
        try:
            # Run inference
            results = self.model(image, verbose=False)
            
            detected_faces = []
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get bounding box coordinates (xyxy format)
                        coords = box.xyxy.cpu().numpy().flatten()
                        x1, y1, x2, y2 = coords.astype(int)
                        
                        # Get confidence score
                        confidence = float(box.conf.cpu().numpy()[0])
                        
                        # Filter by confidence threshold
                        if confidence < self.confidence_threshold:
                            continue
                        
                        # Filter by minimum face size
                        face_width = x2 - x1
                        face_height = y2 - y1
                        if face_width < min_face_size or face_height < min_face_size:
                            continue
                        
                        # Ensure coordinates are within image bounds
                        h, w = image.shape[:2]
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(x1+1, min(x2, w))
                        y2 = max(y1+1, min(y2, h))
                        
                        # Extract face region
                        face_image = image[y1:y2, x1:x2]
                        
                        # Skip if face region is empty
                        if face_image.size == 0:
                            continue
                        
                        detected_face = DetectedFace(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            face_image=face_image
                        )
                        detected_faces.append(detected_face)
            
            return detected_faces
            
        except Exception as e:
            self.logger.error(f"YOLOv8 face detection failed: {e}")
            return []

class CharacterRecognizer:
    """Feature-based character recognition"""
    
    def __init__(self, character_database: CharacterDatabase, similarity_threshold: float = 0.6):
        self.character_database = character_database
        self.similarity_threshold = similarity_threshold
        self.feature_detector = character_database.feature_detector
        self.logger = logging.getLogger(__name__)
        
        # Create feature matcher
        if character_database.feature_method == "orb":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def recognize_character(self, face_image: np.ndarray) -> List[CharacterMatch]:
        """Recognize character in face image"""
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        
        # Extract features from face
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray_face, None)
        
        if descriptors is None:
            return []
        
        character_matches = []
        
        # Compare against each character in database
        for char_name, char_data in self.character_database.characters.items():
            char_descriptors = char_data['descriptors']
            
            # Match features
            matches = self.matcher.match(descriptors, char_descriptors)
            
            if len(matches) > 0:
                # Sort matches by distance (lower is better)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Calculate confidence based on good matches - stricter criteria
                good_matches = [m for m in matches if m.distance < 35]  # Reduced threshold for better precision
                
                if len(good_matches) > 10:  # Increased minimum matches required
                    # Calculate confidence score with improved formula
                    avg_distance = np.mean([m.distance for m in good_matches])
                    confidence = max(0, 1.0 - (avg_distance / 80.0))  # More sensitive normalization
                    
                    if confidence >= self.similarity_threshold:
                        character_match = CharacterMatch(
                            character_name=char_name,
                            confidence=confidence,
                            match_count=len(good_matches)
                        )
                        character_matches.append(character_match)
        
        # Sort by confidence (highest first)
        character_matches.sort(key=lambda x: x.confidence, reverse=True)
        
        return character_matches

def create_face_detector(detection_config: Dict) -> Union[HaarFaceDetector, YOLOv8AnimeFaceDetector]:
    """Factory function to create face detector based on configuration"""
    method = detection_config.get('face_detection_method', 'haar')
    logger = logging.getLogger(__name__)
    
    if method == 'haar':
        logger.info("Creating Haar cascade face detector")
        return HaarFaceDetector(
            scale_factor=detection_config.get('haar_scale_factor', 1.1),
            min_neighbors=detection_config.get('haar_min_neighbors', 5),
            min_size=tuple(detection_config.get('haar_min_size', [30, 30]))
        )
    elif method == 'yolo_anime':
        logger.info("Creating YOLOv8 anime face detector")
        yolo_config = detection_config.get('yolo_anime', {})
        return YOLOv8AnimeFaceDetector(
            model_repo=yolo_config.get('model_repo', 'Fuyucch1/yolov8_animeface'),
            model_file=yolo_config.get('model_file', 'yolov8x6_animeface.pt'),
            confidence_threshold=yolo_config.get('confidence_threshold', 0.5),
            device=yolo_config.get('device', 'auto'),
            cache_model=yolo_config.get('cache_model', True)
        )
    else:
        raise ValueError(f"Unknown face detection method: {method}. Supported methods: 'haar', 'yolo_anime'")

class CharacterDetector:
    """Main character detection pipeline"""
    
    def __init__(self, config: Dict, resume_from: str):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.project_root = Path(config['input']['project_root'])
        self.characters_dir = self.project_root / config['input']['character_references_dir']
        
        # Detection settings
        detection_config = config['detection']
        self.feature_method = detection_config.get('feature_method', 'orb')
        self.similarity_threshold = detection_config.get('similarity_threshold', 0.6)
        self.max_faces_per_frame = detection_config.get('max_faces_per_frame', 5)
        self.min_face_size = detection_config.get('min_face_size', 50)
        self.face_padding = detection_config.get('face_padding', 0.2)
        self.batch_size = detection_config.get('batch_size', 10)
        self.character_embedding_cache = detection_config.get('character_embedding_cache', True)
        self.save_detected_faces_enabled = detection_config.get('save_detected_faces', False)
        self.save_detected_frames_enabled = detection_config.get('save_detected_frames', False)
        self.save_character_matches_only = detection_config.get('save_character_matches_only', False)
        self.character_colors = detection_config.get('character_colors', {})
        
        # Haar cascade settings
        self.haar_scale_factor = detection_config.get('haar_scale_factor', 1.1)
        self.haar_min_neighbors = detection_config.get('haar_min_neighbors', 5)
        self.haar_min_size = tuple(detection_config.get('haar_min_size', [30, 30]))
        
        # Validate and set run directory
        self.run_dir = self._validate_resume_directory(resume_from)
        self.output_dir = self.run_dir / "detection"
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.character_database = None
        self.face_detector = None
        self.character_recognizer = None
        
    def _validate_resume_directory(self, resume_from: str) -> Path:
        """Validate resume directory and return Path object"""
        resume_path = Path(resume_from)
        
        # Check if resume directory exists
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume directory does not exist: {resume_path}")
        
        if not resume_path.is_dir():
            raise NotADirectoryError(f"Resume path is not a directory: {resume_path}")
        
        # Check if filtering results exist
        filtering_dir = resume_path / "filtering"
        if not filtering_dir.exists():
            raise FileNotFoundError(f"No filtering directory found in resume directory: {filtering_dir}")
        
        filtered_json_path = filtering_dir / "filtered.json"
        if not filtered_json_path.exists():
            raise FileNotFoundError(f"No filtering results found in resume directory: {filtered_json_path}")
        
        self.logger.info(f"Validated resume directory: {resume_path}")
        return resume_path
    
    def _initialize_components(self):
        """Initialize character database, face detector, and recognizer"""
        # Initialize character database
        self.logger.info("Initializing character database...")
        self.character_database = CharacterDatabase(self.characters_dir, self.feature_method)
        
        # Try to load from cache
        cache_path = self.output_dir / "character_database.pkl"
        if self.character_embedding_cache and cache_path.exists():
            if self.character_database.load_cache(cache_path):
                self.logger.info("Loaded character database from cache")
            else:
                self.character_database.load_characters()
                self.character_database.save_cache(cache_path)
        else:
            self.character_database.load_characters()
            if self.character_embedding_cache:
                self.character_database.save_cache(cache_path)
        
        # Initialize face detector using factory
        self.logger.info("Initializing face detector...")
        try:
            self.face_detector = create_face_detector(self.config['detection'])
        except Exception as e:
            self.logger.error(f"Failed to initialize face detector: {e}")
            # Fallback to Haar cascade if configured method fails
            self.logger.warning("Falling back to Haar cascade face detector")
            self.face_detector = HaarFaceDetector(
                scale_factor=self.haar_scale_factor,
                min_neighbors=self.haar_min_neighbors,
                min_size=self.haar_min_size
            )
        
        # Initialize character recognizer
        self.logger.info("Initializing character recognizer...")
        self.character_recognizer = CharacterRecognizer(
            self.character_database,
            self.similarity_threshold
        )
    
    def load_filtered_frames(self) -> List[Dict]:
        """Load filtered frames from the resume directory"""
        filtered_json_path = self.run_dir / "filtering" / "filtered.json"
        
        with open(filtered_json_path, 'r') as f:
            filtered_frames = json.load(f)
        
        self.logger.info(f"Loaded {len(filtered_frames)} filtered frames from: {filtered_json_path}")
        return filtered_frames
    
    def process_frame(self, frame_data: Dict) -> FrameDetection:
        """Process a single frame for character detection"""
        frame_id = frame_data['id']
        frame_path = self.project_root / frame_data['path']
        
        if not frame_path.exists():
            self.logger.warning(f"Frame file not found: {frame_path}")
            return FrameDetection(frame_id, str(frame_path), [], [])
        
        # Load image
        image = cv2.imread(str(frame_path))
        if image is None:
            self.logger.warning(f"Could not load image: {frame_path}")
            return FrameDetection(frame_id, str(frame_path), [], [])
        
        # Detect faces
        detected_faces = self.face_detector.detect_faces(image, self.min_face_size)
        
        # Limit number of faces per frame
        if len(detected_faces) > self.max_faces_per_frame:
            detected_faces = sorted(detected_faces, key=lambda x: x.confidence, reverse=True)[:self.max_faces_per_frame]
        
        # Recognize characters in detected faces
        character_matches = []
        for face in detected_faces:
            matches = self.character_recognizer.recognize_character(face.face_image)
            character_matches.extend(matches)
        
        # Remove duplicate character matches (keep highest confidence)
        unique_matches = {}
        for match in character_matches:
            if match.character_name not in unique_matches or match.confidence > unique_matches[match.character_name].confidence:
                unique_matches[match.character_name] = match
        
        final_matches = list(unique_matches.values())
        final_matches.sort(key=lambda x: x.confidence, reverse=True)
        
        return FrameDetection(frame_id, str(frame_path), detected_faces, final_matches)
    
    def save_detected_faces(self, detections: List[FrameDetection]) -> Path:
        """Save detected face crops for visual inspection"""
        faces_dir = self.output_dir / "faces"
        faces_dir.mkdir(exist_ok=True)
        
        face_count = 0
        for detection in detections:
            if not detection.faces:
                continue
                
            frame_image = cv2.imread(detection.frame_path)
            if frame_image is None:
                continue
            
            for i, face in enumerate(detection.faces):
                x1, y1, x2, y2 = face.bbox
                
                # Add padding
                padding = int(min(x2-x1, y2-y1) * self.face_padding)
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame_image.shape[1], x2 + padding)
                y2 = min(frame_image.shape[0], y2 + padding)
                
                # Extract face with padding
                face_crop = frame_image[y1:y2, x1:x2]
                
                # Save face crop
                face_filename = f"{detection.frame_id}_face_{i:02d}_conf_{face.confidence:.2f}.jpg"
                face_path = faces_dir / face_filename
                cv2.imwrite(str(face_path), face_crop)
                face_count += 1
        
        self.logger.info(f"Saved {face_count} detected faces to: {faces_dir}")
        return faces_dir
    
    def save_detected_frames(self, detections: List[FrameDetection]) -> Path:
        """Save frames with character matches, using colored bounding boxes and character labels"""
        # Choose directory name based on configuration
        if self.save_character_matches_only:
            detected_frames_dir = self.output_dir / "character_matches"
        else:
            detected_frames_dir = self.output_dir / "detected_frames"
        
        detected_frames_dir.mkdir(exist_ok=True)
        
        frame_count = 0
        character_match_count = 0
        
        for detection in detections:
            # Skip frames with no character matches if configured to do so
            if self.save_character_matches_only and not detection.character_matches:
                continue
            
            # Skip frames with no faces at all
            if not detection.faces:
                continue
                
            frame_image = cv2.imread(detection.frame_path)
            if frame_image is None:
                continue
            
            # Create a copy to draw on
            annotated_frame = frame_image.copy()
            
            # Create a mapping of character names to faces for visualization
            character_to_faces = {}
            
            # Map character matches to detected faces
            # For simplicity, we'll assign the first character match to the first face, etc.
            # In a more sophisticated implementation, we could try to match faces to characters
            # based on spatial proximity or other heuristics
            for i, match in enumerate(detection.character_matches):
                if i < len(detection.faces):
                    character_to_faces[match.character_name] = {
                        'face': detection.faces[i],
                        'match': match
                    }
            
            # Draw bounding boxes for character matches
            for char_name, char_data in character_to_faces.items():
                face = char_data['face']
                match = char_data['match']
                x1, y1, x2, y2 = face.bbox
                
                # Get color for this character
                color = self.character_colors.get(char_name, self.character_colors.get('default', [0, 255, 255]))
                
                # Draw colored rectangle around face
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Create character label with confidence
                char_label = f"{char_name} ({match.confidence:.2f})"
                
                # Calculate label position (above the bounding box)
                label_size = cv2.getTextSize(char_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                label_y = max(y1 - 10, label_size[1] + 5)
                label_x = x1
                
                # Draw background rectangle for label
                cv2.rectangle(annotated_frame, 
                             (label_x, label_y - label_size[1] - 5), 
                             (label_x + label_size[0] + 10, label_y + 5), 
                             color, -1)
                
                # Draw character name on the bounding box
                cv2.putText(annotated_frame, char_label, (label_x + 5, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw bounding boxes for faces without character matches (if not in character-only mode)
            if not self.save_character_matches_only:
                matched_face_indices = set(range(len(detection.character_matches)))
                for i, face in enumerate(detection.faces):
                    if i not in matched_face_indices:
                        x1, y1, x2, y2 = face.bbox
                        
                        # Draw gray rectangle for unmatched faces
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                        
                        # Add face ID label
                        face_label = f"Face {i}"
                        label_y = max(y1 - 10, 20)
                        cv2.putText(annotated_frame, face_label, (x1, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # Save annotated frame
            if detection.character_matches:
                # Include character names in filename for easy identification
                char_names = "_".join([match.character_name for match in detection.character_matches[:3]])  # Limit to 3 names
                frame_filename = f"{detection.frame_id}_{char_names}.jpg"
                character_match_count += 1
            else:
                frame_filename = f"{detection.frame_id}_detected.jpg"
            
            frame_path = detected_frames_dir / frame_filename
            cv2.imwrite(str(frame_path), annotated_frame)
            frame_count += 1
        
        if self.save_character_matches_only:
            self.logger.info(f"Saved {character_match_count} frames with character matches to: {detected_frames_dir}")
        else:
            self.logger.info(f"Saved {frame_count} annotated frames ({character_match_count} with character matches) to: {detected_frames_dir}")
        
        return detected_frames_dir
    
    def save_detection_results(self, detections: List[FrameDetection]) -> Path:
        """Save detection results to JSON file"""
        output_path = self.output_dir / "detected.json"
        
        # Convert to serializable format
        output_data = []
        for detection in detections:
            detection_data = {
                "frame_id": detection.frame_id,
                "frame_path": detection.frame_path,
                "faces": [
                    {
                        "bbox": [int(x) for x in face.bbox],
                        "confidence": float(face.confidence)
                    }
                    for face in detection.faces
                ],
                "character_matches": [
                    {
                        "character_name": match.character_name,
                        "confidence": float(match.confidence),
                        "match_count": int(match.match_count)
                    }
                    for match in detection.character_matches
                ]
            }
            output_data.append(detection_data)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Detection results saved to: {output_path}")
        return output_path
    
    def run(self) -> Dict:
        """Run the character detection stage"""
        try:
            self.logger.info("Starting character detection...")
            
            # Initialize components
            self._initialize_components()
            
            # Load filtered frames
            filtered_frames = self.load_filtered_frames()
            
            if not filtered_frames:
                return {
                    "success": False,
                    "error": "No filtered frames to process",
                    "run_directory": str(self.run_dir)
                }
            
            # Process frames in batches
            detections = []
            total_batches = (len(filtered_frames) + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"Processing {len(filtered_frames)} frames in {total_batches} batches of {self.batch_size}")
            
            for i in range(0, len(filtered_frames), self.batch_size):
                batch = filtered_frames[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                for frame_data in batch:
                    detection = self.process_frame(frame_data)
                    detections.append(detection)
            
            # Count results
            total_faces = sum(len(d.faces) for d in detections)
            total_character_matches = sum(len(d.character_matches) for d in detections)
            frames_with_detections = sum(1 for d in detections if d.faces or d.character_matches)
            
            self.logger.info(f"Detection complete: {total_faces} faces detected, {total_character_matches} character matches in {frames_with_detections} frames")
            
            # Save results
            output_path = self.save_detection_results(detections)
            
            # Save detected faces if enabled
            faces_dir = None
            if self.save_detected_faces_enabled:
                faces_dir = self.save_detected_faces(detections)
            
            # Save detected frames if enabled
            detected_frames_dir = None
            if self.save_detected_frames_enabled:
                detected_frames_dir = self.save_detected_frames(detections)
            
            result = {
                "success": True,
                "frames_processed": len(filtered_frames),
                "faces_detected": total_faces,
                "character_matches": total_character_matches,
                "frames_with_detections": frames_with_detections,
                "output_path": str(output_path),
                "faces_directory": str(faces_dir) if faces_dir else None,
                "detected_frames_directory": str(detected_frames_dir) if detected_frames_dir else None,
                "run_directory": str(self.run_dir),
                "detection_directory": str(self.output_dir)
            }
            
            self.logger.info(f"Character detection completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Character detection failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "run_directory": str(self.run_dir) if hasattr(self, 'run_dir') else None
            }

def main():
    """Main entry point for detection stage"""
    parser = argparse.ArgumentParser(description="Detect characters in filtered frames")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--resume-from", required=True, help="Path to run directory to resume from (e.g., outputs/run_20250902_165141)")
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
        print(f"Error: Configuration file not found: {args.config}")
        return 1
    except yaml.YAMLError as e:
        print(f"Error: Error parsing configuration file: {e}")
        return 1
    
    # Run detection
    try:
        detector = CharacterDetector(config, args.resume_from)
        result = detector.run()
        
        if result["success"]:
            print(f"Character detection completed successfully!")
            print(f"Frames processed: {result['frames_processed']}")
            print(f"Faces detected: {result['faces_detected']}")
            print(f"Character matches: {result['character_matches']}")
            print(f"Frames with detections: {result['frames_with_detections']}")
            print(f"Output: {result['output_path']}")
            if result['faces_directory']:
                print(f"Faces directory: {result['faces_directory']}")
            print(f"Run directory: {result['run_directory']}")
            return 0
        else:
            print(f"Error: Character detection failed: {result['error']}")
            if result.get('run_directory'):
                print(f"Run directory: {result['run_directory']}")
            return 1
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except NotADirectoryError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: Character detection failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
