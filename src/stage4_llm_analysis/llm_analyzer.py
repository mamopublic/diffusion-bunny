"""
Stage 4.5: LLM Analysis
Main analyzer that combines face detection results with LLM-based character identification and scene description.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import argparse
import yaml
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .composite_creator import CompositeImageCreator, CompositeConfig
from .openrouter_client import OpenRouterClient, OpenRouterConfig, LLMResponse

@dataclass
class FrameLLMAnalysis:
    """LLM analysis results for a single frame"""
    frame_id: str
    frame_path: str
    composite_path: str
    characters_identified: List[str]
    scene_description: str
    sd_caption: str
    confidence: float
    processing_time: float
    model_used: str
    local_detections: List[Dict]  # Original face detections from Stage 3
    llm_raw_response: str

class LLMAnalyzer:
    """Main LLM analysis pipeline"""
    
    def __init__(self, config: Dict, resume_from: str):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration
        self.project_root = Path(config['input']['project_root'])
        self.strip_data_dir = self.project_root / "strip_data"
        
        # LLM analysis settings
        llm_config = config.get('llm_analysis', {})
        self.enabled = llm_config.get('enabled', True)
        self.process_frames_with_faces_only = llm_config.get('process_frames_with_faces_only', True)
        self.min_face_confidence = llm_config.get('min_face_confidence', 0.5)
        self.batch_size = llm_config.get('batch_size', 5)
        self.enable_caching = llm_config.get('enable_caching', True)
        self.similarity_threshold_for_caching = llm_config.get('similarity_threshold_for_caching', 0.95)
        
        # Debug settings
        self.debug_mode = llm_config.get('debug_mode', False)
        self.debug_max_frames = llm_config.get('debug_max_frames', 3)
        
        # Validate and set run directory
        self.run_dir = self._validate_resume_directory(resume_from)
        self.output_dir = self.run_dir / "llm_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Create composites directory
        self.composites_dir = self.output_dir / "composites"
        self.composites_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.composite_creator = None
        self.openrouter_client = None
        self.character_names = []
        
        if not self.enabled:
            self.logger.info("LLM analysis is disabled in configuration")
            return
        
        self._initialize_components()
    
    def _validate_resume_directory(self, resume_from: str) -> Path:
        """Validate resume directory and return Path object"""
        resume_path = Path(resume_from)
        
        # Check if resume directory exists
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume directory does not exist: {resume_path}")
        
        if not resume_path.is_dir():
            raise NotADirectoryError(f"Resume path is not a directory: {resume_path}")
        
        # Check if detection results exist
        detection_dir = resume_path / "detection"
        if not detection_dir.exists():
            raise FileNotFoundError(f"No detection directory found in resume directory: {detection_dir}")
        
        detected_json_path = detection_dir / "detected.json"
        if not detected_json_path.exists():
            raise FileNotFoundError(f"No detection results found in resume directory: {detected_json_path}")
        
        self.logger.info(f"Validated resume directory: {resume_path}")
        return resume_path
    
    def _initialize_components(self):
        """Initialize composite creator and OpenRouter client"""
        try:
            # Initialize composite creator
            self.logger.info("Initializing composite image creator...")
            
            # Create composite config from settings
            llm_config = self.config.get('llm_analysis', {})
            composite_config = CompositeConfig(
                strip_width=llm_config.get('strip_width', 150),
                character_image_size=llm_config.get('character_image_size', 150),
                strip_background_color=tuple(llm_config.get('strip_background_color', [240, 240, 240]))
            )
            
            self.composite_creator = CompositeImageCreator(
                strip_data_dir=self.strip_data_dir,
                config=composite_config
            )
            
            self.character_names = self.composite_creator.get_character_names()
            self.logger.info(f"Available characters: {self.character_names}")
            
            # Initialize OpenRouter client
            self.logger.info("Initializing OpenRouter client...")
            
            openrouter_config_dict = llm_config.get('openrouter', {})
            
            # Get API key from environment
            api_key_env = openrouter_config_dict.get('api_key_env', 'OPENROUTER_API_KEY')
            api_key = os.getenv(api_key_env)
            
            if not api_key:
                raise ValueError(f"OpenRouter API key not found in environment variable: {api_key_env}")
            
            openrouter_config = OpenRouterConfig(
                api_key=api_key,
                base_url=openrouter_config_dict.get('base_url', 'https://openrouter.ai/api/v1'),
                model=llm_config.get('model', 'anthropic/claude-3-haiku'),
                max_tokens=llm_config.get('max_tokens', 300),
                temperature=llm_config.get('temperature', 0.7),
                timeout=openrouter_config_dict.get('timeout', 30),
                retry_attempts=openrouter_config_dict.get('retry_attempts', 3),
                retry_delay=openrouter_config_dict.get('retry_delay', 1.0),
                max_requests_per_minute=llm_config.get('max_requests_per_minute', 30)
            )
            
            self.openrouter_client = OpenRouterClient(
                config=openrouter_config,
                enable_caching=self.enable_caching
            )
            
            self.logger.info("LLM analysis components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM analysis components: {e}")
            raise
    
    def load_detection_results(self) -> List[Dict]:
        """Load detection results from the resume directory"""
        detected_json_path = self.run_dir / "detection" / "detected.json"
        
        with open(detected_json_path, 'r') as f:
            detection_results = json.load(f)
        
        self.logger.info(f"Loaded {len(detection_results)} detection results from: {detected_json_path}")
        return detection_results
    
    def filter_frames_for_analysis(self, detection_results: List[Dict]) -> List[Dict]:
        """Filter frames that should be analyzed by LLM"""
        if not self.process_frames_with_faces_only:
            return detection_results
        
        filtered_frames = []
        
        for frame_data in detection_results:
            faces = frame_data.get('faces', [])
            
            # Check if frame has faces with sufficient confidence
            has_good_faces = any(
                face.get('confidence', 0) >= self.min_face_confidence 
                for face in faces
            )
            
            if has_good_faces:
                filtered_frames.append(frame_data)
        
        self.logger.info(f"Filtered {len(filtered_frames)} frames with faces from {len(detection_results)} total frames")
        return filtered_frames
    
    def create_composite_for_frame(self, frame_data: Dict) -> Optional[Path]:
        """Create composite image for a frame"""
        frame_id = frame_data['frame_id']
        frame_path = Path(frame_data['frame_path'])
        
        # Create composite filename
        composite_filename = f"{frame_id}_composite.jpg"
        composite_path = self.composites_dir / composite_filename
        
        # Skip if composite already exists
        if composite_path.exists():
            self.logger.debug(f"Composite already exists: {composite_path}")
            return composite_path
        
        # Create composite
        faces = frame_data.get('faces', [])
        success = self.composite_creator.create_and_save_composite(
            frame_path=frame_path,
            output_path=composite_path,
            face_detections=faces
        )
        
        if success:
            self.logger.debug(f"Created composite: {composite_path}")
            return composite_path
        else:
            self.logger.warning(f"Failed to create composite for frame: {frame_id}")
            return None
    
    def analyze_frame_with_llm(self, frame_data: Dict, composite_path: Path) -> FrameLLMAnalysis:
        """Analyze a single frame using LLM"""
        frame_id = frame_data['frame_id']
        frame_path = frame_data['frame_path']
        
        try:
            # Analyze with LLM
            llm_response = self.openrouter_client.analyze_image(
                image_path=composite_path,
                character_names=self.character_names
            )
            
            # Create analysis result
            analysis = FrameLLMAnalysis(
                frame_id=frame_id,
                frame_path=frame_path,
                composite_path=str(composite_path),
                characters_identified=llm_response.characters_present,
                scene_description=llm_response.scene_description,
                sd_caption=llm_response.sd_caption,
                confidence=llm_response.confidence,
                processing_time=llm_response.processing_time,
                model_used=llm_response.model_used,
                local_detections=frame_data.get('faces', []),
                llm_raw_response=llm_response.raw_response
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze frame {frame_id} with LLM: {e}")
            
            # Return error analysis
            return FrameLLMAnalysis(
                frame_id=frame_id,
                frame_path=frame_path,
                composite_path=str(composite_path),
                characters_identified=[],
                scene_description=f"Analysis failed: {str(e)}",
                sd_caption=f"Analysis failed: {str(e)}",
                confidence=0.0,
                processing_time=0.0,
                model_used="error",
                local_detections=frame_data.get('faces', []),
                llm_raw_response=""
            )
    
    def process_frames_batch(self, frames_batch: List[Dict]) -> List[FrameLLMAnalysis]:
        """Process a batch of frames"""
        analyses = []
        
        for frame_data in frames_batch:
            frame_id = frame_data['frame_id']
            
            # Create composite
            composite_path = self.create_composite_for_frame(frame_data)
            if not composite_path:
                self.logger.warning(f"Skipping frame {frame_id} - could not create composite")
                continue
            
            # Analyze with LLM
            analysis = self.analyze_frame_with_llm(frame_data, composite_path)
            analyses.append(analysis)
            
            self.logger.debug(f"Analyzed frame {frame_id}: {analysis.characters_identified}")
        
        return analyses
    
    def save_analysis_results(self, analyses: List[FrameLLMAnalysis]) -> Path:
        """Save LLM analysis results to JSON file"""
        output_path = self.output_dir / "llm_analysis.json"
        
        # Convert to serializable format
        output_data = []
        for analysis in analyses:
            analysis_data = {
                "frame_id": analysis.frame_id,
                "frame_path": analysis.frame_path,
                "composite_path": analysis.composite_path,
                "characters_identified": analysis.characters_identified,
                "scene_description": analysis.scene_description,
                "sd_caption": analysis.sd_caption,
                "confidence": float(analysis.confidence),
                "processing_time": float(analysis.processing_time),
                "model_used": analysis.model_used,
                "local_detections": analysis.local_detections,
                "llm_raw_response": analysis.llm_raw_response,
                "analyzed_at": datetime.now().isoformat()
            }
            output_data.append(analysis_data)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"LLM analysis results saved to: {output_path}")
        return output_path
    
    def generate_summary_report(self, analyses: List[FrameLLMAnalysis]) -> Dict[str, Any]:
        """Generate summary report of LLM analysis"""
        total_frames = len(analyses)
        successful_analyses = [a for a in analyses if a.confidence > 0]
        
        # Character statistics
        character_counts = {}
        for analysis in successful_analyses:
            for character in analysis.characters_identified:
                character_counts[character] = character_counts.get(character, 0) + 1
        
        # Confidence statistics
        confidences = [a.confidence for a in successful_analyses]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Processing time statistics
        processing_times = [a.processing_time for a in analyses]
        total_processing_time = sum(processing_times)
        avg_processing_time = total_processing_time / len(processing_times) if processing_times else 0
        
        summary = {
            "total_frames_analyzed": total_frames,
            "successful_analyses": len(successful_analyses),
            "success_rate": len(successful_analyses) / total_frames if total_frames > 0 else 0,
            "character_appearances": character_counts,
            "average_confidence": avg_confidence,
            "total_processing_time": total_processing_time,
            "average_processing_time": avg_processing_time,
            "frames_with_characters": len([a for a in analyses if a.characters_identified]),
            "unique_characters_found": len(character_counts),
            "model_used": analyses[0].model_used if analyses else "unknown"
        }
        
        return summary
    
    def run(self) -> Dict[str, Any]:
        """Run the LLM analysis stage"""
        if not self.enabled:
            return {
                "success": True,
                "message": "LLM analysis is disabled",
                "frames_analyzed": 0,
                "run_directory": str(self.run_dir)
            }
        
        try:
            self.logger.info("Starting LLM analysis...")
            
            # Load detection results
            detection_results = self.load_detection_results()
            
            # Filter frames for analysis
            frames_to_analyze = self.filter_frames_for_analysis(detection_results)
            
            if not frames_to_analyze:
                return {
                    "success": True,
                    "message": "No frames meet criteria for LLM analysis",
                    "frames_analyzed": 0,
                    "run_directory": str(self.run_dir)
                }
            
            # Apply debug frame limit if in debug mode
            if self.debug_mode:
                original_count = len(frames_to_analyze)
                frames_to_analyze = frames_to_analyze[:self.debug_max_frames]
                self.logger.warning(f"🐛 DEBUG MODE: Limited to {len(frames_to_analyze)} frames (from {original_count} total)")
            
            # Process frames in batches
            all_analyses = []
            total_batches = (len(frames_to_analyze) + self.batch_size - 1) // self.batch_size
            
            self.logger.info(f"Processing {len(frames_to_analyze)} frames in {total_batches} batches of {self.batch_size}")
            
            for i in range(0, len(frames_to_analyze), self.batch_size):
                batch = frames_to_analyze[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                batch_analyses = self.process_frames_batch(batch)
                all_analyses.extend(batch_analyses)
            
            # Save results
            output_path = self.save_analysis_results(all_analyses)
            
            # Generate summary
            summary = self.generate_summary_report(all_analyses)
            
            self.logger.info(f"LLM analysis complete: {summary['successful_analyses']}/{summary['total_frames_analyzed']} frames analyzed successfully")
            self.logger.info(f"Characters found: {summary['character_appearances']}")
            
            result = {
                "success": True,
                "frames_analyzed": len(all_analyses),
                "successful_analyses": summary['successful_analyses'],
                "character_appearances": summary['character_appearances'],
                "average_confidence": summary['average_confidence'],
                "total_processing_time": summary['total_processing_time'],
                "output_path": str(output_path),
                "composites_directory": str(self.composites_dir),
                "run_directory": str(self.run_dir),
                "summary": summary
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "run_directory": str(self.run_dir) if hasattr(self, 'run_dir') else None
            }

def main():
    """Main entry point for LLM analysis stage"""
    parser = argparse.ArgumentParser(description="Analyze frames with LLM for character identification")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--resume-from", required=True, help="Path to run directory to resume from")
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
    
    # Run LLM analysis
    try:
        analyzer = LLMAnalyzer(config, args.resume_from)
        result = analyzer.run()
        
        if result["success"]:
            print(f"LLM analysis completed successfully!")
            print(f"Frames analyzed: {result['frames_analyzed']}")
            if 'successful_analyses' in result:
                print(f"Successful analyses: {result['successful_analyses']}")
                print(f"Character appearances: {result['character_appearances']}")
                print(f"Average confidence: {result['average_confidence']:.3f}")
                print(f"Total processing time: {result['total_processing_time']:.2f}s")
                print(f"Output: {result['output_path']}")
            print(f"Run directory: {result['run_directory']}")
            return 0
        else:
            print(f"Error: LLM analysis failed: {result['error']}")
            if result.get('run_directory'):
                print(f"Run directory: {result['run_directory']}")
            return 1
            
    except Exception as e:
        print(f"Error: LLM analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
