"""
Diffusion Bunny Pipeline
Main orchestrator for the movie-to-diffusion pipeline.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from stage1_extraction.extractor import FrameExtractor

class Pipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Available stages
        self.available_stages = {
            "extraction": self.run_extraction,
            "filtering": self.run_filtering,
            "detection": self.run_detection,
            "captioning": self.run_captioning,
            "finetuning": self.run_finetuning,
            "inference": self.run_inference
        }
        
    def load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
            
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('project', {}).get('log_level', 'INFO')
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "pipeline.log"),
                logging.StreamHandler()
            ]
        )
        
    def validate_project_structure(self) -> bool:
        """Validate that the project structure exists"""
        project_root = Path(self.config['input']['project_root'])
        movie_dir = project_root / self.config['input']['movie_dir']
        
        if not project_root.exists():
            self.logger.error(f"Project root not found: {project_root}")
            return False
            
        if not movie_dir.exists():
            self.logger.error(f"Movie directory not found: {movie_dir}")
            return False
            
        self.logger.info(f"Project structure validated: {project_root}")
        return True
        
    def run_extraction(self) -> Dict:
        """Run Stage 1: Frame Extraction"""
        self.logger.info("=== Stage 1: Frame Extraction ===")
        
        extractor = FrameExtractor(self.config)
        result = extractor.run()
        
        if result['success']:
            self.logger.info(f"[SUCCESS] Extraction completed: {result['frames_extracted']} frames")
        else:
            self.logger.error(f"[FAILED] Extraction failed: {result['error']}")
            
        return result
        
    def run_filtering(self) -> Dict:
        """Run Stage 2: Quality Filtering"""
        self.logger.info("=== Stage 2: Quality Filtering ===")
        try:
            from stage2_filtering.filter import FrameFilter
            
            if hasattr(self, 'current_run_dir') and self.current_run_dir:
                # Resume mode: use existing run directory
                filter_stage = FrameFilter(self.config, resume_run_dir=self.current_run_dir)
            else:
                # New run mode: create new timestamped directory
                filter_stage = FrameFilter(self.config)
            
            result = filter_stage.run()
            
            if result['success']:
                self.logger.info(f"[SUCCESS] Filtering completed: {result['frames_kept']} frames kept")
                # Update current run directory for subsequent stages
                if 'run_directory' in result:
                    self.current_run_dir = Path(result['run_directory'])
            else:
                self.logger.error(f"[FAILED] Filtering failed: {result['error']}")
                
            return result
        except ImportError:
            self.logger.warning("Filtering stage not yet implemented")
            return {"success": False, "error": "Not implemented"}
        
    def run_detection(self) -> Dict:
        """Run Stage 3: Character Detection"""
        self.logger.info("=== Stage 3: Character Detection ===")
        try:
            from stage3_detection.detector import CharacterDetector
            
            if not hasattr(self, 'current_run_dir') or not self.current_run_dir:
                return {"success": False, "error": "Detection stage requires a run directory from previous stages"}
            
            detector = CharacterDetector(self.config, str(self.current_run_dir))
            result = detector.run()
            
            if result['success']:
                self.logger.info(f"[SUCCESS] Detection completed: {result['faces_detected']} faces, {result['character_matches']} character matches")
            else:
                self.logger.error(f"[FAILED] Detection failed: {result['error']}")
                
            return result
        except ImportError:
            self.logger.warning("Detection stage not yet implemented")
            return {"success": False, "error": "Not implemented"}
        
    def run_captioning(self) -> Dict:
        """Run Stage 4: Caption Generation"""
        self.logger.info("=== Stage 4: Caption Generation ===")
        # TODO: Implement captioning stage
        self.logger.warning("Captioning stage not yet implemented")
        return {"success": False, "error": "Not implemented"}
        
    def run_finetuning(self) -> Dict:
        """Run Stage 5: Fine-tuning"""
        self.logger.info("=== Stage 5: Fine-tuning ===")
        # TODO: Implement finetuning stage
        self.logger.warning("Fine-tuning stage not yet implemented")
        return {"success": False, "error": "Not implemented"}
        
    def run_inference(self) -> Dict:
        """Run Stage 6: Inference"""
        self.logger.info("=== Stage 6: Inference ===")
        # TODO: Implement inference stage
        self.logger.warning("Inference stage not yet implemented")
        return {"success": False, "error": "Not implemented"}
        
    def can_resume_stage(self, stage: str, resume_dir: Optional[Path] = None) -> bool:
        """Check if a stage can be resumed (has existing output)"""
        project_root = Path(self.config['input']['project_root'])
        
        if stage == "extraction":
            # Stage 1 outputs to project root
            frames_dir = project_root / self.config['input']['frames_dir']
            metadata_file = project_root / "frames_metadata.json"
            return frames_dir.exists() and metadata_file.exists()
        
        elif stage == "filtering":
            if resume_dir:
                # Check if filtering already completed in this run
                filtering_output = resume_dir / "filtering" / "filtered.json"
                return filtering_output.exists()
            else:
                # Check if Stage 1 completed (prerequisite for new run)
                return self.can_resume_stage("extraction")
        
        elif stage == "detection":
            if resume_dir:
                # Check if detection already completed in this run
                detection_output = resume_dir / "detection" / "detected.json"
                return detection_output.exists()
            else:
                return False  # Detection requires explicit resume directory
        
        elif stage == "captioning":
            if resume_dir:
                # Check if captioning already completed in this run
                captioning_output = resume_dir / "captioning" / "captions.json"
                return captioning_output.exists()
            else:
                return False  # Captioning requires explicit resume directory
        
        # Add other stage checks as needed
        return False
        
    def _detect_resume_stage(self, resume_dir: Path) -> str:
        """Auto-detect which stage to resume from based on directory contents"""
        if (resume_dir / "detection" / "detected.json").exists():
            return "captioning"  # Detection completed, resume from captioning
        elif (resume_dir / "filtering" / "filtered.json").exists():
            return "detection"   # Filtering completed, resume from detection
        else:
            return "filtering"   # Resume from filtering

    def run_pipeline(self, stages: Optional[List[str]] = None, resume_from: Optional[str] = None, resume_from_dir: Optional[str] = None) -> Dict:
        """Run the complete pipeline or specified stages"""
        
        # Validate project structure
        if not self.validate_project_structure():
            return {"success": False, "error": "Invalid project structure"}
        
        # Handle resume directory
        resume_dir_path = None
        if resume_from_dir:
            resume_dir_path = Path(resume_from_dir)
            if not resume_dir_path.exists():
                return {"success": False, "error": f"Resume directory not found: {resume_dir_path}"}
            if not resume_dir_path.is_dir():
                return {"success": False, "error": f"Resume path is not a directory: {resume_dir_path}"}
            
            self.logger.info(f"Resuming from directory: {resume_dir_path}")
            
            # Auto-detect resume stage if not specified
            if not resume_from:
                resume_from = self._detect_resume_stage(resume_dir_path)
                self.logger.info(f"Auto-detected resume stage: {resume_from}")
        
        # Set current run directory for stages to use
        self.current_run_dir = resume_dir_path
            
        # Determine which stages to run
        if stages:
            stages_to_run = stages
        elif resume_from:
            all_stages = self.config['pipeline']['stages']
            try:
                start_index = all_stages.index(resume_from)
                stages_to_run = all_stages[start_index:]
            except ValueError:
                return {"success": False, "error": f"Unknown stage: {resume_from}"}
        else:
            stages_to_run = self.config['pipeline']['stages']
            
        self.logger.info(f"Running pipeline stages: {stages_to_run}")
        
        # Check for force rerun setting
        force_rerun = self.config['pipeline'].get('force_rerun', False)
        
        results = {}
        
        for stage in stages_to_run:
            if stage not in self.available_stages:
                self.logger.error(f"Unknown stage: {stage}")
                results[stage] = {"success": False, "error": f"Unknown stage: {stage}"}
                continue
                
            # Check if stage can be skipped (already completed)
            if not force_rerun and self.can_resume_stage(stage, resume_dir_path):
                self.logger.info(f"[SKIP] Skipping {stage} (already completed, use force_rerun to override)")
                results[stage] = {"success": True, "skipped": True}
                continue
                
            # Run the stage
            try:
                result = self.available_stages[stage]()
                results[stage] = result
                
                # Stop pipeline if stage failed
                if not result['success']:
                    self.logger.error(f"Pipeline stopped due to {stage} failure")
                    break
                    
            except Exception as e:
                self.logger.error(f"Stage {stage} crashed: {str(e)}")
                results[stage] = {"success": False, "error": str(e)}
                break
                
        # Summary
        successful_stages = [stage for stage, result in results.items() if result.get('success')]
        failed_stages = [stage for stage, result in results.items() if not result.get('success')]
        
        self.logger.info(f"Pipeline completed. Successful: {successful_stages}, Failed: {failed_stages}")
        
        return {
            "success": len(failed_stages) == 0,
            "stages_run": list(results.keys()),
            "successful_stages": successful_stages,
            "failed_stages": failed_stages,
            "results": results
        }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Diffusion Bunny Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--stages", help="Comma-separated list of stages to run")
    parser.add_argument("--resume-from", help="Stage to resume from")
    parser.add_argument("--resume-from-dir", help="Directory to resume from (e.g., outputs/run_20250902_165141)")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerun of all stages")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Pipeline(args.config)
    
    # Override force_rerun if specified
    if args.force_rerun:
        pipeline.config['pipeline']['force_rerun'] = True
        
    # Parse stages
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(',')]
        
    # Run pipeline
    result = pipeline.run_pipeline(stages=stages, resume_from=args.resume_from, resume_from_dir=args.resume_from_dir)
    
    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)

if __name__ == "__main__":
    main()
