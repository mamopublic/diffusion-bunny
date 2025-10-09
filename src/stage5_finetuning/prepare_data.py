"""
Data Preparation for PEFT DreamBooth LoRA Training
Converts LLM analysis output to PEFT-compatible format
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DreamBoothDataPreparer:
    """Prepares training data for PEFT DreamBooth LoRA fine-tuning"""
    
    def __init__(
        self,
        llm_analysis_json: Path,
        output_dir: Path = None,
        min_confidence: float = 0.5
    ):
        """
        Args:
            llm_analysis_json: Path to LLM analysis JSON from Stage 4
            output_dir: Where to save prepared data (default: auto-derive from llm_analysis path)
            min_confidence: Minimum confidence score to include frame
        """
        self.llm_analysis_json = Path(llm_analysis_json)
        
        # Auto-derive output directory from LLM analysis path if not specified
        # e.g., outputs/run_20250902_165141/llm_analysis/llm_analysis.json
        # -> outputs/run_20250902_165141/training_data
        if output_dir is None:
            run_dir = self.llm_analysis_json.parent.parent
            self.output_dir = run_dir / "training_data"
            logger.info(f"Auto-derived output directory: {self.output_dir}")
        else:
            self.output_dir = Path(output_dir)
        
        self.min_confidence = min_confidence
        
        # Create output structure
        self.instance_images_dir = self.output_dir / "instance_images"
        self.instance_images_dir.mkdir(parents=True, exist_ok=True)
        
    def load_llm_analysis(self) -> List[Dict]:
        """Load LLM analysis results"""
        with open(self.llm_analysis_json, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} frames from LLM analysis")
        return data
    
    def filter_valid_frames(self, data: List[Dict]) -> List[Dict]:
        """Filter frames with valid captions and sufficient confidence"""
        valid_frames = []
        
        for frame in data:
            # Check if frame has characters identified
            if not frame.get('characters_identified'):
                continue
            
            # Check confidence
            confidence = frame.get('confidence', 0.0)
            if confidence < self.min_confidence:
                continue
            
            # Check if sd_caption exists and is not malformed
            sd_caption = frame.get('sd_caption', '')
            if not sd_caption or len(sd_caption) < 10:
                continue
            
            # Clean up malformed captions (some have JSON artifacts)
            if sd_caption.startswith('{'):
                # Try to extract clean caption from malformed JSON
                scene_desc = frame.get('scene_description', '')
                if scene_desc and not scene_desc.startswith('{'):
                    sd_caption = scene_desc
                else:
                    continue
            
            valid_frames.append({
                'frame_id': frame['frame_id'],
                'frame_path': frame['frame_path'],
                'characters': frame['characters_identified'],
                'caption': sd_caption,
                'confidence': confidence
            })
        
        logger.info(f"Filtered to {len(valid_frames)} valid frames (min confidence: {self.min_confidence})")
        return valid_frames
    
    def copy_images(self, frames: List[Dict]) -> List[Dict]:
        """Copy images to instance_images directory"""
        copied_frames = []
        
        for frame in frames:
            src_path = Path(frame['frame_path'])
            
            if not src_path.exists():
                logger.warning(f"Image not found: {src_path}")
                continue
            
            # Copy to instance_images with same name
            dst_path = self.instance_images_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            
            # Update path
            frame['new_path'] = str(dst_path)
            copied_frames.append(frame)
        
        logger.info(f"Copied {len(copied_frames)} images to {self.instance_images_dir}")
        return copied_frames
    
    def create_metadata_jsonl(self, frames: List[Dict]):
        """Create metadata.jsonl file for PEFT training"""
        metadata_path = self.instance_images_dir / "metadata.jsonl"
        
        with open(metadata_path, 'w') as f:
            for frame in frames:
                # PEFT expects: {"file_name": "image.jpg", "text": "caption"}
                metadata_entry = {
                    "file_name": Path(frame['new_path']).name,
                    "text": frame['caption']
                }
                f.write(json.dumps(metadata_entry) + '\n')
        
        logger.info(f"Created metadata.jsonl with {len(frames)} entries")
    
    def create_summary(self, frames: List[Dict]):
        """Create summary statistics"""
        # Character distribution
        char_counts = {}
        for frame in frames:
            for char in frame['characters']:
                char_counts[char] = char_counts.get(char, 0) + 1
        
        summary = {
            'total_frames': len(frames),
            'character_distribution': char_counts,
            'avg_confidence': sum(f['confidence'] for f in frames) / len(frames),
            'data_dir': str(self.instance_images_dir),
            'metadata_file': str(self.instance_images_dir / "metadata.jsonl")
        }
        
        summary_path = self.output_dir / "data_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\n" + "="*50)
        logger.info("DATA PREPARATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total frames: {summary['total_frames']}")
        logger.info(f"Character distribution:")
        for char, count in sorted(char_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {char}: {count} frames")
        logger.info(f"Average confidence: {summary['avg_confidence']:.2f}")
        logger.info(f"Data directory: {summary['data_dir']}")
        logger.info("="*50 + "\n")
        
        return summary
    
    def prepare(self) -> Dict:
        """Execute full data preparation pipeline"""
        logger.info("Starting data preparation for PEFT DreamBooth LoRA training")
        
        # Load data
        data = self.load_llm_analysis()
        
        # Filter valid frames
        valid_frames = self.filter_valid_frames(data)
        
        if not valid_frames:
            raise ValueError("No valid frames found after filtering!")
        
        # Copy images
        copied_frames = self.copy_images(valid_frames)
        
        if not copied_frames:
            raise ValueError("No images could be copied!")
        
        # Create metadata
        self.create_metadata_jsonl(copied_frames)
        
        # Create summary
        summary = self.create_summary(copied_frames)
        
        logger.info("✅ Data preparation complete!")
        return summary


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for DreamBooth LoRA training")
    parser.add_argument(
        '--llm-analysis',
        type=str,
        required=True,
        help='Path to llm_analysis.json from Stage 4'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='training_data',
        help='Output directory for prepared data'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence score to include frame'
    )
    
    args = parser.parse_args()
    
    preparer = DreamBoothDataPreparer(
        llm_analysis_json=args.llm_analysis,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence
    )
    
    summary = preparer.prepare()
    print(f"\n✅ Prepared {summary['total_frames']} frames for training")
    print(f"📁 Data ready at: {summary['data_dir']}")


if __name__ == "__main__":
    main()
