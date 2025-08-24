# Diffusion Bunny - Project Brief

## Project Overview
Diffusion Bunny is a proof-of-concept pipeline that transforms movies into fine-tuned Stable Diffusion 1.5 models. The system extracts frames from movies, processes them through quality filtering and character detection, generates captions, and creates training datasets for LoRA and DreamBooth fine-tuning.

## Core Objectives
- **Automated Dataset Creation**: Convert movies into labeled training datasets for diffusion models
- **Character-Specific Training**: Use character reference images to create character-aware datasets
- **Compute Efficiency**: Design for environments without GPU access during processing
- **Resumable Pipeline**: Enable skipping/resuming stages with persistent metadata
- **Configuration-Driven**: All parameters controlled via config.yaml

## Key Requirements
- **Modular Architecture**: Each pipeline stage in separate directory under src/
- **Metadata Persistence**: JSON files track frame metadata throughout pipeline
- **Quality Control**: Blur detection and configurable thresholds for frame filtering
- **Character Detection**: Face detection with matching against reference character images
- **Flexible Captioning**: Configurable caption generation (AWS Bedrock or alternatives)
- **Efficient Storage**: Avoid copying frames; use references and metadata

## Pipeline Stages
1. **Frame Extraction**: OpenCV-based keyframe/interval extraction with metadata
2. **Quality Filtering**: Laplacian variance blur detection with batch processing
3. **Character Detection**: Face detection and matching against character references
4. **Caption Generation**: Automated caption creation for training data
5. **Fine-tuning**: LoRA and DreamBooth training (notebook-based initially)
6. **Inference**: Model loading and image generation from prompts

## Success Criteria
- Complete pipeline from movie file to fine-tuned model
- Resumable execution with persistent state
- Efficient processing without GPU requirements during preprocessing
- Quality training datasets with character-specific labels
- Working inference capability with generated models

## Technical Constraints
- No GPU available for processing stages (fine-tuning may use external resources)
- Must handle large movie files efficiently
- Memory-conscious batch processing
- Cross-platform compatibility (Windows primary)

## Project Structure
```
diffusion-bunny/
├── src/
│   ├── pipeline.py           # Main pipeline orchestrator
│   ├── utils.py             # Common utilities
│   ├── stage1_extraction/   # Frame extraction
│   ├── stage2_filtering/    # Quality filtering
│   ├── stage3_detection/    # Character detection
│   ├── stage4_captioning/   # Caption generation
│   ├── stage5_finetuning/   # Model fine-tuning
│   └── stage6_inference/    # Model inference
├── config.yaml             # Main configuration
├── docs/                   # Documentation
├── examples/               # Example notebooks and scripts
└── memory-bank/           # Project memory and context
