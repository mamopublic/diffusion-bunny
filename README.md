# Diffusion Bunny 🐰✨

A proof-of-concept pipeline that transforms movies into fine-tuned Stable Diffusion 1.5 models. Extract frames, filter quality, detect characters, generate captions, and create character-specific AI art models.

## Overview

Diffusion Bunny automates the complete workflow from raw video files to character-specific diffusion models:

1. **Frame Extraction**: Extract keyframes or interval-based frames from movies
2. **Quality Filtering**: Remove blurry or low-quality frames using intelligent filtering
3. **Character Detection**: Detect and label characters using reference images
4. **Caption Generation**: Generate descriptive captions using AWS Bedrock or other providers
5. **Fine-tuning**: Train LoRA adapters and DreamBooth models
6. **Inference**: Generate new images with the fine-tuned models

## Features

- 🎬 **Video Processing**: Support for MP4, AVI, MOV, MKV formats
- 🔍 **Smart Filtering**: Blur detection and quality assessment
- 👤 **Character Recognition**: Face detection and matching against reference images
- 📝 **Automated Captioning**: Multiple caption generation providers (AWS Bedrock, OpenAI, Anthropic)
- 🎯 **Efficient Training**: LoRA and DreamBooth fine-tuning
- ⚡ **Resumable Pipeline**: Skip completed stages and resume from any point
- 🔧 **Configuration-Driven**: All settings controlled via YAML configuration
- 💾 **Memory Efficient**: CPU-only processing with intelligent batching

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/mamopublic/diffusion-bunny.git
cd diffusion-bunny

# Create conda environment with Python 3.10
conda create -p /c/Users/mamop/miniconda3/envs/diffusion python=3.10

# Activate the environment
source activate /c/Users/mamop/miniconda3/envs/diffusion

# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys and settings
```

### 2. Prepare Input Data

```bash
# Place your movie file
mkdir -p data/input
cp your_movie.mp4 data/input/movie.mp4

# Add character reference images
mkdir -p data/input/characters/character1
cp character1_ref1.jpg data/input/characters/character1/
cp character1_ref2.jpg data/input/characters/character1/
```

### 3. Configure Pipeline

Edit `config.yaml` to customize:
- Input video path and character references
- Processing parameters for each stage
- Caption generation settings
- Model training parameters

### 4. Run Pipeline

```bash
# Run complete pipeline
python src/pipeline.py

# Run specific stages
python src/pipeline.py --stages extraction,filtering

# Resume from specific stage
python src/pipeline.py --resume-from detection
```

## Configuration

The pipeline is controlled via `config.yaml`. Key sections:

### Input Configuration
```yaml
input:
  video_path: "./data/input/movie.mp4"
  character_references_dir: "./data/input/characters"
```

### Stage Configuration
```yaml
extraction:
  method: "keyframe"  # or "interval"
  interval_seconds: 2.0
  max_frames: 1000

filtering:
  blur_threshold: 0.3
  batch_size: 10
  keep_best_per_batch: true

detection:
  similarity_threshold: 0.6
  max_faces_per_frame: 5

captioning:
  provider: "bedrock"  # bedrock, openai, anthropic, local
  model: "claude-3-haiku"
  batch_size: 5
```

## Pipeline Stages

### Stage 1: Frame Extraction
- Extracts frames using OpenCV
- Supports keyframe detection or interval-based extraction
- Generates metadata JSON with timestamps and source information

### Stage 2: Quality Filtering
- Applies Laplacian variance for blur detection
- Processes frames in configurable batches
- Keeps best frame per batch based on quality scores

### Stage 3: Character Detection
- Detects faces using face_recognition library
- Matches against character reference images
- Enriches metadata with character labels and confidence scores

### Stage 4: Caption Generation
- Generates descriptive captions using AI services
- Supports multiple providers with fallback options
- Batch processing for efficiency

### Stage 5: Fine-tuning
- Implements LoRA and DreamBooth training
- Uses Hugging Face diffusers and PEFT libraries
- Saves adapters for efficient storage

### Stage 6: Inference
- Loads fine-tuned models and adapters
- Generates images from text prompts
- Configurable generation parameters

## Project Structure

```
diffusion-bunny/
├── src/                     # Source code
│   ├── pipeline.py          # Main pipeline orchestrator
│   ├── utils.py            # Common utilities
│   ├── stage1_extraction/  # Frame extraction
│   ├── stage2_filtering/   # Quality filtering
│   ├── stage3_detection/   # Character detection
│   ├── stage4_captioning/  # Caption generation
│   ├── stage5_finetuning/  # Model fine-tuning
│   └── stage6_inference/   # Model inference
├── config.yaml            # Main configuration
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── data/                  # Data directories
│   ├── input/            # Input movies and references
│   ├── output/           # Pipeline outputs
│   └── models/           # Trained models
├── docs/                 # Documentation
├── examples/             # Example notebooks
├── tests/                # Test suite
└── memory-bank/          # Project documentation
```

## Requirements

### Hardware
- **CPU**: Multi-core processor recommended
- **Memory**: 8GB+ RAM for video processing
- **Storage**: Variable based on video size and frame extraction
- **GPU**: Optional for fine-tuning (can use external services)

### Software
- **Python**: 3.8+
- **Operating System**: Windows, Linux, macOS
- **Dependencies**: See `requirements.txt`

### External Services
- **AWS Bedrock**: For caption generation (optional)
- **OpenAI API**: Alternative caption generation (optional)
- **Anthropic API**: Alternative caption generation (optional)

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Type Checking
```bash
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Face Detection Alternatives

The project currently uses **OpenCV DNN** for face detection in Stage 3 (Character Detection). Here are the available options:

### Current Implementation: OpenCV DNN
- ✅ Already included with opencv-python
- ✅ Fast CPU performance
- ✅ No additional dependencies
- ✅ Good accuracy for most use cases
- ❌ May need separate embedding solution for character matching

### Alternative Options

#### MediaPipe Face Detection
```bash
pip install mediapipe>=0.10.0
```
- ✅ Modern neural network, better accuracy
- ✅ CPU optimized
- ✅ Includes facial landmarks
- ❌ Larger memory footprint

#### dlib + face_recognition
```bash
pip install dlib>=19.24.0 face_recognition>=1.3.0
```
- ✅ Excellent accuracy and face embeddings
- ✅ Built-in character matching capabilities
- ❌ Difficult installation (requires C++ compiler)
- ❌ May fail on some systems

To switch face detection methods, modify the `detection.face_detection_model` setting in `config.yaml`.

## Acknowledgments

- OpenCV for video processing and face detection capabilities
- Hugging Face for diffusion model infrastructure
- AWS Bedrock for caption generation services

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review example notebooks in `examples/`

---

**Note**: This is a proof-of-concept project. Use responsibly and ensure you have appropriate rights to process video content.
