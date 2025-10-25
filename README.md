# Diffusion Bunny 🐰✨

A learning project exploring various techniques for building a Stable Diffusion pipeline on CPU-only hardware. This project documents the exploration of face recognition approaches, from traditional computer vision to neural networks to LLM-based analysis, in an attempt to create character-specific models from video content.

**Note**: This is an exploration and learning project, not a production-ready system. While the pipeline functions, its accuracy is limited due to several factors discussed below. The real value lies in the techniques explored and lessons learned about what's possible with CPU-only processing.

## Project Journey & Lessons Learned

This project began as an exploration: having successfully built stable diffusion pipelines on GPU machines, I became curious about what could be accomplished on CPU-only hardware - what techniques would work, what would fail, and what could be learned in the process.

### The Path Through Different Approaches

I tried several face recognition approaches, each with its own challenges:

**Traditional Computer Vision (Haar/SIFT Features)**: Started with classical approaches using Haar cascades and SIFT feature detection. These performed poorly on the anime/character content, struggling with the stylized nature of animated faces and varying art styles.

**Siamese Neural Networks**: Moved to more modern approaches, implementing Siamese networks with MobileNetV2 backbones for character recognition. While technically functional, these also failed to achieve satisfactory accuracy on the target content.

**LLM-Based Analysis (Claude Haiku)**: Eventually settled on using Claude Haiku for character recognition and scene analysis. This approach worked marginally better than the previous methods, providing more contextual understanding of scenes and characters.

### What Went Wrong (And Why)

The stable diffusion fine-tuning process ultimately failed, but for instructive reasons:

- **Insufficient Data**: The amount of training data extracted was simply not enough for effective fine-tuning
- **Lack of Diversity**: The dataset lacked the variety needed for robust model training
- **Data Quality Issues**: Most shots were expansive scenes with small character faces, making character recognition and extraction challenging
- **Need for Better Preprocessing**: The data required more sophisticated cropping and preprocessing to focus on character features

### What Worked

Despite the challenges, several aspects proved successful:

- **Pipeline Architecture**: The modular, stage-based approach allows for easy experimentation and iteration
- **CPU Processing**: Demonstrated that meaningful video processing and analysis can be done on CPU-only hardware
- **Technique Exploration**: Served as an excellent playground for comparing different face recognition approaches
- **Automatic Labeling**: The LLM-based labeling system, while not perfect, showed promise for automated dataset creation

The project went through various approaches, with failures providing as much insight as successes. The theme throughout was simple: what can we do on CPU, and what can we learn in the process?

## Overview

Diffusion Bunny automates the complete workflow from raw video files to character-specific diffusion models:

0. **Siamese Network Training**: Train a specialized neural network for character recognition using reference images
1. **Frame Extraction**: Extract keyframes or interval-based frames from movies using multiple sampling methods
2. **Quality Filtering & Deduplication**: Remove blurry or low-quality frames and eliminate near-duplicates using intelligent filtering
3. **Character Detection**: Detect faces and identify characters using advanced computer vision or trained Siamese networks
4. **LLM Analysis**: Generate comprehensive scene descriptions and character identification using large language models
5. **LoRA Fine-tuning**: Train efficient LoRA adapters for Stable Diffusion models using processed data
6. **Inference**: Generate new character-specific images with the fine-tuned models

## Features

- 🎬 **Video Processing**: Support for MP4, AVI, MOV, MKV formats
- 🔍 **Smart Filtering**: Blur detection and quality assessment
- 👤 **Character Recognition**: Face detection and matching against reference images
- 📝 **LLM Analysis**: Character identification and scene captioning via OpenRouter API
- 🎯 **Efficient Training**: LoRA and DreamBooth fine-tuning
- ⚡ **Resumable Pipeline**: Skip completed stages and resume from any point
- 🔧 **Configuration-Driven**: All settings controlled via YAML configuration
- 💾 **Memory Efficient**: CPU-only processing with intelligent batching

## Technical Details

Diffusion Bunny is a six-stage pipeline that transforms raw video content into fine-tuned character-specific diffusion models. Each stage processes data sequentially from video analysis to AI model generation. The pipeline supports multiple detection methods, processing approaches, and hardware configurations.

### Stage 0: Siamese Network Training

The pipeline begins with an optional character recognition stage that trains a Siamese neural network for anime character identification. This stage uses a MobileNetV2 backbone architecture to create character embeddings that distinguish between different characters. The network employs contrastive or triplet loss functions to learn representations from character reference images. The Siamese approach generates 128-dimensional normalized embeddings that enable character matching using cosine similarity or Euclidean distance metrics. This trained model provides the foundation for character recognition in later stages.

Training occurs through generated positive and negative pairs from character reference images, with data augmentation including rotation, brightness adjustment, and spatial transformations. The system supports both nested directory structures (characters/name/images) and flat file organization (characters/name.jpg), adapting to different data organization patterns. Upon completion, the stage outputs trained model weights and a character embedding database that can be cached for inference during character detection.

### Stage 1: Frame Extraction

The frame extraction stage processes input video files using OpenCV to generate candidate frames for analysis. The system supports three extraction methods: uniform sampling for temporal coverage, interval-based extraction for regular time-spaced frames, and keyframe detection using frame difference analysis with configurable thresholds. Each method serves different use cases - uniform sampling ensures coverage across the video, interval sampling provides predictable temporal spacing, and keyframe detection focuses on scene changes and visual transitions.

The extraction process handles multiple video formats (MP4, AVI, MOV, MKV) and includes frame preprocessing with optional resizing and quality-controlled JPEG compression. All extracted frames are accompanied by metadata including timestamps, frame numbers, source video information, and extraction parameters. This metadata enables tracking and reproducibility throughout the pipeline. The system can handle large video files by streaming processing and provides configurable limits to manage storage requirements while maintaining temporal representation.

### Stage 2: Quality Filtering & Deduplication

The quality filtering stage implements a multi-metric assessment system to identify and retain quality frames from the extraction stage. The primary quality metrics include Laplacian variance blur detection (identifying sharp, well-focused frames), brightness analysis to eliminate over/under-exposed content, and contrast measurement to ensure visual richness. The system processes frames in configurable batches, optionally keeping only the best frame per batch to reduce dataset size while maintaining quality distribution.

The deduplication subsystem uses perceptual hashing (pHash) to identify and remove near-duplicate frames that commonly occur in video content. The implementation supports configurable similarity thresholds, hash precision levels, and group size requirements for deduplication decisions. When duplicates are detected, the system retains the highest-quality frame based on blur scores and other metrics. This two-stage approach (quality filtering followed by deduplication) reduces dataset size while preserving visual diversity and quality.

### Stage 3: Character Detection

The character detection stage combines face detection with character recognition to identify and label characters within filtered frames. The system supports multiple face detection approaches: traditional Haar cascade classifiers for compatibility, and YOLOv8-based anime face detection for animated content. The YOLOv8 implementation uses models trained on anime faces, downloaded from Hugging Face repositories, and supports both CPU and GPU inference with configurable confidence thresholds.

Character recognition offers two approaches: traditional computer vision using ORB, SIFT, or SURF feature descriptors with feature matching algorithms, and the Siamese network approach trained in Stage 0. The Siamese method uses learned embeddings rather than hand-crafted features for anime characters. Recognition results include confidence scores, character names, and bounding box coordinates. The system generates multiple output formats including detected face crops, annotated frames with character labels, and JSON metadata. Color-coded bounding boxes and character visualizations aid in result inspection and pipeline debugging.

### Stage 4: LLM Analysis

The LLM analysis stage combines local face detection results with large language model capabilities for character identification and scene description. The system creates composite images that merge detected faces with character reference strips, providing visual context for LLM analysis. These composites are processed through OpenRouter API integration, supporting vision-language models including Claude and GPT-4 Vision.

The LLM analysis generates three outputs: character identification (cross-referencing detected faces with known characters), scene descriptions (capturing setting, poses, clothing, and interactions), and Stable Diffusion-optimized captions (formatted for AI art generation). The system includes caching mechanisms to avoid duplicate API calls, configurable confidence thresholds for processing decisions, and error handling with fallbacks. Rate limiting and batch processing manage API usage while maintaining cost control. Results are stored with traceability including raw LLM responses, processing times, and confidence metrics.

### Stage 5: LoRA Fine-tuning

The final stage implements Low-Rank Adaptation (LoRA) fine-tuning for Stable Diffusion models, creating character-specific AI art generators from the processed video content. The implementation uses the PEFT (Parameter-Efficient Fine-Tuning) library to add trainable adapter layers to the UNet component while keeping the text encoder frozen, reducing computational requirements and memory usage. The system prepares training data from LLM analysis results, creating DreamBooth-compatible datasets with character-specific captions and instance images.

Training occurs on CPU with optimizations including gradient checkpointing, mixed precision training, and data loading pipelines. The LoRA configuration uses rank-8 adapters targeting specific UNet modules (attention layers), balancing adaptation capability and computational efficiency. The training process includes checkpoint saving, validation image generation, and logging. Final outputs include trained LoRA weights compatible with inference tools, enabling generation of character-specific artwork that maintains consistency with the source video content. The stage is designed as a self-contained process that can resume from previous pipeline stages and adapt to different hardware configurations.

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
# Edit .env with your OpenRouter API key for Stage 4 LLM Analysis
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
  project_root: "./pipeline_data/sprite"
  movie_dir: "movie"
  frames_dir: "frames"
  character_references_dir: "characters"
```

### Stage Configuration

#### Stage 0: Siamese Network Training
```yaml
siamese:
  enabled: true
  backbone: "mobilenetv2"
  embedding_dim: 128
  training:
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
    loss_type: "contrastive"
    pairs_per_epoch: 1000
```

#### Stage 1: Frame Extraction
```yaml
extraction:
  method: "uniform"  # uniform, interval, keyframe
  target_frames: 4000
  interval_seconds: 2.0
  keyframe_threshold: 0.3
  output_format: "jpg"
  quality: 95
```

#### Stage 2: Quality Filtering & Deduplication
```yaml
filtering:
  blur_threshold: 50.0
  batch_size: 10
  keep_best_per_batch: false
  brightness_threshold: [10, 245]
  contrast_threshold: 0.05
  deduplication:
    enabled: true
    similarity_threshold: 0.95
```

#### Stage 3: Character Detection
```yaml
detection:
  face_detection_method: "yolo_anime"  # haar, yolo_anime
  feature_method: "siamese"  # orb, sift, surf, siamese
  similarity_threshold: 0.8
  max_faces_per_frame: 15
  save_detected_faces: true
```

#### Stage 4: LLM Analysis
```yaml
llm_analysis:
  enabled: true
  model: "anthropic/claude-3-haiku"
  max_tokens: 500
  batch_size: 5
  process_frames_with_faces_only: true
  enable_caching: true
```

#### Stage 5: LoRA Fine-tuning
```yaml
finetuning:
  base_model: "stable-diffusion-v1-5/stable-diffusion-v1-5"
  method: "lora"
  device: "cpu"
  lora:
    rank: 8
    alpha: 16
  training:
    learning_rate: 5.0e-5
    batch_size: 1
    max_train_steps: 500
```

## Pipeline Stages

### Stage 0: Siamese Network Training
- **Input**: Character reference images
- **Process**: Train neural network for character recognition
- **Output**: Model weights and character embedding database

### Stage 1: Frame Extraction
- **Input**: Video files (MP4, AVI, MOV, MKV)
- **Process**: Extract frames at uniform intervals, keyframes, or time-based sampling
- **Output**: Frame images with metadata (timestamps, frame numbers)

### Stage 2: Quality Filtering & Deduplication
- **Input**: Extracted frame images
- **Process**: Filter by blur, brightness, contrast; remove near-duplicates
- **Output**: Quality-filtered unique frames

### Stage 3: Character Detection
- **Input**: Quality-filtered frames + character reference images
- **Process**: Detect faces and identify characters
- **Output**: Frames with character labels, face crops, bounding boxes

### Stage 4: LLM Analysis
- **Input**: Frames with detected characters
- **Process**: Generate scene descriptions and character identification via LLM
- **Output**: Character IDs, scene descriptions, Stable Diffusion captions

### Stage 5: LoRA Fine-tuning
- **Input**: LLM captions and character-labeled images
- **Process**: Train LoRA adapters for Stable Diffusion
- **Output**: Character-specific LoRA weights for image generation

## Project Structure

```
diffusion-bunny/
├── src/                     # Source code
│   ├── pipeline.py          # Main pipeline orchestrator
│   ├── stage0_siamese/      # Siamese network training
│   ├── stage1_extraction/   # Frame extraction
│   ├── stage2_filtering/    # Quality filtering
│   ├── stage3_detection/    # Character detection
│   ├── stage4_llm_analysis/ # LLM analysis and captioning
│   └── stage5_finetuning/   # LoRA fine-tuning
├── pipeline_data/           # Project-specific data
│   └── [project_name]/      # e.g., sprite/
│       ├── movie/           # Source video files
│       ├── frames/          # Extracted frames
│       ├── characters/      # Character reference images
│       ├── siamese/         # Trained Siamese models
│       └── strip_data/      # Character reference strips
├── outputs/                 # Pipeline run outputs
│   └── [run_timestamp]/     # e.g., run_20250902_165141/
│       ├── filtering/       # Quality filtering results
│       ├── detection/       # Character detection results
│       ├── llm_analysis/    # LLM analysis results
│       └── lora_weights/    # Trained LoRA weights
├── scripts/                 # Utility scripts
├── memory-bank/             # Project documentation
├── config.yaml             # Main configuration
├── requirements.txt         # Python dependencies
└── .env.example             # Environment variables template
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
- **OpenRouter API**: For LLM-based character identification and scene analysis (required for Stage 4)

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
- OpenRouter for LLM API access and integration

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review example notebooks in `examples/`

---

**Note**: This is a learning and exploration project that documents various approaches to CPU-based diffusion pipeline development. While not immediately successful in its original goals, it serves as a valuable resource for understanding the challenges and techniques involved in character recognition, automated labeling, and diffusion model fine-tuning on constrained hardware. Use responsibly and ensure you have appropriate rights to process video content.
