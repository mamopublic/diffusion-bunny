# Diffusion Bunny 🐰✨

An end-to-end pipeline for automatically constructing character-specific LoRA fine-tuning datasets from video, using a hybrid **Siamese network / YOLOv8 detection + vision-LLM auto-labeling** strategy, designed for CPU-constrained environments.

The central question driving this project: *Can a fully automated pipeline — no manual labeling, no GPU at preprocessing time — produce a usable DreamBooth-style training dataset from raw video footage?*

## Experimental Results

The pipeline was run end-to-end on anime video content, producing **65 frames** that passed quality filtering and character detection thresholds, which then served as the LoRA training set.

### What Worked

- **Detection pipeline**: YOLOv8 (anime-domain) + Siamese recognizer consistently out-performed the ORB/SIFT/SURF baseline — substantially fewer false-positive character matches on stylized art. The composite-image LLM strategy (face crops + reference strip sent together) further reduced hallucination on ambiguous crops.
- **Auto-labeling quality**: Claude Haiku's character identification was meaningfully better than both the Siamese alone and raw frame analysis, particularly for partially-occluded faces and wide establishing shots.
- **Pipeline engineering**: Resumable stage execution, timestamped run directories, config provenance copying — all functioned as designed.

### What Didn't Work (and Why)

| Method | Result | Root Cause |
|---|---|---|
| Haar cascade | Poor recall on anime faces | Haar features tuned for photorealistic faces; stylized geometry doesn't generalize |
| ORB/SIFT feature matching | High false-positive rate | Low-texture anime art produces unstable keypoints; match-count thresholds difficult to tune |
| Siamese network | Functional but limited | 128-dim embeddings distinguish major characters but struggle with same-character pose variation |
| LoRA fine-tuning (65 frames) | Insufficient character consistency | Dataset size was the binding constraint — DreamBooth-style training typically requires 200–500+ images |

**Core finding**: The *pipeline* works — it can go from raw video to a labeled dataset automatically. The *dataset* it produced (65 frames, limited pose diversity, expansive scenes with small faces) was insufficient for reliable fine-tuning. The bottleneck is upstream data quality and volume, not the auto-labeling method.

### Design Decisions

**Why Siamese network over ORB/SIFT/SURF for character recognition?**

Traditional feature descriptors fail on anime-style art because: (1) stylized faces have low texture, making keypoint detection unstable; (2) the color palette and line-art rendering make descriptor distances uninformative across different scenes. ORB in particular is designed for geometric repeatability — which anime faces don't have across lighting and art-style variation. The Siamese approach learns an embedding space directly from character reference images, using contrastive/triplet loss to separate character identity regardless of pose. It's still imperfect (same-character variation is a real challenge), which is why the LLM fallback (Stage 4) exists.

**Why the composite-image LLM strategy?**

Sending raw frames to a VLM risks hallucinated character names — the model may identify characters it "knows" from its training data rather than from the actual reference set. Compositing detected face crops alongside a visual reference strip of candidate characters grounds the identification problem: the LLM is explicitly shown who the candidates are and asked to match, not freely identify. This is a constrained visual comparison task rather than open-ended recognition, which substantially reduces hallucination on domain-specific characters.

**Why run both Siamese recognition (Stage 3) and LLM recognition (Stage 4)?**

They are sequential, not alternatives, and they don't do the same job. Stage 3 does two things: (1) detect face locations via YOLOv8 and (2) attempt a fast, offline character identity match via Siamese embeddings. Stage 4 takes those face crops and independently re-identifies characters using the LLM with a reference strip — and, critically, also generates the Stable Diffusion training captions that Stage 5 needs. The Siamese output is preserved alongside the LLM output for comparison, but the LLM identification is what becomes the training label. The Siamese is not a prerequisite for Stage 4 to work — Stage 4 would function with just the face locations from YOLOv8. What Siamese adds is a cheap, immediate first-pass identity signal that runs without an API call, useful for inspection and for cases where LLM access isn't available.

**Why LoRA on UNet only (frozen text encoder)?**

With ~65 training frames, fine-tuning the text encoder risks catastrophic forgetting — the model may corrupt its learned concept space for common tokens. Rank-8 UNet LoRA targeting attention layers gives character-specific weight updates at minimal parameter cost (~3M trainable vs ~860M base), sufficient for style anchoring with a small dataset. The text encoder is intentionally left frozen to preserve compositional generation capability.

## Overview

Diffusion Bunny automates the complete workflow from raw video files to character-specific diffusion models:

0. **Siamese Network Training**: Train a specialized neural network for character recognition using reference images
1. **Frame Extraction**: Extract keyframes or interval-based frames from movies using multiple sampling methods
2. **Quality Filtering & Deduplication**: Remove blurry or low-quality frames and eliminate near-duplicates using perceptual hashing
3. **Character Detection**: Detect faces and identify characters using YOLOv8 (anime) or Haar cascade, with Siamese or feature-based recognition
4. **LLM Analysis**: Generate scene descriptions and character identification via vision-language model (composite-image strategy)
5. **LoRA Fine-tuning**: Train efficient LoRA adapters for Stable Diffusion models using processed data
6. **Inference**: Generate new character-specific images with the fine-tuned models

## Features

- 🎬 **Video Processing**: Support for MP4, AVI, MOV, MKV formats
- 🔍 **Smart Filtering**: Laplacian blur detection, brightness/contrast assessment, pHash deduplication
- 👤 **Character Recognition**: YOLOv8 anime face detection + Siamese embedding matching against reference images
- 📝 **LLM Auto-labeling**: VLM-based character ID and SD-caption generation via OpenRouter API (composite-image strategy)
- 🎯 **LoRA Fine-tuning**: UNet-only LoRA with PEFT, DreamBooth-style dataset preparation
- ⚡ **Resumable Pipeline**: Timestamped run directories, per-stage idempotency checks, resume from any checkpoint
- 🔧 **Configuration-Driven**: All parameters controlled via `config.yaml`, config copied to each run directory for provenance
- 💾 **CPU-First Design**: All preprocessing stages run without GPU; fine-tuning offloadable to Colab/Kaggle

## Technical Details

Diffusion Bunny is a six-stage pipeline that transforms raw video content into fine-tuned character-specific diffusion models. Each stage processes data sequentially and writes structured JSON artifacts that downstream stages consume.

### Stage 0: Siamese Network Training

The pipeline begins with an optional character recognition stage that trains a Siamese neural network for anime character identification. This stage uses a MobileNetV2 backbone architecture to create character embeddings that distinguish between different characters. The network employs contrastive or triplet loss functions to learn representations from character reference images. The Siamese approach generates 128-dimensional normalized embeddings that enable character matching using cosine similarity or Euclidean distance metrics. This trained model provides the foundation for character recognition in later stages.

Training occurs through generated positive and negative pairs from character reference images, with data augmentation including rotation, brightness adjustment, and spatial transformations. The system supports both nested directory structures (characters/name/images) and flat file organization (characters/name.jpg), adapting to different data organization patterns. Upon completion, the stage outputs trained model weights and a character embedding database that can be cached for inference during character detection.

### Stage 1: Frame Extraction

The frame extraction stage processes input video files using OpenCV to generate candidate frames for analysis. The system supports three extraction methods: uniform sampling for temporal coverage, interval-based extraction for regular time-spaced frames, and keyframe detection using frame difference analysis with configurable thresholds. Each method serves different use cases — uniform sampling ensures coverage across the video, interval sampling provides predictable temporal spacing, and keyframe detection focuses on scene changes and visual transitions.

The extraction process handles multiple video formats (MP4, AVI, MOV, MKV) and includes frame preprocessing with optional resizing and quality-controlled JPEG compression. All extracted frames are accompanied by metadata including timestamps, frame numbers, source video information, and extraction parameters. This metadata enables tracking and reproducibility throughout the pipeline.

### Stage 2: Quality Filtering & Deduplication

The quality filtering stage implements a multi-metric assessment system to identify and retain quality frames from the extraction stage. The primary quality metrics include Laplacian variance blur detection (identifying sharp, well-focused frames), brightness analysis to eliminate over/under-exposed content, and contrast measurement to ensure visual richness. The system processes frames in configurable batches, optionally keeping only the best frame per batch to reduce dataset size while maintaining quality distribution.

The deduplication subsystem uses perceptual hashing (pHash) to identify and remove near-duplicate frames that commonly occur in video content. The implementation supports configurable similarity thresholds, hash precision levels, and group size requirements for deduplication decisions. When duplicates are detected, the system retains the highest-quality frame based on blur scores and other metrics.

### Stage 3: Character Detection

The character detection stage combines face detection with character recognition to identify and label characters within filtered frames. The system supports multiple face detection approaches: traditional Haar cascade classifiers for compatibility, and YOLOv8-based anime face detection for animated content. The YOLOv8 implementation uses models trained on anime faces, downloaded from Hugging Face repositories, and supports both CPU and GPU inference with configurable confidence thresholds.

Character recognition offers two approaches: traditional computer vision using ORB, SIFT, or SURF feature descriptors, and the Siamese network approach trained in Stage 0. The Siamese method uses learned embeddings rather than hand-crafted features for anime characters. Recognition results include confidence scores, character names, and bounding box coordinates. Color-coded bounding boxes and character visualizations aid in result inspection and pipeline debugging.

### Stage 4: LLM Analysis

The LLM analysis stage combines local face detection results with large language model capabilities for character identification and scene description. The system creates **composite images** that merge detected face crops with character reference strips, providing visual context for LLM analysis. These composites are processed through OpenRouter API integration, supporting vision-language models including Claude and GPT-4 Vision.

The LLM analysis generates three outputs: character identification (constrained to the reference set shown in the composite), scene descriptions (capturing setting, poses, clothing, and interactions), and Stable Diffusion-optimized captions (formatted for training). The system includes caching mechanisms, configurable confidence thresholds, rate limiting, and batch processing for API cost control. Results are stored with traceability including raw LLM responses, processing times, and confidence metrics.

### Stage 5: LoRA Fine-tuning

The final stage implements Low-Rank Adaptation (LoRA) fine-tuning for Stable Diffusion models, creating character-specific AI art generators from the processed video content. The implementation uses the PEFT library to add trainable adapter layers to the UNet component while keeping the text encoder frozen. The system auto-discovers the most recent LLM analysis output and prepares a DreamBooth-compatible dataset, then runs on CPU with gradient checkpointing enabled.

The LoRA configuration uses rank-8 adapters targeting UNet attention layers, with `init_lora_weights="gaussian"`. The training process includes checkpoint saving at configurable intervals and config provenance copying to the run directory. Final outputs are PEFT-compatible LoRA weights (`adapter_config.json` + `adapter_model.bin`) ready for loading with standard diffusers tooling.

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/mamopublic/diffusion-bunny.git
cd diffusion-bunny

# Create conda environment
conda create -n diffusion-bunny python=3.10
conda activate diffusion-bunny

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

Edit `config.yaml` to set:
- Input video path and character references directory
- Extraction method and frame count targets
- Detection method (`yolo_anime` recommended for animation)
- LLM model and API settings
- LoRA rank, learning rate, training steps

### 4. Run Pipeline

```bash
# Run complete pipeline
python src/pipeline.py

# Run specific stages only
python src/pipeline.py --stages extraction,filtering

# Resume from a previous run's checkpoint
python src/pipeline.py --resume-from-dir outputs/run_20250902_165141

# Force rerun of all stages
python src/pipeline.py --force-rerun
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
  feature_method: "siamese"            # orb, sift, surf, siamese
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
- **Process**: Train MobileNetV2-backbone Siamese network with contrastive/triplet loss
- **Output**: `model_weights.pth`, `character_embeddings.pkl`

### Stage 1: Frame Extraction
- **Input**: Video files (MP4, AVI, MOV, MKV)
- **Process**: Extract frames at uniform intervals, keyframes, or time-based sampling
- **Output**: Frame images + `frames_metadata.json`

### Stage 2: Quality Filtering & Deduplication
- **Input**: Extracted frame images + metadata
- **Process**: Laplacian blur filter, brightness/contrast filter, pHash deduplication
- **Output**: `filtering/filtered.json`, quality-filtered frame set

### Stage 3: Character Detection
- **Input**: Filtered frames + character reference images
- **Process**: YOLOv8/Haar face detection + Siamese/feature-based character matching
- **Output**: `detection/detected.json` with bounding boxes, confidence scores, character labels

### Stage 4: LLM Analysis
- **Input**: Detected frames + character reference strips
- **Process**: Composite image creation → VLM character ID + scene captioning
- **Output**: `llm_analysis/llm_analysis.json` with character IDs, scene descriptions, SD captions

### Stage 5: LoRA Fine-tuning
- **Input**: LLM captions + character-labeled images
- **Process**: Data preparation → UNet-only LoRA training (text encoder frozen)
- **Output**: `lora_weights/final/unet/` — PEFT-compatible adapter weights

## Project Structure

```
diffusion-bunny/
├── src/                     # Source code
│   ├── pipeline.py          # Main pipeline orchestrator
│   ├── stage0_siamese/      # Siamese network training
│   ├── stage1_extraction/   # Frame extraction
│   ├── stage2_filtering/    # Quality filtering & deduplication
│   ├── stage3_detection/    # Character detection (YOLOv8 + Siamese)
│   ├── stage4_llm_analysis/ # Composite-image LLM auto-labeling
│   └── stage5_finetuning/   # LoRA fine-tuning
├── scripts/                 # Utility scripts
├── pipeline_data/           # Project-specific input data
│   └── [project_name]/      # e.g., sprite/
│       ├── movie/           # Source video files
│       ├── frames/          # Extracted frames
│       ├── characters/      # Character reference images
│       └── siamese/         # Trained Siamese model + embeddings
├── outputs/                 # Pipeline run outputs (timestamped)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── filtering/       # filtered.json
│       ├── detection/       # detected.json + face crops
│       ├── llm_analysis/    # llm_analysis.json + composites
│       ├── training_data/   # DreamBooth-format dataset
│       └── lora_weights/    # Trained LoRA adapter weights
├── config.yaml              # Main configuration (single source of truth)
├── requirements.txt         # Python dependencies
└── .env.example             # Environment variable template
```

## Requirements

### Hardware
- **CPU**: Multi-core processor recommended for preprocessing
- **Memory**: 8GB+ RAM for video processing
- **Storage**: Variable based on video size and frame extraction volume
- **GPU**: Optional for fine-tuning (pipeline designed to offload to Colab/Kaggle)

### Software
- **Python**: 3.10+
- **Operating System**: Windows, Linux, macOS

### External Services
- **OpenRouter API**: Required for Stage 4 LLM-based character identification and captioning

## Running Tests

```bash
pytest tests/
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with OpenCV, PyTorch, Hugging Face (PEFT, diffusers, ultralytics), and OpenRouter API.
