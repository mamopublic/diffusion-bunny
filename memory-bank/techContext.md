# Technical Context - Diffusion Bunny

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary development language
- **OpenCV**: Video processing and frame extraction
- **NumPy**: Numerical computations and array operations
- **Pillow (PIL)**: Image processing and manipulation
- **PyYAML**: Configuration file parsing
- **JSON**: Metadata storage and interchange

### Computer Vision & ML
- **OpenCV**: Video decoding, frame extraction, blur detection
- **face_recognition**: Face detection and encoding (dlib-based)
- **scikit-image**: Image quality metrics and analysis
- **MediaPipe**: Alternative face detection option
- **MTCNN**: Multi-task CNN for face detection (optional)

### Diffusion Model Components
- **Diffusers**: Hugging Face diffusion models library
- **Transformers**: Model loading and tokenization
- **Accelerate**: Training optimization and device management
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA implementation)
- **Torch**: PyTorch for model operations

### External Services
- **AWS Bedrock**: Caption generation (primary option)
- **OpenAI API**: Alternative caption generation
- **Anthropic Claude**: Alternative caption generation
- **Local LLMs**: Offline caption generation options

### Development Tools
- **Conda**: Environment management
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting
- **mypy**: Type checking

## Dependencies & Requirements

### Core Dependencies
```yaml
# Core processing
opencv-python>=4.8.0
numpy>=1.21.0
Pillow>=9.0.0
PyYAML>=6.0

# Computer vision
face-recognition>=1.3.0
scikit-image>=0.19.0
mediapipe>=0.10.0  # optional

# ML/AI
torch>=2.0.0
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
peft>=0.4.0

# Utilities
tqdm>=4.64.0
click>=8.0.0
python-dotenv>=1.0.0
```

### Optional Dependencies
```yaml
# AWS integration
boto3>=1.26.0
botocore>=1.29.0

# Alternative APIs
openai>=1.0.0
anthropic>=0.3.0

# Development
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=1.0.0
```

## Development Setup

### Environment Configuration
- **Conda Environment**: Isolated Python environment per project
- **Environment Variables**: Stored in `.env` file for sensitive data
- **Configuration**: `config.yaml` for project settings
- **Platform**: Windows 11 primary, cross-platform compatible

### Directory Structure
```
diffusion-bunny/
├── environment.yml          # Conda environment specification
├── requirements.txt         # Pip requirements
├── config.yaml             # Main configuration
├── .env.example            # Environment variables template
├── .env                    # Local environment variables (gitignored)
├── src/                    # Source code
├── docs/                   # Documentation
├── examples/               # Example notebooks and scripts
├── tests/                  # Test suite
├── data/                   # Data directories (gitignored)
│   ├── input/              # Input movies and references
│   ├── output/             # Pipeline outputs
│   └── models/             # Trained models
└── memory-bank/           # Project memory and context
```

## Technical Constraints

### Hardware Limitations
- **No GPU**: Processing stages must run on CPU only
- **Memory**: Efficient batch processing for large video files
- **Storage**: Minimize disk usage through metadata references
- **Network**: Handle API rate limits and timeouts gracefully

### Performance Requirements
- **Video Processing**: Handle HD video files efficiently
- **Batch Processing**: Configurable batch sizes for memory management
- **Resumability**: Fast startup and state recovery
- **Scalability**: Process multiple movies in sequence

### Compatibility Requirements
- **Python Versions**: Support Python 3.8+
- **Operating Systems**: Windows primary, Linux/macOS compatible
- **Video Formats**: Support common formats (MP4, AVI, MOV, MKV)
- **Image Formats**: JPEG, PNG output formats

## Configuration Management

### Configuration Hierarchy
1. **Default Configuration**: Built-in defaults in code
2. **Project Configuration**: `config.yaml` in project root
3. **Environment Variables**: Override sensitive settings
4. **Command Line Arguments**: Runtime parameter overrides

### Configuration Schema
```yaml
# Project settings
project:
  name: "diffusion-bunny"
  version: "0.1.0"
  data_dir: "./data"

# Pipeline configuration
pipeline:
  stages: ["extraction", "filtering", "detection", "captioning"]
  force_rerun: false
  parallel_processing: true

# Stage-specific settings
extraction:
  method: "keyframe"  # keyframe | interval
  interval_seconds: 2.0
  output_format: "jpg"
  quality: 95

filtering:
  blur_threshold: 0.3
  batch_size: 10
  keep_best_per_batch: true

detection:
  face_detection_model: "hog"  # hog | cnn | mediapipe
  similarity_threshold: 0.6
  max_faces_per_frame: 5

captioning:
  provider: "bedrock"  # bedrock | openai | anthropic | local
  model: "claude-3-haiku"
  max_tokens: 100
  batch_size: 5

# External service configuration
aws:
  region: "us-east-1"
  profile: "default"

# Model configuration
models:
  base_model: "runwayml/stable-diffusion-v1-5"
  lora_rank: 16
  learning_rate: 1e-4
  training_steps: 1000
```

## Tool Usage Patterns

### Video Processing
- **OpenCV**: Primary tool for video decoding and frame extraction
- **FFmpeg**: Alternative for complex video operations (via subprocess)
- **Batch Processing**: Process frames in configurable batches

### Face Detection
- **face_recognition**: Primary face detection and encoding
- **MediaPipe**: Alternative for faster processing
- **Caching**: Store face embeddings to avoid recomputation

### API Integration
- **Retry Logic**: Exponential backoff for API failures
- **Rate Limiting**: Respect API rate limits
- **Fallback Options**: Multiple caption generation providers

### Model Training
- **Jupyter Notebooks**: Initial training experiments
- **Python Scripts**: Production training pipelines
- **Checkpointing**: Save training progress for resumability

## Security Considerations

### API Key Management
- Store sensitive keys in `.env` file
- Use environment variables in production
- Rotate keys regularly
- Implement key validation

### Data Privacy
- Process data locally when possible
- Secure API communications (HTTPS)
- Clean up temporary files
- Respect data retention policies

### Input Validation
- Validate video file formats and sizes
- Sanitize file paths and names
- Validate configuration parameters
- Handle malformed input gracefully

## Performance Optimization

### Memory Management
- Stream video processing to avoid loading entire files
- Use generators for large datasets
- Implement garbage collection hints
- Monitor memory usage during processing

### I/O Optimization
- Minimize file system operations
- Use efficient image formats
- Batch file operations
- Implement progress tracking

### Compute Optimization
- Leverage NumPy vectorization
- Use multiprocessing for CPU-bound tasks
- Cache expensive computations
- Profile bottlenecks regularly
