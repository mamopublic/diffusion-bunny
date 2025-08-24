# System Patterns - Diffusion Bunny

## Architecture Overview

### Pipeline Pattern
The system follows a **Linear Pipeline Architecture** where each stage processes data sequentially and passes enriched metadata to the next stage. This pattern ensures:
- Clear separation of concerns
- Easy debugging and monitoring
- Resumable execution at any stage
- Independent testing of components

### Data Flow Architecture
```
Movie File → Frame Extraction → Quality Filtering → Character Detection → Caption Generation → Fine-tuning → Inference
     ↓              ↓                ↓                    ↓                   ↓             ↓           ↓
  Raw Video    Frame Files +    Filtered Metadata +  Character Labels +   Captioned Data + Model Files + Generated Images
              Metadata JSON     Quality Scores      Face Embeddings     Training Captions   (LoRA/DB)
```

## Core Design Patterns

### 1. Stage Pattern
Each pipeline stage implements a consistent interface:
```python
class PipelineStage:
    def __init__(self, config: dict, input_dir: str, output_dir: str)
    def can_resume(self) -> bool
    def execute(self) -> StageResult
    def get_progress(self) -> float
```

**Benefits:**
- Uniform stage management
- Easy addition of new stages
- Consistent error handling
- Progress tracking across pipeline

### 2. Metadata Enrichment Pattern
Each stage enriches a central metadata JSON structure rather than copying files:
```json
{
  "frames": [
    {
      "id": "frame_001",
      "source_path": "extracted/frame_001.jpg",
      "timestamp": 12.5,
      "blur_score": 0.85,
      "characters": [{"name": "character1", "confidence": 0.92}],
      "caption": "A person standing in a room"
    }
  ]
}
```

**Benefits:**
- Minimal disk usage
- Fast processing
- Rich contextual information
- Easy filtering and querying

### 3. Configuration-Driven Pattern
All behavior controlled via hierarchical configuration:
```yaml
pipeline:
  stages: [extraction, filtering, detection, captioning]
  
extraction:
  method: keyframe  # or interval
  interval_seconds: 2.0
  
filtering:
  blur_threshold: 0.3
  batch_size: 10
```

**Benefits:**
- No code changes for different use cases
- Easy experimentation with parameters
- Reproducible runs
- Environment-specific configurations

### 4. Resumable Execution Pattern
Each stage checks for existing output and can skip or resume:
```python
def can_resume(self) -> bool:
    return os.path.exists(self.output_metadata_path)

def execute(self) -> StageResult:
    if self.can_resume() and not self.config.get('force_rerun'):
        return self.load_existing_result()
    return self.run_stage()
```

**Benefits:**
- Efficient development iteration
- Recovery from failures
- Partial pipeline execution
- Cost savings on expensive operations

## Component Relationships

### Core Components
- **Pipeline Orchestrator** (`pipeline.py`): Manages stage execution and flow
- **Stage Implementations**: Independent processing modules
- **Utilities** (`utils.py`): Shared functionality across stages
- **Configuration Manager**: Handles config loading and validation

### Data Dependencies
```
Stage 1 (Extraction) → metadata.json
Stage 2 (Filtering) → metadata.json + quality_scores
Stage 3 (Detection) → metadata.json + character_data
Stage 4 (Captioning) → metadata.json + captions
Stage 5 (Fine-tuning) → training_dataset
Stage 6 (Inference) → model_files
```

## Critical Implementation Paths

### 1. Frame Processing Path
```
Video Input → OpenCV Extraction → File Storage → Metadata Creation
```
- **Key Decision**: Store frames as individual files for random access
- **Performance**: Batch processing for memory efficiency
- **Quality**: Keyframe detection vs. interval sampling

### 2. Quality Assessment Path
```
Frame Files → Blur Detection → Batch Comparison → Quality Scoring → Filtering
```
- **Key Decision**: Laplacian variance for blur detection
- **Performance**: Process in configurable batches
- **Quality**: Keep best frame per batch, configurable thresholds

### 3. Character Recognition Path
```
Filtered Frames → Face Detection → Feature Extraction → Reference Matching → Labeling
```
- **Key Decision**: Use face embeddings for character matching
- **Performance**: Cache reference embeddings
- **Quality**: Confidence thresholds and multiple character support

### 4. Training Data Path
```
Labeled Frames → Caption Generation → Dataset Formatting → Model Training
```
- **Key Decision**: Flexible captioning backends (AWS Bedrock, local)
- **Performance**: Batch caption generation
- **Quality**: Structured prompts for consistent captions

## Error Handling Patterns

### Graceful Degradation
- Continue processing if individual frames fail
- Log errors with context for debugging
- Provide partial results when possible

### Retry Mechanisms
- Configurable retry counts for external services
- Exponential backoff for API calls
- Circuit breaker pattern for failing services

### Validation Patterns
- Input validation at stage boundaries
- Output verification before stage completion
- Configuration validation at startup

## Performance Optimization Patterns

### Memory Management
- Stream processing for large video files
- Configurable batch sizes based on available memory
- Lazy loading of frame data

### I/O Optimization
- Minimize file copying (use references)
- Batch file operations
- Asynchronous processing where possible

### Compute Optimization
- CPU-based processing for accessibility
- Parallel processing within stages
- Caching of expensive computations (embeddings, models)
