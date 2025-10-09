# Stage 4.5: LLM Analysis

Remote character identification and scene description using OpenRouter LLMs.

## Overview

This stage enhances the existing face detection pipeline by sending composite images to Large Language Models (LLMs) via OpenRouter for advanced character identification and scene analysis. It creates composite images with character reference strips alongside detected faces and uses structured prompts to get reliable character identification and scene descriptions.

## Features

- **Composite Image Creation**: Combines character reference strip with frame containing detected faces
- **OpenRouter Integration**: Uses cost-efficient models like Claude 3 Haiku for analysis
- **Smart Caching**: Avoids duplicate API calls for similar images
- **Rate Limiting**: Respects API rate limits to control costs
- **Batch Processing**: Processes multiple frames efficiently
- **Error Handling**: Robust error handling with retry mechanisms

## Architecture

```
Detection Results → Composite Creator → OpenRouter LLM → Analysis Results
                                    ↓
                            Character Strip + Frame
```

### Components

1. **CompositeImageCreator**: Creates composite images with character reference strips
2. **OpenRouterClient**: Handles API communication with rate limiting and caching
3. **LLMAnalyzer**: Main orchestrator that processes frames and manages the pipeline

## Configuration

Add to your `config.yaml`:

```yaml
llm_analysis:
  enabled: true
  model: "anthropic/claude-3-haiku"
  max_tokens: 300
  temperature: 0.7
  
  # Processing settings
  process_frames_with_faces_only: true
  min_face_confidence: 0.5
  batch_size: 5
  
  # Composite image settings
  strip_width: 150
  character_image_size: 150
  strip_background_color: [240, 240, 240]
  
  # Cost optimization
  enable_caching: true
  max_requests_per_minute: 30
  
  # OpenRouter API settings
  openrouter:
    api_key_env: "OPENROUTER_API_KEY"
    base_url: "https://openrouter.ai/api/v1"
    timeout: 30
    retry_attempts: 3
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   
   Copy `.env.example` to `.env` and add your OpenRouter API key:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and set:
   ```
   OPENROUTER_API_KEY=your_actual_openrouter_api_key_here
   ```
   
   The application will automatically load environment variables from the `.env` file using `python-dotenv`.

3. **Prepare Character Strip Images**:
   Place character reference images in `pipeline_data/sprite/strip_data/`:
   - `ellie.jpg`
   - `phil.jpg` 
   - `rex.jpg`
   - `victoria.jpg`

## Usage

### Standalone Usage

```python
from src.stage4_llm_analysis import LLMAnalyzer
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run LLM analysis
analyzer = LLMAnalyzer(config, "outputs/run_20250902_165141")
result = analyzer.run()

print(f"Analyzed {result['frames_analyzed']} frames")
print(f"Characters found: {result['character_appearances']}")
```

### Command Line Usage

```bash
python src/stage4_llm_analysis/llm_analyzer.py \
    --config config.yaml \
    --resume-from outputs/run_20250902_165141
```

### Testing Composite Creation

```bash
cd src/stage4_llm_analysis
python test_composite.py
```

## Output

The stage creates:

1. **Composite Images**: `{run_dir}/llm_analysis/composites/`
   - Character reference strip + annotated frame
   - Used for LLM analysis

2. **Analysis Results**: `{run_dir}/llm_analysis/llm_analysis.json`
   ```json
   {
     "frame_id": "frame_000699",
     "characters_identified": ["ellie", "phil"],
     "scene_description": "Two characters talking in a room",
     "confidence": 0.85,
     "processing_time": 2.3
   }
   ```

## Cost Optimization

- **Caching**: Identical images are cached to avoid duplicate API calls
- **Rate Limiting**: Respects OpenRouter rate limits (30 requests/minute default)
- **Smart Filtering**: Only processes frames with detected faces
- **Batch Processing**: Groups requests efficiently
- **Model Selection**: Uses cost-efficient models like Claude 3 Haiku
- **Debug Mode**: Test with limited frames before full run

### Estimated Costs

Per frame analysis:
- **Input**: ~15,200 tokens (composite image + prompt)
- **Output**: ~150 tokens (JSON response)
- **Cost per frame**: ~$0.004 (less than half a cent)

For a typical movie analysis:
- **Frames with faces**: ~181 frames (based on detection results)
- **Model**: Claude 3 Haiku ($0.25 per 1M input tokens, $1.25 per 1M output)
- **Estimated Cost**: ~$0.72 for 181 frames, ~$4 for 1000 frames

### Debug Mode for Cost-Effective Testing

Enable debug mode in `config.yaml` to test with limited frames:

```yaml
llm_analysis:
  debug_mode: true  # Enable debug mode
  debug_max_frames: 3  # Only process first 3 frames with faces
```

This allows you to:
- **Test the pipeline** with minimal cost (~$0.012 for 3 frames)
- **Verify API connectivity** and authentication
- **Check prompt effectiveness** before full run
- **Validate output format** and quality

Example debug run output:
```
🐛 DEBUG MODE: Limited to 3 frames (from 181 total)
Processing 3 frames in 1 batches of 5
```

**Remember to set `debug_mode: false` for production runs!**

## Integration with Pipeline

This stage integrates between Stage 3 (Detection) and Stage 4 (Captioning):

```
Stage 3: Detection → Stage 4.5: LLM Analysis → Stage 4: Captioning
```

The LLM analysis results can enhance the captioning stage by providing:
- Verified character identifications
- Scene context descriptions
- Confidence scores for filtering

## Error Handling

- **API Failures**: Automatic retries with exponential backoff
- **Rate Limiting**: Automatic waiting when limits are reached
- **Invalid Responses**: Fallback parsing for non-JSON responses
- **Missing Files**: Graceful handling of missing frames or character images

## Troubleshooting

### Common Issues

1. **API Key Not Found**:
   ```
   ValueError: OpenRouter API key not found in environment variable: OPENROUTER_API_KEY
   ```
   **Solution**: Set the environment variable with your OpenRouter API key

2. **Character Images Not Found**:
   ```
   FileNotFoundError: No character images found in pipeline_data/sprite/strip_data
   ```
   **Solution**: Ensure character reference images are in the correct directory

3. **Rate Limit Exceeded**:
   ```
   INFO: Rate limit reached, waiting 45.2 seconds
   ```
   **Solution**: This is normal - the system will automatically wait and continue

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- **Multiple Model Support**: Support for different LLM providers
- **Advanced Prompting**: More sophisticated prompt engineering
- **Confidence Thresholding**: Filter results by confidence scores
- **Batch Optimization**: More efficient batching strategies
- **Cost Tracking**: Detailed cost monitoring and reporting
