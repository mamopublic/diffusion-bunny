# Active Context - Diffusion Bunny

## Current Work Focus
**Stage 1 Implementation Phase**: Implemented frame extraction stage with new pipeline_data organization scheme. The project now has a working foundation with the first stage operational.

## Recent Changes
- **Pipeline Data Organization**: Implemented new directory structure
  - `pipeline_data/sprite/movie/`: Contains source video files
  - `pipeline_data/sprite/frames/`: Will contain extracted frames
  - Project-specific organization with configurable start points
- **Stage 1 Implementation**: Complete frame extraction functionality
  - `src/stage1_extraction/extractor.py`: Full OpenCV-based frame extraction
  - Keyframe detection and interval-based extraction methods
  - Comprehensive metadata generation and JSON output
- **Pipeline Orchestrator**: Basic pipeline.py with stage management
  - Resumable execution with stage completion detection
  - Command-line interface with stage selection
  - Logging and error handling framework

## Next Steps
1. **Complete Memory Bank**: Finish `progress.md` to establish current status baseline
2. **Project Structure Setup**: Create the core directory structure (`src/`, `docs/`, `examples/`)
3. **Configuration Templates**: Set up `config.yaml` and environment configuration files
4. **Pipeline Foundation**: Implement the main `pipeline.py` orchestrator
5. **Stage Scaffolding**: Create basic structure for all 6 pipeline stages
6. **Utilities Module**: Implement common utilities in `utils.py`

## Active Decisions and Considerations

### Architecture Decisions Made
- **Linear Pipeline Pattern**: Sequential stage processing with metadata enrichment
- **Resumable Execution**: Each stage can skip/resume based on existing output
- **Configuration-Driven**: All behavior controlled via `config.yaml`
- **Metadata-Centric**: Avoid file copying; use JSON metadata with file references
- **CPU-Only Processing**: Design for environments without GPU access

### Technology Choices Made
- **Python 3.8+**: Primary development language for broad compatibility
- **OpenCV**: Video processing and frame extraction
- **OpenCV DNN**: Face detection for character matching (replaced dlib/face_recognition due to installation issues)
- **AWS Bedrock**: Primary caption generation service
- **Conda**: Environment management for reproducible setups

### Pending Decisions
- **Face Detection Library**: Choose between face_recognition, MediaPipe, or MTCNN
- **Caption Generation Fallbacks**: Define priority order for caption services
- **Batch Processing Strategy**: Determine optimal batch sizes for different stages
- **Error Recovery**: Define retry strategies and failure handling approaches

## Important Patterns and Preferences

### Code Organization
- **Modular Design**: Each stage in separate directory with clear interfaces
- **Consistent Naming**: Use descriptive, domain-specific names
- **Error Handling**: Comprehensive error handling with informative messages
- **Documentation**: Inline comments for complex logic, comprehensive README

### Development Workflow
- **Memory Bank First**: Always update memory bank before major changes
- **Configuration-Driven**: Avoid hardcoded values; use config.yaml
- **Resumable Development**: Design for interrupted development sessions
- **Testing Strategy**: Unit tests for business logic, integration tests for stages

### Performance Priorities
1. **Memory Efficiency**: Handle large video files without excessive memory usage
2. **Resumability**: Fast startup and state recovery
3. **Batch Processing**: Configurable batch sizes for different hardware
4. **I/O Optimization**: Minimize file operations and copying

## Learnings and Project Insights

### Key Technical Insights
- **Metadata Strategy**: Using JSON metadata instead of file copying dramatically reduces storage requirements and improves processing speed
- **Stage Independence**: Each stage operating independently enables better debugging and parallel development
- **Configuration Flexibility**: Hierarchical configuration allows for easy experimentation and environment-specific settings

### Development Insights
- **Memory Bank Value**: Comprehensive documentation enables seamless session resumption
- **PoC Focus**: Prioritize functionality over polish for proof-of-concept phase
- **Modular Testing**: Independent stages enable focused testing and validation

### User Experience Insights
- **Single Command Execution**: Users want simple `python pipeline.py` execution
- **Progress Visibility**: Clear progress tracking across all stages is essential
- **Resumable Processing**: Ability to resume from any stage is critical for long-running processes

## Current Challenges

### Technical Challenges
- **Video Format Support**: Ensuring broad compatibility across video formats
- **Memory Management**: Handling large video files efficiently on CPU-only systems
- **API Integration**: Managing rate limits and failures for external caption services
- **Character Recognition Accuracy**: Balancing speed vs. accuracy for face detection

### Development Challenges
- **Dependency Management**: Ensuring consistent environment setup across platforms
- **Configuration Complexity**: Balancing flexibility with simplicity
- **Error Handling**: Providing meaningful error messages for complex pipeline failures

## Environment Context
- **Platform**: Windows 11 primary development environment
- **Hardware**: CPU-only processing (no GPU available)
- **Python**: Version 3.8+ for broad compatibility
- **Development Tools**: Conda for environment management, VSCode for development

## Integration Points
- **AWS Bedrock**: Primary caption generation service
- **Hugging Face**: Diffusion model hosting and fine-tuning
- **OpenCV**: Core video processing capabilities
- **Face Recognition Libraries**: Character detection and matching

## Quality Standards
- **Code Quality**: Follow PEP 8, use type hints, comprehensive error handling
- **Documentation**: README for setup, inline comments for complex logic
- **Testing**: Unit tests for core logic, integration tests for pipeline stages
- **Configuration**: All behavior configurable via YAML, no hardcoded values

## Success Metrics for Current Phase
- **Memory Bank Completeness**: All core documentation files created and comprehensive
- **Project Structure**: Clean, logical directory organization established
- **Configuration Framework**: Flexible, hierarchical configuration system
- **Pipeline Foundation**: Basic orchestrator with stage management capabilities
- **Development Environment**: Reproducible setup with conda environment
