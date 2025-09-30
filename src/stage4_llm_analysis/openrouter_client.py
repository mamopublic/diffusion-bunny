"""
OpenRouter API Client for LLM Analysis
Handles communication with OpenRouter API for image analysis and character identification.
"""

import os
import json
import base64
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter API client"""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "anthropic/claude-3-haiku"
    max_tokens: int = 300
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_requests_per_minute: int = 30

@dataclass
class LLMResponse:
    """Response from LLM analysis"""
    characters_present: List[str]
    scene_description: str
    confidence: float
    raw_response: str
    processing_time: float
    model_used: str

class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.logger = logging.getLogger(__name__)
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = datetime.now()
        
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < timedelta(minutes=1)]
        
        # Check if we need to wait
        if len(self.requests) >= self.max_requests:
            oldest_request = min(self.requests)
            wait_time = 60 - (now - oldest_request).total_seconds()
            
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                # Clean up old requests after waiting
                now = datetime.now()
                self.requests = [req_time for req_time in self.requests 
                               if now - req_time < timedelta(minutes=1)]
        
        # Record this request
        self.requests.append(now)

class ResponseCache:
    """Simple cache for LLM responses to avoid duplicate API calls"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(".cache/llm_responses")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_key(self, image_data: bytes, prompt: str) -> str:
        """Generate cache key from image and prompt"""
        combined = image_data + prompt.encode('utf-8')
        return hashlib.md5(combined).hexdigest()
    
    def get(self, image_data: bytes, prompt: str) -> Optional[LLMResponse]:
        """Get cached response if available"""
        cache_key = self._get_cache_key(image_data, prompt)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                response = LLMResponse(
                    characters_present=data['characters_present'],
                    scene_description=data['scene_description'],
                    confidence=data['confidence'],
                    raw_response=data['raw_response'],
                    processing_time=data['processing_time'],
                    model_used=data['model_used']
                )
                
                self.logger.debug(f"Cache hit for key: {cache_key}")
                return response
        except Exception as e:
            self.logger.warning(f"Failed to load cached response: {e}")
        
        return None
    
    def set(self, image_data: bytes, prompt: str, response: LLMResponse):
        """Cache response"""
        cache_key = self._get_cache_key(image_data, prompt)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            data = {
                'characters_present': response.characters_present,
                'scene_description': response.scene_description,
                'confidence': response.confidence,
                'raw_response': response.raw_response,
                'processing_time': response.processing_time,
                'model_used': response.model_used,
                'cached_at': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Cached response for key: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Failed to cache response: {e}")

class OpenRouterClient:
    """Client for OpenRouter API"""
    
    def __init__(self, config: OpenRouterConfig, enable_caching: bool = True):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(config.max_requests_per_minute)
        
        # Initialize cache if enabled
        self.cache = ResponseCache() if enable_caching else None
        
        # Validate API key
        if not config.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.logger.info(f"Initialized OpenRouter client with model: {config.model}")
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
    def _create_prompt(self, character_names: List[str]) -> str:
        """Create structured prompt for character identification"""
        char_list = ", ".join(character_names)
        
        prompt = f"""You are analyzing a frame from an animated movie. On the LEFT is a reference strip showing {len(character_names)} characters: {char_list}. On the RIGHT is the current frame with detected faces marked by colored boxes.

Your task:
1. IDENTIFY: Which characters from the reference strip appear in the current frame? Look carefully at facial features, hair, clothing, and other distinguishing characteristics.
2. DESCRIBE: What is happening in this scene? Be concise but descriptive.

Format your response as JSON:
{{
  "characters_present": ["character_name1", "character_name2"],
  "scene_description": "Brief description of what's happening",
  "confidence": 0.85
}}

Important:
- Only include characters you can confidently identify
- Use exact character names from the reference strip
- Confidence should be between 0.0 and 1.0
- Keep scene description under 100 words
- If no characters are clearly identifiable, return empty array for characters_present"""

        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data"""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Validate required fields
                if 'characters_present' in parsed and 'scene_description' in parsed:
                    return {
                        'characters_present': parsed.get('characters_present', []),
                        'scene_description': parsed.get('scene_description', ''),
                        'confidence': float(parsed.get('confidence', 0.5))
                    }
            
            # Fallback: try to extract information manually
            self.logger.warning("Could not parse JSON response, attempting manual extraction")
            return self._manual_parse(response_text)
            
        except Exception as e:
            self.logger.error(f"Failed to parse response: {e}")
            return {
                'characters_present': [],
                'scene_description': response_text[:200] + "..." if len(response_text) > 200 else response_text,
                'confidence': 0.0
            }
    
    def _manual_parse(self, response_text: str) -> Dict[str, Any]:
        """Manual parsing fallback for non-JSON responses"""
        # Simple keyword extraction - this is a fallback
        characters = []
        confidence = 0.3  # Lower confidence for manual parsing
        
        # Look for character names (you might want to make this more sophisticated)
        common_names = ['ellie', 'phil', 'rex', 'victoria']
        for name in common_names:
            if name.lower() in response_text.lower():
                characters.append(name)
        
        return {
            'characters_present': characters,
            'scene_description': response_text[:200] + "..." if len(response_text) > 200 else response_text,
            'confidence': confidence
        }
    
    def _make_api_request(self, image_b64: str, prompt: str) -> Dict[str, Any]:
        """Make request to OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mamopublic/diffusion-bunny",
            "X-Title": "Diffusion Bunny - Character Analysis"
        }
        
        # Determine image format
        image_format = "image/jpeg"  # Default assumption
        
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_format};base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        response = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config.timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def analyze_image(self, image_path: Path, character_names: List[str]) -> LLMResponse:
        """Analyze image and identify characters"""
        start_time = time.time()
        
        try:
            # Encode image
            image_b64 = self._encode_image(image_path)
            image_data = base64.b64decode(image_b64)
            
            # Create prompt
            prompt = self._create_prompt(character_names)
            
            # Check cache first
            if self.cache:
                cached_response = self.cache.get(image_data, prompt)
                if cached_response:
                    self.logger.debug(f"Using cached response for {image_path}")
                    return cached_response
            
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Make API request with retries
            last_exception = None
            for attempt in range(self.config.retry_attempts):
                try:
                    api_response = self._make_api_request(image_b64, prompt)
                    break
                except Exception as e:
                    last_exception = e
                    if attempt < self.config.retry_attempts - 1:
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        self.logger.warning(f"API request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        raise e
            
            # Extract response content
            raw_response = api_response['choices'][0]['message']['content']
            
            # Parse response
            parsed_data = self._parse_response(raw_response)
            
            # Create response object
            processing_time = time.time() - start_time
            response = LLMResponse(
                characters_present=parsed_data['characters_present'],
                scene_description=parsed_data['scene_description'],
                confidence=parsed_data['confidence'],
                raw_response=raw_response,
                processing_time=processing_time,
                model_used=self.config.model
            )
            
            # Cache response
            if self.cache:
                self.cache.set(image_data, prompt, response)
            
            self.logger.info(f"Analyzed {image_path} in {processing_time:.2f}s - Found: {response.characters_present}")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to analyze {image_path}: {e}")
            
            # Return error response
            return LLMResponse(
                characters_present=[],
                scene_description=f"Analysis failed: {str(e)}",
                confidence=0.0,
                raw_response="",
                processing_time=processing_time,
                model_used=self.config.model
            )

def main():
    """Test the OpenRouter client"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenRouter client")
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--model", default="anthropic/claude-3-haiku", help="Model to use")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create client
    config = OpenRouterConfig(
        api_key=args.api_key,
        model=args.model
    )
    
    client = OpenRouterClient(config)
    
    # Test analysis
    character_names = ["ellie", "phil", "rex", "victoria"]
    response = client.analyze_image(Path(args.image), character_names)
    
    print(f"Characters found: {response.characters_present}")
    print(f"Scene description: {response.scene_description}")
    print(f"Confidence: {response.confidence}")
    print(f"Processing time: {response.processing_time:.2f}s")

if __name__ == "__main__":
    main()
