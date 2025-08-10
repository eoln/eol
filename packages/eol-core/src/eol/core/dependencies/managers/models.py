"""Model Manager - Manages LLM model dependencies"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os
import asyncio
import time


@dataclass
class ModelConfig:
    """Configuration for an LLM model"""
    name: str
    provider: str
    purpose: str
    version: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    cost: Optional[Dict[str, float]] = None
    fallback: Optional[str] = None
    endpoint: Optional[str] = None


@dataclass
class ModelUsage:
    """Track model usage and costs"""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    total_cost: float = 0.0
    last_used: Optional[float] = None


class ModelClient(ABC):
    """Abstract base class for model clients"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion from prompt"""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if model is accessible"""
        pass


class AnthropicModelClient(ModelClient):
    """Client for Anthropic Claude models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.model_name = config.version or "claude-3-opus-20240229"
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate using Claude"""
        
        # Mock implementation - replace with actual Anthropic SDK
        await asyncio.sleep(0.1)  # Simulate API call
        
        return {
            'content': f"Claude response to: {prompt[:50]}...",
            'usage': {
                'input_tokens': len(prompt) // 4,
                'output_tokens': 100,
                'total_tokens': len(prompt) // 4 + 100
            },
            'model': self.model_name
        }
    
    async def embed(self, text: str) -> List[float]:
        """Claude doesn't provide embeddings directly"""
        raise NotImplementedError("Use a dedicated embedding model")
    
    async def health_check(self) -> bool:
        """Check Anthropic API accessibility"""
        
        if not self.api_key:
            return False
        
        # Mock health check - replace with actual API call
        return True


class OpenAIModelClient(ModelClient):
    """Client for OpenAI models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = config.version or "gpt-4-turbo-preview"
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate using OpenAI"""
        
        # Mock implementation - replace with actual OpenAI SDK
        await asyncio.sleep(0.1)  # Simulate API call
        
        return {
            'content': f"GPT response to: {prompt[:50]}...",
            'usage': {
                'input_tokens': len(prompt) // 4,
                'output_tokens': 100,
                'total_tokens': len(prompt) // 4 + 100
            },
            'model': self.model_name
        }
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI"""
        
        # Mock implementation - replace with actual embedding call
        await asyncio.sleep(0.05)
        
        # Return mock embedding vector
        import random
        return [random.random() for _ in range(1536)]
    
    async def health_check(self) -> bool:
        """Check OpenAI API accessibility"""
        
        if not self.api_key:
            return False
        
        # Mock health check - replace with actual API call
        return True


class LocalModelClient(ModelClient):
    """Client for local models (e.g., Ollama)"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.endpoint = config.endpoint or "http://localhost:11434"
        self.model_name = config.name
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate using local model"""
        
        # Mock implementation - replace with actual HTTP call to local model
        await asyncio.sleep(0.05)  # Simulate local inference
        
        return {
            'content': f"Local model response to: {prompt[:50]}...",
            'usage': {
                'input_tokens': len(prompt) // 4,
                'output_tokens': 100,
                'total_tokens': len(prompt) // 4 + 100
            },
            'model': self.model_name
        }
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using local model"""
        
        # Mock implementation
        await asyncio.sleep(0.02)
        
        import random
        return [random.random() for _ in range(768)]
    
    async def health_check(self) -> bool:
        """Check if local model is running"""
        
        # Mock check - replace with actual endpoint check
        return True


class ModelManager:
    """Manages LLM model dependencies"""
    
    def __init__(self):
        self.models: Dict[str, ModelClient] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.usage: Dict[str, ModelUsage] = {}
        self.models_by_purpose: Dict[str, List[str]] = {}
    
    async def resolve(self, node) -> ModelClient:
        """Resolve model dependency"""
        
        config = ModelConfig(
            name=node.name,
            provider=node.config.get('provider', 'openai'),
            purpose=node.config.get('purpose', 'general'),
            version=node.config.get('version'),
            config=node.config.get('config', {}),
            cost=node.config.get('cost'),
            fallback=node.config.get('fallback'),
            endpoint=node.config.get('endpoint')
        )
        
        # Create appropriate client
        client = await self._create_client(config)
        
        # Test connection
        if not await client.health_check():
            if config.fallback:
                # Try fallback
                fallback_config = ModelConfig(
                    name=config.fallback,
                    provider=config.provider,
                    purpose=config.purpose
                )
                client = await self._create_client(fallback_config)
                if not await client.health_check():
                    raise ConnectionError(f"Failed to connect to model {node.name} and fallback")
            else:
                raise ConnectionError(f"Failed to connect to model {node.name}")
        
        # Store model
        self.models[node.name] = client
        self.configs[node.name] = config
        
        # Track by purpose
        if config.purpose not in self.models_by_purpose:
            self.models_by_purpose[config.purpose] = []
        self.models_by_purpose[config.purpose].append(node.name)
        
        # Initialize usage tracking
        self.usage[node.name] = ModelUsage(model=node.name)
        
        return client
    
    async def _create_client(self, config: ModelConfig) -> ModelClient:
        """Create model client based on provider"""
        
        if config.provider == 'anthropic':
            return AnthropicModelClient(config)
        elif config.provider == 'openai':
            return OpenAIModelClient(config)
        elif config.provider == 'local':
            return LocalModelClient(config)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
    
    async def generate(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate using specific model"""
        
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        client = self.models[model_name]
        config = self.configs[model_name]
        
        # Merge configurations
        generation_config = {**config.config, **kwargs}
        
        # Track start time
        start_time = time.time()
        
        # Generate
        result = await client.generate(prompt, **generation_config)
        
        # Track usage
        usage = self.usage[model_name]
        usage.requests += 1
        usage.last_used = time.time()
        
        if 'usage' in result:
            usage.input_tokens += result['usage'].get('input_tokens', 0)
            usage.output_tokens += result['usage'].get('output_tokens', 0)
            usage.total_tokens += result['usage'].get('total_tokens', 0)
            
            # Calculate cost if configured
            if config.cost:
                input_cost = (usage.input_tokens / 1000) * config.cost.get('input', 0)
                output_cost = (usage.output_tokens / 1000) * config.cost.get('output', 0)
                usage.total_cost = input_cost + output_cost
        
        # Add timing
        result['latency'] = time.time() - start_time
        
        return result
    
    async def generate_by_purpose(self, purpose: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate using model selected by purpose"""
        
        if purpose not in self.models_by_purpose:
            raise ValueError(f"No model configured for purpose: {purpose}")
        
        # Get primary model for purpose
        model_name = self.models_by_purpose[purpose][0]
        
        return await self.generate(model_name, prompt, **kwargs)
    
    async def embed(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """Generate embeddings"""
        
        # Find embedding model if not specified
        if not model_name:
            if 'embeddings' in self.models_by_purpose:
                model_name = self.models_by_purpose['embeddings'][0]
            else:
                # Find any model that supports embeddings
                for name, client in self.models.items():
                    try:
                        return await client.embed(text)
                    except NotImplementedError:
                        continue
                raise ValueError("No embedding model available")
        
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        return await self.models[model_name].embed(text)
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get usage report for all models"""
        
        report = {}
        
        for model_name, usage in self.usage.items():
            report[model_name] = {
                'requests': usage.requests,
                'input_tokens': usage.input_tokens,
                'output_tokens': usage.output_tokens,
                'total_tokens': usage.total_tokens,
                'total_cost': usage.total_cost,
                'last_used': usage.last_used
            }
        
        # Calculate totals
        report['totals'] = {
            'requests': sum(u.requests for u in self.usage.values()),
            'total_tokens': sum(u.total_tokens for u in self.usage.values()),
            'total_cost': sum(u.total_cost for u in self.usage.values())
        }
        
        return report
    
    async def health_check(self, client: ModelClient) -> Dict[str, Any]:
        """Check health of a model client"""
        
        is_healthy = await client.health_check()
        
        return {
            'status': 'healthy' if is_healthy else 'unhealthy',
            'accessible': is_healthy
        }