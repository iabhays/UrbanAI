"""
LLM-based reasoning module.

Uses large language models to generate natural language explanations.
"""

from typing import Dict, List, Optional
import os
from loguru import logger

from ...utils.config import get_config


class LLMReasoner:
    """
    LLM-based reasoning module.
    
    Generates natural language explanations using LLMs.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM reasoner.
        
        Args:
            provider: LLM provider (openai, anthropic, local)
            model: Model name
            api_key: API key (or use environment variable)
        """
        self.config = get_config()
        explainability_config = self.config.get_section("explainability")
        llm_config = explainability_config.get("llm", {})
        
        self.provider = llm_config.get("provider", provider)
        self.model = llm_config.get("model", model)
        self.temperature = llm_config.get("temperature", 0.7)
        self.max_tokens = llm_config.get("max_tokens", 500)
        
        # Get API key
        api_key_env = llm_config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = api_key or os.getenv(api_key_env)
        
        # Initialize client based on provider
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client."""
        if self.provider == "openai":
            try:
                import openai
                if self.api_key:
                    return openai.OpenAI(api_key=self.api_key)
                else:
                    logger.warning("OpenAI API key not found")
                    return None
            except ImportError:
                logger.error("OpenAI library not installed")
                return None
        
        elif self.provider == "anthropic":
            try:
                import anthropic
                if self.api_key:
                    return anthropic.Anthropic(api_key=self.api_key)
                else:
                    logger.warning("Anthropic API key not found")
                    return None
            except ImportError:
                logger.error("Anthropic library not installed")
                return None
        
        else:
            logger.warning(f"Unsupported provider: {self.provider}")
            return None
    
    def explain(
        self,
        event_type: str,
        event_data: Dict,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate explanation for event.
        
        Args:
            event_type: Type of event (crowd_crush, accident, anomaly, etc.)
            event_data: Event data dictionary
            context: Optional context dictionary
        
        Returns:
            Natural language explanation
        """
        if self.client is None:
            return self._fallback_explanation(event_type, event_data)
        
        prompt = self._build_prompt(event_type, event_data, context)
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an AI assistant that explains security and safety incidents in a clear, professional manner."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        
        except Exception as e:
            logger.error(f"LLM reasoning error: {e}")
            return self._fallback_explanation(event_type, event_data)
    
    def _build_prompt(
        self,
        event_type: str,
        event_data: Dict,
        context: Optional[Dict]
    ) -> str:
        """Build prompt for LLM."""
        prompt = f"Explain the following {event_type} event in clear, professional language:\n\n"
        prompt += f"Event Data: {event_data}\n\n"
        
        if context:
            prompt += f"Context: {context}\n\n"
        
        prompt += "Provide a concise explanation suitable for security personnel, including:\n"
        prompt += "1. What happened\n"
        prompt += "2. Why it's significant\n"
        prompt += "3. Recommended actions\n"
        
        return prompt
    
    def _fallback_explanation(self, event_type: str, event_data: Dict) -> str:
        """Generate fallback explanation without LLM."""
        if event_type == "crowd_crush":
            risk_score = event_data.get("risk_score", 0.0)
            return f"Crowd crush risk detected with score {risk_score:.2f}. "
        
        elif event_type == "accident":
            return f"Traffic accident detected. {len(event_data.get('vehicles_involved', []))} vehicles involved."
        
        elif event_type == "anomaly":
            return f"Anomalous behavior detected with score {event_data.get('anomaly_score', 0.0):.2f}."
        
        else:
            return f"{event_type} event detected."
