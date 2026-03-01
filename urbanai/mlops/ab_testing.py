"""
A/B testing framework for models.

Enables comparing model versions in production.
"""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger


@dataclass
class ModelVariant:
    """Model variant for A/B testing."""
    model_id: str
    version: str
    weight: float  # Traffic weight (0-1)


class ABTestingFramework:
    """
    A/B testing framework for models.
    
    Enables comparing multiple model versions with traffic splitting.
    """
    
    def __init__(self):
        """Initialize A/B testing framework."""
        self.experiments: Dict[str, Dict[str, ModelVariant]] = {}
        self.results: Dict[str, List[Dict]] = {}
    
    def create_experiment(
        self,
        experiment_id: str,
        variants: List[ModelVariant]
    ):
        """
        Create A/B test experiment.
        
        Args:
            experiment_id: Experiment identifier
            variants: List of model variants
        """
        # Normalize weights
        total_weight = sum(v.weight for v in variants)
        if total_weight != 1.0:
            # Normalize
            for v in variants:
                v.weight /= total_weight
        
        self.experiments[experiment_id] = {
            v.version: v for v in variants
        }
        
        self.results[experiment_id] = []
        
        logger.info(f"Created A/B test experiment: {experiment_id} with {len(variants)} variants")
    
    def select_variant(
        self,
        experiment_id: str,
        user_id: Optional[str] = None
    ) -> Optional[ModelVariant]:
        """
        Select model variant for request.
        
        Args:
            experiment_id: Experiment identifier
            user_id: Optional user ID for consistent assignment
        
        Returns:
            Selected model variant
        """
        if experiment_id not in self.experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return None
        
        variants = list(self.experiments[experiment_id].values())
        
        if user_id:
            # Consistent assignment based on user ID
            random.seed(hash(user_id))
        
        # Weighted random selection
        r = random.random()
        cumulative = 0.0
        
        for variant in variants:
            cumulative += variant.weight
            if r <= cumulative:
                return variant
        
        # Fallback to last variant
        return variants[-1]
    
    def record_result(
        self,
        experiment_id: str,
        variant_version: str,
        metrics: Dict
    ):
        """
        Record experiment result.
        
        Args:
            experiment_id: Experiment identifier
            variant_version: Variant version
            metrics: Performance metrics
        """
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "variant": variant_version,
            "metrics": metrics
        }
        
        self.results[experiment_id].append(result)
        
        logger.debug(f"Recorded result for {experiment_id} variant {variant_version}")
    
    def get_experiment_stats(self, experiment_id: str) -> Dict:
        """
        Get experiment statistics.
        
        Args:
            experiment_id: Experiment identifier
        
        Returns:
            Experiment statistics
        """
        if experiment_id not in self.results:
            return {}
        
        results = self.results[experiment_id]
        
        # Group by variant
        variant_stats = {}
        for result in results:
            variant = result["variant"]
            if variant not in variant_stats:
                variant_stats[variant] = {
                    "count": 0,
                    "metrics": []
                }
            
            variant_stats[variant]["count"] += 1
            variant_stats[variant]["metrics"].append(result["metrics"])
        
        # Calculate statistics
        stats = {}
        for variant, data in variant_stats.items():
            stats[variant] = {
                "request_count": data["count"],
                "metrics_summary": self._summarize_metrics(data["metrics"])
            }
        
        return stats
    
    def _summarize_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Summarize metrics list."""
        if not metrics_list:
            return {}
        
        # Extract common metrics
        summary = {}
        
        # Average inference time
        inference_times = [m.get("inference_time_ms") for m in metrics_list if m.get("inference_time_ms")]
        if inference_times:
            summary["avg_inference_time_ms"] = sum(inference_times) / len(inference_times)
        
        # Accuracy
        accuracies = [m.get("accuracy") for m in metrics_list if m.get("accuracy")]
        if accuracies:
            summary["avg_accuracy"] = sum(accuracies) / len(accuracies)
        
        return summary
