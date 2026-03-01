"""
Anomaly detection engine.

Detects anomalous behaviors and patterns using statistical and ML methods.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import IsolationForest
from loguru import logger

from ...utils.config import get_config


class AnomalyDetector:
    """
    Anomaly detection engine.
    
    Detects anomalous behaviors, movements, and patterns.
    """
    
    def __init__(
        self,
        threshold: float = 0.75,
        window_size: int = 60,
        contamination: float = 0.1
    ):
        """
        Initialize anomaly detector.
        
        Args:
            threshold: Anomaly score threshold
            window_size: Window size for analysis
            contamination: Expected proportion of anomalies
        """
        self.config = get_config()
        intelligence_config = self.config.get_section("intelligence")
        anomaly_config = intelligence_config.get("anomaly_detection", {})
        
        self.threshold = anomaly_config.get("threshold", threshold)
        self.window_size = anomaly_config.get("window_size", window_size)
        self.contamination = contamination
        
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.is_fitted = False
        self.feature_history: List[np.ndarray] = []
    
    def detect(
        self,
        features: np.ndarray,
        context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Detect anomalies in features.
        
        Args:
            features: Feature vector or array
            context: Optional context dictionary
        
        Returns:
            Dictionary with:
            - is_anomaly: Boolean indicating anomaly
            - anomaly_score: Anomaly score (0-1, higher is more anomalous)
            - features: Contributing features
        """
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Add to history
        self.feature_history.append(features[0])
        if len(self.feature_history) > self.window_size:
            self.feature_history.pop(0)
        
        # Need enough history to fit model
        if len(self.feature_history) < 10:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "reason": "Insufficient history"
            }
        
        # Fit model if needed
        if not self.is_fitted or len(self.feature_history) % 20 == 0:
            self._fit_model()
        
        # Predict anomaly
        prediction = self.isolation_forest.predict(features)
        score = self.isolation_forest.score_samples(features)
        
        # Convert score to 0-1 range (lower score = more anomalous)
        # Isolation Forest returns negative scores for anomalies
        normalized_score = 1.0 / (1.0 + np.exp(-score[0]))  # Sigmoid normalization
        
        is_anomaly = prediction[0] == -1 or normalized_score > self.threshold
        
        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(normalized_score),
            "raw_score": float(score[0]),
            "prediction": int(prediction[0])
        }
    
    def _fit_model(self):
        """Fit isolation forest model."""
        if len(self.feature_history) < 10:
            return
        
        X = np.array(self.feature_history)
        self.isolation_forest.fit(X)
        self.is_fitted = True
        logger.debug("Anomaly detector model refitted")
    
    def detect_behavior_anomaly(
        self,
        behavior_embedding: np.ndarray,
        historical_embeddings: Optional[List[np.ndarray]] = None
    ) -> Dict[str, any]:
        """
        Detect anomalies in behavior embeddings.
        
        Args:
            behavior_embedding: Current behavior embedding
            historical_embeddings: Optional historical embeddings
        
        Returns:
            Anomaly detection result
        """
        # Calculate distance from historical mean
        if historical_embeddings and len(historical_embeddings) > 0:
            hist_array = np.array(historical_embeddings)
            mean_embedding = np.mean(hist_array, axis=0)
            
            # Cosine distance
            norm_current = behavior_embedding / (np.linalg.norm(behavior_embedding) + 1e-8)
            norm_mean = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)
            cosine_distance = 1 - np.dot(norm_current, norm_mean)
            
            # Euclidean distance
            euclidean_distance = np.linalg.norm(behavior_embedding - mean_embedding)
            
            # Combine distances
            anomaly_score = (cosine_distance + euclidean_distance / 10.0) / 2.0
            
            is_anomaly = anomaly_score > self.threshold
            
            return {
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(anomaly_score),
                "cosine_distance": float(cosine_distance),
                "euclidean_distance": float(euclidean_distance)
            }
        
        # Fallback to standard detection
        return self.detect(behavior_embedding)
    
    def reset(self):
        """Reset detector state."""
        self.feature_history.clear()
        self.is_fitted = False
