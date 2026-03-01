"""
Incident summarization module.

Summarizes multiple incidents into concise reports.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from loguru import logger

from .llm_reasoner import LLMReasoner
from ...utils.config import get_config


class IncidentSummarizer:
    """
    Incident summarizer.
    
    Summarizes multiple incidents into concise reports.
    """
    
    def __init__(self, llm_reasoner: Optional[LLMReasoner] = None):
        """
        Initialize incident summarizer.
        
        Args:
            llm_reasoner: Optional LLM reasoner instance
        """
        self.llm_reasoner = llm_reasoner or LLMReasoner()
        self.config = get_config()
        explainability_config = self.config.get_section("explainability")
        self.max_incidents = explainability_config.get(
            "incident_summarization", {}
        ).get("max_incidents", 100)
    
    def summarize(
        self,
        incidents: List[Dict],
        time_window: Optional[timedelta] = None
    ) -> Dict:
        """
        Summarize incidents.
        
        Args:
            incidents: List of incident dictionaries
            time_window: Optional time window to filter incidents
        
        Returns:
            Summary dictionary
        """
        if not incidents:
            return {
                "summary": "No incidents to summarize",
                "total_incidents": 0,
                "time_range": None
            }
        
        # Filter by time window if provided
        if time_window:
            cutoff_time = datetime.now() - time_window
            incidents = [
                inc for inc in incidents
                if self._parse_timestamp(inc.get("timestamp")) > cutoff_time
            ]
        
        # Limit incidents
        incidents = incidents[:self.max_incidents]
        
        # Group by type
        by_type = {}
        for incident in incidents:
            incident_type = incident.get("type", "unknown")
            if incident_type not in by_type:
                by_type[incident_type] = []
            by_type[incident_type].append(incident)
        
        # Calculate statistics
        total_incidents = len(incidents)
        critical_count = sum(
            1 for inc in incidents
            if inc.get("severity") == "CRITICAL"
        )
        high_count = sum(
            1 for inc in incidents
            if inc.get("severity") == "HIGH"
        )
        
        # Get time range
        timestamps = [
            self._parse_timestamp(inc.get("timestamp"))
            for inc in incidents
            if inc.get("timestamp")
        ]
        time_range = None
        if timestamps:
            time_range = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat()
            }
        
        # Generate summary text
        summary_text = self._generate_summary_text(
            incidents, by_type, total_incidents, critical_count, high_count
        )
        
        return {
            "summary": summary_text,
            "total_incidents": total_incidents,
            "critical_count": critical_count,
            "high_count": high_count,
            "by_type": {k: len(v) for k, v in by_type.items()},
            "time_range": time_range,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_summary_text(
        self,
        incidents: List[Dict],
        by_type: Dict[str, List[Dict]],
        total: int,
        critical: int,
        high: int
    ) -> str:
        """Generate summary text."""
        parts = [f"Summary of {total} incidents"]
        
        if critical > 0:
            parts.append(f"{critical} critical")
        if high > 0:
            parts.append(f"{high} high severity")
        
        parts.append("incidents detected.")
        
        # Add type breakdown
        if len(by_type) > 1:
            type_breakdown = ", ".join(
                f"{len(v)} {k}" for k, v in by_type.items()
            )
            parts.append(f"Breakdown: {type_breakdown}.")
        
        # Use LLM for detailed summary if available
        if self.llm_reasoner and self.llm_reasoner.client and len(incidents) <= 20:
            try:
                llm_summary = self.llm_reasoner.explain(
                    "incident_summary",
                    {
                        "incidents": incidents[:10],  # Limit for LLM
                        "statistics": {
                            "total": total,
                            "critical": critical,
                            "high": high
                        }
                    }
                )
                parts.append(llm_summary)
            except Exception as e:
                logger.error(f"LLM summarization error: {e}")
        
        return " ".join(parts)
    
    def _parse_timestamp(self, timestamp: Optional[str]) -> datetime:
        """Parse timestamp string to datetime."""
        if timestamp is None:
            return datetime.now()
        
        try:
            if isinstance(timestamp, str):
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return timestamp
        except Exception:
            return datetime.now()
