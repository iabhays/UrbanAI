"""
Audit logging service.

Provides security audit logging for compliance and forensics.
"""

import json
from datetime import datetime
from typing import Dict, Optional, Any
from enum import Enum
from pathlib import Path
from loguru import logger


class AuditEventType(Enum):
    """Audit event types."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"


class AuditLogger:
    """
    Security audit logger.
    
    Logs security-relevant events for compliance and forensics.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
        """
        self.log_file = log_file or "logs/audit.log"
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str],
        action: str,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        ip_address: Optional[str] = None
    ):
        """
        Log audit event.
        
        Args:
            event_type: Type of audit event
            user_id: User identifier (None for system events)
            action: Action performed
            resource: Resource accessed/modified
            details: Additional details
            success: Whether action succeeded
            ip_address: Client IP address
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "success": success,
            "ip_address": ip_address,
            "details": details or {}
        }
        
        # Log to file
        self._write_log(audit_entry)
        
        # Also log to logger
        log_level = "INFO" if success else "WARNING"
        logger.log(
            log_level,
            f"Audit: {event_type.value} | User: {user_id} | Action: {action} | Success: {success}"
        )
    
    def _write_log(self, entry: Dict[str, Any]):
        """Write audit entry to file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def log_authentication(
        self,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        """Log authentication event."""
        self.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            user_id=user_id,
            action="login" if success else "login_failed",
            success=success,
            ip_address=ip_address,
            details=details
        )
    
    def log_authorization(
        self,
        user_id: str,
        action: str,
        resource: str,
        granted: bool,
        ip_address: Optional[str] = None
    ):
        """Log authorization event."""
        self.log_event(
            event_type=AuditEventType.AUTHORIZATION,
            user_id=user_id,
            action=action,
            resource=resource,
            success=granted,
            ip_address=ip_address
        )
    
    def log_data_access(
        self,
        user_id: str,
        resource: str,
        ip_address: Optional[str] = None
    ):
        """Log data access event."""
        self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=user_id,
            action="access",
            resource=resource,
            ip_address=ip_address
        )
