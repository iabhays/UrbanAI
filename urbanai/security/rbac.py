"""
Role-Based Access Control (RBAC) service.

Manages roles, permissions, and access control.
"""

from typing import List, Dict, Set, Optional
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class Permission(Enum):
    """System permissions."""
    # Camera permissions
    VIEW_CAMERAS = "view_cameras"
    MANAGE_CAMERAS = "manage_cameras"
    
    # Alert permissions
    VIEW_ALERTS = "view_alerts"
    ACKNOWLEDGE_ALERTS = "acknowledge_alerts"
    MANAGE_ALERTS = "manage_alerts"
    
    # Analytics permissions
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_ANALYTICS = "export_analytics"
    
    # User management
    VIEW_USERS = "view_users"
    MANAGE_USERS = "manage_users"
    
    # System administration
    MANAGE_SYSTEM = "manage_system"
    VIEW_LOGS = "view_logs"
    MANAGE_MODELS = "manage_models"


class Role(Enum):
    """System roles."""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    ANALYST = "analyst"


@dataclass
class RoleDefinition:
    """Role definition with permissions."""
    role: Role
    permissions: Set[Permission]
    description: str


class RBACService:
    """
    Role-Based Access Control service.
    
    Manages roles, permissions, and access control checks.
    """
    
    def __init__(self):
        """Initialize RBAC service."""
        self.role_definitions = self._initialize_roles()
    
    def _initialize_roles(self) -> Dict[Role, RoleDefinition]:
        """Initialize role definitions."""
        return {
            Role.ADMIN: RoleDefinition(
                role=Role.ADMIN,
                permissions=set(Permission),  # All permissions
                description="Full system access"
            ),
            Role.OPERATOR: RoleDefinition(
                role=Role.OPERATOR,
                permissions={
                    Permission.VIEW_CAMERAS,
                    Permission.MANAGE_CAMERAS,
                    Permission.VIEW_ALERTS,
                    Permission.ACKNOWLEDGE_ALERTS,
                    Permission.VIEW_ANALYTICS
                },
                description="Operational access for monitoring and alerts"
            ),
            Role.VIEWER: RoleDefinition(
                role=Role.VIEWER,
                permissions={
                    Permission.VIEW_CAMERAS,
                    Permission.VIEW_ALERTS,
                    Permission.VIEW_ANALYTICS
                },
                description="Read-only access"
            ),
            Role.ANALYST: RoleDefinition(
                role=Role.ANALYST,
                permissions={
                    Permission.VIEW_CAMERAS,
                    Permission.VIEW_ALERTS,
                    Permission.VIEW_ANALYTICS,
                    Permission.EXPORT_ANALYTICS
                },
                description="Analytics and reporting access"
            )
        }
    
    def has_permission(
        self,
        roles: List[Role],
        permission: Permission
    ) -> bool:
        """
        Check if roles have permission.
        
        Args:
            roles: List of user roles
            permission: Permission to check
        
        Returns:
            True if any role has permission
        """
        for role in roles:
            if role in self.role_definitions:
                role_def = self.role_definitions[role]
                if permission in role_def.permissions:
                    return True
        
        return False
    
    def get_permissions(self, roles: List[Role]) -> Set[Permission]:
        """
        Get all permissions for roles.
        
        Args:
            roles: List of user roles
        
        Returns:
            Set of permissions
        """
        permissions = set()
        for role in roles:
            if role in self.role_definitions:
                role_def = self.role_definitions[role]
                permissions.update(role_def.permissions)
        
        return permissions
    
    def is_admin(self, roles: List[Role]) -> bool:
        """
        Check if user is admin.
        
        Args:
            roles: List of user roles
        
        Returns:
            True if admin role present
        """
        return Role.ADMIN in roles
    
    def require_permission(
        self,
        roles: List[Role],
        permission: Permission
    ):
        """
        Require permission (raise exception if not granted).
        
        Args:
            roles: List of user roles
            permission: Required permission
        
        Raises:
            PermissionError: If permission not granted
        """
        if not self.has_permission(roles, permission):
            raise PermissionError(
                f"Permission {permission.value} required. "
                f"User roles: {[r.value for r in roles]}"
            )
