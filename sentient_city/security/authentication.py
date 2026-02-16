"""
Authentication service for JWT-based authentication.

Provides secure authentication and token management.
"""

import jwt
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger

from ..utils.config import get_config


class AuthenticationService:
    """
    JWT-based authentication service.
    
    Handles user authentication, token generation, and validation.
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        expiration_hours: int = 24
    ):
        """
        Initialize authentication service.
        
        Args:
            secret_key: JWT secret key (or from config)
            algorithm: JWT algorithm
            expiration_hours: Token expiration time in hours
        """
        self.config = get_config()
        backend_config = self.config.get_section("backend")
        auth_config = backend_config.get("authentication", {})
        
        self.secret_key = secret_key or auth_config.get("jwt_secret", "change-me-in-production")
        self.algorithm = auth_config.get("jwt_algorithm", algorithm)
        self.expiration_hours = auth_config.get("jwt_expiration_hours", expiration_hours)
        
        if self.secret_key == "change-me-in-production":
            logger.warning("Using default JWT secret key. Change in production!")
    
    def generate_token(
        self,
        user_id: str,
        username: str,
        roles: list,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate JWT token.
        
        Args:
            user_id: User identifier
            username: Username
            roles: List of user roles
            additional_claims: Additional JWT claims
        
        Returns:
            JWT token string
        """
        now = datetime.utcnow()
        expiration = now + timedelta(hours=self.expiration_hours)
        
        payload = {
            "user_id": user_id,
            "username": username,
            "roles": roles,
            "iat": now,
            "exp": expiration,
            "iss": "sentientcity-ai"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token string
        
        Returns:
            Decoded payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256 (in production, use bcrypt/argon2).
        
        Args:
            password: Plain text password
        
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password
            hashed: Hashed password
        
        Returns:
            True if password matches
        """
        return self.hash_password(password) == hashed
    
    def refresh_token(self, token: str) -> Optional[str]:
        """
        Refresh JWT token.
        
        Args:
            token: Current token
        
        Returns:
            New token or None if refresh failed
        """
        payload = self.verify_token(token)
        if payload is None:
            return None
        
        # Generate new token with same claims
        return self.generate_token(
            user_id=payload["user_id"],
            username=payload["username"],
            roles=payload["roles"]
        )
