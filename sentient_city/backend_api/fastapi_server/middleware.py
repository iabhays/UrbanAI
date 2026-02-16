"""
Security middleware for FastAPI.

Provides authentication, rate limiting, and security headers.
"""

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Callable, Optional
from loguru import logger
import time
from collections import defaultdict, deque

from ...security import AuthenticationService, RBACService
from ...security.rbac import Permission


security = HTTPBearer(auto_error=False)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    
    Limits requests per IP address.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application
            requests_per_minute: Requests per minute limit
            requests_per_hour: Requests per hour limit
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.request_times: defaultdict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests
        self.request_times[client_ip] = deque(
            t for t in self.request_times[client_ip]
            if current_time - t < 3600  # Keep last hour
        )
        
        # Check minute limit
        minute_requests = [
            t for t in self.request_times[client_ip]
            if current_time - t < 60
        ]
        
        if len(minute_requests) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return Response(
                content="Rate limit exceeded",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS
            )
        
        # Check hour limit
        if len(self.request_times[client_ip]) >= self.requests_per_hour:
            logger.warning(f"Hourly rate limit exceeded for {client_ip}")
            return Response(
                content="Hourly rate limit exceeded",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS
            )
        
        # Record request
        self.request_times[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware.
    
    Validates JWT tokens and attaches user info to request.
    """
    
    def __init__(self, app, auth_service: Optional[AuthenticationService] = None):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application
            auth_service: Authentication service instance
        """
        super().__init__(app)
        self.auth_service = auth_service or AuthenticationService()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with authentication."""
        # Skip authentication for public endpoints
        if request.url.path in ["/api/v1/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Get token from header
        authorization = request.headers.get("Authorization")
        
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            payload = self.auth_service.verify_token(token)
            
            if payload:
                # Attach user info to request state
                request.state.user_id = payload.get("user_id")
                request.state.username = payload.get("username")
                request.state.roles = payload.get("roles", [])
                request.state.authenticated = True
            else:
                request.state.authenticated = False
        else:
            request.state.authenticated = False
        
        response = await call_next(request)
        return response


def require_permission(permission: Permission):
    """
    Decorator to require permission.
    
    Args:
        permission: Required permission
    
    Returns:
        Decorator function
    """
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            if not hasattr(request.state, "authenticated") or not request.state.authenticated:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            rbac = RBACService()
            roles = [r for r in request.state.roles]  # Convert to Role enum
            
            if not rbac.has_permission(roles, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission {permission.value} required"
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator
