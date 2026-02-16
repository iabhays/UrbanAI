"""Tests for security modules."""

import pytest
from sentient_city.security import AuthenticationService, RBACService, EncryptionService
from sentient_city.security.rbac import Role, Permission


def test_authentication_service():
    """Test authentication service."""
    auth = AuthenticationService(secret_key="test-secret-key")
    
    # Generate token
    token = auth.generate_token(
        user_id="user123",
        username="testuser",
        roles=["admin"]
    )
    
    assert token is not None
    
    # Verify token
    payload = auth.verify_token(token)
    assert payload is not None
    assert payload["user_id"] == "user123"
    assert payload["username"] == "testuser"


def test_rbac_service():
    """Test RBAC service."""
    rbac = RBACService()
    
    # Test permissions
    assert rbac.has_permission([Role.ADMIN], Permission.MANAGE_SYSTEM)
    assert rbac.has_permission([Role.OPERATOR], Permission.VIEW_CAMERAS)
    assert not rbac.has_permission([Role.VIEWER], Permission.MANAGE_CAMERAS)
    
    # Test admin check
    assert rbac.is_admin([Role.ADMIN])
    assert not rbac.is_admin([Role.VIEWER])


def test_encryption_service():
    """Test encryption service."""
    enc = EncryptionService()
    
    # Test string encryption
    plaintext = "sensitive data"
    encrypted = enc.encrypt_string(plaintext)
    decrypted = enc.decrypt_string(encrypted)
    
    assert decrypted == plaintext
    
    # Test bytes encryption
    data = b"binary data"
    encrypted_bytes = enc.encrypt(data)
    decrypted_bytes = enc.decrypt(encrypted_bytes)
    
    assert decrypted_bytes == data
