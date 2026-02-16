"""
Encryption service for data protection.

Provides encryption at rest and in transit.
"""

import os
import base64
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger


class EncryptionService:
    """
    Encryption service for data protection.
    
    Provides symmetric encryption using Fernet (AES-128).
    """
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize encryption service.
        
        Args:
            key: Encryption key (or generate from password)
        """
        if key is None:
            # Generate key from environment or default
            key_str = os.getenv("ENCRYPTION_KEY", self._generate_default_key())
            key = key_str.encode() if isinstance(key_str, str) else key_str
        
        # Ensure key is proper Fernet key (32 bytes, base64)
        if len(key) != 44:  # Fernet keys are 44 bytes base64-encoded
            key = self._derive_key(key)
        
        self.key = key
        self.cipher = Fernet(key)
    
    def _generate_default_key(self) -> str:
        """Generate default key (for development only)."""
        logger.warning("Using default encryption key. Set ENCRYPTION_KEY in production!")
        return Fernet.generate_key().decode()
    
    def _derive_key(self, password: bytes) -> bytes:
        """
        Derive Fernet key from password.
        
        Args:
            password: Password bytes
        
        Returns:
            Fernet key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'sentientcity_salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt(self, data: bytes) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
        
        Returns:
            Encrypted data
        """
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
        
        Returns:
            Decrypted data
        """
        return self.cipher.decrypt(encrypted_data)
    
    def encrypt_string(self, text: str) -> str:
        """
        Encrypt string.
        
        Args:
            text: Text to encrypt
        
        Returns:
            Base64-encoded encrypted string
        """
        encrypted = self.encrypt(text.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """
        Decrypt string.
        
        Args:
            encrypted_text: Base64-encoded encrypted string
        
        Returns:
            Decrypted text
        """
        encrypted_bytes = base64.b64decode(encrypted_text.encode())
        decrypted = self.decrypt(encrypted_bytes)
        return decrypted.decode()
