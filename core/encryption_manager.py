"""Secure encryption management for keylogger data."""

import os
import base64
import hashlib
from pathlib import Path
from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manages encryption keys and data encryption/decryption."""
    
    def __init__(self, key_file: str = "encryption.key", password: Optional[str] = None):
        self.key_file = Path(key_file)
        self.password = password
        self._cipher_suite: Optional[Fernet] = None
        self._initialize_encryption()
    
    def _initialize_encryption(self) -> None:
        """Initialize encryption with existing or new key."""
        try:
            if self.key_file.exists():
                self._load_key()
            else:
                self._generate_and_save_key()
            logger.info("Encryption initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def _generate_key(self, password: Optional[str] = None) -> bytes:
        """Generate a new encryption key."""
        if password:
            # Derive key from password
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            # Store salt with key for later derivation
            return salt + key
        else:
            # Generate random key
            return Fernet.generate_key()
    
    def _generate_and_save_key(self) -> None:
        """Generate a new key and save it securely."""
        key = self._generate_key(self.password)
        
        # Ensure key directory exists
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save key with restricted permissions
        with open(self.key_file, 'wb') as f:
            f.write(key)
        
        # Set restrictive permissions (owner read/write only)
        if os.name != 'nt':  # Unix-like systems
            os.chmod(self.key_file, 0o600)
        
        self._cipher_suite = Fernet(key if not self.password else key[16:])
        logger.info(f"New encryption key generated and saved to {self.key_file}")
    
    def _load_key(self) -> None:
        """Load existing encryption key."""
        try:
            with open(self.key_file, 'rb') as f:
                key_data = f.read()
            
            if self.password:
                # Extract salt and derive key from password
                salt = key_data[:16]
                stored_key = key_data[16:]
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                derived_key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
                if derived_key != stored_key:
                    raise ValueError("Invalid password")
                key = stored_key
            else:
                key = key_data
            
            self._cipher_suite = Fernet(key)
            logger.info(f"Encryption key loaded from {self.key_file}")
        except Exception as e:
            logger.error(f"Failed to load encryption key: {e}")
            raise
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data."""
        if not self._cipher_suite:
            raise RuntimeError("Encryption not initialized")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        try:
            return self._cipher_suite.encrypt(data)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        if not self._cipher_suite:
            raise RuntimeError("Encryption not initialized")
        
        try:
            return self._cipher_suite.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_file(self, file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """Encrypt a file."""
        file_path = Path(file_path)
        if not output_path:
            output_path = file_path.with_suffix(file_path.suffix + '.enc')
        else:
            output_path = Path(output_path)
        
        try:
            with open(file_path, 'rb') as infile:
                data = infile.read()
            
            encrypted_data = self.encrypt(data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(encrypted_data)
            
            logger.info(f"File encrypted: {file_path} -> {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            raise
    
    def decrypt_file(self, encrypted_file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
        """Decrypt a file."""
        encrypted_file_path = Path(encrypted_file_path)
        if not output_path:
            # Remove .enc extension if present
            if encrypted_file_path.suffix == '.enc':
                output_path = encrypted_file_path.with_suffix('')
            else:
                output_path = encrypted_file_path.with_suffix('.decrypted')
        else:
            output_path = Path(output_path)
        
        try:
            with open(encrypted_file_path, 'rb') as infile:
                encrypted_data = infile.read()
            
            decrypted_data = self.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(decrypted_data)
            
            logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            raise
    
    def hash_data(self, data: Union[str, bytes], algorithm: str = 'sha256') -> str:
        """Hash data using specified algorithm."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == 'sha256':
            return hashlib.sha256(data).hexdigest()
        elif algorithm == 'md5':
            return hashlib.md5(data).hexdigest()
        elif algorithm == 'sha1':
            return hashlib.sha1(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    def verify_integrity(self, data: Union[str, bytes], expected_hash: str, algorithm: str = 'sha256') -> bool:
        """Verify data integrity using hash comparison."""
        actual_hash = self.hash_data(data, algorithm)
        return actual_hash == expected_hash
    
    def rotate_key(self, new_password: Optional[str] = None) -> None:
        """Rotate encryption key (generate new key)."""
        # Backup old key
        if self.key_file.exists():
            backup_path = self.key_file.with_suffix('.key.backup')
            self.key_file.rename(backup_path)
            logger.info(f"Old key backed up to {backup_path}")
        
        # Generate new key
        self.password = new_password
        self._generate_and_save_key()
        logger.info("Encryption key rotated successfully")
    
    def is_initialized(self) -> bool:
        """Check if encryption is properly initialized."""
        return self._cipher_suite is not None
    
    def get_key_info(self) -> dict:
        """Get information about the current key."""
        if not self.key_file.exists():
            return {"exists": False}
        
        stat = self.key_file.stat()
        return {
            "exists": True,
            "file_path": str(self.key_file),
            "size_bytes": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "password_protected": self.password is not None
        }