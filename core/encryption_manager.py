"""Secure encryption management for keylogger data."""

import os
import base64
import hashlib
import logging
from typing import Optional, Union, Tuple
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import secrets

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manages encryption and decryption of keylogger data."""
    
    def __init__(self, config_or_path: Union[str, os.PathLike, object] = 'keys/encryption.key'):
        # Support either a Config-like object with `.get` or a direct key file path
        algorithm = 'AES-256-GCM'
        if hasattr(config_or_path, 'get') and callable(getattr(config_or_path, 'get')):
            try:
                key_path = config_or_path.get('encryption.key_file', 'keys/encryption.key')
                algorithm = config_or_path.get('encryption.algorithm', algorithm)
            except Exception:
                key_path = 'keys/encryption.key'
        else:
            key_path = config_or_path
        
        self.key_file = Path(key_path) if not isinstance(key_path, Path) else key_path
        self.algorithm = algorithm
        self.key: Optional[bytes] = None
        self.fernet: Optional[Fernet] = None
        self.backend = default_backend()
        
        # Ensure key directory exists
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        self._initialize_encryption()

    def _initialize_encryption(self) -> bool:
        """Wrapper to initialize encryption during construction."""
        try:
            return self.init_encryption()
        except Exception as e:
            logger.error(f"Failed to initialize encryption in constructor: {e}")
            return False
    
    def init_encryption(self) -> bool:
        """Initialize encryption system."""
        try:
            if self.key_file.exists():
                self.key = self._load_key()
            else:
                logger.info("No encryption key found, generating new key")
                self.key = self._generate_key()
                self._save_key(self.key)
            
            if self.key:
                self.fernet = Fernet(base64.urlsafe_b64encode(self.key[:32]))
                logger.info("Encryption system initialized successfully")
                return True
            else:
                logger.error("Failed to initialize encryption key")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing encryption: {e}")
            return False
    
    def _generate_key(self) -> bytes:
        """Generate a new encryption key."""
        try:
            # Generate 32 bytes (256 bits) for AES-256
            key = secrets.token_bytes(32)
            logger.info("New encryption key generated")
            return key
            
        except Exception as e:
            logger.error(f"Error generating encryption key: {e}")
            raise
    
    def generate_key_from_password(self, password: str, salt: Optional[bytes] = None, iterations: int = 100_000) -> bytes:
        """Derive a 32-byte key from a password using PBKDF2-HMAC-SHA256."""
        if salt is None:
            salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=self.backend,
        )
        key = kdf.derive(password.encode('utf-8'))
        return key

    def _save_key(self, key: bytes) -> bool:
        """Save encryption key to file."""
        try:
            # Encode key as base64 for storage
            encoded_key = base64.b64encode(key).decode('utf-8')
            
            with open(self.key_file, 'w', encoding='utf-8') as f:
                f.write(encoded_key)
            
            # Set restrictive permissions on Unix-like systems
            if os.name != 'nt':
                os.chmod(self.key_file, 0o600)
            
            logger.info(f"Encryption key saved to {self.key_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving encryption key: {e}")
            return False

    def save_key(self, key: bytes) -> bool:
        """Save a specific key to file."""
        return self._save_key(key)
    
    def _load_key(self) -> Optional[bytes]:
        """Load encryption key from file."""
        try:
            with open(self.key_file, 'r', encoding='utf-8') as f:
                encoded_key = f.read().strip()
            
            key = base64.b64decode(encoded_key.encode('utf-8'))
            logger.info(f"Encryption key loaded from {self.key_file}")
            return key
            
        except Exception as e:
            logger.error(f"Error loading encryption key: {e}")
            return None

    def load_key(self) -> Optional[bytes]:
        """Public wrapper to load the encryption key from file."""
        return self._load_key()
    
    def encrypt_data(self, data: Union[str, bytes]) -> Optional[bytes]:
        """Encrypt data using AES-256-GCM."""
        try:
            if not self.key:
                logger.error("No encryption key available")
                return None
            
            # Convert string to bytes if necessary
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Generate random IV (12 bytes for GCM)
            iv = secrets.token_bytes(12)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.GCM(iv),
                backend=self.backend
            )
            
            # Encrypt data
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Combine IV + tag + ciphertext
            encrypted_data = iv + encryptor.tag + ciphertext
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: bytes) -> Optional[str]:
        """Decrypt data using AES-256-GCM and return UTF-8 string."""
        try:
            if not self.key:
                logger.error("No encryption key available")
                return None
            
            if len(encrypted_data) < 28:  # 12 (IV) + 16 (tag) minimum
                logger.error("Invalid encrypted data length")
                return None
            
            # Extract components
            iv = encrypted_data[:12]
            tag = encrypted_data[12:28]
            ciphertext = encrypted_data[28:]
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.GCM(iv, tag),
                backend=self.backend
            )
            
            # Decrypt data
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return None
    
    def encrypt_string(self, text: str) -> Optional[str]:
        """Encrypt string and return base64 encoded result."""
        try:
            encrypted_bytes = self.encrypt_data(text)
            if encrypted_bytes:
                return base64.b64encode(encrypted_bytes).decode('utf-8')
            return None
            
        except Exception as e:
            logger.error(f"Error encrypting string: {e}")
            return None
    
    def decrypt_string(self, encrypted_text: str) -> Optional[str]:
        """Decrypt base64 encoded string and return UTF-8 string."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
            decrypted_text = self.decrypt_data(encrypted_bytes)
            return decrypted_text
            
        except Exception as e:
            logger.error(f"Error decrypting string: {e}")
            return None
    
    def encrypt_file(self, input_file: str, output_file: str) -> bool:
        """Encrypt entire file."""
        try:
            with open(input_file, 'rb') as f:
                data = f.read()
            
            encrypted_data = self.encrypt_data(data)
            if not encrypted_data:
                return False
            
            with open(output_file, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info(f"File encrypted: {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error encrypting file: {e}")
            return False
    
    def decrypt_file(self, input_file: str, output_file: str) -> bool:
        """Decrypt entire file."""
        try:
            with open(input_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_text = self.decrypt_data(encrypted_data)
            if decrypted_text is None:
                return False
            
            to_write = decrypted_text.encode('utf-8') if isinstance(decrypted_text, str) else decrypted_text
            with open(output_file, 'wb') as f:
                f.write(to_write)
            
            logger.info(f"File decrypted: {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error decrypting file: {e}")
            return False
    
    def hash_data(self, data: Union[str, bytes]) -> str:
        """Create SHA-256 hash of data."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
            digest.update(data)
            hash_bytes = digest.finalize()
            
            return base64.b64encode(hash_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error hashing data: {e}")
            return ""
    
    def generate_random_key(self) -> bytes:
        """Generate a new random encryption key."""
        return self._generate_key()
    
    def rotate_key(self) -> bool:
        """Generate and save a new encryption key."""
        try:
            # Backup old key if it exists
            if self.key_file.exists():
                backup_file = self.key_file.with_suffix('.key.backup')
                self.key_file.rename(backup_file)
                logger.info(f"Old key backed up to {backup_file}")
            
            # Generate new key
            new_key = self._generate_key()
            if self._save_key(new_key):
                self.key = new_key
                self.fernet = Fernet(base64.urlsafe_b64encode(self.key[:32]))
                logger.info("Encryption key rotated successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error rotating encryption key: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if encryption is properly initialized."""
        return self.key is not None and self.fernet is not None
    
    def get_key_info(self) -> dict:
        """Get information about the current encryption key."""
        try:
            info = {
                'key_file': str(self.key_file),
                'key_exists': self.key_file.exists(),
                'initialized': self.is_initialized(),
                'algorithm': self.algorithm,
            }
            
            if self.key_file.exists():
                stat = self.key_file.stat()
                info['key_size'] = stat.st_size
                info['created'] = stat.st_ctime
                info['modified'] = stat.st_mtime
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting key info: {e}")
            return {'error': str(e)}
    
    def verify_key(self) -> bool:
        """Verify that the encryption key works correctly."""
        try:
            test_data = "encryption_test_data_12345"
            
            # Test encryption
            encrypted = self.encrypt_string(test_data)
            if not encrypted:
                return False
            
            # Test decryption
            decrypted = self.decrypt_string(encrypted)
            if decrypted != test_data:
                return False
            
            logger.info("Encryption key verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Encryption key verification failed: {e}")
            return False
    
    def secure_delete_key(self) -> bool:
        """Securely delete the encryption key file."""
        try:
            if not self.key_file.exists():
                return True
            
            # Overwrite file with random data multiple times
            file_size = self.key_file.stat().st_size
            
            with open(self.key_file, 'r+b') as f:
                for _ in range(3):  # Overwrite 3 times
                    f.seek(0)
                    f.write(secrets.token_bytes(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Delete the file
            self.key_file.unlink()
            
            # Clear memory
            self.key = None
            self.fernet = None
            
            logger.info("Encryption key securely deleted")
            return True
            
        except Exception as e:
            logger.error(f"Error securely deleting key: {e}")
            return False