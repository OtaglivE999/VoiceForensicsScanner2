"""
Audio Stream Manager
Real-time audio capture with buffering and segmentation.

Version: 1.0
"""

import numpy as np
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)

# Try to import sounddevice
try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    logger.warning("sounddevice not available")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StreamConfig:
    """Configuration for audio stream."""
    
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = 'float32'
    
    # Device selection
    device_id: Optional[int] = None
    device_name: Optional[str] = None
    
    # Buffering
    buffer_seconds: float = 30.0
    segment_duration_ms: int = 2000
    overlap_ms: int = 500
    blocksize_ms: int = 100
    
    # Processing
    normalize: bool = True
    remove_dc: bool = True


# ============================================================================
# Audio Stream Manager
# ============================================================================

class AudioStreamManager:
    """
    Manages real-time audio capture with buffering.
    
    Features:
    - Device auto-detection and selection
    - Circular buffer for continuous capture
    - Segmented output with configurable overlap
    - Thread-safe operation
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        """
        Initialize the audio stream manager.
        
        Args:
            config: Stream configuration
        """
        if not SD_AVAILABLE:
            raise RuntimeError("sounddevice library not available")
        
        self.config = config or StreamConfig()
        
        # State
        self._running = False
        self._stream = None
        self._device_info: Optional[Dict] = None
        
        # Buffer
        buffer_samples = int(self.config.buffer_seconds * self.config.sample_rate)
        self._buffer = np.zeros(buffer_samples, dtype=np.float32)
        self._buffer_pos = 0
        self._buffer_lock = threading.Lock()
        
        # Segmentation
        self._segment_samples = int(self.config.segment_duration_ms * self.config.sample_rate / 1000)
        self._overlap_samples = int(self.config.overlap_ms * self.config.sample_rate / 1000)
        self._last_segment_end = 0
        
        # Callback
        self._segment_callback: Optional[Callable] = None
        
        # Worker
        self._segment_thread: Optional[threading.Thread] = None
        
        # Select device
        self._select_device()
    
    def _select_device(self):
        """Select audio input device."""
        devices = sd.query_devices()
        
        # If device_name specified, search for it
        if self.config.device_name:
            for i, dev in enumerate(devices):
                if self.config.device_name.lower() in dev['name'].lower():
                    if dev['max_input_channels'] > 0:
                        self.config.device_id = i
                        self._device_info = dev
                        logger.info(f"Selected device: {dev['name']} (ID: {i})")
                        return
            logger.warning(f"Device '{self.config.device_name}' not found")
        
        # If device_id specified, use it
        if self.config.device_id is not None:
            self._device_info = devices[self.config.device_id]
            logger.info(f"Using device ID {self.config.device_id}: {self._device_info['name']}")
            return
        
        # Use default input device
        default_input = sd.default.device[0]
        if default_input is not None:
            self.config.device_id = default_input
            self._device_info = devices[default_input]
            logger.info(f"Using default device: {self._device_info['name']}")
        else:
            raise RuntimeError("No audio input device available")
    
    def list_devices(self) -> List[Dict]:
        """List available audio input devices."""
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                devices.append({
                    'id': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate']
                })
        return devices
    
    def start(self, callback: Optional[Callable] = None):
        """
        Start audio capture.
        
        Args:
            callback: Function called with (audio_array, timestamp) for each segment
        """
        if self._running:
            logger.warning("Stream already running")
            return
        
        self._segment_callback = callback
        self._running = True
        
        # Calculate blocksize
        blocksize = int(self.config.blocksize_ms * self.config.sample_rate / 1000)
        
        # Create input stream
        self._stream = sd.InputStream(
            device=self.config.device_id,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            dtype=self.config.dtype,
            blocksize=blocksize,
            callback=self._audio_callback
        )
        
        # Start segmentation thread
        self._segment_thread = threading.Thread(target=self._segmentation_worker, daemon=True)
        self._segment_thread.start()
        
        # Start stream
        self._stream.start()
        
        logger.info(f"Audio stream started: {self._device_info['name']} @ {self.config.sample_rate}Hz")
    
    def stop(self):
        """Stop audio capture."""
        if not self._running:
            return
        
        self._running = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        if self._segment_thread:
            self._segment_thread.join(timeout=2.0)
        
        logger.info("Audio stream stopped")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for incoming audio data."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert to mono if needed
        if indata.ndim > 1:
            audio = indata[:, 0].copy()
        else:
            audio = indata.flatten().copy()
        
        # Normalize
        if self.config.normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
        
        # Remove DC offset
        if self.config.remove_dc:
            audio = audio - np.mean(audio)
        
        # Add to buffer
        with self._buffer_lock:
            n_samples = len(audio)
            
            # Handle wrap-around
            if self._buffer_pos + n_samples <= len(self._buffer):
                self._buffer[self._buffer_pos:self._buffer_pos + n_samples] = audio
            else:
                # Split across buffer boundary
                first_part = len(self._buffer) - self._buffer_pos
                self._buffer[self._buffer_pos:] = audio[:first_part]
                self._buffer[:n_samples - first_part] = audio[first_part:]
            
            self._buffer_pos = (self._buffer_pos + n_samples) % len(self._buffer)
    
    def _segmentation_worker(self):
        """Worker thread for extracting segments from buffer."""
        segment_interval = (self.config.segment_duration_ms - self.config.overlap_ms) / 1000
        
        while self._running:
            time.sleep(segment_interval)
            
            if not self._segment_callback:
                continue
            
            # Extract segment from buffer
            segment = self._extract_segment()
            
            if segment is not None:
                timestamp = datetime.now(timezone.utc)
                try:
                    self._segment_callback(segment, timestamp)
                except Exception as e:
                    logger.error(f"Segment callback error: {e}")
    
    def _extract_segment(self) -> Optional[np.ndarray]:
        """Extract a segment from the circular buffer."""
        with self._buffer_lock:
            # Calculate segment boundaries
            end_pos = self._buffer_pos
            start_pos = (end_pos - self._segment_samples) % len(self._buffer)
            
            # Extract segment (handle wrap-around)
            if start_pos < end_pos:
                segment = self._buffer[start_pos:end_pos].copy()
            else:
                segment = np.concatenate([
                    self._buffer[start_pos:],
                    self._buffer[:end_pos]
                ])
            
            return segment
    
    def get_current_buffer(self) -> np.ndarray:
        """Get a copy of the current buffer content."""
        with self._buffer_lock:
            # Reorder buffer to be chronological
            return np.concatenate([
                self._buffer[self._buffer_pos:],
                self._buffer[:self._buffer_pos]
            ])
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def device_name(self) -> str:
        return self._device_info['name'] if self._device_info else "Unknown"
    
    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate


# ============================================================================
# Utility Functions
# ============================================================================

def list_audio_devices() -> List[Dict]:
    """List all available audio input devices."""
    if not SD_AVAILABLE:
        return []
    
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            devices.append({
                'id': i,
                'name': dev['name'],
                'channels': dev['max_input_channels'],
                'sample_rate': dev['default_samplerate'],
                'is_default': i == sd.default.device[0]
            })
    return devices


def find_device_by_name(name: str) -> Optional[int]:
    """Find device ID by partial name match."""
    if not SD_AVAILABLE:
        return None
    
    for i, dev in enumerate(sd.query_devices()):
        if name.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
            return i
    return None
