"""
Soft Voice Far-Field Scanner
=============================

Enhanced scanner optimized for detecting:
- SOFT/FAINT voices at far-field and ultra-far-field distances
- Whispers and low-volume speech
- Voices in noisy/crowd environments
- Multiple speakers with quick interchanges
- Multi-language support (embedding-based)

Features:
- Adaptive gain control for faint audio
- Multi-band energy detection
- Lower SNR thresholds for soft voices
- Spectral analysis for voice detection
- Enhanced sensitivity mode

Output Format:
[HH:MM:SS] 🔇 No voice detected
[HH:MM:SS] 🔄 Speaker Name | FIELD | SNR: XXdB | Conf: XX% [verifying]
[HH:MM:SS] 🆕 NEW VOICE_XXXX | FIELD | SNR: XXdB | Conf: XX%
[HH:MM:SS] ✅ Speaker Name (ID) | FIELD | SNR: XXdB | Conf: XX% [xN]
[HH:MM:SS] 🔊 SOFT Speaker Name (ID) | ULTRA-FAR | SNR: XdB | Conf: XX% [soft]

Usage:
    python soft_voice_scanner.py                           # Default
    python soft_voice_scanner.py --ultra-sensitive        # Max sensitivity
    python soft_voice_scanner.py --min-snr 2              # Very low SNR threshold
"""

import argparse
import sys
import os
import json
import time
import threading
import queue
import wave
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import sounddevice as sd
from scipy.ndimage import uniform_filter1d
from scipy import signal
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# Import resemblyzer
try:
    from resemblyzer import VoiceEncoder
    ENCODER = None
except ImportError:
    print("Error: resemblyzer not installed. Run: pip install resemblyzer")
    sys.exit(1)

# Paths
BASE_DIR = Path(__file__).parent
LIBRARY_DIR = BASE_DIR / "voice_library"
LIBRARY_DIR.mkdir(exist_ok=True)
LIBRARY_FILE = LIBRARY_DIR / "voice_fingerprints.json"
SPEAKERS_DB = BASE_DIR / "speakers" / "discovered_speakers.json"
SCAN_LOG_DIR = LIBRARY_DIR / "soft_voice_logs"
SCAN_LOG_DIR.mkdir(exist_ok=True)
SHARED_STATE_FILE = BASE_DIR / "scanner_state.json"

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1

# Field type thresholds (SNR-based) - EXTENDED for soft voices
FIELD_THRESHOLDS = {
    'NEAR': 35,           # > 35 dB (0-1m)
    'MEDIUM': 25,         # 25-35 dB (1-3m)
    'FAR': 15,            # 15-25 dB (3-6m)
    'ULTRA-FAR': 6,       # 6-15 dB (6m+)
    'EXTREME': 2,         # 2-6 dB (crowd/very far)
    'WHISPER': 0          # < 2 dB (whisper/faint)
}

# Voice frequency bands (Hz)
VOICE_BANDS = {
    'low': (80, 300),      # Fundamental frequency
    'mid': (300, 2000),    # Primary speech
    'high': (2000, 4000)   # Consonants/sibilants
}

# PersonaPlex settings
MIMI_SAMPLE_RATE = 24000  # Mimi codec native rate


class NeuralEnhancer:
    """
    Neural audio enhancer for far-field voice enhancement.
    Uses spectral subtraction and noise reduction for degraded audio.
    """
    
    def __init__(self):
        self.available = False
        self.method = 'none'
        self._initialize()
    
    def _initialize(self):
        """Initialize neural enhancer"""
        try:
            # Try noisereduce (best for far-field)
            import noisereduce as nr
            self.nr = nr
            self.available = True
            self.method = 'noisereduce'
            logger.info("✓ Neural enhancer loaded (noisereduce)")
            logger.info("   Optimized for far-field/degraded audio")
            return
        except ImportError:
            pass
        
        # Fallback to spectral enhancement
        try:
            from scipy import signal as scipy_signal
            self.scipy_signal = scipy_signal
            self.available = True
            self.method = 'spectral'
            logger.info("✓ Neural enhancer loaded (spectral)")
            return
        except:
            pass
        
        logger.info("Neural enhancement disabled (install: pip install noisereduce)")
    
    def enhance(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """
        Enhance far-field audio using neural noise reduction.
        """
        if not self.available or len(audio) < 1600:  # Need 0.1s minimum
            return audio
        
        try:
            if self.method == 'noisereduce':
                # Advanced noise reduction for far-field
                enhanced = self.nr.reduce_noise(
                    y=audio.astype(np.float32),
                    sr=sample_rate,
                    stationary=False,  # Adaptive for varying noise
                    prop_decrease=0.8,  # Aggressive reduction for degraded audio
                    freq_mask_smooth_hz=500,  # Smooth frequency response
                    time_mask_smooth_ms=50  # Temporal smoothing
                )
                return enhanced.astype(np.float32)
            
            elif self.method == 'spectral':
                # Spectral subtraction for noise reduction
                return self._spectral_enhance(audio, sample_rate)
            
        except Exception as e:
            logger.debug(f"Neural enhancement failed: {e}")
        
        return audio
    
    def _spectral_enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Spectral subtraction enhancement"""
        # Apply Wiener filtering
        try:
            # Estimate noise from silent portions
            rms = np.sqrt(np.mean(audio ** 2))
            noise_threshold = rms * 0.1
            
            # Simple spectral subtraction
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            phase = np.angle(fft)
            
            # Estimate noise floor
            noise_floor = np.percentile(magnitude, 10)
            
            # Subtract noise (with floor)
            enhanced_mag = np.maximum(magnitude - noise_floor * 1.5, magnitude * 0.1)
            
            # Reconstruct
            enhanced_fft = enhanced_mag * np.exp(1j * phase)
            enhanced = np.fft.irfft(enhanced_fft, len(audio))
            
            return enhanced.astype(np.float32)
        except:
            return audio


class ContinuousRecorder:
    """
    Continuous audio recorder that saves long recording files.
    Records both raw and enhanced audio with speaker annotations.
    """
    
    def __init__(self, output_dir: Path = None, max_file_duration: int = 3600,
                 enable_neural: bool = True):
        """
        Args:
            output_dir: Directory to save recordings
            max_file_duration: Max duration per file in seconds (default 1 hour)
            enable_neural: Enable neural enhancement (noisereduce)
        """
        self.output_dir = output_dir or (BASE_DIR / "recordings")
        self.output_dir.mkdir(exist_ok=True)
        
        self.max_file_duration = max_file_duration  # 1 hour per file
        self.sample_rate = SAMPLE_RATE
        
        # Initialize Neural Enhancer
        self.neural = NeuralEnhancer() if enable_neural else None
        self.neural_enabled = enable_neural and (self.neural and self.neural.available)
        
        # Current recording state
        self.recording = False
        self.current_raw_file = None
        self.current_enhanced_file = None
        self.current_neural_file = None
        self.current_metadata_file = None
        self.raw_writer = None
        self.enhanced_writer = None
        self.neural_writer = None
        
        self.file_start_time = None
        self.total_samples = 0
        self.session_start = None
        
        # Detection log for metadata
        self.detections = []
        
        # Buffers for audio
        self.raw_buffer = []
        self.enhanced_buffer = []
        self.neural_buffer = []
        
        logger.info(f"📼 ContinuousRecorder initialized: {self.output_dir}")
        if self.neural_enabled:
            logger.info(f"   🧠 Neural enhancement: ENABLED ({self.neural.method})")
        else:
            logger.info("   🧠 Neural enhancement: DISABLED")
    
    def start(self):
        """Start a new recording session."""
        self.recording = True
        self.session_start = datetime.now()
        self._start_new_file()
        logger.info(f"🔴 Recording started: {self.current_raw_file}")
    
    def _start_new_file(self):
        """Start a new recording file pair."""
        # Close existing files if any
        self._close_current_files()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create file paths
        self.current_raw_file = self.output_dir / f"raw_{timestamp}.wav"
        self.current_enhanced_file = self.output_dir / f"enhanced_{timestamp}.wav"
        self.current_metadata_file = self.output_dir / f"metadata_{timestamp}.json"
        
        # Neural enhancement file
        if self.neural_enabled:
            self.current_neural_file = self.output_dir / f"neural_{timestamp}.wav"
        
        # Open WAV writers
        self.raw_writer = wave.open(str(self.current_raw_file), 'wb')
        self.raw_writer.setnchannels(1)
        self.raw_writer.setsampwidth(2)  # 16-bit
        self.raw_writer.setframerate(self.sample_rate)
        
        self.enhanced_writer = wave.open(str(self.current_enhanced_file), 'wb')
        self.enhanced_writer.setnchannels(1)
        self.enhanced_writer.setsampwidth(2)
        self.enhanced_writer.setframerate(self.sample_rate)
        
        # Neural enhancement writer
        if self.neural_enabled:
            self.neural_writer = wave.open(str(self.current_neural_file), 'wb')
            self.neural_writer.setnchannels(1)
            self.neural_writer.setsampwidth(2)
            self.neural_writer.setframerate(self.sample_rate)
        
        self.file_start_time = time.time()
        self.total_samples = 0
        self.detections = []
    
    def write(self, raw_audio: np.ndarray, enhanced_audio: np.ndarray = None,
              detection: dict = None):
        """
        Write audio chunk to recording files.
        
        Args:
            raw_audio: Raw audio samples (float32)
            enhanced_audio: Enhanced audio samples (optional, uses raw if None)
            detection: Detection metadata (speaker, field, etc.)
        """
        if not self.recording:
            return
        
        # Check if we need to rotate files (max duration reached)
        elapsed = time.time() - self.file_start_time
        if elapsed >= self.max_file_duration:
            self._save_metadata()
            self._start_new_file()
            logger.info(f"📼 Rotated to new file: {self.current_raw_file}")
        
        # Convert to 16-bit PCM
        raw_pcm = (np.clip(raw_audio, -1, 1) * 32767).astype(np.int16)
        
        if enhanced_audio is not None:
            enhanced_pcm = (np.clip(enhanced_audio, -1, 1) * 32767).astype(np.int16)
        else:
            enhanced_pcm = raw_pcm
        
        # Write to files
        self.raw_writer.writeframes(raw_pcm.tobytes())
        self.enhanced_writer.writeframes(enhanced_pcm.tobytes())
        
        # Write neural-enhanced audio
        if self.neural_enabled and self.neural_writer:
            # Enhance through neural enhancer
            neural_audio = self.neural.enhance(enhanced_audio if enhanced_audio is not None else raw_audio)
            neural_pcm = (np.clip(neural_audio, -1, 1) * 32767).astype(np.int16)
            self.neural_writer.writeframes(neural_pcm.tobytes())
        
        self.total_samples += len(raw_audio)
        
        # Log detection
        if detection:
            detection['sample_offset'] = self.total_samples - len(raw_audio)
            detection['timestamp'] = time.time() - self.file_start_time
            self.detections.append(detection)
    
    def _save_metadata(self):
        """Save metadata JSON file."""
        if self.current_metadata_file:
            metadata = {
                'file_start': self.file_start_time,
                'duration_seconds': self.total_samples / self.sample_rate,
                'total_samples': self.total_samples,
                'sample_rate': self.sample_rate,
                'raw_file': self.current_raw_file.name,
                'enhanced_file': self.current_enhanced_file.name,
                'neural_file': self.current_neural_file.name if self.neural_enabled else None,
                'neural_enabled': self.neural_enabled,
                'detections': self.detections
            }
            with open(self.current_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    
    def _close_current_files(self):
        """Close current WAV files."""
        if self.raw_writer:
            self.raw_writer.close()
            self.raw_writer = None
        if self.enhanced_writer:
            self.enhanced_writer.close()
            self.enhanced_writer = None
        if self.neural_writer:
            self.neural_writer.close()
            self.neural_writer = None
    
    def stop(self):
        """Stop recording and finalize files."""
        if not self.recording:
            return
        
        self.recording = False
        self._save_metadata()
        self._close_current_files()
        
        duration = self.total_samples / self.sample_rate
        logger.info(f"⏹️  Recording stopped. Duration: {duration:.1f}s")
        logger.info(f"   Raw: {self.current_raw_file}")
        logger.info(f"   Enhanced: {self.current_enhanced_file}")
        if self.neural_enabled:
            logger.info(f"   Neural: {self.current_neural_file}")
        logger.info(f"   Metadata: {self.current_metadata_file}")
    
    @property
    def is_recording(self):
        return self.recording
    
    @property
    def current_duration(self):
        """Get current file duration in seconds."""
        return self.total_samples / self.sample_rate


def get_encoder():
    """Lazy load encoder."""
    global ENCODER
    if ENCODER is None:
        logger.info("Loading voice encoder...")
        ENCODER = VoiceEncoder()
        logger.info("✓ Voice encoder loaded")
    return ENCODER


@dataclass
class SpeakerTrack:
    """Tracks a speaker across time."""
    id: str
    name: str
    embedding: np.ndarray
    last_seen: float = 0.0
    first_seen: float = 0.0
    utterance_count: int = 0
    total_duration: float = 0.0
    confidence_history: List[float] = field(default_factory=list)
    field_types: Dict[str, int] = field(default_factory=dict)
    active: bool = True
    is_new: bool = False
    is_verifying: bool = True
    consecutive_count: int = 0
    soft_voice_count: int = 0  # Track soft voice detections
    
    @property
    def avg_confidence(self) -> float:
        if not self.confidence_history:
            return 0.0
        return float(np.mean(self.confidence_history[-20:]))
    
    @property
    def display_name(self) -> str:
        if self.name.startswith('VOICE_'):
            num = self.name.split('_')[-1]
            return f"Unknown Speaker {int(num)}"
        return self.name
    
    def update(self, embedding: np.ndarray, confidence: float, 
               field_type: str, duration: float, is_soft: bool = False):
        """Update speaker track."""
        alpha = 0.1
        self.embedding = (1 - alpha) * self.embedding + alpha * embedding
        self.embedding /= np.linalg.norm(self.embedding)
        
        self.last_seen = time.time()
        self.utterance_count += 1
        self.total_duration += duration
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 50:
            self.confidence_history = self.confidence_history[-50:]
        
        self.field_types[field_type] = self.field_types.get(field_type, 0) + 1
        
        if is_soft:
            self.soft_voice_count += 1
        
        if self.utterance_count >= 2:
            self.is_verifying = False


class AutomaticGainControl:
    """
    Automatic Gain Control (AGC) for handling hot/clipped audio input.
    Normalizes audio to optimal levels for voice detection.
    """
    
    def __init__(self, target_rms: float = 0.1, attack_time: float = 0.01,
                 release_time: float = 0.1, max_gain: float = 10.0,
                 min_gain: float = 0.01):
        self.target_rms = target_rms
        self.attack_coef = 1.0 - np.exp(-1.0 / (attack_time * SAMPLE_RATE))
        self.release_coef = 1.0 - np.exp(-1.0 / (release_time * SAMPLE_RATE))
        self.max_gain = max_gain
        self.min_gain = min_gain
        self.current_gain = 1.0
        self.envelope = 0.0
        self.enabled = True
        self.clipping_detected = False
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply AGC to audio signal."""
        if not self.enabled or len(audio) == 0:
            return audio
        
        # Detect if audio is clipping/saturated
        max_val = np.max(np.abs(audio))
        self.clipping_detected = max_val > 0.95
        
        # Calculate current RMS
        current_rms = np.sqrt(np.mean(audio ** 2))
        
        if current_rms < 1e-10:
            return audio
        
        # Calculate desired gain
        desired_gain = self.target_rms / current_rms
        desired_gain = np.clip(desired_gain, self.min_gain, self.max_gain)
        
        # Smooth gain changes (attack/release)
        if desired_gain < self.current_gain:
            # Attack (fast reduction for loud signals)
            coef = self.attack_coef
        else:
            # Release (slow increase for quiet signals)
            coef = self.release_coef
        
        self.current_gain = (1 - coef) * self.current_gain + coef * desired_gain
        
        # Apply gain
        processed = audio * self.current_gain
        
        # Soft limiting to prevent clipping
        processed = np.tanh(processed * 1.5) / 1.5
        
        return processed.astype(np.float32)
    
    def get_status(self) -> dict:
        """Get AGC status for display."""
        return {
            'gain': self.current_gain,
            'gain_db': 20 * np.log10(self.current_gain + 1e-10),
            'clipping': self.clipping_detected,
            'enabled': self.enabled
        }


class AdaptiveNoiseEstimator:
    """Enhanced noise estimator for soft voice detection."""
    
    def __init__(self, window_sec: float = 2.0, update_rate: float = 0.05):
        self.window_samples = int(window_sec * SAMPLE_RATE)
        self.update_rate = update_rate
        self.noise_floor = None
        self.noise_history = deque(maxlen=200)
        self.min_noise_floor = 0.0005  # Very low floor for soft voices
        
    def update(self, audio: np.ndarray) -> float:
        """Update noise estimate with lower floor for soft voices."""
        frame_size = int(0.02 * SAMPLE_RATE)
        energies = []
        
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i+frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            energies.append(energy)
        
        if not energies:
            return self.noise_floor or self.min_noise_floor
        
        # Use 5th percentile for better soft voice detection
        noise_estimate = np.percentile(energies, 5)
        self.noise_history.append(noise_estimate)
        
        if self.noise_floor is None:
            self.noise_floor = max(noise_estimate, self.min_noise_floor)
        else:
            # Slower adaptation for stability
            self.noise_floor = (1 - self.update_rate) * self.noise_floor + \
                               self.update_rate * noise_estimate
            self.noise_floor = max(self.noise_floor, self.min_noise_floor)
        
        return self.noise_floor
    
    def get_snr(self, signal_energy: float) -> float:
        """Calculate SNR - enhanced for low signals."""
        if self.noise_floor is None or self.noise_floor < 1e-10:
            return 20.0
        ratio = signal_energy / self.noise_floor
        if ratio <= 0:
            return 0.0
        return 20 * np.log10(ratio + 1e-10)


class SoftVoiceDetector:
    """Enhanced voice detector for soft/faint voices."""
    
    def __init__(self, ultra_sensitive: bool = False):
        self.ultra_sensitive = ultra_sensitive
        
        # Lower thresholds for soft voice detection
        self.base_threshold = 0.008 if ultra_sensitive else 0.012
        self.soft_threshold = 0.004 if ultra_sensitive else 0.006
        
        # Spectral parameters
        self.voice_freq_low = 80
        self.voice_freq_high = 4000
        
    def compute_spectral_energy(self, audio: np.ndarray) -> Tuple[float, float, float]:
        """Compute energy in voice frequency bands."""
        if len(audio) < 512:
            return 0.0, 0.0, 0.0
        
        # Compute FFT
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)
        
        # Band energies
        low_mask = (freqs >= VOICE_BANDS['low'][0]) & (freqs < VOICE_BANDS['low'][1])
        mid_mask = (freqs >= VOICE_BANDS['mid'][0]) & (freqs < VOICE_BANDS['mid'][1])
        high_mask = (freqs >= VOICE_BANDS['high'][0]) & (freqs < VOICE_BANDS['high'][1])
        
        low_energy = np.sqrt(np.mean(fft[low_mask] ** 2)) if np.any(low_mask) else 0
        mid_energy = np.sqrt(np.mean(fft[mid_mask] ** 2)) if np.any(mid_mask) else 0
        high_energy = np.sqrt(np.mean(fft[high_mask] ** 2)) if np.any(high_mask) else 0
        
        return low_energy, mid_energy, high_energy
    
    def is_voice_like(self, audio: np.ndarray) -> Tuple[bool, bool, float]:
        """
        Detect if audio contains voice-like content.
        Returns: (is_voice, is_soft_voice, voice_score)
        """
        if len(audio) < 320:  # 20ms minimum
            return False, False, 0.0
        
        # Time-domain energy
        energy = np.sqrt(np.mean(audio ** 2))
        
        # Spectral analysis
        low_e, mid_e, high_e = self.compute_spectral_energy(audio)
        
        # Voice typically has strong mid-frequency content
        spectral_ratio = mid_e / (low_e + high_e + 1e-10)
        
        # Zero-crossing rate (voice has moderate ZCR)
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        voice_zcr_range = (0.02, 0.25)  # Typical voice ZCR range
        
        # Voice score based on multiple features
        voice_score = 0.0
        
        # Energy contribution
        if energy > self.base_threshold:
            voice_score += 0.3
        elif energy > self.soft_threshold:
            voice_score += 0.15
        
        # Spectral ratio contribution (voice has strong mids)
        if spectral_ratio > 1.5:
            voice_score += 0.3
        elif spectral_ratio > 0.8:
            voice_score += 0.15
        
        # ZCR contribution
        if voice_zcr_range[0] <= zcr <= voice_zcr_range[1]:
            voice_score += 0.2
        
        # Mid-frequency energy contribution
        if mid_e > self.soft_threshold:
            voice_score += 0.2
        
        is_voice = voice_score >= 0.4
        is_soft = is_voice and energy < self.base_threshold
        
        return is_voice, is_soft, voice_score
    
    def detect_segments(self, audio: np.ndarray, noise_floor: float,
                        min_samples: int) -> List[Tuple[int, int, bool]]:
        """
        Detect voice segments with soft voice identification.
        Returns: List of (start, end, is_soft)
        """
        frame_size = int(0.02 * SAMPLE_RATE)  # 20ms frames
        hop_size = int(0.01 * SAMPLE_RATE)    # 10ms hop
        
        # Analyze frames
        frame_data = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            is_voice, is_soft, score = self.is_voice_like(frame)
            frame_data.append({
                'energy': energy,
                'is_voice': is_voice,
                'is_soft': is_soft,
                'score': score
            })
        
        if not frame_data:
            return []
        
        # Smooth voice detection
        scores = np.array([f['score'] for f in frame_data])
        smoothed = uniform_filter1d(scores, size=5)
        
        # Adaptive threshold based on sensitivity
        threshold = 0.3 if self.ultra_sensitive else 0.4
        voice_mask = smoothed >= threshold
        
        # Find segments
        segments = []
        in_segment = False
        start_idx = 0
        segment_soft = False
        
        for i, is_voice in enumerate(voice_mask):
            if is_voice and not in_segment:
                start_idx = i
                in_segment = True
                segment_soft = frame_data[i]['is_soft']
            elif is_voice and in_segment:
                # Track if any frame is soft
                segment_soft = segment_soft or frame_data[i]['is_soft']
            elif not is_voice and in_segment:
                end_idx = i
                start_sample = start_idx * hop_size
                end_sample = min(end_idx * hop_size + frame_size, len(audio))
                
                if end_sample - start_sample >= min_samples:
                    # Check if segment is predominantly soft
                    segment_scores = scores[start_idx:end_idx]
                    avg_energy = np.mean([frame_data[j]['energy'] 
                                         for j in range(start_idx, end_idx)])
                    is_soft_segment = avg_energy < self.base_threshold
                    
                    segments.append((start_sample, end_sample, is_soft_segment))
                
                in_segment = False
        
        # Handle segment at end
        if in_segment:
            end_sample = len(audio)
            if end_sample - start_idx * hop_size >= min_samples:
                segments.append((start_idx * hop_size, end_sample, segment_soft))
        
        return segments


class SpeakerDatabase:
    """Speaker database with soft voice support."""
    
    def __init__(self, similarity_threshold: float = 0.65):
        self.speakers: Dict[str, SpeakerTrack] = {}
        self.similarity_threshold = similarity_threshold
        self.active_speakers: Set[str] = set()
        self.speaker_timeout = 30.0
        self.next_voice_id = 1
        self.load()
        
    def load(self):
        """Load speakers from files."""
        max_id = 0
        
        if LIBRARY_FILE.exists():
            try:
                with open(LIBRARY_FILE) as f:
                    data = json.load(f)
                for fp in data.get('fingerprints', []):
                    if fp.get('embedding'):
                        sid = fp['id']
                        self.speakers[sid] = SpeakerTrack(
                            id=sid,
                            name=fp.get('name', sid),
                            embedding=np.array(fp['embedding'], dtype=np.float32),
                            utterance_count=fp.get('detection_count', 0),
                            total_duration=float(fp.get('total_duration_sec', 0)),
                            field_types=fp.get('field_types', {}),
                            is_verifying=False,
                            soft_voice_count=fp.get('soft_voice_count', 0)
                        )
                        if sid.startswith('VOICE_'):
                            try:
                                vid = int(sid.split('_')[-1])
                                max_id = max(max_id, vid)
                            except:
                                pass
                logger.info(f"Loaded {len(self.speakers)} speakers from library")
            except Exception as e:
                logger.error(f"Error loading library: {e}")
        
        if SPEAKERS_DB.exists():
            try:
                with open(SPEAKERS_DB) as f:
                    data = json.load(f)
                for sp in data.get('speakers', []):
                    if sp.get('centroid') and sp['id'] not in self.speakers:
                        sid = sp['id']
                        self.speakers[sid] = SpeakerTrack(
                            id=sid,
                            name=sp.get('name', sid),
                            embedding=np.array(sp['centroid'], dtype=np.float32),
                            utterance_count=sp.get('appearance_count', 0),
                            field_types=sp.get('field_types', {}),
                            is_verifying=False
                        )
                        if sid.startswith('VOICE_'):
                            try:
                                vid = int(sid.split('_')[-1])
                                max_id = max(max_id, vid)
                            except:
                                pass
                logger.info(f"Total speakers after merge: {len(self.speakers)}")
            except Exception as e:
                logger.error(f"Error loading discovered speakers: {e}")
        
        self.next_voice_id = max_id + 1
    
    def identify(self, embedding: np.ndarray, field_type: str = 'UNKNOWN',
                 is_soft: bool = False) -> Tuple[Optional[SpeakerTrack], float]:
        """Identify speaker with adjusted thresholds for soft voices."""
        if not self.speakers:
            return None, 0.0
        
        # Adjust threshold based on field type and soft voice
        threshold = self.similarity_threshold
        
        if field_type == 'FAR':
            threshold -= 0.05
        elif field_type == 'ULTRA-FAR':
            threshold -= 0.10
        elif field_type in ('EXTREME', 'WHISPER'):
            threshold -= 0.15
        
        # Further reduce threshold for soft voices
        if is_soft:
            threshold -= 0.05
        
        # Minimum threshold
        threshold = max(threshold, 0.45)
        
        best_match = None
        best_score = 0.0
        
        for sid, speaker in self.speakers.items():
            score = float(np.dot(embedding, speaker.embedding))
            if score > best_score:
                best_score = score
                best_match = speaker
        
        if best_match and best_score >= threshold:
            return best_match, best_score
        
        return None, best_score
    
    def add_speaker(self, embedding: np.ndarray, field_type: str,
                    is_soft: bool = False) -> SpeakerTrack:
        """Add new speaker."""
        speaker_id = f"VOICE_{self.next_voice_id:04d}"
        self.next_voice_id += 1
        
        speaker = SpeakerTrack(
            id=speaker_id,
            name=speaker_id,
            embedding=embedding,
            first_seen=time.time(),
            last_seen=time.time(),
            is_new=True,
            is_verifying=True,
            soft_voice_count=1 if is_soft else 0
        )
        speaker.field_types[field_type] = 1
        
        self.speakers[speaker_id] = speaker
        self.active_speakers.add(speaker_id)
        
        logger.info(f"📝 New fingerprint enrolled: {speaker_id}" + 
                   (" [soft voice]" if is_soft else ""))
        
        return speaker
    
    def update_active_speakers(self):
        """Update active speaker list."""
        current_time = time.time()
        for sid in list(self.active_speakers):
            if sid in self.speakers:
                if current_time - self.speakers[sid].last_seen > self.speaker_timeout:
                    self.active_speakers.discard(sid)
                    self.speakers[sid].active = False
    
    def save(self):
        """Save database."""
        fingerprints = []
        for sid, speaker in self.speakers.items():
            fingerprints.append({
                'id': speaker.id,
                'name': speaker.name,
                'embedding': speaker.embedding.tolist(),
                'detection_count': int(speaker.utterance_count),
                'total_duration_sec': float(speaker.total_duration),
                'avg_confidence': float(speaker.avg_confidence),
                'field_types': {k: int(v) for k, v in speaker.field_types.items()},
                'soft_voice_count': int(speaker.soft_voice_count),
                'first_seen': speaker.first_seen,
                'last_seen': speaker.last_seen
            })
        
        data = {
            'version': '2.1-soft-voice',
            'updated': datetime.now().isoformat(),
            'total_speakers': len(fingerprints),
            'fingerprints': fingerprints
        }
        
        with open(LIBRARY_FILE, 'w') as f:
            json.dump(data, f, indent=2)


class SoftVoiceScanner:
    """
    Scanner optimized for soft/faint voice detection at far-field distances.
    """
    
    def __init__(self,
                 ultra_sensitive: bool = False,
                 min_snr: float = 2.0,
                 auto_enroll: bool = True,
                 fast_switch: bool = True,
                 enable_recording: bool = False,
                 output_dir: Path = None,
                 enable_neural: bool = False):
        
        self.ultra_sensitive = ultra_sensitive
        self.min_snr = min_snr
        self.auto_enroll = auto_enroll
        self.fast_switch = fast_switch
        
        # Components
        self.encoder = get_encoder()
        self.noise_estimator = AdaptiveNoiseEstimator()
        self.voice_detector = SoftVoiceDetector(ultra_sensitive=ultra_sensitive)
        self.speaker_db = SpeakerDatabase(
            similarity_threshold=0.60 if ultra_sensitive else 0.65
        )
        
        # AGC (Automatic Gain Control) for hot/clipped inputs
        self.agc = AutomaticGainControl(
            target_rms=0.1,
            attack_time=0.01,
            release_time=0.1,
            max_gain=10.0,
            min_gain=0.01
        )
        self.agc_enabled = True
        
        # Continuous Recording (1 hour per file for long recordings)
        self.enable_recording = enable_recording
        self.recorder = None
        if enable_recording:
            self.recorder = ContinuousRecorder(
                output_dir=output_dir,
                max_file_duration=3600,  # 1 hour per file
                enable_neural=enable_neural
            )
        
        # Audio settings - shorter for fast response
        self.min_segment_sec = 0.25 if fast_switch else 0.4
        self.max_segment_sec = 3.0
        self.min_samples = int(self.min_segment_sec * SAMPLE_RATE)
        self.max_samples = int(self.max_segment_sec * SAMPLE_RATE)
        
        self.chunk_duration = 2.0
        self.chunk_samples = int(self.chunk_duration * SAMPLE_RATE)
        self.audio_queue = queue.Queue()
        
        # State
        self.running = False
        self.current_chunk = np.array([], dtype=np.float32)
        self.last_speaker_id = None
        self.last_detection_time = 0
        self.no_voice_printed = False
        self.consecutive_counts: Dict[str, int] = {}
        
        # Statistics
        self.stats = {
            'total_segments': 0,
            'identified_segments': 0,
            'new_speakers': 0,
            'speaker_transitions': 0,
            'soft_voice_detections': 0,
            'field_distribution': {},
            'start_time': None
        }
        
        # Logging
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = SCAN_LOG_DIR / f"soft_voice_{self.session_id}.jsonl"
        
        # Events for dashboard
        self.events: List[dict] = []
    
    def apply_agc(self, audio: np.ndarray) -> np.ndarray:
        """Apply automatic gain control for hot/clipped inputs."""
        if self.agc_enabled:
            return self.agc.process(audio)
        return audio
    
    def classify_field(self, snr: float) -> str:
        """Classify field type with extended ranges for soft voices."""
        if snr >= FIELD_THRESHOLDS['NEAR']:
            return 'NEAR'
        elif snr >= FIELD_THRESHOLDS['MEDIUM']:
            return 'MEDIUM'
        elif snr >= FIELD_THRESHOLDS['FAR']:
            return 'FAR'
        elif snr >= FIELD_THRESHOLDS['ULTRA-FAR']:
            return 'ULTRA-FAR'
        elif snr >= FIELD_THRESHOLDS['EXTREME']:
            return 'EXTREME'
        else:
            return 'WHISPER'
    
    def extract_embedding(self, audio: np.ndarray, is_soft: bool = False) -> Optional[np.ndarray]:
        """Extract embedding with AGC for soft voices."""
        try:
            if len(audio) < self.min_samples:
                return None
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            
            audio = audio.astype(np.float32)
            
            # Apply AGC for soft voices
            if is_soft:
                audio = self.apply_agc(audio)
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            embedding = self.encoder.embed_utterance(audio)
            return embedding
        except Exception as e:
            logger.debug(f"Embedding error: {e}")
            return None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback."""
        if status:
            logger.warning(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def update_shared_state(self, event: dict = None):
        """Update dashboard state."""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'running': bool(self.running),
                'mode': 'soft-voice-detection',
                'ultra_sensitive': bool(self.ultra_sensitive),
                'stats': {
                    'total_segments': int(self.stats['total_segments']),
                    'identified_segments': int(self.stats['identified_segments']),
                    'new_speakers': int(self.stats['new_speakers']),
                    'speaker_transitions': int(self.stats['speaker_transitions']),
                    'soft_voice_detections': int(self.stats['soft_voice_detections']),
                    'field_distribution': {k: int(v) for k, v in self.stats['field_distribution'].items()}
                },
                'active_speakers': list(self.speaker_db.active_speakers),
                'total_speakers': int(len(self.speaker_db.speakers)),
                'speakers': [
                    {
                        'id': str(sp.id),
                        'name': str(sp.display_name),
                        'utterance_count': int(sp.utterance_count),
                        'avg_confidence': float(sp.avg_confidence),
                        'soft_voice_count': int(sp.soft_voice_count),
                        'last_seen': float(sp.last_seen),
                        'active': bool(sp.id in self.speaker_db.active_speakers)
                    }
                    for sp in sorted(self.speaker_db.speakers.values(),
                                    key=lambda x: x.last_seen, reverse=True)[:20]
                ],
                'last_event': event,
                'recent_events': self.events[-50:]
            }
            
            with open(SHARED_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2, cls=NumpyEncoder)
        except:
            pass
    
    def display_detection(self, speaker: SpeakerTrack, confidence: float,
                          snr: float, field_type: str, is_new: bool = False,
                          is_soft: bool = False, is_transition: bool = False):
        """Display detection with soft voice indicator."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.no_voice_printed = False
        
        if speaker.id not in self.consecutive_counts:
            self.consecutive_counts[speaker.id] = 0
        self.consecutive_counts[speaker.id] += 1
        count = self.consecutive_counts[speaker.id]
        
        if is_transition:
            for sid in list(self.consecutive_counts.keys()):
                if sid != speaker.id:
                    self.consecutive_counts[sid] = 0
        
        conf_pct = int(confidence * 100)
        snr_int = int(round(snr))
        
        event = {
            'timestamp': timestamp,
            'speaker_id': speaker.id,
            'speaker_name': speaker.display_name,
            'confidence': int(conf_pct),
            'snr': int(snr_int),
            'field_type': str(field_type),
            'count': int(count),
            'is_new': bool(is_new),
            'is_soft': bool(is_soft),
            'is_verifying': bool(speaker.is_verifying)
        }
        self.events.append(event)
        if len(self.events) > 100:
            self.events = self.events[-100:]
        
        # Build output line
        soft_tag = " [soft]" if is_soft else ""
        soft_icon = "🔊" if is_soft else ""
        
        if is_new:
            line = f"[{timestamp}] 🆕 NEW {speaker.id} | {field_type} | SNR: {snr_int}dB | Conf: {conf_pct}%{soft_tag}"
        elif speaker.is_verifying:
            line = f"[{timestamp}] 🔄 {speaker.display_name} | {field_type} | SNR: {snr_int}dB | Conf: {conf_pct}% [verifying]{soft_tag}"
        elif is_soft:
            line = f"[{timestamp}] {soft_icon} SOFT {speaker.display_name} ({speaker.id}) | {field_type} | SNR: {snr_int}dB | Conf: {conf_pct}% [x{count}]"
        elif count >= 3:
            line = f"[{timestamp}] ✅✅ {speaker.display_name} ({speaker.id}) | {field_type} | SNR: {snr_int}dB | Conf: {conf_pct}% [x{count}]"
        elif count >= 2:
            line = f"[{timestamp}] ✅ {speaker.display_name} ({speaker.id}) | {field_type} | SNR: {snr_int}dB | Conf: {conf_pct}% [x{count}]"
        else:
            line = f"[{timestamp}] ✅ {speaker.display_name} ({speaker.id}) | {field_type} | SNR: {snr_int}dB | Conf: {conf_pct}%"
        
        print(line)
        self.update_shared_state(event)
        self.log_detection(speaker, confidence, snr, field_type, is_soft)
    
    def display_no_voice(self):
        """Display no voice message."""
        if not self.no_voice_printed:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] 🔇 No voice detected")
            self.no_voice_printed = True
            self.consecutive_counts = {}
    
    def log_detection(self, speaker: SpeakerTrack, confidence: float,
                      snr: float, field_type: str, is_soft: bool):
        """Log to file."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'speaker_id': str(speaker.id),
            'speaker_name': str(speaker.display_name),
            'confidence': float(confidence),
            'snr_db': float(snr),
            'field_type': str(field_type),
            'is_soft_voice': bool(is_soft),
            'utterance_count': int(speaker.utterance_count)
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, cls=NumpyEncoder) + '\n')
    
    def process_chunk(self, audio: np.ndarray):
        """Process chunk with soft voice detection."""
        # Store raw audio for recording before AGC
        raw_audio = audio.copy()
        
        # Apply AGC first to handle hot/clipped inputs
        audio = self.apply_agc(audio)
        
        # Record audio if enabled (both raw and AGC-processed)
        if self.recorder and self.recorder.is_recording:
            # Pass raw and processed audio to recorder
            self.recorder.write(raw_audio, audio)
        
        # Update noise estimate
        noise_floor = self.noise_estimator.update(audio)
        
        # Detect voice segments (including soft voices)
        segments = self.voice_detector.detect_segments(
            audio, noise_floor, self.min_samples
        )
        
        if not segments:
            if time.time() - self.last_detection_time > 3.0:
                self.display_no_voice()
            return []
        
        results = []
        
        for start_sample, end_sample, is_soft in segments:
            segment_audio = audio[start_sample:end_sample]
            
            # Calculate SNR
            signal_energy = np.sqrt(np.mean(segment_audio ** 2))
            snr = self.noise_estimator.get_snr(signal_energy)
            
            # Allow lower SNR for soft voices
            effective_min_snr = self.min_snr - 2 if is_soft else self.min_snr
            if snr < effective_min_snr:
                continue
            
            field_type = self.classify_field(snr)
            
            # Extract embedding (with AGC for soft voices)
            embedding = self.extract_embedding(segment_audio, is_soft=is_soft)
            if embedding is None:
                continue
            
            # Identify speaker
            speaker, confidence = self.speaker_db.identify(
                embedding, field_type, is_soft=is_soft
            )
            
            segment_duration = len(segment_audio) / SAMPLE_RATE
            is_new = False
            is_transition = False
            
            if speaker:
                speaker.update(embedding, confidence, field_type, 
                              segment_duration, is_soft=is_soft)
                self.speaker_db.active_speakers.add(speaker.id)
                
                if self.last_speaker_id and self.last_speaker_id != speaker.id:
                    is_transition = True
                    self.stats['speaker_transitions'] += 1
                
                self.last_speaker_id = speaker.id
                self.stats['identified_segments'] += 1
                
            elif self.auto_enroll:
                speaker = self.speaker_db.add_speaker(embedding, field_type, is_soft)
                speaker.update(embedding, 0.80, field_type, segment_duration, is_soft)
                confidence = 0.80
                is_new = True
                self.stats['new_speakers'] += 1
                
                if self.last_speaker_id:
                    is_transition = True
                    self.stats['speaker_transitions'] += 1
                
                self.last_speaker_id = speaker.id
            
            # Update stats
            self.stats['total_segments'] += 1
            if is_soft:
                self.stats['soft_voice_detections'] += 1
            self.stats['field_distribution'][field_type] = \
                self.stats['field_distribution'].get(field_type, 0) + 1
            
            if speaker:
                self.last_detection_time = time.time()
                self.display_detection(speaker, confidence, snr, field_type,
                                       is_new=is_new, is_soft=is_soft,
                                       is_transition=is_transition)
                results.append((speaker, confidence, snr, field_type, is_soft))
                
                # Add detection metadata to recording
                if self.recorder and self.recorder.is_recording:
                    detection_meta = {
                        'speaker_id': speaker.id,
                        'speaker_name': speaker.display_name,
                        'confidence': float(confidence),
                        'snr_db': float(snr),
                        'field_type': field_type,
                        'is_soft': is_soft,
                        'is_new': is_new
                    }
                    self.recorder.write(np.array([]), detection=detection_meta)
        
        return results
    
    def process_loop(self):
        """Processing loop."""
        while self.running:
            try:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if chunk.ndim > 1:
                    chunk = chunk.flatten()
                
                self.current_chunk = np.concatenate([self.current_chunk, chunk])
                
                if len(self.current_chunk) >= self.chunk_samples:
                    self.process_chunk(self.current_chunk[:self.chunk_samples])
                    
                    overlap = int(0.5 * SAMPLE_RATE)
                    self.current_chunk = self.current_chunk[-overlap:]
                    
                    self.speaker_db.update_active_speakers()
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def display_status(self):
        """Display status."""
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        print("\n" + "=" * 70)
        print("📊 SOFT VOICE SCANNER STATUS")
        print("=" * 70)
        
        active = [self.speaker_db.speakers[sid] for sid in self.speaker_db.active_speakers
                  if sid in self.speaker_db.speakers]
        
        print(f"\n🎯 Active Speakers ({len(active)}):")
        for sp in sorted(active, key=lambda x: x.last_seen, reverse=True)[:5]:
            soft_pct = (sp.soft_voice_count / max(1, sp.utterance_count)) * 100
            print(f"   • {sp.display_name} ({sp.id}): {sp.utterance_count} detections, "
                  f"soft: {soft_pct:.0f}%, avg conf: {sp.avg_confidence*100:.0f}%")
        
        print(f"\n📈 Statistics:")
        print(f"   Running time: {elapsed/60:.1f} min")
        print(f"   Total segments: {self.stats['total_segments']}")
        print(f"   Soft voice detections: {self.stats['soft_voice_detections']}")
        print(f"   Identified: {self.stats['identified_segments']}")
        print(f"   New speakers: {self.stats['new_speakers']}")
        
        # AGC status
        agc_status = self.agc.get_status()
        print(f"\n🎚️  AGC Status:")
        print(f"   Gain: {agc_status['gain_db']:.1f} dB")
        print(f"   Clipping: {'⚠️ YES' if agc_status['clipping'] else '✓ No'}")
        
        print(f"\n📡 Field Distribution:")
        for field, count in sorted(self.stats['field_distribution'].items()):
            pct = count / max(1, self.stats['total_segments']) * 100
            bar = '█' * int(pct / 5)
            print(f"   {field:12s}: {bar} {count} ({pct:.0f}%)")
        
        print(f"\n💾 Total speakers: {len(self.speaker_db.speakers)}")
        print("=" * 70 + "\n")
    
    def run(self):
        """Main loop."""
        print("\n" + "=" * 70)
        print("🔊 SOFT VOICE FAR-FIELD SCANNER")
        print("=" * 70)
        print(f"\n⚙️  Configuration:")
        print(f"   Ultra-sensitive: {'ON' if self.ultra_sensitive else 'OFF'}")
        print(f"   Fast switch: {'ON' if self.fast_switch else 'OFF'}")
        print(f"   Auto-enroll: {'ON' if self.auto_enroll else 'OFF'}")
        print(f"   Min SNR: {self.min_snr} dB")
        print(f"   AGC (Auto Gain): {'ON' if self.agc_enabled else 'OFF'}")
        print(f"   Multi-language: YES (embedding-based)")
        print(f"   Recording: {'🔴 ON' if self.enable_recording else 'OFF'}")
        
        print(f"\n📚 Speaker Database: {len(self.speaker_db.speakers)} speakers")
        for sid, sp in list(self.speaker_db.speakers.items())[:5]:
            print(f"   • {sp.display_name} ({sp.id}): {sp.utterance_count} prior")
        if len(self.speaker_db.speakers) > 5:
            print(f"   ... and {len(self.speaker_db.speakers) - 5} more")
        
        print(f"\n📡 Field Detection (SNR → Distance):")
        print(f"   NEAR:      ≥{FIELD_THRESHOLDS['NEAR']}dB (0-1m)")
        print(f"   MEDIUM:    {FIELD_THRESHOLDS['MEDIUM']}-{FIELD_THRESHOLDS['NEAR']}dB (1-3m)")
        print(f"   FAR:       {FIELD_THRESHOLDS['FAR']}-{FIELD_THRESHOLDS['MEDIUM']}dB (3-6m)")
        print(f"   ULTRA-FAR: {FIELD_THRESHOLDS['ULTRA-FAR']}-{FIELD_THRESHOLDS['FAR']}dB (6m+)")
        print(f"   EXTREME:   {FIELD_THRESHOLDS['EXTREME']}-{FIELD_THRESHOLDS['ULTRA-FAR']}dB (very far)")
        print(f"   WHISPER:   <{FIELD_THRESHOLDS['EXTREME']}dB (faint/whisper)")
        
        print("\n" + "=" * 70)
        print("🌐 Dashboard: http://localhost:5050")
        print("Commands: [Enter]=Status | [s]=Save | [q]=Quit")
        print("=" * 70)
        print("\n🎤 Scanning for voices (including soft/faint)...\n")
        
        self.running = True
        self.stats['start_time'] = time.time()
        self.update_shared_state()
        
        # Start recording if enabled
        if self.recorder:
            self.recorder.start()
            print(f"📼 Recording to: {self.recorder.output_dir}")
            print(f"   Streams: raw + agc" + (" + neural" if self.recorder.neural_enabled else ""))
            print(f"   Max file duration: 1 hour (continuous)")
            if self.recorder.neural_enabled:
                print(f"   🧠 Neural enhancement: ENABLED (noisereduce)")
            else:
                print(f"   🧠 Neural enhancement: DISABLED")
        
        process_thread = threading.Thread(target=self.process_loop, daemon=True)
        process_thread.start()
        
        # Get device setting
        device_id = getattr(self, 'audio_device', None)
        device_name = "default"
        if device_id is not None:
            try:
                device_info = sd.query_devices(device_id)
                device_name = device_info['name']
            except:
                pass
        print(f"\n🎤 Using audio device: {device_name}")
        
        try:
            with sd.InputStream(
                device=device_id,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=int(SAMPLE_RATE * 0.1),
                dtype=np.float32,
                callback=self.audio_callback
            ):
                while self.running:
                    try:
                        cmd = input()
                        if cmd.lower() == 'q':
                            break
                        elif cmd.lower() == 's':
                            self.speaker_db.save()
                            print("✓ Database saved")
                        else:
                            self.display_status()
                    except EOFError:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            print("\n🛑 Stopping scanner...")
            
            # Stop recording
            if self.recorder:
                self.recorder.stop()
            
            self.speaker_db.save()
            print("✓ Database saved")
            self.display_status()


def main():
    parser = argparse.ArgumentParser(
        description='Soft Voice Far-Field Scanner with Continuous Recording'
    )
    parser.add_argument('--ultra-sensitive', action='store_true',
                        help='Maximum sensitivity for very faint voices')
    parser.add_argument('--min-snr', type=float, default=2.0,
                        help='Minimum SNR for detection (dB), default=2')
    parser.add_argument('--no-enroll', action='store_true',
                        help='Disable auto-enrollment')
    parser.add_argument('--no-agc', action='store_true',
                        help='Disable automatic gain control')
    parser.add_argument('--record', '-r', action='store_true',
                        help='Enable continuous recording with enhancement')
    parser.add_argument('--neural', action='store_true',
                        help='Enable neural enhancement for far-field audio (noisereduce)')
    parser.add_argument('--device', '-d', type=int, default=None,
                        help='Audio input device ID (use --list-devices to see available)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio input devices and exit')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory for recordings')
    
    args = parser.parse_args()
    
    # List devices and exit
    if args.list_devices:
        import sounddevice as sd
        print("Available Audio Input Devices:")
        print("=" * 70)
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                print(f"{i:3d}: {d['name']}")
        return
    
    scanner = SoftVoiceScanner(
        ultra_sensitive=args.ultra_sensitive,
        min_snr=args.min_snr,
        auto_enroll=not args.no_enroll,
        fast_switch=True,
        enable_recording=args.record,
        output_dir=args.output_dir,
        enable_neural=args.neural
    )
    
    if args.no_agc:
        scanner.agc_enabled = False
    
    # Set audio device
    if args.device is not None:
        scanner.audio_device = args.device
    
    scanner.run()


if __name__ == '__main__':
    main()
