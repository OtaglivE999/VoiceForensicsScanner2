"""
Field-Type Audio Processor Module
Process audio for near-field, far-field, and ultra-far-field
voice detection with adaptive enhancement.

Version: 1.0
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import torch
import torchaudio
from scipy import signal

logger = logging.getLogger(__name__)


# ============================================================================
# Field Type Definitions
# ============================================================================

class FieldType(Enum):
    """Audio field types based on distance from source."""
    NEAR_FIELD = "near_field"       # 0-3 feet (direct speech)
    FAR_FIELD = "far_field"         # 3-15 feet (room conversation)
    ULTRA_FAR = "ultra_far"         # 15+ feet (surveillance, ambient)


@dataclass
class FieldTypeConfig:
    """Configuration for a specific field type."""
    field_type: FieldType
    distance_min_ft: float
    distance_max_ft: float
    
    # Enhancement settings
    enhancement_level: str = "none"  # none, light, moderate, aggressive
    denoise_strength: float = 0.0    # 0.0 to 1.0
    gain_db: float = 0.0
    
    # Quality thresholds
    min_snr_db: float = 10.0
    min_voice_confidence: float = 0.5
    
    # Match thresholds
    match_threshold: float = 0.75
    high_confidence_threshold: float = 0.90


# Default field type configurations
FIELD_TYPE_CONFIGS = {
    FieldType.NEAR_FIELD: FieldTypeConfig(
        field_type=FieldType.NEAR_FIELD,
        distance_min_ft=0,
        distance_max_ft=3,
        enhancement_level="none",
        denoise_strength=0.0,
        gain_db=0.0,
        min_snr_db=20.0,
        min_voice_confidence=0.8,
        match_threshold=0.85,
        high_confidence_threshold=0.95
    ),
    FieldType.FAR_FIELD: FieldTypeConfig(
        field_type=FieldType.FAR_FIELD,
        distance_min_ft=3,
        distance_max_ft=15,
        enhancement_level="moderate",
        denoise_strength=0.5,
        gain_db=6.0,
        min_snr_db=10.0,
        min_voice_confidence=0.6,
        match_threshold=0.75,
        high_confidence_threshold=0.88
    ),
    FieldType.ULTRA_FAR: FieldTypeConfig(
        field_type=FieldType.ULTRA_FAR,
        distance_min_ft=15,
        distance_max_ft=100,
        enhancement_level="aggressive",
        denoise_strength=0.8,
        gain_db=12.0,
        min_snr_db=5.0,
        min_voice_confidence=0.4,
        match_threshold=0.65,
        high_confidence_threshold=0.80
    )
}


@dataclass
class ProcessorConfig:
    """Configuration for the field-type processor."""
    sample_rate: int = 16000
    
    # VAD settings
    vad_mode: int = 3  # 0-3, higher = more aggressive
    vad_frame_ms: int = 30
    
    # Enhancement
    use_deepfilternet: bool = True
    deepfilternet_model: str = "DeepFilterNet3"
    
    # Analysis
    estimate_distance: bool = True
    estimate_snr: bool = True
    
    # Field configs
    field_configs: Dict[FieldType, FieldTypeConfig] = field(
        default_factory=lambda: FIELD_TYPE_CONFIGS.copy()
    )


# ============================================================================
# Audio Analysis
# ============================================================================

class AudioAnalyzer:
    """Analyze audio characteristics for field-type classification."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estimate Signal-to-Noise Ratio (SNR) in dB.
        Uses energy-based estimation.
        """
        # Simple energy-based SNR estimation
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)     # 10ms hop
        
        # Calculate frame energies
        num_frames = (len(audio) - frame_length) // hop_length + 1
        energies = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]
            energies[i] = np.sum(frame ** 2) / frame_length
        
        # Sort energies
        sorted_energies = np.sort(energies)
        
        # Estimate noise from bottom 10%
        noise_percentile = max(1, int(0.1 * len(sorted_energies)))
        noise_energy = np.mean(sorted_energies[:noise_percentile])
        
        # Estimate signal from top 20%
        signal_percentile = max(1, int(0.2 * len(sorted_energies)))
        signal_energy = np.mean(sorted_energies[-signal_percentile:])
        
        if noise_energy > 0:
            snr_db = 10 * np.log10((signal_energy + 1e-10) / (noise_energy + 1e-10))
        else:
            snr_db = 40.0  # Very clean
        
        return float(np.clip(snr_db, 0, 60))
    
    def estimate_distance(self, audio: np.ndarray) -> Tuple[float, str]:
        """
        Estimate approximate distance based on audio characteristics.
        Returns (estimated_feet, field_type_name).
        
        This is a heuristic based on:
        - Direct-to-reverberant ratio
        - High frequency content (attenuates with distance)
        - Energy distribution
        """
        # Calculate spectral features
        n_fft = 2048
        freqs = np.fft.rfftfreq(n_fft, 1/self.sample_rate)
        spectrum = np.abs(np.fft.rfft(audio, n_fft))
        
        # High frequency energy (above 4kHz)
        hf_mask = freqs > 4000
        hf_energy = np.mean(spectrum[hf_mask]) if np.any(hf_mask) else 0
        
        # Low frequency energy (below 1kHz)
        lf_mask = freqs < 1000
        lf_energy = np.mean(spectrum[lf_mask]) if np.any(lf_mask) else 1
        
        # HF/LF ratio (higher = closer)
        hf_lf_ratio = (hf_energy + 1e-10) / (lf_energy + 1e-10)
        
        # Simple heuristic mapping
        if hf_lf_ratio > 0.5:
            estimated_feet = 2.0
            field_type = "near_field"
        elif hf_lf_ratio > 0.2:
            estimated_feet = 8.0
            field_type = "far_field"
        else:
            estimated_feet = 25.0
            field_type = "ultra_far"
        
        return estimated_feet, field_type
    
    def detect_voice_activity(self, audio: np.ndarray) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Detect voice activity regions.
        Returns (voice_ratio, [(start_sec, end_sec), ...]).
        """
        try:
            import webrtcvad
            
            vad = webrtcvad.Vad(3)  # Aggressive mode
            frame_duration_ms = 30
            frame_samples = int(self.sample_rate * frame_duration_ms / 1000)
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            voice_frames = []
            num_frames = len(audio_int16) // frame_samples
            
            for i in range(num_frames):
                start = i * frame_samples
                frame = audio_int16[start:start + frame_samples].tobytes()
                is_speech = vad.is_speech(frame, self.sample_rate)
                voice_frames.append(is_speech)
            
            # Find voice regions
            regions = []
            in_voice = False
            start_frame = 0
            
            for i, is_voice in enumerate(voice_frames):
                if is_voice and not in_voice:
                    in_voice = True
                    start_frame = i
                elif not is_voice and in_voice:
                    in_voice = False
                    start_sec = start_frame * frame_duration_ms / 1000
                    end_sec = i * frame_duration_ms / 1000
                    regions.append((start_sec, end_sec))
            
            # Handle case where audio ends during voice
            if in_voice:
                start_sec = start_frame * frame_duration_ms / 1000
                end_sec = num_frames * frame_duration_ms / 1000
                regions.append((start_sec, end_sec))
            
            voice_ratio = sum(voice_frames) / len(voice_frames) if voice_frames else 0
            
            return voice_ratio, regions
            
        except ImportError:
            # Fallback: energy-based VAD
            frame_length = int(0.030 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            
            threshold = np.percentile(np.abs(audio), 70)
            
            num_frames = (len(audio) - frame_length) // hop_length + 1
            voice_frames = []
            
            for i in range(num_frames):
                start = i * hop_length
                frame = audio[start:start + frame_length]
                is_voice = np.max(np.abs(frame)) > threshold
                voice_frames.append(is_voice)
            
            voice_ratio = sum(voice_frames) / len(voice_frames) if voice_frames else 0
            
            return voice_ratio, []


# ============================================================================
# Audio Enhancement
# ============================================================================

class AudioEnhancer:
    """Enhance audio using DeepFilterNet and DSP techniques."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self._deepfilter = None
        self._deepfilter_loaded = False
    
    def _load_deepfilter(self) -> bool:
        """Load DeepFilterNet model."""
        if self._deepfilter_loaded:
            return self._deepfilter is not None
        
        try:
            from df import enhance, init_df
            
            self._df_model, self._df_state, _ = init_df()
            self._deepfilter = lambda audio, sr: enhance(
                self._df_model, self._df_state, audio
            )
            
            self._deepfilter_loaded = True
            logger.info("DeepFilterNet3 loaded successfully")
            return True
            
        except ImportError:
            logger.warning("DeepFilterNet not installed. Install with: pip install deepfilternet")
            self._deepfilter_loaded = True
            return False
        except Exception as e:
            logger.error(f"Failed to load DeepFilterNet: {e}")
            self._deepfilter_loaded = True
            return False
    
    def enhance_audio(self,
                     audio: np.ndarray,
                     field_type: FieldType,
                     sample_rate: int = 16000) -> np.ndarray:
        """
        Enhance audio based on field type configuration.
        
        Args:
            audio: Input audio (mono)
            field_type: Field type for enhancement settings
            sample_rate: Sample rate
        
        Returns:
            Enhanced audio
        """
        field_config = self.config.field_configs.get(field_type)
        if field_config is None or field_config.enhancement_level == "none":
            return audio
        
        enhanced = audio.copy()
        
        # Apply DeepFilterNet denoising
        if self.config.use_deepfilternet and field_config.denoise_strength > 0:
            if self._load_deepfilter() and self._deepfilter is not None:
                try:
                    # DeepFilterNet expects torch tensor
                    audio_tensor = torch.from_numpy(enhanced).float().unsqueeze(0)
                    enhanced_tensor = self._deepfilter(audio_tensor, sample_rate)
                    enhanced = enhanced_tensor.squeeze().numpy()
                    
                    # Blend based on denoise strength
                    enhanced = (
                        field_config.denoise_strength * enhanced +
                        (1 - field_config.denoise_strength) * audio
                    )
                except Exception as e:
                    logger.error(f"DeepFilterNet enhancement failed: {e}")
        
        # Apply gain
        if field_config.gain_db != 0:
            gain_linear = 10 ** (field_config.gain_db / 20)
            enhanced = enhanced * gain_linear
            
            # Soft clip to prevent distortion
            enhanced = np.tanh(enhanced)
        
        # Normalize
        max_val = np.max(np.abs(enhanced))
        if max_val > 0:
            enhanced = enhanced / max_val * 0.95
        
        return enhanced.astype(np.float32)
    
    def bandpass_filter(self,
                       audio: np.ndarray,
                       low_freq: float = 100,
                       high_freq: float = 8000,
                       sample_rate: int = 16000,
                       order: int = 5) -> np.ndarray:
        """Apply bandpass filter for voice frequencies."""
        nyquist = sample_rate / 2
        low = low_freq / nyquist
        high = min(high_freq / nyquist, 0.99)
        
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, audio).astype(np.float32)
    
    def spectral_subtraction(self,
                            audio: np.ndarray,
                            noise_estimate: np.ndarray = None,
                            alpha: float = 2.0,
                            beta: float = 0.01) -> np.ndarray:
        """
        Apply spectral subtraction for noise reduction.
        Classic technique for stationary noise.
        """
        n_fft = 2048
        hop_length = 512
        
        # STFT
        stft = np.fft.rfft(audio, n_fft)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from first 100ms if not provided
        if noise_estimate is None:
            noise_samples = int(0.1 * len(audio))
            noise_stft = np.fft.rfft(audio[:noise_samples], n_fft)
            noise_estimate = np.abs(noise_stft)
        
        # Spectral subtraction
        enhanced_mag = magnitude - alpha * noise_estimate
        enhanced_mag = np.maximum(enhanced_mag, beta * magnitude)
        
        # Reconstruct
        enhanced = np.fft.irfft(enhanced_mag * np.exp(1j * phase))
        
        return enhanced[:len(audio)].astype(np.float32)


# ============================================================================
# Field-Type Processor
# ============================================================================

@dataclass
class ProcessingResult:
    """Result from field-type processing."""
    # Original and enhanced audio
    original_audio: np.ndarray
    enhanced_audio: np.ndarray
    sample_rate: int
    
    # Classification
    detected_field_type: FieldType
    estimated_distance_ft: float
    
    # Quality metrics
    original_snr_db: float
    enhanced_snr_db: float
    voice_confidence: float
    voice_regions: List[Tuple[float, float]]
    
    # Processing info
    enhancement_applied: str
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'detected_field_type': self.detected_field_type.value,
            'estimated_distance_ft': self.estimated_distance_ft,
            'original_snr_db': self.original_snr_db,
            'enhanced_snr_db': self.enhanced_snr_db,
            'voice_confidence': self.voice_confidence,
            'voice_regions': self.voice_regions,
            'enhancement_applied': self.enhancement_applied,
            'processing_time_ms': self.processing_time_ms
        }


class FieldTypeProcessor:
    """
    Process audio with field-type detection and adaptive enhancement.
    """
    
    def __init__(self, config: ProcessorConfig = None):
        """
        Initialize the field-type processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessorConfig()
        self.analyzer = AudioAnalyzer(self.config.sample_rate)
        self.enhancer = AudioEnhancer(self.config)
    
    def classify_field_type(self, 
                           audio: np.ndarray,
                           known_distance_ft: float = None) -> Tuple[FieldType, float]:
        """
        Classify audio into field type.
        
        Args:
            audio: Audio data
            known_distance_ft: Known recording distance (if available)
        
        Returns:
            (field_type, estimated_distance_ft)
        """
        if known_distance_ft is not None:
            # Use known distance
            if known_distance_ft <= 3:
                return FieldType.NEAR_FIELD, known_distance_ft
            elif known_distance_ft <= 15:
                return FieldType.FAR_FIELD, known_distance_ft
            else:
                return FieldType.ULTRA_FAR, known_distance_ft
        
        # Estimate from audio characteristics
        estimated_ft, field_name = self.analyzer.estimate_distance(audio)
        field_type = FieldType(field_name)
        
        return field_type, estimated_ft
    
    def process(self,
               audio: Union[str, Path, np.ndarray, torch.Tensor],
               sample_rate: int = None,
               known_distance_ft: float = None,
               force_field_type: FieldType = None) -> ProcessingResult:
        """
        Process audio with field-type detection and enhancement.
        
        Args:
            audio: Audio file path, numpy array, or tensor
            sample_rate: Sample rate (if audio is array/tensor)
            known_distance_ft: Known recording distance
            force_field_type: Force specific field type
        
        Returns:
            ProcessingResult with enhanced audio and metrics
        """
        import time
        start_time = time.time()
        
        # Load audio if needed
        if isinstance(audio, (str, Path)):
            waveform, sr = torchaudio.load(str(audio))
            audio_np = waveform.numpy().squeeze()
            sample_rate = sr
        elif isinstance(audio, torch.Tensor):
            audio_np = audio.numpy().squeeze()
        else:
            audio_np = np.asarray(audio).squeeze()
        
        sample_rate = sample_rate or self.config.sample_rate
        
        # Resample if needed
        if sample_rate != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.config.sample_rate
            )
            audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
            audio_np = resampler(audio_tensor).squeeze().numpy()
            sample_rate = self.config.sample_rate
        
        # Analyze original audio
        original_snr = self.analyzer.estimate_snr(audio_np)
        voice_ratio, voice_regions = self.analyzer.detect_voice_activity(audio_np)
        
        # Classify field type
        if force_field_type is not None:
            field_type = force_field_type
            estimated_distance = self.config.field_configs[field_type].distance_min_ft + 1
        else:
            field_type, estimated_distance = self.classify_field_type(
                audio_np, known_distance_ft
            )
        
        # Get field config
        field_config = self.config.field_configs[field_type]
        
        # Apply enhancement
        enhanced_audio = self.enhancer.enhance_audio(
            audio_np, field_type, sample_rate
        )
        
        # Analyze enhanced audio
        enhanced_snr = self.analyzer.estimate_snr(enhanced_audio)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ProcessingResult(
            original_audio=audio_np,
            enhanced_audio=enhanced_audio,
            sample_rate=sample_rate,
            detected_field_type=field_type,
            estimated_distance_ft=estimated_distance,
            original_snr_db=original_snr,
            enhanced_snr_db=enhanced_snr,
            voice_confidence=voice_ratio,
            voice_regions=voice_regions,
            enhancement_applied=field_config.enhancement_level,
            processing_time_ms=processing_time
        )
    
    def process_multi_field(self,
                           audio: np.ndarray,
                           sample_rate: int = None) -> Dict[FieldType, ProcessingResult]:
        """
        Process audio for all field types and return results for each.
        Useful for enrollment when field type is unknown.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
        
        Returns:
            Dictionary of field_type -> ProcessingResult
        """
        results = {}
        
        for field_type in FieldType:
            results[field_type] = self.process(
                audio=audio,
                sample_rate=sample_rate,
                force_field_type=field_type
            )
        
        return results
    
    def get_best_field_type_for_match(self,
                                     test_audio: np.ndarray,
                                     enrolled_field_types: List[FieldType]) -> FieldType:
        """
        Determine which enrolled field type is best for matching test audio.
        
        Args:
            test_audio: Audio to match
            enrolled_field_types: Available enrolled field types
        
        Returns:
            Best field type for matching
        """
        if not enrolled_field_types:
            return FieldType.FAR_FIELD  # Default
        
        # Classify test audio
        detected_type, _ = self.classify_field_type(test_audio)
        
        # Use exact match if available
        if detected_type in enrolled_field_types:
            return detected_type
        
        # Find closest match
        type_order = [FieldType.NEAR_FIELD, FieldType.FAR_FIELD, FieldType.ULTRA_FAR]
        detected_idx = type_order.index(detected_type)
        
        # Find nearest available
        min_distance = float('inf')
        best_match = enrolled_field_types[0]
        
        for available in enrolled_field_types:
            available_idx = type_order.index(available)
            distance = abs(detected_idx - available_idx)
            if distance < min_distance:
                min_distance = distance
                best_match = available
        
        return best_match


# ============================================================================
# Convenience Functions
# ============================================================================

_processor: Optional[FieldTypeProcessor] = None

def get_field_processor(config: ProcessorConfig = None) -> FieldTypeProcessor:
    """Get the global field processor instance."""
    global _processor
    if _processor is None:
        _processor = FieldTypeProcessor(config or ProcessorConfig())
    return _processor


def process_audio_for_field(audio_path: str,
                           known_distance: float = None) -> ProcessingResult:
    """Quick function to process audio file."""
    processor = get_field_processor()
    return processor.process(audio_path, known_distance_ft=known_distance)


def enhance_far_field_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Quick function to enhance far-field audio."""
    processor = get_field_processor()
    result = processor.process(
        audio, sample_rate, force_field_type=FieldType.FAR_FIELD
    )
    return result.enhanced_audio


if __name__ == '__main__':
    # Test the field processor
    processor = FieldTypeProcessor()
    
    # Generate test audio (simulated far-field with noise)
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simple voiced signal + noise
    signal = np.sin(2 * np.pi * 200 * t) * 0.3  # Fundamental
    signal += np.sin(2 * np.pi * 400 * t) * 0.15  # Harmonic
    noise = np.random.randn(len(t)) * 0.1
    test_audio = (signal + noise).astype(np.float32)
    
    # Process
    result = processor.process(test_audio, sample_rate)
    
    print(f"Detected field type: {result.detected_field_type.value}")
    print(f"Estimated distance: {result.estimated_distance_ft:.1f} ft")
    print(f"Original SNR: {result.original_snr_db:.1f} dB")
    print(f"Enhanced SNR: {result.enhanced_snr_db:.1f} dB")
    print(f"Voice confidence: {result.voice_confidence:.1%}")
    print(f"Enhancement applied: {result.enhancement_applied}")
    print(f"Processing time: {result.processing_time_ms:.1f} ms")


# ============================================================================
# Backwards Compatibility Aliases
# ============================================================================

# Alias for compatibility with code that uses 'FieldProcessor'
FieldProcessor = FieldTypeProcessor
