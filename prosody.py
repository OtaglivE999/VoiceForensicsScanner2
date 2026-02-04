"""
Prosodic Analysis Module
Analyzes pitch, rhythm, and voice quality indicators using Parselmouth (Praat)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("Warning: Parselmouth not available. Install with: pip install praat-parselmouth")

try:
    import soundfile as sf
except ImportError:
    import scipy.io.wavfile as wavfile
    sf = None


@dataclass
class ProsodySegment:
    """Prosodic features for a speech segment."""
    start_time: float
    end_time: float
    speaker_id: Optional[str]
    
    # Pitch features (F0)
    pitch_mean: float
    pitch_std: float
    pitch_min: float
    pitch_max: float
    pitch_range: float
    
    # Intensity features
    intensity_mean: float
    intensity_std: float
    intensity_max: float
    
    # Voice quality
    jitter_local: float      # Pitch perturbation (voice tremor)
    shimmer_local: float     # Amplitude perturbation (voice quality)
    hnr: float              # Harmonics-to-noise ratio (breathiness)
    
    # Speech rate
    speech_rate: float       # Syllables per second estimate
    pause_ratio: float       # Ratio of silence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'timestamp': self._format_time(self.start_time),
            'speaker_id': self.speaker_id,
            'pitch': {
                'mean': self.pitch_mean,
                'std': self.pitch_std,
                'min': self.pitch_min,
                'max': self.pitch_max,
                'range': self.pitch_range
            },
            'intensity': {
                'mean': self.intensity_mean,
                'std': self.intensity_std,
                'max': self.intensity_max
            },
            'voice_quality': {
                'jitter': self.jitter_local,
                'shimmer': self.shimmer_local,
                'hnr': self.hnr
            },
            'rhythm': {
                'speech_rate': self.speech_rate,
                'pause_ratio': self.pause_ratio
            }
        }
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    
    def get_stress_indicators(self) -> Dict[str, Any]:
        """Identify potential stress/distress indicators."""
        indicators = []
        severity = 'low'
        
        # High pitch variation can indicate stress
        if self.pitch_std > 50:
            indicators.append('high_pitch_variation')
            severity = 'moderate'
        
        # High jitter indicates voice tremor (nervousness/fear)
        if self.jitter_local > 0.02:
            indicators.append('voice_tremor')
            severity = 'moderate'
        
        # Very high jitter indicates severe distress
        if self.jitter_local > 0.04:
            indicators.append('severe_voice_tremor')
            severity = 'high'
        
        # High shimmer indicates voice instability
        if self.shimmer_local > 0.1:
            indicators.append('voice_instability')
        
        # Low HNR can indicate breathy/strained voice
        if self.hnr < 10:
            indicators.append('strained_voice')
        
        # Very high intensity could indicate shouting
        if self.intensity_max > 80:
            indicators.append('raised_voice')
            severity = 'high' if severity != 'high' else severity
        
        # Fast speech rate can indicate agitation
        if self.speech_rate > 6:
            indicators.append('rapid_speech')
        
        # Slow speech with high pause ratio may indicate emotional suppression
        if self.speech_rate < 2 and self.pause_ratio > 0.5:
            indicators.append('hesitant_speech')
        
        return {
            'indicators': indicators,
            'severity': severity,
            'indicator_count': len(indicators)
        }


@dataclass
class SpeakerProsodyProfile:
    """Aggregated prosodic profile for a speaker."""
    speaker_id: str
    segments_analyzed: int
    
    # Average features
    avg_pitch_mean: float
    avg_pitch_std: float
    avg_pitch_range: float
    avg_intensity: float
    avg_jitter: float
    avg_shimmer: float
    avg_hnr: float
    avg_speech_rate: float
    
    # Variability across segments
    pitch_variability: float
    intensity_variability: float
    
    # Stress indicators
    total_stress_indicators: int
    stress_severity: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'speaker_id': self.speaker_id,
            'segments_analyzed': self.segments_analyzed,
            'pitch_profile': {
                'avg_mean': self.avg_pitch_mean,
                'avg_std': self.avg_pitch_std,
                'avg_range': self.avg_pitch_range,
                'variability': self.pitch_variability
            },
            'intensity_profile': {
                'avg': self.avg_intensity,
                'variability': self.intensity_variability
            },
            'voice_quality': {
                'avg_jitter': self.avg_jitter,
                'avg_shimmer': self.avg_shimmer,
                'avg_hnr': self.avg_hnr
            },
            'speech_rate': {
                'avg': self.avg_speech_rate
            },
            'stress_assessment': {
                'total_indicators': self.total_stress_indicators,
                'severity': self.stress_severity
            }
        }


class ProsodyAnalyzer:
    """
    Analyzes prosodic (suprasegmental) features of speech using Praat/Parselmouth.
    These features reveal emotional state, stress, and speaking patterns.
    """
    
    def __init__(self,
                 pitch_floor: float = 75.0,
                 pitch_ceiling: float = 600.0,
                 time_step: float = 0.0):
        """
        Initialize the prosody analyzer.
        
        Args:
            pitch_floor: Minimum F0 in Hz (use lower for male voices)
            pitch_ceiling: Maximum F0 in Hz (use higher for female/child voices)
            time_step: Time step for analysis (0 = auto)
        """
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.time_step = time_step
        
        if not PARSELMOUTH_AVAILABLE:
            print("⚠️ Parselmouth not available - using fallback analysis")
    
    def analyze_segment(self,
                       audio_data: np.ndarray,
                       sample_rate: int,
                       start_time: float = 0,
                       speaker_id: str = None) -> ProsodySegment:
        """
        Analyze prosodic features of an audio segment.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate of audio
            start_time: Start time in original file
            speaker_id: Speaker ID if known
        
        Returns:
            ProsodySegment with extracted features
        """
        duration = len(audio_data) / sample_rate
        end_time = start_time + duration
        
        if PARSELMOUTH_AVAILABLE:
            return self._analyze_parselmouth(
                audio_data, sample_rate, start_time, end_time, speaker_id
            )
        else:
            return self._analyze_fallback(
                audio_data, sample_rate, start_time, end_time, speaker_id
            )
    
    def _analyze_parselmouth(self,
                            audio_data: np.ndarray,
                            sample_rate: int,
                            start_time: float,
                            end_time: float,
                            speaker_id: str) -> ProsodySegment:
        """Analyze using Parselmouth/Praat."""
        
        # Create Sound object
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float64)
        
        sound = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)
        
        # Extract pitch
        pitch = call(sound, "To Pitch", self.time_step, self.pitch_floor, self.pitch_ceiling)
        
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced
        
        if len(pitch_values) > 0:
            pitch_mean = float(np.mean(pitch_values))
            pitch_std = float(np.std(pitch_values))
            pitch_min = float(np.min(pitch_values))
            pitch_max = float(np.max(pitch_values))
            pitch_range = pitch_max - pitch_min
        else:
            pitch_mean = pitch_std = pitch_min = pitch_max = pitch_range = 0.0
        
        # Extract intensity
        intensity = call(sound, "To Intensity", self.pitch_floor, self.time_step)
        intensity_values = call(intensity, "Get all values")
        
        if len(intensity_values) > 0:
            intensity_mean = float(np.mean(intensity_values))
            intensity_std = float(np.std(intensity_values))
            intensity_max = float(np.max(intensity_values))
        else:
            intensity_mean = intensity_std = intensity_max = 0.0
        
        # Extract voice quality (point process needed)
        try:
            point_process = call(sound, "To PointProcess (periodic, cc)", 
                               self.pitch_floor, self.pitch_ceiling)
            
            # Jitter (local)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter = jitter if not np.isnan(jitter) else 0.0
            
            # Shimmer (local)
            shimmer = call([sound, point_process], "Get shimmer (local)", 
                          0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer = shimmer if not np.isnan(shimmer) else 0.0
            
        except Exception:
            jitter = shimmer = 0.0
        
        # Harmonics-to-noise ratio
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", self.time_step, 
                             self.pitch_floor, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            hnr = hnr if not np.isnan(hnr) else 0.0
        except Exception:
            hnr = 0.0
        
        # Estimate speech rate from intensity envelope
        speech_rate, pause_ratio = self._estimate_speech_rate(intensity_values)
        
        return ProsodySegment(
            start_time=start_time,
            end_time=end_time,
            speaker_id=speaker_id,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            pitch_range=pitch_range,
            intensity_mean=intensity_mean,
            intensity_std=intensity_std,
            intensity_max=intensity_max,
            jitter_local=float(jitter),
            shimmer_local=float(shimmer),
            hnr=float(hnr),
            speech_rate=speech_rate,
            pause_ratio=pause_ratio
        )
    
    def _analyze_fallback(self,
                         audio_data: np.ndarray,
                         sample_rate: int,
                         start_time: float,
                         end_time: float,
                         speaker_id: str) -> ProsodySegment:
        """Fallback analysis when Parselmouth is not available."""
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Basic intensity from RMS
        frame_size = int(sample_rate * 0.025)
        hop_size = int(sample_rate * 0.010)
        
        frames = [audio_data[i:i+frame_size] 
                  for i in range(0, len(audio_data)-frame_size, hop_size)]
        
        rms_values = [np.sqrt(np.mean(f ** 2)) for f in frames if len(f) > 0]
        
        if rms_values:
            # Convert to dB-like scale
            intensity_values = [20 * np.log10(max(r, 1e-10)) + 100 for r in rms_values]
            intensity_mean = np.mean(intensity_values)
            intensity_std = np.std(intensity_values)
            intensity_max = np.max(intensity_values)
        else:
            intensity_mean = intensity_std = intensity_max = 0.0
        
        # Zero-crossing rate as rough pitch proxy
        zcr = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
        rough_pitch = zcr * sample_rate / 2
        
        # Estimate speech rate and pause ratio
        speech_rate, pause_ratio = self._estimate_speech_rate_simple(audio_data, sample_rate)
        
        return ProsodySegment(
            start_time=start_time,
            end_time=end_time,
            speaker_id=speaker_id,
            pitch_mean=min(max(rough_pitch, 80), 300),  # Clamp to reasonable range
            pitch_std=0.0,
            pitch_min=0.0,
            pitch_max=0.0,
            pitch_range=0.0,
            intensity_mean=intensity_mean,
            intensity_std=intensity_std,
            intensity_max=intensity_max,
            jitter_local=0.0,
            shimmer_local=0.0,
            hnr=0.0,
            speech_rate=speech_rate,
            pause_ratio=pause_ratio
        )
    
    def _estimate_speech_rate(self, intensity_values: np.ndarray) -> Tuple[float, float]:
        """Estimate speech rate from intensity envelope."""
        if len(intensity_values) < 2:
            return 0.0, 1.0
        
        # Find intensity threshold
        threshold = np.mean(intensity_values) - np.std(intensity_values)
        
        # Count syllable-like peaks (intensity maxima above threshold)
        above_threshold = intensity_values > threshold
        peaks = 0
        in_peak = False
        silence_frames = 0
        
        for val, above in zip(intensity_values, above_threshold):
            if above and not in_peak:
                peaks += 1
                in_peak = True
            elif not above:
                in_peak = False
                silence_frames += 1
        
        duration = len(intensity_values) * 0.01  # Assuming 10ms frames
        speech_rate = peaks / duration if duration > 0 else 0
        pause_ratio = silence_frames / len(intensity_values) if len(intensity_values) > 0 else 0
        
        return float(speech_rate), float(pause_ratio)
    
    def _estimate_speech_rate_simple(self, 
                                    audio_data: np.ndarray, 
                                    sample_rate: int) -> Tuple[float, float]:
        """Simple speech rate estimation from audio."""
        frame_size = int(sample_rate * 0.025)
        hop_size = int(sample_rate * 0.010)
        
        # Calculate frame energies
        energies = []
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i:i+frame_size]
            energies.append(np.sqrt(np.mean(frame ** 2)))
        
        if not energies:
            return 0.0, 1.0
        
        threshold = np.mean(energies) * 0.5
        
        # Count transitions (rough syllable count)
        above = [e > threshold for e in energies]
        transitions = sum(1 for i in range(1, len(above)) if above[i] and not above[i-1])
        
        duration = len(audio_data) / sample_rate
        speech_rate = transitions / duration if duration > 0 else 0
        pause_ratio = sum(1 for a in above if not a) / len(above)
        
        return float(speech_rate), float(pause_ratio)
    
    def analyze_audio_file(self,
                          audio_path: str,
                          segments: List[Dict] = None,
                          segment_duration: float = 5.0) -> List[ProsodySegment]:
        """
        Analyze prosody in an audio file.
        
        Args:
            audio_path: Path to audio file
            segments: List of segments with timing info
            segment_duration: Duration if no segments provided
        
        Returns:
            List of ProsodySegment
        """
        # Load audio
        if sf is not None:
            audio_data, sample_rate = sf.read(audio_path)
        else:
            sample_rate, audio_data = wavfile.read(audio_path)
            audio_data = audio_data.astype(np.float64) / 32768.0
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        results = []
        
        # Create segments if not provided
        if segments is None:
            duration = len(audio_data) / sample_rate
            segments = []
            for start in np.arange(0, duration, segment_duration):
                end = min(start + segment_duration, duration)
                segments.append({
                    'start_time': start,
                    'end_time': end,
                    'speaker_id': None
                })
        
        print(f"   Analyzing prosody in {len(segments)} segments...")
        
        for i, seg in enumerate(segments):
            start_sample = int(seg['start_time'] * sample_rate)
            end_sample = int(seg['end_time'] * sample_rate)
            
            segment_audio = audio_data[start_sample:end_sample]
            
            if len(segment_audio) < sample_rate * 0.5:
                continue
            
            result = self.analyze_segment(
                segment_audio,
                sample_rate,
                start_time=seg['start_time'],
                speaker_id=seg.get('speaker_id')
            )
            results.append(result)
            
            if (i + 1) % 50 == 0:
                print(f"      Processed {i + 1}/{len(segments)} segments")
        
        return results
    
    def aggregate_by_speaker(self,
                            segments: List[ProsodySegment]) -> Dict[str, SpeakerProsodyProfile]:
        """
        Aggregate prosody results by speaker.
        
        Args:
            segments: List of ProsodySegment
        
        Returns:
            Dictionary of speaker profiles
        """
        speaker_data = {}
        
        for seg in segments:
            speaker_id = seg.speaker_id or 'UNKNOWN'
            
            if speaker_id not in speaker_data:
                speaker_data[speaker_id] = {
                    'pitch_means': [],
                    'pitch_stds': [],
                    'pitch_ranges': [],
                    'intensities': [],
                    'jitters': [],
                    'shimmers': [],
                    'hnrs': [],
                    'speech_rates': [],
                    'stress_indicators': 0
                }
            
            data = speaker_data[speaker_id]
            data['pitch_means'].append(seg.pitch_mean)
            data['pitch_stds'].append(seg.pitch_std)
            data['pitch_ranges'].append(seg.pitch_range)
            data['intensities'].append(seg.intensity_mean)
            data['jitters'].append(seg.jitter_local)
            data['shimmers'].append(seg.shimmer_local)
            data['hnrs'].append(seg.hnr)
            data['speech_rates'].append(seg.speech_rate)
            data['stress_indicators'] += len(seg.get_stress_indicators()['indicators'])
        
        profiles = {}
        for speaker_id, data in speaker_data.items():
            n = len(data['pitch_means'])
            
            # Determine overall stress severity
            avg_stress = data['stress_indicators'] / n if n > 0 else 0
            if avg_stress > 2:
                stress_severity = 'high'
            elif avg_stress > 1:
                stress_severity = 'moderate'
            else:
                stress_severity = 'low'
            
            profiles[speaker_id] = SpeakerProsodyProfile(
                speaker_id=speaker_id,
                segments_analyzed=n,
                avg_pitch_mean=np.mean(data['pitch_means']) if data['pitch_means'] else 0,
                avg_pitch_std=np.mean(data['pitch_stds']) if data['pitch_stds'] else 0,
                avg_pitch_range=np.mean(data['pitch_ranges']) if data['pitch_ranges'] else 0,
                avg_intensity=np.mean(data['intensities']) if data['intensities'] else 0,
                avg_jitter=np.mean(data['jitters']) if data['jitters'] else 0,
                avg_shimmer=np.mean(data['shimmers']) if data['shimmers'] else 0,
                avg_hnr=np.mean(data['hnrs']) if data['hnrs'] else 0,
                avg_speech_rate=np.mean(data['speech_rates']) if data['speech_rates'] else 0,
                pitch_variability=np.std(data['pitch_means']) if data['pitch_means'] else 0,
                intensity_variability=np.std(data['intensities']) if data['intensities'] else 0,
                total_stress_indicators=data['stress_indicators'],
                stress_severity=stress_severity
            )
        
        return profiles


def analyze_prosody(audio_path: str,
                   segments: List[Dict] = None) -> Tuple[List[ProsodySegment], Dict[str, SpeakerProsodyProfile]]:
    """
    Convenience function to analyze prosody in an audio file.
    
    Args:
        audio_path: Path to audio file
        segments: Optional list of segments with timing and speaker info
    
    Returns:
        Tuple of (prosody segments, speaker profiles)
    """
    from config_enhanced import get_enhanced_config
    config = get_enhanced_config()
    
    analyzer = ProsodyAnalyzer(
        pitch_floor=config.prosody.pitch_floor,
        pitch_ceiling=config.prosody.pitch_ceiling
    )
    
    segments_result = analyzer.analyze_audio_file(audio_path, segments)
    profiles = analyzer.aggregate_by_speaker(segments_result)
    
    return segments_result, profiles


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prosody.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print("=" * 60)
    print("PROSODIC ANALYSIS")
    print("=" * 60)
    print(f"File: {audio_file}")
    print(f"Parselmouth available: {PARSELMOUTH_AVAILABLE}")
    
    segments, profiles = analyze_prosody(audio_file)
    
    print(f"\n📊 PROSODY ANALYSIS RESULTS:")
    print(f"   Segments analyzed: {len(segments)}")
    
    # Find segments with stress indicators
    stress_segments = [s for s in segments if s.get_stress_indicators()['indicator_count'] > 0]
    print(f"   Segments with stress indicators: {len(stress_segments)}")
    
    print(f"\n📢 SPEAKER PROSODY PROFILES:")
    for speaker_id, profile in profiles.items():
        print(f"\n   {speaker_id}:")
        print(f"      Avg pitch: {profile.avg_pitch_mean:.1f} Hz")
        print(f"      Avg intensity: {profile.avg_intensity:.1f} dB")
        print(f"      Voice quality - Jitter: {profile.avg_jitter:.4f}, Shimmer: {profile.avg_shimmer:.4f}")
        print(f"      Speech rate: {profile.avg_speech_rate:.2f} syllables/sec")
        print(f"      Stress indicators: {profile.total_stress_indicators} ({profile.stress_severity})")
