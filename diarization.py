"""
Speaker Diarization Module
Uses pyannote.audio for speaker segmentation and embedding extraction
Identifies WHO is speaking and WHEN in the audio
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import warnings

# Suppress warnings during import
warnings.filterwarnings('ignore')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Diarization will be limited.")

try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Install with: pip install pyannote.audio")

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    print("Warning: resemblyzer not available. Speaker embeddings will be limited.")

import soundfile as sf


@dataclass
class SpeakerSegment:
    """Represents a single speaker segment in the audio."""
    speaker_id: str
    start_time: float
    end_time: float
    duration: float
    confidence: float = 1.0
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'speaker_id': self.speaker_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'confidence': self.confidence,
            'start_formatted': self._format_time(self.start_time),
            'end_formatted': self._format_time(self.end_time)
        }
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


@dataclass
class DiarizationResult:
    """Complete diarization result for an audio file."""
    audio_path: str
    duration: float
    num_speakers: int
    segments: List[SpeakerSegment]
    speaker_stats: Dict[str, Dict[str, float]]
    embeddings: Dict[str, np.ndarray]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'audio_path': self.audio_path,
            'duration': self.duration,
            'num_speakers': self.num_speakers,
            'segments': [s.to_dict() for s in self.segments],
            'speaker_stats': self.speaker_stats
        }
    
    def get_speaker_segments(self, speaker_id: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker."""
        return [s for s in self.segments if s.speaker_id == speaker_id]
    
    def get_segment_at_time(self, time: float) -> Optional[SpeakerSegment]:
        """Get the speaker segment at a specific time."""
        for segment in self.segments:
            if segment.start_time <= time <= segment.end_time:
                return segment
        return None


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio
    Segments audio by speaker and extracts voice embeddings
    """
    
    def __init__(self, 
                 hf_token: str = None,
                 model_name: str = "pyannote/speaker-diarization-3.1",
                 min_speakers: int = 1,
                 max_speakers: int = 10,
                 device: str = None):
        """
        Initialize the speaker diarizer.
        
        Args:
            hf_token: HuggingFace authentication token
            model_name: Pyannote pipeline model name
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers
            device: Compute device ('cuda', 'cpu', or None for auto)
        """
        self.hf_token = hf_token or os.environ.get('HF_TOKEN', '')
        self.model_name = model_name
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Determine device
        if device is None:
            self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.pipeline = None
        self.voice_encoder = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the diarization pipeline."""
        if self._initialized:
            return True
        
        print(f"   Loading diarization model ({self.model_name})...")
        
        # Try pyannote first
        if PYANNOTE_AVAILABLE and self.hf_token:
            try:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    token=self.hf_token
                )
                self.pipeline.to(torch.device(self.device))
                print(f"   ✅ Pyannote pipeline loaded on {self.device}")
            except Exception as e:
                print(f"   ⚠️ Pyannote failed: {e}")
                self.pipeline = None
        
        # Initialize resemblyzer for embeddings
        if RESEMBLYZER_AVAILABLE:
            try:
                self.voice_encoder = VoiceEncoder(device=self.device)
                print("   ✅ Voice encoder loaded")
            except Exception as e:
                print(f"   ⚠️ Voice encoder failed: {e}")
                self.voice_encoder = None
        
        self._initialized = self.pipeline is not None or self.voice_encoder is not None
        return self._initialized
    
    def diarize(self, 
                audio_path: str,
                num_speakers: Optional[int] = None,
                show_progress: bool = True) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Known number of speakers (None for auto-detect)
            show_progress: Show progress bar
        
        Returns:
            DiarizationResult with speaker segments and embeddings
        """
        if not self._initialized:
            self.initialize()
        
        audio_path = str(Path(audio_path).resolve())
        
        # Load audio for duration
        audio_data, sample_rate = sf.read(audio_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono
        duration = len(audio_data) / sample_rate
        
        segments = []
        embeddings = {}
        
        # Run pyannote diarization
        if self.pipeline is not None:
            print(f"   Running speaker diarization...")
            
            try:
                # Configure pipeline
                params = {}
                if num_speakers is not None:
                    params['num_speakers'] = num_speakers
                else:
                    params['min_speakers'] = self.min_speakers
                    params['max_speakers'] = self.max_speakers
                
                # Prepare audio in dict format to avoid AudioDecoder issues with torchcodec
                # pyannote 4.x expects {'waveform': (channel, time) tensor, 'sample_rate': int}
                waveform = torch.from_numpy(audio_data).float().unsqueeze(0)  # (1, num_samples)
                audio_input = {'waveform': waveform, 'sample_rate': sample_rate}
                
                # Run diarization with pre-loaded audio
                if show_progress:
                    with ProgressHook() as hook:
                        diarization_output = self.pipeline(audio_input, hook=hook, **params)
                else:
                    diarization_output = self.pipeline(audio_input, **params)
                
                # Extract segments - pyannote 4.x returns DiarizeOutput with speaker_diarization attribute
                # which contains the Annotation object
                if hasattr(diarization_output, 'speaker_diarization'):
                    # pyannote 4.x format
                    annotation = diarization_output.speaker_diarization
                else:
                    # Older format - direct Annotation
                    annotation = diarization_output
                
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    segment = SpeakerSegment(
                        speaker_id=speaker,
                        start_time=turn.start,
                        end_time=turn.end,
                        duration=turn.end - turn.start
                    )
                    segments.append(segment)
                
                print(f"   ✅ Found {len(set(s.speaker_id for s in segments))} speakers, {len(segments)} segments")
                
            except Exception as e:
                print(f"   ⚠️ Diarization error: {e}")
                # Fall back to single speaker
                segments = [SpeakerSegment(
                    speaker_id='SPEAKER_00',
                    start_time=0,
                    end_time=duration,
                    duration=duration
                )]
        else:
            # No diarization available - treat as single speaker
            print("   ⚠️ No diarization model - treating as single speaker")
            segments = [SpeakerSegment(
                speaker_id='SPEAKER_00',
                start_time=0,
                end_time=duration,
                duration=duration
            )]
        
        # Extract speaker embeddings
        if self.voice_encoder is not None:
            print("   Extracting speaker embeddings...")
            embeddings = self._extract_embeddings(audio_data, sample_rate, segments)
        
        # Calculate speaker statistics
        speaker_stats = self._calculate_speaker_stats(segments, duration)
        
        return DiarizationResult(
            audio_path=audio_path,
            duration=duration,
            num_speakers=len(speaker_stats),
            segments=segments,
            speaker_stats=speaker_stats,
            embeddings=embeddings
        )
    
    def _extract_embeddings(self, 
                           audio_data: np.ndarray, 
                           sample_rate: int,
                           segments: List[SpeakerSegment]) -> Dict[str, np.ndarray]:
        """Extract voice embeddings for each speaker."""
        embeddings = {}
        speaker_audio = {}
        
        # Collect audio for each speaker
        for segment in segments:
            speaker_id = segment.speaker_id
            start_sample = int(segment.start_time * sample_rate)
            end_sample = int(segment.end_time * sample_rate)
            
            audio_chunk = audio_data[start_sample:end_sample]
            
            if speaker_id not in speaker_audio:
                speaker_audio[speaker_id] = []
            speaker_audio[speaker_id].append(audio_chunk)
        
        # Generate embeddings per speaker
        for speaker_id, chunks in speaker_audio.items():
            try:
                # Concatenate all chunks for this speaker
                combined = np.concatenate(chunks)
                
                # Preprocess for resemblyzer (needs 16kHz)
                if sample_rate != 16000:
                    # Simple resampling
                    from scipy import signal
                    combined = signal.resample(combined, int(len(combined) * 16000 / sample_rate))
                
                # Generate embedding
                embedding = self.voice_encoder.embed_utterance(combined)
                embeddings[speaker_id] = embedding
                
            except Exception as e:
                print(f"   ⚠️ Embedding extraction failed for {speaker_id}: {e}")
        
        return embeddings
    
    def _calculate_speaker_stats(self, 
                                 segments: List[SpeakerSegment], 
                                 total_duration: float) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each speaker."""
        stats = {}
        
        for segment in segments:
            speaker_id = segment.speaker_id
            if speaker_id not in stats:
                stats[speaker_id] = {
                    'total_time': 0.0,
                    'segment_count': 0,
                    'percentage': 0.0,
                    'avg_segment_duration': 0.0,
                    'max_segment_duration': 0.0,
                    'first_appearance': float('inf'),
                    'last_appearance': 0.0
                }
            
            stats[speaker_id]['total_time'] += segment.duration
            stats[speaker_id]['segment_count'] += 1
            stats[speaker_id]['max_segment_duration'] = max(
                stats[speaker_id]['max_segment_duration'], 
                segment.duration
            )
            stats[speaker_id]['first_appearance'] = min(
                stats[speaker_id]['first_appearance'],
                segment.start_time
            )
            stats[speaker_id]['last_appearance'] = max(
                stats[speaker_id]['last_appearance'],
                segment.end_time
            )
        
        # Calculate percentages and averages
        for speaker_id in stats:
            stats[speaker_id]['percentage'] = (
                stats[speaker_id]['total_time'] / total_duration * 100
            ) if total_duration > 0 else 0
            
            stats[speaker_id]['avg_segment_duration'] = (
                stats[speaker_id]['total_time'] / stats[speaker_id]['segment_count']
            ) if stats[speaker_id]['segment_count'] > 0 else 0
        
        return stats
    
    def compare_speakers(self, 
                        embedding1: np.ndarray, 
                        embedding2: np.ndarray) -> float:
        """
        Compare two speaker embeddings using cosine similarity.
        
        Returns:
            Similarity score between 0 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


def diarize_audio(audio_path: str,
                  hf_token: str = None,
                  num_speakers: int = None,
                  show_progress: bool = True) -> DiarizationResult:
    """
    Convenience function to diarize an audio file.
    
    Args:
        audio_path: Path to audio file
        hf_token: HuggingFace token (uses env var if not provided)
        num_speakers: Known number of speakers
        show_progress: Show progress bar
    
    Returns:
        DiarizationResult with speaker segments
    """
    from config_enhanced import get_enhanced_config
    
    config = get_enhanced_config()
    
    diarizer = SpeakerDiarizer(
        hf_token=hf_token or config.diarization.hf_token,
        model_name=config.diarization.model_name,
        min_speakers=config.diarization.min_speakers,
        max_speakers=config.diarization.max_speakers
    )
    
    return diarizer.diarize(audio_path, num_speakers, show_progress)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diarization.py <audio_file> [num_speakers]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print("=" * 60)
    print("SPEAKER DIARIZATION")
    print("=" * 60)
    print(f"File: {audio_file}")
    
    result = diarize_audio(audio_file, num_speakers=num_speakers)
    
    print("\n📊 SPEAKER STATISTICS:")
    for speaker_id, stats in result.speaker_stats.items():
        print(f"\n   {speaker_id}:")
        print(f"      Speaking time: {stats['total_time']:.1f}s ({stats['percentage']:.1f}%)")
        print(f"      Segments: {stats['segment_count']}")
        print(f"      Avg segment: {stats['avg_segment_duration']:.1f}s")
    
    print("\n📝 SEGMENTS:")
    for seg in result.segments[:20]:  # Show first 20
        print(f"   [{seg.start_time:6.1f}s - {seg.end_time:6.1f}s] {seg.speaker_id}")
    
    if len(result.segments) > 20:
        print(f"   ... and {len(result.segments) - 20} more segments")
    
    # Save results
    output_file = Path(audio_file).stem + "_diarization.json"
    with open(output_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")
