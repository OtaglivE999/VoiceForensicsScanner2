"""
Live Voice Matcher Module
Real-time speaker identification by matching voice embeddings
against the speaker database with alert triggering.

Version: 1.0
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Match Result
# ============================================================================

@dataclass
class MatchResult:
    """Result of a speaker matching operation."""
    matched: bool
    speaker_id: Optional[str]
    speaker_name: Optional[str]
    similarity: float
    role: Optional[str]
    role_confidence: float
    
    # Matching details
    field_type_used: str
    threshold_used: float
    embedding_norm: float
    
    # Audio info
    audio_duration_ms: float
    voice_confidence: float
    snr_db: float
    
    # Timing
    timestamp: datetime
    processing_time_ms: float
    
    # Candidates
    top_candidates: List[Tuple[str, float]] = field(default_factory=list)
    # [(speaker_id, similarity), ...]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'matched': self.matched,
            'speaker_id': self.speaker_id,
            'speaker_name': self.speaker_name,
            'similarity': self.similarity,
            'role': self.role,
            'role_confidence': self.role_confidence,
            'field_type_used': self.field_type_used,
            'threshold_used': self.threshold_used,
            'embedding_norm': self.embedding_norm,
            'audio_duration_ms': self.audio_duration_ms,
            'voice_confidence': self.voice_confidence,
            'snr_db': self.snr_db,
            'timestamp': self.timestamp.isoformat(),
            'processing_time_ms': self.processing_time_ms,
            'top_candidates': self.top_candidates
        }


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MatcherConfig:
    """Configuration for the live voice matcher."""
    # Matching settings
    default_threshold: float = 0.75
    high_confidence_threshold: float = 0.88
    min_voice_confidence: float = 0.5
    min_audio_duration_ms: float = 500
    
    # Buffer settings
    audio_buffer_seconds: float = 2.0
    match_interval_ms: float = 1000           # How often to attempt matches
    embedding_smoothing: bool = True           # Average recent embeddings
    smoothing_window: int = 3
    
    # Unknown speaker handling
    track_unknown_speakers: bool = True
    unknown_speaker_threshold: float = 0.60    # Lower threshold for unknown tracking
    min_appearances_for_auto_enroll: int = 5
    auto_enroll_unknown: bool = False
    
    # Alert triggering
    alert_on_match: bool = True
    alert_on_role_match: bool = True
    alert_roles: List[str] = field(default_factory=lambda: ["PERPETRATOR"])
    
    # Performance
    use_npu: bool = True
    max_concurrent_matches: int = 4


# ============================================================================
# Live Voice Matcher
# ============================================================================

class LiveVoiceMatcher:
    """
    Real-time speaker identification system.
    Matches incoming audio against the speaker database.
    """
    
    def __init__(self, config: MatcherConfig = None):
        """
        Initialize the live voice matcher.
        
        Args:
            config: Matcher configuration
        """
        self.config = config or MatcherConfig()
        
        # Components (lazy loaded)
        self._embedding_extractor = None
        self._field_processor = None
        self._speaker_database = None
        self._alert_manager = None
        
        # Buffers
        self._audio_buffer: Dict[str, deque] = {}  # stream_id -> audio deque
        self._embedding_buffer: Dict[str, deque] = {}  # speaker_id -> recent embeddings
        
        # Unknown speaker tracking
        self._unknown_embeddings: Dict[str, List[np.ndarray]] = {}
        # temp_id -> [embeddings]
        
        # Match history
        self._match_history: deque = deque(maxlen=1000)
        self._last_match_time: Dict[str, datetime] = {}
        
        # Processing
        self._running = False
        self._match_queue = queue.Queue()
        self._worker_thread = None
        
        # Callbacks
        self.on_match_callbacks: List[Callable[[MatchResult], None]] = []
        self.on_unknown_callbacks: List[Callable[[str, np.ndarray], None]] = []
    
    def _get_embedding_extractor(self):
        """Lazy load embedding extractor."""
        if self._embedding_extractor is None:
            from npu_speaker_embedding import NPUSpeakerEmbedding, EmbeddingConfig
            
            emb_config = EmbeddingConfig(
                model_name='ecapa_tdnn',
                enable_npu=self.config.use_npu
            )
            self._embedding_extractor = NPUSpeakerEmbedding(emb_config)
        
        return self._embedding_extractor
    
    def _get_field_processor(self):
        """Lazy load field processor."""
        if self._field_processor is None:
            from field_processor import FieldTypeProcessor
            self._field_processor = FieldTypeProcessor()
        return self._field_processor
    
    def _get_speaker_database(self):
        """Lazy load speaker database."""
        if self._speaker_database is None:
            from speaker_database_v2 import get_speaker_database_v2
            self._speaker_database = get_speaker_database_v2()
        return self._speaker_database
    
    def _get_alert_manager(self):
        """Lazy load alert manager."""
        if self._alert_manager is None:
            from alerts.alert_manager import get_alert_manager
            self._alert_manager = get_alert_manager()
        return self._alert_manager
    
    def _match_worker(self) -> None:
        """Background worker for processing match requests."""
        while self._running:
            try:
                request = self._match_queue.get(timeout=0.5)
                if request is None:
                    continue
                
                stream_id, audio, timestamp = request
                result = self._process_match(stream_id, audio, timestamp)
                
                if result:
                    self._match_history.append(result)
                    
                    # Notify callbacks
                    for callback in self.on_match_callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"Match callback error: {e}")
                    
                    # Trigger alerts if matched
                    if result.matched and self.config.alert_on_match:
                        self._trigger_alert(result)
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Match worker error: {e}")
    
    def _process_match(self, 
                      stream_id: str, 
                      audio: np.ndarray,
                      timestamp: datetime) -> Optional[MatchResult]:
        """Process a single match request."""
        start_time = time.time()
        
        try:
            # Get components
            extractor = self._get_embedding_extractor()
            processor = self._get_field_processor()
            database = self._get_speaker_database()
            
            # Process audio
            result = processor.process(audio)
            
            # Check voice confidence
            if result.voice_confidence < self.config.min_voice_confidence:
                return None
            
            # Check duration
            duration_ms = len(audio) / result.sample_rate * 1000
            if duration_ms < self.config.min_audio_duration_ms:
                return None
            
            # Extract embedding
            embedding = extractor.extract_embedding(
                result.enhanced_audio,
                result.sample_rate
            )
            
            embedding_norm = float(np.linalg.norm(embedding))
            
            # Apply embedding smoothing if enabled
            if self.config.embedding_smoothing:
                if stream_id not in self._embedding_buffer:
                    self._embedding_buffer[stream_id] = deque(
                        maxlen=self.config.smoothing_window
                    )
                
                self._embedding_buffer[stream_id].append(embedding)
                
                # Average recent embeddings
                if len(self._embedding_buffer[stream_id]) > 1:
                    embedding = np.mean(
                        list(self._embedding_buffer[stream_id]), axis=0
                    )
                    embedding = embedding / np.linalg.norm(embedding)
            
            # Find match in database
            field_type = result.detected_field_type.value
            match = database.find_matching_speaker(
                embedding=embedding,
                field_type=field_type,
                threshold=self.config.default_threshold
            )
            
            # Get top candidates
            top_candidates = self._get_top_candidates(embedding, database, field_type)
            
            processing_time = (time.time() - start_time) * 1000
            
            if match:
                speaker_id, similarity = match
                profile = database.get_speaker(speaker_id)
                
                return MatchResult(
                    matched=True,
                    speaker_id=speaker_id,
                    speaker_name=profile.display_name if profile else None,
                    similarity=similarity,
                    role=profile.role if profile else None,
                    role_confidence=profile.role_confidence if profile else 0.0,
                    field_type_used=field_type,
                    threshold_used=self.config.default_threshold,
                    embedding_norm=embedding_norm,
                    audio_duration_ms=duration_ms,
                    voice_confidence=result.voice_confidence,
                    snr_db=result.enhanced_snr_db,
                    timestamp=timestamp,
                    processing_time_ms=processing_time,
                    top_candidates=top_candidates
                )
            else:
                # Handle unknown speaker
                if self.config.track_unknown_speakers:
                    self._track_unknown(embedding, stream_id)
                
                return MatchResult(
                    matched=False,
                    speaker_id=None,
                    speaker_name=None,
                    similarity=top_candidates[0][1] if top_candidates else 0.0,
                    role=None,
                    role_confidence=0.0,
                    field_type_used=field_type,
                    threshold_used=self.config.default_threshold,
                    embedding_norm=embedding_norm,
                    audio_duration_ms=duration_ms,
                    voice_confidence=result.voice_confidence,
                    snr_db=result.enhanced_snr_db,
                    timestamp=timestamp,
                    processing_time_ms=processing_time,
                    top_candidates=top_candidates
                )
                
        except Exception as e:
            logger.error(f"Match processing error: {e}")
            return None
    
    def _get_top_candidates(self,
                           embedding: np.ndarray,
                           database,
                           field_type: str,
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top matching candidates."""
        candidates = []
        
        for speaker_id, field_embeddings in database.embeddings.items():
            stored = field_embeddings.get(field_type)
            if stored is None:
                stored = next(iter(field_embeddings.values()), None)
            
            if stored is not None:
                similarity = float(np.dot(embedding, stored) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored) + 1e-10
                ))
                candidates.append((speaker_id, similarity))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def _track_unknown(self, embedding: np.ndarray, stream_id: str) -> None:
        """Track unknown speaker embeddings for potential enrollment."""
        # Check if similar to existing unknown
        best_temp_id = None
        best_similarity = 0.0
        
        for temp_id, embeddings in self._unknown_embeddings.items():
            avg_embedding = np.mean(embeddings, axis=0)
            similarity = float(np.dot(embedding, avg_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(avg_embedding) + 1e-10
            ))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_temp_id = temp_id
        
        if best_similarity >= self.config.unknown_speaker_threshold:
            # Add to existing unknown cluster
            self._unknown_embeddings[best_temp_id].append(embedding)
            
            # Check for auto-enrollment
            if (self.config.auto_enroll_unknown and 
                len(self._unknown_embeddings[best_temp_id]) >= 
                self.config.min_appearances_for_auto_enroll):
                
                self._auto_enroll_unknown(best_temp_id)
        else:
            # Create new unknown cluster
            temp_id = f"UNKNOWN_{len(self._unknown_embeddings)}_{int(time.time())}"
            self._unknown_embeddings[temp_id] = [embedding]
            
            # Notify callbacks
            for callback in self.on_unknown_callbacks:
                try:
                    callback(temp_id, embedding)
                except Exception as e:
                    logger.error(f"Unknown callback error: {e}")
    
    def _auto_enroll_unknown(self, temp_id: str) -> None:
        """Auto-enroll an unknown speaker."""
        if temp_id not in self._unknown_embeddings:
            return
        
        embeddings = self._unknown_embeddings[temp_id]
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        database = self._get_speaker_database()
        speaker_id = database.add_speaker(
            embedding=avg_embedding,
            field_type="far_field",
            display_name=f"Auto-enrolled {temp_id[:20]}",
            role="UNKNOWN",
            notes=f"Auto-enrolled after {len(embeddings)} appearances"
        )
        
        # Remove from unknown tracking
        del self._unknown_embeddings[temp_id]
        
        logger.info(f"Auto-enrolled unknown speaker as {speaker_id}")
    
    def _trigger_alert(self, result: MatchResult) -> None:
        """Trigger alert for a match result."""
        if not result.matched:
            return
        
        alert_manager = self._get_alert_manager()
        
        # Check role-based alert
        if (self.config.alert_on_role_match and 
            result.role in self.config.alert_roles):
            
            alert_manager.trigger_role_match(
                speaker_id=result.speaker_id,
                speaker_name=result.speaker_name,
                role=result.role,
                confidence=result.similarity
            )
        else:
            # Standard speaker detection alert
            alert_manager.trigger_speaker_detected(
                speaker_id=result.speaker_id,
                speaker_name=result.speaker_name,
                confidence=result.similarity,
                role=result.role
            )
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    def start(self) -> None:
        """Start the matcher background worker."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._match_worker, daemon=True)
        self._worker_thread.start()
        
        logger.info("Live voice matcher started")
    
    def stop(self) -> None:
        """Stop the matcher."""
        self._running = False
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        
        logger.info("Live voice matcher stopped")
    
    def submit_audio(self,
                    stream_id: str,
                    audio: np.ndarray,
                    timestamp: datetime = None) -> None:
        """
        Submit audio for matching.
        
        Args:
            stream_id: Source stream ID
            audio: Audio data (mono, float32)
            timestamp: Capture timestamp
        """
        if not self._running:
            return
        
        timestamp = timestamp or datetime.now()
        
        # Check rate limiting
        last_match = self._last_match_time.get(stream_id)
        if last_match:
            elapsed = (timestamp - last_match).total_seconds() * 1000
            if elapsed < self.config.match_interval_ms:
                return
        
        self._last_match_time[stream_id] = timestamp
        
        # Queue for processing
        try:
            self._match_queue.put_nowait((stream_id, audio, timestamp))
        except queue.Full:
            logger.warning("Match queue full, dropping request")
    
    def process_chunk(self, chunk) -> Optional[MatchResult]:
        """
        Process an AudioChunk from stream_capture.
        
        Args:
            chunk: AudioChunk object
        
        Returns:
            MatchResult if voice detected, None otherwise
        """
        if not chunk.is_voice or chunk.voice_confidence < self.config.min_voice_confidence:
            return None
        
        # Buffer audio
        if chunk.stream_id not in self._audio_buffer:
            max_samples = int(
                self.config.audio_buffer_seconds * chunk.sample_rate
            )
            self._audio_buffer[chunk.stream_id] = deque(maxlen=max_samples)
        
        self._audio_buffer[chunk.stream_id].extend(chunk.audio)
        
        # Check if enough audio buffered
        buffer = self._audio_buffer[chunk.stream_id]
        min_samples = int(self.config.min_audio_duration_ms / 1000 * chunk.sample_rate)
        
        if len(buffer) >= min_samples:
            audio = np.array(buffer)
            self.submit_audio(chunk.stream_id, audio, chunk.timestamp)
        
        return None  # Result delivered via callback
    
    def match_audio_file(self, 
                        audio_path: str,
                        stream_id: str = "file") -> Optional[MatchResult]:
        """
        Match audio from a file (synchronous).
        
        Args:
            audio_path: Path to audio file
            stream_id: Virtual stream ID
        
        Returns:
            MatchResult
        """
        import torchaudio
        
        waveform, sr = torchaudio.load(audio_path)
        audio = waveform.numpy().squeeze()
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(waveform).numpy().squeeze()
        
        return self._process_match(stream_id, audio, datetime.now())
    
    def enroll_speaker(self,
                      audio_path: str,
                      display_name: str,
                      role: str = "UNKNOWN",
                      field_type: str = "far_field",
                      case_ids: List[str] = None) -> Optional[str]:
        """
        Enroll a new speaker from audio file.
        
        Args:
            audio_path: Path to enrollment audio
            display_name: Speaker name
            role: Speaker role
            field_type: Field type for this sample
            case_ids: Associated case IDs
        
        Returns:
            New speaker ID
        """
        try:
            import torchaudio
            
            extractor = self._get_embedding_extractor()
            processor = self._get_field_processor()
            database = self._get_speaker_database()
            
            # Load and process audio
            waveform, sr = torchaudio.load(audio_path)
            audio = waveform.numpy().squeeze()
            
            result = processor.process(audio, sr)
            
            # Extract embedding
            embedding = extractor.extract_embedding(
                result.enhanced_audio,
                result.sample_rate
            )
            
            # Add to database
            speaker_id = database.add_speaker(
                embedding=embedding,
                field_type=field_type,
                display_name=display_name,
                role=role,
                speaking_time=result.audio_duration_ms / 1000 if hasattr(result, 'audio_duration_ms') else len(audio) / sr,
                case_ids=case_ids,
                wav_path=audio_path,
                recording_conditions={
                    'snr_db': result.enhanced_snr_db,
                    'field_type_detected': result.detected_field_type.value
                }
            )
            
            logger.info(f"Enrolled speaker: {display_name} ({speaker_id})")
            return speaker_id
            
        except Exception as e:
            logger.error(f"Enrollment error: {e}")
            return None
    
    def get_recent_matches(self, 
                          count: int = 50,
                          matched_only: bool = False) -> List[MatchResult]:
        """Get recent match results."""
        results = list(self._match_history)[-count:]
        
        if matched_only:
            results = [r for r in results if r.matched]
        
        return list(reversed(results))
    
    def get_unknown_speakers(self) -> List[Dict[str, Any]]:
        """Get tracked unknown speakers."""
        return [
            {
                'temp_id': temp_id,
                'appearance_count': len(embeddings),
                'needs_enrollment': len(embeddings) >= self.config.min_appearances_for_auto_enroll
            }
            for temp_id, embeddings in self._unknown_embeddings.items()
        ]
    
    def enroll_unknown(self, 
                      temp_id: str, 
                      display_name: str,
                      role: str = "UNKNOWN") -> Optional[str]:
        """Enroll a tracked unknown speaker."""
        if temp_id not in self._unknown_embeddings:
            return None
        
        embeddings = self._unknown_embeddings[temp_id]
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        database = self._get_speaker_database()
        speaker_id = database.add_speaker(
            embedding=avg_embedding,
            field_type="far_field",
            display_name=display_name,
            role=role,
            notes=f"Enrolled from tracked unknown ({len(embeddings)} appearances)"
        )
        
        del self._unknown_embeddings[temp_id]
        return speaker_id
    
    def register_match_callback(self, callback: Callable[[MatchResult], None]) -> None:
        """Register a callback for match results."""
        self.on_match_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get matcher statistics."""
        total = len(self._match_history)
        matched = sum(1 for r in self._match_history if r.matched)
        
        return {
            'total_matches_attempted': total,
            'successful_matches': matched,
            'match_rate': matched / total if total > 0 else 0,
            'unknown_speakers_tracked': len(self._unknown_embeddings),
            'embedding_buffer_streams': len(self._embedding_buffer),
            'is_running': self._running
        }


# ============================================================================
# Convenience Functions
# ============================================================================

_matcher: Optional[LiveVoiceMatcher] = None

def get_voice_matcher(config: MatcherConfig = None) -> LiveVoiceMatcher:
    """Get the global voice matcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = LiveVoiceMatcher(config or MatcherConfig())
    return _matcher


if __name__ == '__main__':
    # Demo the live matcher
    matcher = LiveVoiceMatcher()
    matcher.start()
    
    # Test callback
    def print_match(result: MatchResult):
        if result.matched:
            print(f"✓ MATCH: {result.speaker_name} ({result.similarity:.1%})")
        else:
            print(f"✗ Unknown speaker (top: {result.top_candidates[0] if result.top_candidates else 'none'})")
    
    matcher.register_match_callback(print_match)
    
    # Test with a file if available
    test_file = Path("./speaker_samples/test.wav")
    if test_file.exists():
        result = matcher.match_audio_file(str(test_file))
        print(f"\nFile match result: {result}")
    
    print("\nStatistics:")
    print(matcher.get_statistics())
    
    matcher.stop()
