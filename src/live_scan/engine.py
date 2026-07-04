"""
Live Scan Engine - Main Orchestrator
Coordinates real-time audio capture, processing, and pattern detection.
Includes Voice Biometric Fingerprinting System.

Version: 1.1
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
import threading
import queue
import time
import logging
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .voice_biometrics import VoiceBiometricEngine, BiometricMatch

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ScanConfig:
    """Configuration for the live scan engine."""
    
    # Audio settings
    sample_rate: int = 48000
    channels: int = 1
    device_id: Optional[int] = None
    device_name: Optional[str] = None  # e.g., "Zoom H6"

    # Processing settings
    segment_duration_ms: int = 2000      # Process audio in 2-second segments
    overlap_ms: int = 500                 # Overlap between segments
    buffer_seconds: float = 30.0          # Audio buffer size

    # iPhone microphone enhancement
    iphone_enhancement: bool = False
    iphone_mode: str = 'forensic'  # forensic, voice_isolation, wide_spectrum, interview, surveillance
    
    # Pattern detection
    enable_dialogue_patterns: bool = True
    enable_coercive_detection: bool = True
    enable_emotion_detection: bool = True
    enable_speaker_identification: bool = True
    
    # NPU/GPU acceleration
    use_npu: bool = True
    use_gpu: bool = True
    
    # Output settings
    output_dir: Optional[str] = None
    save_flagged_audio: bool = True
    save_all_transcriptions: bool = True
    
    # Alert settings
    alert_threshold: float = 0.75
    enable_sound_alerts: bool = False
    enable_visual_alerts: bool = True
    
    # Dashboard
    enable_dashboard: bool = True
    dashboard_port: int = 8050
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = str(Path(__file__).parent.parent / "live_scan_output")


# ============================================================================
# Scan Result
# ============================================================================

@dataclass
class ScanResult:
    """Result from processing a single audio segment."""
    
    # Timing
    timestamp: datetime
    segment_start_ms: float
    segment_end_ms: float
    processing_time_ms: float
    
    # Audio info
    audio_duration_ms: float
    has_voice: bool
    voice_confidence: float
    snr_db: float
    
    # Transcription
    transcription: Optional[str] = None
    transcription_confidence: float = 0.0
    language: str = "en"
    
    # Speaker identification
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    speaker_similarity: float = 0.0
    
    # Voice Biometric Fingerprinting
    biometric_match_confidence: float = 0.0
    biometric_formants: Optional[Dict[str, float]] = None  # F1, F2, F3
    biometric_jitter: Optional[Dict[str, float]] = None    # local, RAP, PPQ5
    biometric_shimmer: Optional[Dict[str, float]] = None   # local, APQ3, APQ5
    biometric_mfcc_similarity: float = 0.0
    biometric_spectral_match: float = 0.0
    biometric_false_positive_risk: float = 0.0
    biometric_is_verified: bool = False
    
    # Patterns detected
    patterns_detected: List[Dict[str, Any]] = field(default_factory=list)
    
    # Emotion
    emotion: Optional[str] = None
    emotion_confidence: float = 0.0
    
    # Alerts triggered
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'segment_start_ms': self.segment_start_ms,
            'segment_end_ms': self.segment_end_ms,
            'processing_time_ms': self.processing_time_ms,
            'audio_duration_ms': self.audio_duration_ms,
            'has_voice': self.has_voice,
            'voice_confidence': self.voice_confidence,
            'snr_db': self.snr_db,
            'transcription': self.transcription,
            'transcription_confidence': self.transcription_confidence,
            'language': self.language,
            'speaker_id': self.speaker_id,
            'speaker_name': self.speaker_name,
            'speaker_similarity': self.speaker_similarity,
            'biometric_match_confidence': self.biometric_match_confidence,
            'biometric_formants': self.biometric_formants,
            'biometric_jitter': self.biometric_jitter,
            'biometric_shimmer': self.biometric_shimmer,
            'biometric_mfcc_similarity': self.biometric_mfcc_similarity,
            'biometric_spectral_match': self.biometric_spectral_match,
            'biometric_false_positive_risk': self.biometric_false_positive_risk,
            'biometric_is_verified': self.biometric_is_verified,
            'patterns_detected': self.patterns_detected,
            'emotion': self.emotion,
            'emotion_confidence': self.emotion_confidence,
            'alerts': self.alerts
        }


# ============================================================================
# Live Scan Engine
# ============================================================================

class LiveScanEngine:
    """
    Main orchestrator for live audio scanning with dialogue pattern recognition.
    
    Coordinates:
    - Real-time audio capture
    - NPU/GPU-accelerated processing
    - Pattern detection pipeline
    - Speaker identification
    - Alert management
    - Dashboard updates
    """
    
    def __init__(self, config: Optional[ScanConfig] = None):
        """
        Initialize the live scan engine.
        
        Args:
            config: Scan configuration (uses defaults if None)
        """
        self.config = config or ScanConfig()
        
        # State
        self._running = False
        self._paused = False
        self._start_time: Optional[datetime] = None
        
        # Components (lazy loaded to avoid import errors)
        self._audio_stream = None
        self._pattern_pipeline = None
        self._alert_manager = None
        self._dashboard = None
        
        # Existing module integrations (lazy loaded)
        self._live_matcher = None
        self._field_processor = None
        self._conversation_dynamics = None
        self._whisper_npu = None
        self._sentiment_npu = None
        
        # Voice Biometric Fingerprinting System
        self._biometric_engine: Optional[VoiceBiometricEngine] = None
        
        # Processing
        self._audio_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        self._worker_threads: List[threading.Thread] = []
        
        # History and statistics
        self._results_history: deque = deque(maxlen=10000)
        self._session_stats = {
            'start_time': None,
            'segments_processed': 0,
            'voice_segments': 0,
            'patterns_detected': 0,
            'alerts_triggered': 0,
            'speakers_identified': set(),
            'total_audio_duration_ms': 0,
            'total_processing_time_ms': 0,
            'biometric_matches': 0,
            'enrolled_voiceprints': 0
        }
        
        # Callbacks
        self._on_result_callbacks: List[Callable[[ScanResult], None]] = []
        self._on_alert_callbacks: List[Callable[[Dict], None]] = []
        self._on_pattern_callbacks: List[Callable[[Dict], None]] = []
        
        # Create output directory
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LiveScanEngine initialized with config: {self.config}")
    
    # ========================================================================
    # Component Initialization
    # ========================================================================
    
    def _init_audio_stream(self):
        """Initialize audio stream manager."""
        if self._audio_stream is None:
            try:
                from .audio_stream import AudioStreamManager, StreamConfig

                stream_config = StreamConfig(
                    sample_rate=self.config.sample_rate,
                    channels=self.config.channels,
                    device_id=self.config.device_id,
                    device_name=self.config.device_name,
                    buffer_seconds=self.config.buffer_seconds,
                    iphone_enhancement=self.config.iphone_enhancement,
                    iphone_mode=self.config.iphone_mode,
                )
                self._audio_stream = AudioStreamManager(stream_config)
                logger.info("Audio stream manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize audio stream: {e}")
                raise
    
    def _init_pattern_pipeline(self):
        """Initialize pattern detection pipeline."""
        if self._pattern_pipeline is None:
            try:
                from .pattern_pipeline import PatternPipeline, PipelineConfig
                
                pipeline_config = PipelineConfig(
                    enable_dialogue_patterns=self.config.enable_dialogue_patterns,
                    enable_coercive_detection=self.config.enable_coercive_detection,
                    enable_emotion_detection=self.config.enable_emotion_detection
                )
                self._pattern_pipeline = PatternPipeline(pipeline_config)
                logger.info("Pattern pipeline initialized")
            except Exception as e:
                logger.warning(f"Pattern pipeline not available: {e}")
    
    def _init_alert_manager(self):
        """Initialize alert manager."""
        if self._alert_manager is None:
            try:
                from .alert_manager import AlertManager, AlertConfig, AlertLevel
                
                alert_config = AlertConfig(
                    min_alert_level=AlertLevel.MEDIUM,
                    cooldown_seconds=5.0,
                    enable_console_alerts=True,
                    enable_file_logging=True,
                    enable_sound_alerts=self.config.enable_sound_alerts,
                    enable_callback_alerts=True
                )
                self._alert_manager = AlertManager(alert_config)
                logger.info("Alert manager initialized")
            except Exception as e:
                logger.warning(f"Alert manager not available: {e}")
    
    def _init_existing_modules(self):
        """Initialize integrations with existing audio_capture_enhanced modules."""
        
        # Live matcher for speaker identification
        if self.config.enable_speaker_identification:
            try:
                from live_matcher import LiveVoiceMatcher, MatcherConfig
                self._live_matcher = LiveVoiceMatcher(MatcherConfig(use_npu=self.config.use_npu))
                logger.info("Live matcher integrated")
            except ImportError:
                logger.warning("live_matcher module not available")
        
        # Field processor for audio enhancement
        try:
            from field_processor import FieldProcessor, ProcessorConfig
            self._field_processor = FieldProcessor(ProcessorConfig())
            logger.info("Field processor integrated")
        except ImportError:
            logger.warning("field_processor module not available")
        
        # Conversation dynamics
        if self.config.enable_dialogue_patterns:
            try:
                from conversation_dynamics import ConversationAnalyzer
                self._conversation_dynamics = ConversationAnalyzer()
                logger.info("Conversation dynamics integrated")
            except ImportError:
                logger.warning("conversation_dynamics module not available")
        
        # NPU modules for acceleration
        if self.config.use_npu:
            try:
                from npu import WhisperNPU, SentimentNPU
                self._whisper_npu = WhisperNPU()
                self._sentiment_npu = SentimentNPU()
                logger.info("NPU acceleration enabled")
            except ImportError:
                logger.warning("NPU modules not available, using CPU fallback")
    
    def _init_biometric_engine(self):
        """Initialize Voice Biometric Fingerprinting System."""
        if self._biometric_engine is None:
            try:
                voiceprint_dir = self._output_dir / 'voiceprints'
                voiceprint_dir.mkdir(parents=True, exist_ok=True)
                
                self._biometric_engine = VoiceBiometricEngine(
                    voiceprint_dir=voiceprint_dir,
                    sample_rate=self.config.sample_rate,
                    min_samples_for_enrollment=3,
                    distance_invariant=True
                )
                
                # Check for existing voiceprints
                existing_prints = list(voiceprint_dir.glob('*.json'))
                num_loaded = len(self._biometric_engine._voiceprints)
                if num_loaded > 0:
                    logger.info(f"Voice Biometric Engine initialized - loaded {num_loaded} existing voiceprints")
                    self._session_stats['enrolled_voiceprints'] = num_loaded
                else:
                    logger.info("Voice Biometric Engine initialized - no existing voiceprints")
                    
            except Exception as e:
                logger.error(f"Failed to initialize biometric engine: {e}")
                self._biometric_engine = None

    # ========================================================================
    # Main Control
    # ========================================================================
    
    def start(self, blocking: bool = True):
        """
        Start the live scan engine.
        
        Args:
            blocking: If True, blocks until stopped. If False, runs in background.
        """
        if self._running:
            logger.warning("Engine already running")
            return
        
        logger.info("Starting live scan engine...")
        
        # Initialize components
        self._init_audio_stream()
        self._init_pattern_pipeline()
        self._init_alert_manager()
        self._init_existing_modules()
        self._init_biometric_engine()
        
        # Reset state
        self._running = True
        self._paused = False
        self._start_time = datetime.now(timezone.utc)
        self._session_stats['start_time'] = self._start_time.isoformat()
        
        # Start worker threads
        self._start_workers()
        
        # Start audio capture
        self._audio_stream.start(callback=self._on_audio_segment)
        
        # Start dashboard if enabled
        if self.config.enable_dashboard:
            self._start_dashboard()
        
        logger.info("Live scan engine started")
        
        # Print banner with field detection legend
        print("\n" + "=" * 70)
        print("🎙️  LIVE SCAN + DIALOGUE PATTERN RECOGNITION")
        print("    🇺🇸 English / 🇪🇸 Español (Auto-detect)")
        print("    🔐 Voice Biometric Fingerprinting System")
        print("=" * 70)
        print()
        print("Field Detection (SNR → Distance):")
        print("   NEAR:      ≥35dB (0-1m)")
        print("   MEDIUM:    25-35dB (1-3m)")
        print("   FAR:       15-25dB (3-6m)")
        print("   ULTRA-FAR: 6-15dB (6m+)")
        print("   EXTREME:   2-6dB (very far)")
        print("   WHISPER:   <2dB (faint)")
        print()
        print("Voice Biometric Features:")
        print("   🎤 Formants (F1, F2, F3) - Vocal tract anatomy")
        print("   📊 Jitter - Pitch perturbation patterns")
        print("   📈 Shimmer - Amplitude variation analysis")
        print("   🔊 MFCC - Timbral characteristics (13 coefficients)")
        print("   🌈 Spectral - Voice color/brightness profile")
        print("   🔐 Identity Score - 0-100% match confidence")
        print("   ⚠️  FP-Risk - False positive risk assessment")
        print()
        print("Speaker Status:")
        print("   🔐 VERIFIED - Biometrically confirmed identity")
        print("   🔑 PARTIAL - Partial biometric match")
        print("   👤 NEW - New speaker (auto-enrolled)")
        print()
        print("=" * 70)
        print(f"📍 Output: {self._output_dir}")
        print(f"🎯 Patterns: {'ON' if self.config.enable_dialogue_patterns else 'OFF'}")
        print(f"👤 Speaker ID: {'ON' if self.config.enable_speaker_identification else 'OFF'}")
        bio_status = "ON" if self._biometric_engine else "OFF"
        enrolled = self._session_stats.get('enrolled_voiceprints', 0)
        print(f"🔐 Biometrics: {bio_status} ({enrolled} voiceprints)")
        iphone_status = f"ON ({self.config.iphone_mode})" if self.config.iphone_enhancement else "OFF"
        print(f"📱 iPhone Mic: {iphone_status}")
        print(f"⚡ NPU: {'ON' if self.config.use_npu else 'OFF'}")
        if self.config.enable_dashboard:
            print(f"📊 Dashboard: http://localhost:{self.config.dashboard_port}")
        print("=" * 70)
        print("Commands: [Ctrl+C]=Stop")
        print("=" * 70)
        print("\n🎤 Scanning...\n")
        
        if blocking:
            try:
                while self._running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.stop()
    
    def stop(self):
        """Stop the live scan engine."""
        if not self._running:
            return
        
        logger.info("Stopping live scan engine...")
        self._running = False
        
        # Stop audio capture
        if self._audio_stream:
            self._audio_stream.stop()
        
        # Wait for workers
        for thread in self._worker_threads:
            thread.join(timeout=2.0)
        
        # Save session summary
        self._save_session_summary()
        
        logger.info("Live scan engine stopped")
        print("\n" + "=" * 60)
        print("🛑 LIVE SCAN ENGINE STOPPED")
        print("=" * 60)
        print(f"📊 Segments processed: {self._session_stats['segments_processed']}")
        print(f"🗣️  Voice segments: {self._session_stats['voice_segments']}")
        print(f"🔍 Patterns detected: {self._session_stats['patterns_detected']}")
        print(f"⚠️  Alerts triggered: {self._session_stats['alerts_triggered']}")
        print(f"👥 Unique speakers: {len(self._session_stats['speakers_identified'])}")
        print(f"🔐 Biometric matches: {self._session_stats.get('biometric_matches', 0)}")
        print(f"📝 Enrolled voiceprints: {self._session_stats.get('enrolled_voiceprints', 0)}")
        print("=" * 60 + "\n")
    
    def pause(self):
        """Pause processing (audio still captured to buffer)."""
        self._paused = True
        logger.info("Scan paused")
    
    def resume(self):
        """Resume processing."""
        self._paused = False
        logger.info("Scan resumed")
    
    # ========================================================================
    # Signal Processing Utilities
    # ========================================================================
    
    def _simple_vad_snr(self, audio: np.ndarray) -> tuple:
        """
        Simple VAD with SNR estimation.
        
        Returns:
            tuple: (snr_db, voice_confidence, has_voice)
        """
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / 32768.0
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Estimate noise floor (use quietest 10% of frames)
        frame_size = int(0.02 * self.config.sample_rate)  # 20ms frames
        n_frames = len(audio) // frame_size
        
        if n_frames > 5:
            frame_energies = []
            for i in range(n_frames):
                frame = audio[i * frame_size:(i + 1) * frame_size]
                frame_energies.append(np.sqrt(np.mean(frame ** 2)))
            
            # Sort and take bottom 10% as noise estimate
            sorted_energies = sorted(frame_energies)
            noise_frames = max(1, n_frames // 10)
            noise_floor = np.mean(sorted_energies[:noise_frames])
            
            # Signal energy from top 50%
            signal_frames = max(1, n_frames // 2)
            signal_level = np.mean(sorted_energies[-signal_frames:])
        else:
            noise_floor = rms * 0.1  # Assume noise is 10% of signal
            signal_level = rms
        
        # Calculate SNR in dB
        if noise_floor > 1e-10:
            snr_linear = signal_level / noise_floor
            snr_db = 20 * np.log10(max(snr_linear, 1e-10))
        else:
            snr_db = 60.0  # Very clean signal
        
        # Clamp SNR to reasonable range
        snr_db = max(0, min(60, snr_db))
        
        # Voice confidence based on energy and SNR
        energy_conf = min(1.0, rms * 20)  # Scale RMS to confidence
        snr_conf = min(1.0, snr_db / 40)  # Higher SNR = higher confidence
        voice_confidence = (energy_conf + snr_conf) / 2
        
        # Voice detection threshold
        has_voice = rms > 0.005 and snr_db > 5
        
        return snr_db, voice_confidence, has_voice
    
    # ========================================================================
    # Processing Workers
    # ========================================================================
    
    def _start_workers(self):
        """Start processing worker threads."""
        # Main processing worker
        worker = threading.Thread(target=self._processing_worker, daemon=True)
        worker.start()
        self._worker_threads.append(worker)
        
        # Result handler worker
        result_worker = threading.Thread(target=self._result_worker, daemon=True)
        result_worker.start()
        self._worker_threads.append(result_worker)
    
    def _processing_worker(self):
        """Worker thread for processing audio segments."""
        while self._running:
            try:
                # Get audio segment from queue
                segment_data = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if self._paused:
                continue
            
            try:
                # Process the segment
                result = self._process_segment(segment_data)
                
                # Queue result for handling
                self._result_queue.put(result)
                
            except Exception as e:
                logger.error(f"Error processing segment: {e}")
    
    def _result_worker(self):
        """Worker thread for handling results."""
        while self._running:
            try:
                result = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            try:
                # Update statistics
                self._update_stats(result)
                
                # Add to history
                self._results_history.append(result)
                
                # Check for alerts
                self._check_alerts(result)
                
                # Invoke callbacks
                for callback in self._on_result_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Log significant results with field detection format
                if result.has_voice:
                    self._print_detection(result)
                
            except Exception as e:
                logger.error(f"Error handling result: {e}")
    
    def _get_field_zone(self, snr_db: float) -> str:
        """
        Get field detection zone based on SNR.
        
        Field Detection (SNR → Distance):
           NEAR:      ≥35dB (0-1m)
           MEDIUM:    25-35dB (1-3m)
           FAR:       15-25dB (3-6m)
           ULTRA-FAR: 6-15dB (6m+)
           EXTREME:   2-6dB (very far)
           WHISPER:   <2dB (faint)
        """
        if snr_db >= 35:
            return "NEAR"
        elif snr_db >= 25:
            return "MEDIUM"
        elif snr_db >= 15:
            return "FAR"
        elif snr_db >= 6:
            return "ULTRA-FAR"
        elif snr_db >= 2:
            return "EXTREME"
        else:
            return "WHISPER"
    
    def _print_detection(self, result: ScanResult):
        """Print detection in field scanner format with Voice Biometric details."""
        # Track speaker consecutive detections
        if not hasattr(self, '_speaker_counts'):
            self._speaker_counts = {}
        
        speaker_id = result.speaker_id or "UNKNOWN"
        speaker_name = result.speaker_name or speaker_id
        
        # Update count
        if speaker_id not in self._speaker_counts:
            self._speaker_counts[speaker_id] = 0
        self._speaker_counts[speaker_id] += 1
        count = self._speaker_counts[speaker_id]
        
        # Get field zone
        zone = self._get_field_zone(result.snr_db)
        
        # Format timestamp
        ts = result.timestamp.strftime("%H:%M:%S")
        
        # Build output line
        # Biometric verification status
        if result.biometric_is_verified:
            emoji = "🔐✅" if count >= 3 else "🔐"  # Verified biometrically
            id_status = f"VERIFIED ({result.biometric_match_confidence:.0f}%)"
        elif result.biometric_match_confidence > 0:
            emoji = "🔑" # Partial match
            id_status = f"PARTIAL ({result.biometric_match_confidence:.0f}%)"
        else:
            emoji = "👤"  # New/unknown speaker
            id_status = "NEW"
        
        # Base output with biometric confidence
        conf_pct = int(result.voice_confidence * 100)
        output = f"[{ts}] {emoji} {speaker_name} | {zone} | SNR: {result.snr_db:.0f}dB | Conf: {conf_pct}%"
        
        # Add biometric verification status
        if result.biometric_is_verified:
            output += f" | ✓ BIO-ID: {result.biometric_match_confidence:.0f}%"
            
            # Show false positive risk assessment
            if result.biometric_false_positive_risk > 0:
                risk_level = "LOW" if result.biometric_false_positive_risk < 10 else \
                            "MED" if result.biometric_false_positive_risk < 25 else "HIGH"
                output += f" | FP-Risk: {risk_level}"
        elif result.biometric_match_confidence > 0:
            output += f" | ⚠ BIO: {result.biometric_match_confidence:.0f}%"
        else:
            output += " | 🆕 NEW SPEAKER"
        
        if count > 1:
            output += f" [x{count}]"
        
        print(output)
        
        # Show biometric feature details if verified or partial match
        if result.biometric_match_confidence > 0 and (result.biometric_is_verified or result.biometric_match_confidence > 50):
            bio_details = []
            
            # MFCC similarity
            if result.biometric_mfcc_similarity > 0:
                bio_details.append(f"MFCC:{result.biometric_mfcc_similarity:.0f}%")
            
            # Spectral match
            if result.biometric_spectral_match > 0:
                bio_details.append(f"Spectral:{result.biometric_spectral_match:.0f}%")
            
            # Jitter score
            if result.biometric_jitter and result.biometric_jitter.get('match_score', 0) > 0:
                bio_details.append(f"Jitter:{result.biometric_jitter['match_score']:.0f}%")
            
            # Shimmer score
            if result.biometric_shimmer and result.biometric_shimmer.get('match_score', 0) > 0:
                bio_details.append(f"Shimmer:{result.biometric_shimmer['match_score']:.0f}%")
            
            # Formant match
            if result.biometric_formants and result.biometric_formants.get('F1_match', 0) > 0:
                bio_details.append(f"Formants:{result.biometric_formants['F1_match']:.0f}%")
            
            if bio_details:
                print(f"    🎤 Biometrics: {' | '.join(bio_details)}")
        
        # Show transcription if available with language flag
        if result.transcription:
            lang = getattr(result, 'language', 'en') or 'en'
            lang_flags = {
                'en': '🇺🇸',
                'es': '🇪🇸',
                'english': '🇺🇸',
                'spanish': '🇪🇸',
            }
            flag = lang_flags.get(lang.lower(), '🌐')
            print(f"    {flag} 💬 \"{result.transcription}\"")
        
        # Show pattern alerts
        if result.patterns_detected:
            for pattern in result.patterns_detected:
                ptype = pattern.get('type', 'unknown')
                pconf = pattern.get('confidence', 0)
                print(f"    🔍 Pattern: {ptype} ({pconf:.0%})")
    
    # ========================================================================
    # Audio Processing
    # ========================================================================
    
    def _on_audio_segment(self, audio_data: np.ndarray, timestamp: datetime):
        """Callback when audio segment is ready."""
        self._audio_queue.put({
            'audio': audio_data,
            'timestamp': timestamp,
            'sample_rate': self.config.sample_rate
        })
    
    def _process_segment(self, segment_data: Dict) -> ScanResult:
        """
        Process a single audio segment through the full pipeline.
        
        Args:
            segment_data: Dict with 'audio', 'timestamp', 'sample_rate'
            
        Returns:
            ScanResult with all analysis results
        """
        start_time = time.perf_counter()
        
        audio = segment_data['audio']
        timestamp = segment_data['timestamp']
        sample_rate = segment_data['sample_rate']
        
        audio_duration_ms = len(audio) / sample_rate * 1000
        
        # Initialize result
        result = ScanResult(
            timestamp=timestamp,
            segment_start_ms=0,
            segment_end_ms=audio_duration_ms,
            processing_time_ms=0,
            audio_duration_ms=audio_duration_ms,
            has_voice=False,
            voice_confidence=0.0,
            snr_db=0.0
        )
        
        # Step 1: Field processing (enhancement, SNR estimation)
        if self._field_processor:
            try:
                field_result = self._field_processor.process(audio, sample_rate)
                audio = field_result.get('enhanced_audio', audio)
                result.snr_db = field_result.get('snr_db', 0.0)
                result.voice_confidence = field_result.get('voice_confidence', 0.0)
                result.has_voice = result.voice_confidence > 0.5
            except Exception as e:
                logger.debug(f"Field processing failed: {e}")
                # Simple VAD fallback with SNR
                result.snr_db, result.voice_confidence, result.has_voice = self._simple_vad_snr(audio)
        else:
            # Simple VAD fallback with SNR estimation
            result.snr_db, result.voice_confidence, result.has_voice = self._simple_vad_snr(audio)
        
        # Only continue if voice detected
        if not result.has_voice:
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return result
        
        # Step 2: Transcription
        transcription = None
        if self._whisper_npu:
            try:
                trans_result = self._whisper_npu.transcribe(audio, sample_rate)
                transcription = trans_result.get('text', '').strip()
                result.transcription = transcription
                result.transcription_confidence = trans_result.get('confidence', 0.0)
                result.language = trans_result.get('language', 'en')
            except Exception as e:
                logger.debug(f"NPU transcription failed: {e}")
        
        # Fallback to standard whisper if NPU failed
        if not transcription:
            try:
                import whisper
                if not hasattr(self, '_whisper_model'):
                    logger.info("Loading Whisper model (base) for multilingual transcription (EN/ES)...")
                    self._whisper_model = whisper.load_model('base')
                
                # Whisper expects float32 normalized audio
                audio_float = audio.astype(np.float32)
                if np.abs(audio_float).max() > 1.0:
                    audio_float = audio_float / 32768.0
                
                # Auto-detect language (supports English and Spanish)
                trans_result = self._whisper_model.transcribe(
                    audio_float, 
                    language=None,  # Auto-detect language
                    fp16=False,
                    task='transcribe'  # Keep original language (not translate)
                )
                transcription = trans_result.get('text', '').strip()
                result.transcription = transcription
                result.transcription_confidence = 0.8  # Assume reasonable confidence
                result.language = trans_result.get('language', 'unknown')
            except ImportError:
                logger.debug("Whisper not installed for fallback transcription")
            except Exception as e:
                logger.debug(f"Whisper fallback failed: {e}")
        
        # Step 3: Voice Biometric Fingerprinting System
        if self._biometric_engine and self.config.enable_speaker_identification:
            try:
                # Extract biometric features and match against enrolled voiceprints
                bio_match = self._biometric_engine.match(audio, snr_db=result.snr_db)
                
                if bio_match and bio_match.matched and bio_match.speaker_id:
                    # We have a match - populate biometric results
                    result.speaker_id = bio_match.speaker_id
                    result.speaker_name = bio_match.speaker_name or bio_match.speaker_id
                    result.speaker_similarity = bio_match.confidence
                    result.biometric_match_confidence = bio_match.confidence * 100  # Convert to percentage
                    result.biometric_mfcc_similarity = bio_match.mfcc_score * 100
                    result.biometric_spectral_match = bio_match.spectral_score * 100
                    result.biometric_false_positive_risk = bio_match.false_positive_risk * 100
                    result.biometric_is_verified = bio_match.is_verified
                    
                    # Store formant scores (convert from 0-1.0 to percentage)
                    result.biometric_formants = {
                        'F1_match': bio_match.formant_score * 100,
                        'F2_match': bio_match.formant_score * 100,
                        'F3_match': bio_match.formant_score * 100
                    }
                    
                    # Store jitter/shimmer scores
                    result.biometric_jitter = {
                        'match_score': bio_match.jitter_score * 100
                    }
                    result.biometric_shimmer = {
                        'match_score': bio_match.shimmer_score * 100
                    }
                    
                    if bio_match.is_verified:
                        self._session_stats['biometric_matches'] += 1
                        self._session_stats['speakers_identified'].add(bio_match.speaker_id)
                else:
                    # No match found - auto-enroll as new speaker
                    new_id = self._biometric_engine.enroll_sample(audio, snr_db=result.snr_db)
                    if new_id:
                        result.speaker_id = new_id
                        result.speaker_name = f"Speaker_{new_id[:8]}"
                        result.biometric_match_confidence = 0.0
                        result.biometric_is_verified = False
                        self._session_stats['enrolled_voiceprints'] = len(self._biometric_engine._voiceprints) if hasattr(self._biometric_engine, '_voiceprints') else 0
                        logger.info(f"New speaker enrolled: {result.speaker_name}")
                    else:
                        result.speaker_id = "UNKNOWN"
                        result.speaker_name = "UNKNOWN"
                        
            except Exception as e:
                logger.debug(f"Biometric processing failed: {e}")
                result.speaker_id = "UNKNOWN"
                result.speaker_name = "UNKNOWN"
        else:
            # Fallback to UNKNOWN if biometric engine not available
            result.speaker_id = "UNKNOWN"
            result.speaker_name = "UNKNOWN"
        
        # Step 4: Pattern detection
        if self._pattern_pipeline and transcription:
            try:
                patterns = self._pattern_pipeline.detect(
                    transcription=transcription,
                    audio=audio,
                    speaker_id=result.speaker_id,
                    timestamp=timestamp
                )
                result.patterns_detected = patterns
            except Exception as e:
                logger.debug(f"Pattern detection failed: {e}")
        
        # Step 5: Emotion detection
        if self._sentiment_npu and self.config.enable_emotion_detection:
            try:
                emotion_result = self._sentiment_npu.analyze(transcription or "")
                result.emotion = emotion_result.get('label', None)
                result.emotion_confidence = emotion_result.get('score', 0.0)
            except Exception as e:
                logger.debug(f"Emotion detection failed: {e}")
        
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    # ========================================================================
    # Alerts and Statistics
    # ========================================================================
    
    def _check_alerts(self, result: ScanResult):
        """Check result for alert conditions."""
        if not self._alert_manager:
            return
        
        for pattern in result.patterns_detected:
            confidence = pattern.get('confidence', 0)
            if confidence >= self.config.alert_threshold:
                alert = self._alert_manager.create_alert(
                    pattern_type=pattern.get('type', 'unknown'),
                    confidence=confidence,
                    details=pattern,
                    result=result
                )
                result.alerts.append(alert.to_dict() if hasattr(alert, 'to_dict') else alert)
                self._session_stats['alerts_triggered'] += 1
                
                # Invoke alert callbacks
                for callback in self._on_alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")
    
    def _update_stats(self, result: ScanResult):
        """Update session statistics."""
        self._session_stats['segments_processed'] += 1
        self._session_stats['total_audio_duration_ms'] += result.audio_duration_ms
        self._session_stats['total_processing_time_ms'] += result.processing_time_ms
        
        if result.has_voice:
            self._session_stats['voice_segments'] += 1
        
        if result.patterns_detected:
            self._session_stats['patterns_detected'] += len(result.patterns_detected)
        
        if result.speaker_id:
            self._session_stats['speakers_identified'].add(result.speaker_id)
    
    def _save_session_summary(self):
        """Save session summary to file."""
        summary_file = self._output_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert set to list for JSON
        stats = self._session_stats.copy()
        stats['speakers_identified'] = list(stats['speakers_identified'])
        stats['end_time'] = datetime.now(timezone.utc).isoformat()
        
        with open(summary_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Session summary saved to {summary_file}")
    
    # ========================================================================
    # Dashboard
    # ========================================================================
    
    def _start_dashboard(self):
        """Start the real-time dashboard in a separate thread."""
        try:
            from .dashboard import start_dashboard
            dashboard_thread = threading.Thread(
                target=start_dashboard,
                args=(self, self.config.dashboard_port),
                daemon=True
            )
            dashboard_thread.start()
            logger.info(f"Dashboard started on port {self.config.dashboard_port}")
        except ImportError:
            logger.warning("Dashboard module not available")
    
    # ========================================================================
    # Callbacks
    # ========================================================================
    
    def on_result(self, callback: Callable[[ScanResult], None]):
        """Register callback for scan results."""
        self._on_result_callbacks.append(callback)
    
    def on_alert(self, callback: Callable[[Dict], None]):
        """Register callback for alerts."""
        self._on_alert_callbacks.append(callback)
    
    def on_pattern(self, callback: Callable[[Dict], None]):
        """Register callback for pattern detections."""
        self._on_pattern_callbacks.append(callback)
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def is_paused(self) -> bool:
        return self._paused
    
    @property
    def stats(self) -> Dict[str, Any]:
        return self._session_stats.copy()
    
    @property
    def results_history(self) -> List[ScanResult]:
        return list(self._results_history)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Scan Engine")
    parser.add_argument('--device', type=str, help='Audio device name')
    parser.add_argument('--device-id', type=int, help='Audio device ID')
    parser.add_argument('--sample-rate', type=int, default=16000)
    parser.add_argument('--no-patterns', action='store_true', help='Disable pattern detection')
    parser.add_argument('--no-speaker-id', action='store_true', help='Disable speaker identification')
    parser.add_argument('--no-npu', action='store_true', help='Disable NPU acceleration')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable dashboard')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    parser.add_argument('--iphone', action='store_true', help='Enable iPhone microphone enhancement')
    parser.add_argument('--iphone-mode', type=str, default='forensic',
                        choices=['forensic', 'voice_isolation', 'wide_spectrum', 'interview', 'surveillance'],
                        help='iPhone enhancement mode')
    parser.add_argument('--iphone-export-swift', type=str, help='Export Swift setup code to file')

    args = parser.parse_args()

    if args.iphone_export_swift:
        from iphone_microphone import IPhoneMicrophoneEnhancer, create_iphone_enhanced_stream_config
        iphone_cfg = create_iphone_enhanced_stream_config(args.iphone_mode)
        enhancer = IPhoneMicrophoneEnhancer(iphone_cfg)
        code = enhancer.export_swift_file(args.iphone_export_swift)
        print(f"Swift configuration exported to {args.iphone_export_swift}")
        return

    config = ScanConfig(
        device_name=args.device,
        device_id=args.device_id,
        sample_rate=args.sample_rate,
        enable_dialogue_patterns=not args.no_patterns,
        enable_speaker_identification=not args.no_speaker_id,
        use_npu=not args.no_npu,
        enable_dashboard=not args.no_dashboard,
        output_dir=args.output,
        dashboard_port=args.port,
        iphone_enhancement=args.iphone,
        iphone_mode=args.iphone_mode,
    )
    
    engine = LiveScanEngine(config)
    engine.start(blocking=True)


if __name__ == '__main__':
    main()
