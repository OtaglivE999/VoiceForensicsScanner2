"""
Pattern Detection Pipeline
Real-time dialogue pattern recognition and analysis.

Version: 1.0
"""

import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Pattern Types
# ============================================================================

class PatternCategory(Enum):
    """Categories of dialogue patterns."""
    DYNAMICS = "dynamics"           # Conversation flow patterns
    COERCIVE = "coercive"           # Coercive control indicators
    EMOTIONAL = "emotional"         # Emotional state patterns
    LINGUISTIC = "linguistic"       # Language-based patterns
    PROSODIC = "prosodic"          # Speech rhythm/tone patterns


class PatternSeverity(Enum):
    """Severity levels for detected patterns."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the pattern pipeline."""
    
    # Enable/disable pattern categories
    enable_dialogue_patterns: bool = True
    enable_coercive_detection: bool = True
    enable_emotion_detection: bool = True
    enable_linguistic_patterns: bool = True
    enable_prosodic_patterns: bool = True
    
    # Thresholds
    min_confidence: float = 0.5
    alert_threshold: float = 0.75
    
    # Context window
    context_window_seconds: float = 60.0
    max_history_items: int = 1000
    
    # Speaker tracking
    track_speaker_patterns: bool = True
    speaker_pattern_window: int = 10  # Utterances per speaker


# ============================================================================
# Pattern Definitions
# ============================================================================

# Dynamics patterns (conversation flow)
DYNAMICS_PATTERNS = {
    "dominance": {
        "description": "One speaker controlling the conversation",
        "indicators": ["turn_imbalance", "interruptions", "long_monologues"],
        "severity": PatternSeverity.MEDIUM,
        "thresholds": {
            "turn_ratio_threshold": 0.75,
            "interruption_rate_threshold": 0.3,
            "monologue_duration_seconds": 30
        }
    },
    "withdrawal": {
        "description": "Speaker becoming passive or silent",
        "indicators": ["short_responses", "long_pauses", "filler_heavy"],
        "severity": PatternSeverity.MEDIUM,
        "thresholds": {
            "avg_turn_length_threshold": 2.0,
            "pause_ratio_threshold": 0.5
        }
    },
    "escalation": {
        "description": "Tension increasing over time",
        "indicators": ["rising_volume", "faster_speech", "overlapping_increases"],
        "severity": PatternSeverity.HIGH,
        "thresholds": {
            "volume_increase_percent": 20,
            "tempo_increase_percent": 15
        }
    },
    "de_escalation": {
        "description": "Tension decreasing over time",
        "indicators": ["lowering_volume", "slower_speech", "longer_pauses"],
        "severity": PatternSeverity.INFO,
        "thresholds": {
            "volume_decrease_percent": 15,
            "tempo_decrease_percent": 10
        }
    }
}

# Coercive control patterns
COERCIVE_PATTERNS = {
    "intimidation": {
        "description": "Threatening or intimidating language",
        "keywords": [
            "you better", "or else", "i'll", "don't make me",
            "you're going to regret", "watch what happens",
            "i'm warning you", "you have no idea"
        ],
        "severity": PatternSeverity.HIGH,
        "prosody_indicators": {"intensity": "high", "pitch": "variable"}
    },
    "isolation": {
        "description": "Attempts to isolate from support",
        "keywords": [
            "don't talk to", "stay away from", "they don't care",
            "only i", "no one else", "your friends", "your family",
            "they're not your real", "i'm the only one"
        ],
        "severity": PatternSeverity.HIGH,
        "prosody_indicators": {}
    },
    "gaslighting": {
        "description": "Reality distortion or denial",
        "keywords": [
            "that never happened", "you're imagining", "you're crazy",
            "you're being paranoid", "that's not what i said",
            "you always", "you never", "you're making things up",
            "you're too sensitive", "you're overreacting"
        ],
        "severity": PatternSeverity.CRITICAL,
        "prosody_indicators": {"tone": "dismissive"}
    },
    "blame_shifting": {
        "description": "Deflecting responsibility",
        "keywords": [
            "it's your fault", "you made me", "because of you",
            "if you hadn't", "you're the reason", "look what you did",
            "this is on you", "you caused this"
        ],
        "severity": PatternSeverity.HIGH,
        "prosody_indicators": {}
    },
    "minimization": {
        "description": "Downplaying concerns or experiences",
        "keywords": [
            "it's not a big deal", "you're overreacting",
            "it was just", "i was only", "calm down",
            "relax", "it's nothing", "stop being dramatic"
        ],
        "severity": PatternSeverity.MEDIUM,
        "prosody_indicators": {}
    },
    "control": {
        "description": "Exerting control over decisions",
        "keywords": [
            "you can't", "you're not allowed", "i decide",
            "i said no", "because i said so", "don't question",
            "you have to", "you must", "you will"
        ],
        "severity": PatternSeverity.HIGH,
        "prosody_indicators": {"intensity": "high"}
    }
}

# Emotional state patterns
EMOTIONAL_PATTERNS = {
    "distress": {
        "description": "Signs of emotional distress",
        "linguistic_markers": [
            "help", "please", "stop", "i can't", "no more",
            "why are you", "leave me alone"
        ],
        "prosody_indicators": ["trembling", "crying", "breathless"],
        "severity": PatternSeverity.HIGH
    },
    "fear": {
        "description": "Indicators of fear",
        "linguistic_markers": [
            "i'm scared", "don't hurt", "please don't",
            "i'm afraid", "what are you going to"
        ],
        "prosody_indicators": ["whispered", "hesitant", "quiet"],
        "severity": PatternSeverity.CRITICAL
    },
    "compliance": {
        "description": "Fearful compliance language",
        "linguistic_markers": [
            "okay okay", "i'll do it", "whatever you say",
            "fine", "alright", "yes", "sorry", "i'm sorry"
        ],
        "prosody_indicators": ["submissive_tone", "rushed"],
        "severity": PatternSeverity.MEDIUM
    },
    "confusion": {
        "description": "Signs of confusion or disorientation",
        "linguistic_markers": [
            "i don't understand", "what do you mean",
            "but you said", "i thought", "wait",
            "that doesn't make sense"
        ],
        "prosody_indicators": ["questioning_tone"],
        "severity": PatternSeverity.LOW
    }
}


# ============================================================================
# Pattern Detection Result
# ============================================================================

@dataclass
class PatternMatch:
    """A detected pattern match."""
    
    pattern_id: str
    category: PatternCategory
    severity: PatternSeverity
    confidence: float
    
    # Details
    description: str
    matched_indicators: List[str]
    matched_keywords: List[str] = field(default_factory=list)
    
    # Context
    transcription_snippet: str = ""
    speaker_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'type': self.pattern_id,  # Alias for compatibility
            'category': self.category.value,
            'severity': self.severity.value,
            'confidence': self.confidence,
            'description': self.description,
            'matched_indicators': self.matched_indicators,
            'matched_keywords': self.matched_keywords,
            'transcription_snippet': self.transcription_snippet,
            'speaker_id': self.speaker_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata
        }


# ============================================================================
# Conversation State Tracker
# ============================================================================

class ConversationState:
    """Tracks conversation state for pattern detection."""
    
    def __init__(self, max_history: int = 1000):
        self.utterances: deque = deque(maxlen=max_history)
        self.speaker_stats: Dict[str, Dict] = {}
        self.turn_count: int = 0
        self.start_time: Optional[datetime] = None
        self.last_speaker: Optional[str] = None
        self.interruption_count: int = 0
        
        # Tracking
        self.volume_history: deque = deque(maxlen=100)
        self.tempo_history: deque = deque(maxlen=100)
    
    def add_utterance(
        self,
        transcription: str,
        speaker_id: Optional[str],
        timestamp: datetime,
        duration_ms: float,
        volume: float = 0.0,
        tempo: float = 0.0
    ):
        """Add an utterance to the state."""
        if self.start_time is None:
            self.start_time = timestamp
        
        # Track turn
        self.turn_count += 1
        
        # Check for interruption (quick speaker change)
        if self.last_speaker and speaker_id and self.last_speaker != speaker_id:
            if self.utterances:
                last = self.utterances[-1]
                time_gap = (timestamp - last['timestamp']).total_seconds()
                if time_gap < 0.5:  # Less than 500ms gap
                    self.interruption_count += 1
        
        self.last_speaker = speaker_id
        
        # Store utterance
        self.utterances.append({
            'transcription': transcription,
            'speaker_id': speaker_id,
            'timestamp': timestamp,
            'duration_ms': duration_ms,
            'volume': volume,
            'tempo': tempo,
            'turn_number': self.turn_count
        })
        
        # Update speaker stats
        if speaker_id:
            if speaker_id not in self.speaker_stats:
                self.speaker_stats[speaker_id] = {
                    'turn_count': 0,
                    'total_duration_ms': 0,
                    'word_count': 0,
                    'interruptions_made': 0,
                    'interruptions_received': 0
                }
            
            stats = self.speaker_stats[speaker_id]
            stats['turn_count'] += 1
            stats['total_duration_ms'] += duration_ms
            stats['word_count'] += len(transcription.split())
        
        # Track volume/tempo
        if volume > 0:
            self.volume_history.append(volume)
        if tempo > 0:
            self.tempo_history.append(tempo)
    
    def get_turn_ratio(self, speaker_id: str) -> float:
        """Get turn ratio for a speaker."""
        if not self.speaker_stats or speaker_id not in self.speaker_stats:
            return 0.0
        total_turns = sum(s['turn_count'] for s in self.speaker_stats.values())
        if total_turns == 0:
            return 0.0
        return self.speaker_stats[speaker_id]['turn_count'] / total_turns
    
    def get_volume_trend(self) -> str:
        """Get volume trend (increasing/decreasing/stable)."""
        if len(self.volume_history) < 10:
            return "insufficient_data"
        
        recent = list(self.volume_history)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        
        change = (second_half - first_half) / (first_half + 0.001) * 100
        
        if change > 15:
            return "increasing"
        elif change < -15:
            return "decreasing"
        return "stable"
    
    def get_tempo_trend(self) -> str:
        """Get speech tempo trend."""
        if len(self.tempo_history) < 10:
            return "insufficient_data"
        
        recent = list(self.tempo_history)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        
        change = (second_half - first_half) / (first_half + 0.001) * 100
        
        if change > 10:
            return "increasing"
        elif change < -10:
            return "decreasing"
        return "stable"
    
    def get_interruption_rate(self) -> float:
        """Get interruption rate."""
        if self.turn_count < 2:
            return 0.0
        return self.interruption_count / (self.turn_count - 1)


# ============================================================================
# Pattern Pipeline
# ============================================================================

class PatternPipeline:
    """
    Pipeline for detecting dialogue patterns in real-time.
    
    Processes transcriptions and audio features to detect:
    - Conversation dynamics (dominance, withdrawal, escalation)
    - Coercive control indicators
    - Emotional state patterns
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pattern pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Conversation state
        self.state = ConversationState(max_history=self.config.max_history_items)
        
        # Pattern definitions
        self.dynamics_patterns = DYNAMICS_PATTERNS
        self.coercive_patterns = COERCIVE_PATTERNS
        self.emotional_patterns = EMOTIONAL_PATTERNS
        
        # Compile keyword patterns for efficiency
        self._compile_patterns()
        
        logger.info("Pattern pipeline initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for keyword matching."""
        self._coercive_regexes = {}
        for pattern_id, pattern in self.coercive_patterns.items():
            keywords = pattern.get('keywords', [])
            if keywords:
                # Create case-insensitive pattern
                regex_str = '|'.join(re.escape(kw) for kw in keywords)
                self._coercive_regexes[pattern_id] = re.compile(regex_str, re.IGNORECASE)
        
        self._emotional_regexes = {}
        for pattern_id, pattern in self.emotional_patterns.items():
            markers = pattern.get('linguistic_markers', [])
            if markers:
                regex_str = '|'.join(re.escape(m) for m in markers)
                self._emotional_regexes[pattern_id] = re.compile(regex_str, re.IGNORECASE)
    
    def detect(
        self,
        transcription: str,
        audio: Optional[np.ndarray] = None,
        speaker_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        duration_ms: float = 2000,
        volume: float = 0.0,
        tempo: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns in a transcription.
        
        Args:
            transcription: Text transcription
            audio: Optional audio array for prosodic analysis
            speaker_id: Optional speaker identifier
            timestamp: Timestamp of the segment
            duration_ms: Duration in milliseconds
            volume: Average volume level
            tempo: Speech tempo (words per minute)
            
        Returns:
            List of detected patterns as dictionaries
        """
        timestamp = timestamp or datetime.now(timezone.utc)
        
        # Update conversation state
        self.state.add_utterance(
            transcription=transcription,
            speaker_id=speaker_id,
            timestamp=timestamp,
            duration_ms=duration_ms,
            volume=volume,
            tempo=tempo
        )
        
        patterns = []
        
        # Detect coercive patterns
        if self.config.enable_coercive_detection:
            coercive = self._detect_coercive_patterns(transcription, speaker_id, timestamp)
            patterns.extend(coercive)
        
        # Detect emotional patterns
        if self.config.enable_emotion_detection:
            emotional = self._detect_emotional_patterns(transcription, speaker_id, timestamp)
            patterns.extend(emotional)
        
        # Detect dynamics patterns
        if self.config.enable_dialogue_patterns:
            dynamics = self._detect_dynamics_patterns(speaker_id, timestamp)
            patterns.extend(dynamics)
        
        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= self.config.min_confidence]
        
        # Convert to dicts
        return [p.to_dict() for p in patterns]
    
    def _detect_coercive_patterns(
        self,
        transcription: str,
        speaker_id: Optional[str],
        timestamp: datetime
    ) -> List[PatternMatch]:
        """Detect coercive control patterns in text."""
        patterns = []
        
        text_lower = transcription.lower()
        
        for pattern_id, regex in self._coercive_regexes.items():
            matches = regex.findall(text_lower)
            if matches:
                pattern_def = self.coercive_patterns[pattern_id]
                
                # Calculate confidence based on match count and specificity
                base_confidence = 0.6
                confidence = min(1.0, base_confidence + len(matches) * 0.1)
                
                patterns.append(PatternMatch(
                    pattern_id=pattern_id,
                    category=PatternCategory.COERCIVE,
                    severity=pattern_def['severity'],
                    confidence=confidence,
                    description=pattern_def['description'],
                    matched_indicators=['keyword_match'],
                    matched_keywords=list(set(matches)),
                    transcription_snippet=transcription[:100],
                    speaker_id=speaker_id,
                    timestamp=timestamp
                ))
        
        return patterns
    
    def _detect_emotional_patterns(
        self,
        transcription: str,
        speaker_id: Optional[str],
        timestamp: datetime
    ) -> List[PatternMatch]:
        """Detect emotional state patterns in text."""
        patterns = []
        
        text_lower = transcription.lower()
        
        for pattern_id, regex in self._emotional_regexes.items():
            matches = regex.findall(text_lower)
            if matches:
                pattern_def = self.emotional_patterns[pattern_id]
                
                base_confidence = 0.5
                confidence = min(1.0, base_confidence + len(matches) * 0.15)
                
                patterns.append(PatternMatch(
                    pattern_id=pattern_id,
                    category=PatternCategory.EMOTIONAL,
                    severity=pattern_def['severity'],
                    confidence=confidence,
                    description=pattern_def['description'],
                    matched_indicators=['linguistic_marker'],
                    matched_keywords=list(set(matches)),
                    transcription_snippet=transcription[:100],
                    speaker_id=speaker_id,
                    timestamp=timestamp
                ))
        
        return patterns
    
    def _detect_dynamics_patterns(
        self,
        speaker_id: Optional[str],
        timestamp: datetime
    ) -> List[PatternMatch]:
        """Detect conversation dynamics patterns."""
        patterns = []
        
        # Need sufficient conversation history
        if len(self.state.utterances) < 5:
            return patterns
        
        # Check dominance
        if speaker_id:
            turn_ratio = self.state.get_turn_ratio(speaker_id)
            dominance_def = self.dynamics_patterns['dominance']
            
            if turn_ratio >= dominance_def['thresholds']['turn_ratio_threshold']:
                patterns.append(PatternMatch(
                    pattern_id='dominance',
                    category=PatternCategory.DYNAMICS,
                    severity=dominance_def['severity'],
                    confidence=min(1.0, turn_ratio),
                    description=dominance_def['description'],
                    matched_indicators=['turn_imbalance'],
                    speaker_id=speaker_id,
                    timestamp=timestamp,
                    metadata={'turn_ratio': turn_ratio}
                ))
        
        # Check escalation
        volume_trend = self.state.get_volume_trend()
        tempo_trend = self.state.get_tempo_trend()
        
        if volume_trend == "increasing" or tempo_trend == "increasing":
            escalation_def = self.dynamics_patterns['escalation']
            confidence = 0.6
            indicators = []
            
            if volume_trend == "increasing":
                confidence += 0.2
                indicators.append('rising_volume')
            if tempo_trend == "increasing":
                confidence += 0.2
                indicators.append('faster_speech')
            
            interruption_rate = self.state.get_interruption_rate()
            if interruption_rate > 0.3:
                confidence += 0.1
                indicators.append('high_interruptions')
            
            if confidence >= self.config.min_confidence:
                patterns.append(PatternMatch(
                    pattern_id='escalation',
                    category=PatternCategory.DYNAMICS,
                    severity=escalation_def['severity'],
                    confidence=min(1.0, confidence),
                    description=escalation_def['description'],
                    matched_indicators=indicators,
                    timestamp=timestamp,
                    metadata={
                        'volume_trend': volume_trend,
                        'tempo_trend': tempo_trend,
                        'interruption_rate': interruption_rate
                    }
                ))
        
        # Check de-escalation
        if volume_trend == "decreasing" and tempo_trend in ["decreasing", "stable"]:
            de_escalation_def = self.dynamics_patterns['de_escalation']
            patterns.append(PatternMatch(
                pattern_id='de_escalation',
                category=PatternCategory.DYNAMICS,
                severity=de_escalation_def['severity'],
                confidence=0.7,
                description=de_escalation_def['description'],
                matched_indicators=['lowering_volume', 'slower_speech'],
                timestamp=timestamp
            ))
        
        return patterns
    
    def reset_state(self):
        """Reset conversation state."""
        self.state = ConversationState(max_history=self.config.max_history_items)
        logger.info("Pattern pipeline state reset")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation state."""
        return {
            'turn_count': self.state.turn_count,
            'speaker_count': len(self.state.speaker_stats),
            'interruption_rate': self.state.get_interruption_rate(),
            'volume_trend': self.state.get_volume_trend(),
            'tempo_trend': self.state.get_tempo_trend(),
            'speaker_stats': self.state.speaker_stats
        }
