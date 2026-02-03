"""
Alert Manager
Real-time alert system for detected patterns.

Version: 1.0
"""

import asyncio
from typing import Optional, Dict, List, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum
import json
import logging
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# Alert Types and Severity
# ============================================================================

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def __lt__(self, other):
        order = [AlertLevel.INFO, AlertLevel.LOW, AlertLevel.MEDIUM, 
                 AlertLevel.HIGH, AlertLevel.CRITICAL]
        return order.index(self) < order.index(other)
    
    def __le__(self, other):
        return self == other or self < other


class AlertType(Enum):
    """Types of alerts."""
    PATTERN_DETECTED = "pattern_detected"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    SPEAKER_IDENTIFIED = "speaker_identified"
    SYSTEM_EVENT = "system_event"
    ESCALATION = "escalation"


# ============================================================================
# Alert Configuration
# ============================================================================

@dataclass
class AlertConfig:
    """Configuration for alert management."""
    
    # Minimum level to raise alerts
    min_alert_level: AlertLevel = AlertLevel.MEDIUM
    
    # Cooldown to prevent alert spam (seconds)
    cooldown_seconds: float = 5.0
    
    # Enable different alert channels
    enable_console_alerts: bool = True
    enable_file_logging: bool = True
    enable_sound_alerts: bool = False
    enable_callback_alerts: bool = True
    
    # File logging
    log_file_path: Optional[str] = None
    max_log_entries: int = 10000
    
    # Alert history
    max_history: int = 1000
    
    # Escalation settings
    escalation_threshold: int = 3  # Number of alerts before escalation
    escalation_window_seconds: float = 60.0


# ============================================================================
# Alert Data Structure
# ============================================================================

@dataclass
class Alert:
    """An alert instance."""
    
    id: str
    level: AlertLevel
    alert_type: AlertType
    
    title: str
    message: str
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Related data
    pattern_id: Optional[str] = None
    speaker_id: Optional[str] = None
    transcription_snippet: Optional[str] = None
    confidence: float = 1.0
    
    # State
    acknowledged: bool = False
    dismissed: bool = False
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'level': self.level.value,
            'alert_type': self.alert_type.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'pattern_id': self.pattern_id,
            'speaker_id': self.speaker_id,
            'transcription_snippet': self.transcription_snippet,
            'confidence': self.confidence,
            'acknowledged': self.acknowledged,
            'dismissed': self.dismissed,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        return f"[{self.level.value.upper()}] {self.title}: {self.message}"


# ============================================================================
# Alert Callback Type
# ============================================================================

AlertCallback = Callable[[Alert], Awaitable[None]]


# ============================================================================
# Alert Manager
# ============================================================================

class AlertManager:
    """
    Manages real-time alerts from pattern detection.
    
    Features:
    - Multi-channel alerts (console, file, callbacks)
    - Cooldown to prevent spam
    - Alert history and statistics
    - Escalation handling
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize the alert manager.
        
        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()
        
        # Alert history
        self._history: deque = deque(maxlen=self.config.max_history)
        self._alert_counter = 0
        
        # Cooldown tracking
        self._last_alert_by_pattern: Dict[str, datetime] = {}
        
        # Callbacks
        self._callbacks: List[AlertCallback] = []
        
        # File logging
        self._log_file = None
        if self.config.enable_file_logging and self.config.log_file_path:
            self._init_log_file()
        
        # Statistics
        self._stats = {
            'total_alerts': 0,
            'alerts_by_level': {level.value: 0 for level in AlertLevel},
            'alerts_by_type': {atype.value: 0 for atype in AlertType},
            'alerts_by_pattern': {}
        }
        
        # Escalation tracking
        self._recent_alerts: deque = deque(maxlen=100)
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        logger.info("Alert manager initialized")
    
    def _init_log_file(self):
        """Initialize log file."""
        try:
            log_path = Path(self.config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_path, 'a', encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to open alert log file: {e}")
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"ALT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._alert_counter:05d}"
    
    def register_callback(self, callback: AlertCallback):
        """Register an alert callback."""
        self._callbacks.append(callback)
        logger.debug(f"Registered alert callback: {callback}")
    
    def unregister_callback(self, callback: AlertCallback):
        """Unregister an alert callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _check_cooldown(self, pattern_id: Optional[str]) -> bool:
        """Check if alert should be suppressed due to cooldown."""
        if not pattern_id:
            return False
        
        with self._lock:
            if pattern_id in self._last_alert_by_pattern:
                last_time = self._last_alert_by_pattern[pattern_id]
                elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
                if elapsed < self.config.cooldown_seconds:
                    return True  # Suppress
            
            # Update last alert time
            self._last_alert_by_pattern[pattern_id] = datetime.now(timezone.utc)
        
        return False
    
    def _check_escalation(self) -> Optional[Alert]:
        """Check if escalation alert is needed."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.config.escalation_window_seconds)
        
        # Count recent high-severity alerts
        high_severity_count = sum(
            1 for alert in self._recent_alerts
            if alert.timestamp >= window_start and alert.level >= AlertLevel.HIGH
        )
        
        if high_severity_count >= self.config.escalation_threshold:
            return Alert(
                id=self._generate_alert_id(),
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.ESCALATION,
                title="Escalation Alert",
                message=f"Multiple high-severity alerts detected ({high_severity_count} in {self.config.escalation_window_seconds}s)",
                metadata={'triggering_alert_count': high_severity_count}
            )
        
        return None
    
    async def raise_alert(
        self,
        level: AlertLevel,
        alert_type: AlertType,
        title: str,
        message: str,
        pattern_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        transcription_snippet: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Alert]:
        """
        Raise an alert.
        
        Args:
            level: Alert severity level
            alert_type: Type of alert
            title: Alert title
            message: Alert message
            pattern_id: Related pattern ID
            speaker_id: Related speaker ID
            transcription_snippet: Relevant transcription
            confidence: Confidence score
            metadata: Additional metadata
            
        Returns:
            The raised Alert, or None if suppressed
        """
        # Check minimum level
        if level < self.config.min_alert_level:
            return None
        
        # Check cooldown
        if self._check_cooldown(pattern_id):
            logger.debug(f"Alert suppressed due to cooldown: {pattern_id}")
            return None
        
        # Create alert
        alert = Alert(
            id=self._generate_alert_id(),
            level=level,
            alert_type=alert_type,
            title=title,
            message=message,
            pattern_id=pattern_id,
            speaker_id=speaker_id,
            transcription_snippet=transcription_snippet,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Process alert
        await self._process_alert(alert)
        
        # Check for escalation
        escalation = self._check_escalation()
        if escalation:
            await self._process_alert(escalation)
        
        return alert
    
    async def raise_from_pattern(self, pattern: Dict[str, Any]) -> Optional[Alert]:
        """
        Raise alert from a detected pattern.
        
        Args:
            pattern: Pattern dict from PatternPipeline
            
        Returns:
            The raised Alert, or None if suppressed
        """
        # Map pattern severity to alert level
        severity_map = {
            'info': AlertLevel.INFO,
            'low': AlertLevel.LOW,
            'medium': AlertLevel.MEDIUM,
            'high': AlertLevel.HIGH,
            'critical': AlertLevel.CRITICAL
        }
        
        level = severity_map.get(pattern.get('severity', 'medium'), AlertLevel.MEDIUM)
        
        return await self.raise_alert(
            level=level,
            alert_type=AlertType.PATTERN_DETECTED,
            title=f"Pattern: {pattern.get('pattern_id', 'Unknown')}",
            message=pattern.get('description', 'Pattern detected'),
            pattern_id=pattern.get('pattern_id'),
            speaker_id=pattern.get('speaker_id'),
            transcription_snippet=pattern.get('transcription_snippet'),
            confidence=pattern.get('confidence', 1.0),
            metadata={
                'category': pattern.get('category'),
                'matched_keywords': pattern.get('matched_keywords', []),
                'matched_indicators': pattern.get('matched_indicators', [])
            }
        )
    
    async def _process_alert(self, alert: Alert):
        """Process an alert through all channels."""
        # Add to history
        with self._lock:
            self._history.append(alert)
            self._recent_alerts.append(alert)
            
            # Update stats
            self._stats['total_alerts'] += 1
            self._stats['alerts_by_level'][alert.level.value] += 1
            self._stats['alerts_by_type'][alert.alert_type.value] += 1
            
            if alert.pattern_id:
                if alert.pattern_id not in self._stats['alerts_by_pattern']:
                    self._stats['alerts_by_pattern'][alert.pattern_id] = 0
                self._stats['alerts_by_pattern'][alert.pattern_id] += 1
        
        # Console alert
        if self.config.enable_console_alerts:
            self._console_alert(alert)
        
        # File logging
        if self.config.enable_file_logging and self._log_file:
            self._file_alert(alert)
        
        # Callbacks
        if self.config.enable_callback_alerts:
            await self._callback_alert(alert)
    
    def _console_alert(self, alert: Alert):
        """Output alert to console."""
        level_colors = {
            AlertLevel.INFO: '\033[94m',      # Blue
            AlertLevel.LOW: '\033[96m',       # Cyan
            AlertLevel.MEDIUM: '\033[93m',    # Yellow
            AlertLevel.HIGH: '\033[91m',      # Red
            AlertLevel.CRITICAL: '\033[95m'   # Magenta
        }
        reset = '\033[0m'
        
        color = level_colors.get(alert.level, '')
        print(f"{color}[ALERT] {alert}{reset}")
    
    def _file_alert(self, alert: Alert):
        """Write alert to log file."""
        try:
            self._log_file.write(json.dumps(alert.to_dict()) + '\n')
            self._log_file.flush()
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
    
    async def _callback_alert(self, alert: Alert):
        """Send alert to registered callbacks."""
        for callback in self._callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._history:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    return True
        return False
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert."""
        with self._lock:
            for alert in self._history:
                if alert.id == alert_id:
                    alert.dismissed = True
                    return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unacknowledged, undismissed alerts."""
        with self._lock:
            return [
                a for a in self._history
                if not a.acknowledged and not a.dismissed
            ]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by level."""
        with self._lock:
            return [a for a in self._history if a.level == level]
    
    def get_alerts_by_pattern(self, pattern_id: str) -> List[Alert]:
        """Get alerts by pattern ID."""
        with self._lock:
            return [a for a in self._history if a.pattern_id == pattern_id]
    
    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """Get most recent alerts."""
        with self._lock:
            return list(self._history)[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            return {
                **self._stats,
                'active_alerts': len([a for a in self._history if not a.acknowledged and not a.dismissed]),
                'acknowledged_alerts': len([a for a in self._history if a.acknowledged]),
                'dismissed_alerts': len([a for a in self._history if a.dismissed])
            }
    
    def clear_history(self):
        """Clear alert history."""
        with self._lock:
            self._history.clear()
            self._recent_alerts.clear()
            self._stats = {
                'total_alerts': 0,
                'alerts_by_level': {level.value: 0 for level in AlertLevel},
                'alerts_by_type': {atype.value: 0 for atype in AlertType},
                'alerts_by_pattern': {}
            }
    
    def create_alert(
        self,
        pattern_type: str,
        confidence: float,
        details: Dict[str, Any],
        result: Any = None
    ) -> Alert:
        """
        Create an alert synchronously (for non-async code paths).
        
        Args:
            pattern_type: Type of pattern detected
            confidence: Confidence score (0-1)
            details: Pattern details dictionary
            result: Optional ScanResult
            
        Returns:
            Created Alert
        """
        # Map confidence to alert level
        if confidence >= 0.9:
            level = AlertLevel.CRITICAL
        elif confidence >= 0.8:
            level = AlertLevel.HIGH
        elif confidence >= 0.7:
            level = AlertLevel.MEDIUM
        elif confidence >= 0.5:
            level = AlertLevel.LOW
        else:
            level = AlertLevel.INFO
        
        # Create alert
        alert = Alert(
            id=self._generate_alert_id(),
            level=level,
            alert_type=AlertType.PATTERN_DETECTED,
            title=f"Pattern: {pattern_type}",
            message=details.get('description', f'{pattern_type} detected'),
            pattern_id=details.get('pattern_id', pattern_type),
            speaker_id=details.get('speaker_id'),
            transcription_snippet=details.get('transcription_snippet'),
            confidence=confidence,
            metadata={
                'category': details.get('category'),
                'matched_keywords': details.get('matched_keywords', []),
                'matched_indicators': details.get('matched_indicators', []),
                'full_details': details
            }
        )
        
        # Add to history (sync)
        with self._lock:
            self._history.append(alert)
            self._recent_alerts.append(alert)
            
            # Update stats
            self._stats['total_alerts'] += 1
            self._stats['alerts_by_level'][alert.level.value] += 1
            self._stats['alerts_by_type'][alert.alert_type.value] += 1
            
            if alert.pattern_id:
                if alert.pattern_id not in self._stats['alerts_by_pattern']:
                    self._stats['alerts_by_pattern'][alert.pattern_id] = 0
                self._stats['alerts_by_pattern'][alert.pattern_id] += 1
        
        # Console alert (sync)
        if self.config.enable_console_alerts:
            self._console_alert(alert)
        
        # File logging (sync)
        if self.config.enable_file_logging and self._log_file:
            self._file_alert(alert)
        
        return alert

    def close(self):
        """Close the alert manager."""
        if self._log_file:
            self._log_file.close()
        self._callbacks.clear()
        logger.info("Alert manager closed")


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_default_manager(
    log_path: Optional[str] = None,
    min_level: AlertLevel = AlertLevel.MEDIUM
) -> AlertManager:
    """
    Create an alert manager with default settings.
    
    Args:
        log_path: Path for alert log file
        min_level: Minimum alert level
        
    Returns:
        Configured AlertManager
    """
    config = AlertConfig(
        min_alert_level=min_level,
        enable_file_logging=log_path is not None,
        log_file_path=log_path
    )
    
    return AlertManager(config)
