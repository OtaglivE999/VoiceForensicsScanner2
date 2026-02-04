"""
Live Scan Module
Real-time audio scanning with dialogue pattern recognition.

This module provides:
- LiveScanEngine: Main orchestrator for live scanning
- AudioStreamManager: Real-time audio capture
- PatternPipeline: Pattern detection processing
- AlertManager: Real-time alert system

Usage:
    from live_scan import LiveScanEngine, ScanConfig
    
    config = ScanConfig()
    engine = LiveScanEngine(config)
    engine.start()
"""

from .engine import LiveScanEngine, ScanConfig
from .audio_stream import AudioStreamManager, StreamConfig
from .pattern_pipeline import PatternPipeline, PipelineConfig
from .alert_manager import AlertManager, AlertConfig, Alert, AlertLevel

__all__ = [
    'LiveScanEngine',
    'ScanConfig',
    'AudioStreamManager',
    'StreamConfig',
    'PatternPipeline',
    'PipelineConfig',
    'AlertManager',
    'AlertConfig',
    'Alert',
    'AlertLevel'
]

__version__ = '1.0.0'
