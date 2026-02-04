"""
Configuration Management for Enhanced Audio Recording System
Supports environment variable overrides and .env files
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal

# Try to load python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    # Load .env file from current directory or parent
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def get_env(key: str, default: str = '') -> str:
    """Get environment variable with fallback."""
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    val = os.environ.get(key, '').lower()
    if val in ('true', '1', 'yes', 'on'):
        return True
    if val in ('false', '0', 'no', 'off'):
        return False
    return default


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


@dataclass
class AudioConfig:
    """Audio capture configuration."""
    sample_rate: int = field(default_factory=lambda: get_env_int('AUDIO_SAMPLE_RATE', 16000))
    chunk_duration: int = field(default_factory=lambda: get_env_int('AUDIO_CHUNK_DURATION', 30))
    channels: int = field(default_factory=lambda: get_env_int('AUDIO_CHANNELS', 1))
    dtype: str = field(default_factory=lambda: get_env('AUDIO_DTYPE', 'float32'))
    device_id: Optional[int] = field(default=None)
    blocksize_ms: int = field(default_factory=lambda: get_env_int('AUDIO_BLOCKSIZE_MS', 100))
    
    # Sample rate presets
    RATE_PRESETS = {
        'speech': 16000,
        'cd': 44100,
        'professional': 48000,
        'high_res': 96000
    }
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.sample_rate not in [16000, 22050, 44100, 48000, 96000]:
            print(f"Warning: Non-standard sample rate {self.sample_rate}")
        if self.chunk_duration < 1 or self.chunk_duration > 300:
            raise ValueError(f"Chunk duration must be 1-300 seconds, got {self.chunk_duration}")
        if self.channels not in [1, 2]:
            raise ValueError(f"Channels must be 1 or 2, got {self.channels}")


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    energy_threshold: float = field(default_factory=lambda: get_env_float('VAD_ENERGY_THRESHOLD', 0.01))
    zcr_threshold: float = field(default_factory=lambda: get_env_float('VAD_ZCR_THRESHOLD', 0.3))
    min_voice_percentage: float = field(default_factory=lambda: get_env_float('VAD_MIN_VOICE_PERCENT', 10.0))
    frame_length_ms: int = field(default_factory=lambda: get_env_int('VAD_FRAME_LENGTH_MS', 25))
    use_spectral: bool = field(default_factory=lambda: get_env_bool('VAD_USE_SPECTRAL', True))
    calibration_seconds: float = field(default_factory=lambda: get_env_float('VAD_CALIBRATION_SECONDS', 2.0))


@dataclass 
class WhisperConfig:
    """Whisper transcription configuration."""
    model_name: str = field(default_factory=lambda: get_env('WHISPER_MODEL', 'base'))
    language: Optional[str] = field(default_factory=lambda: get_env('WHISPER_LANGUAGE', '') or None)
    task: Literal['transcribe', 'translate'] = field(
        default_factory=lambda: get_env('WHISPER_TASK', 'transcribe')  # type: ignore
    )
    fp16: bool = field(default_factory=lambda: get_env_bool('WHISPER_FP16', True))
    
    # Available models: tiny, base, small, medium, large, large-v2, large-v3
    AVAILABLE_MODELS = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    
    def __post_init__(self):
        if self.model_name not in self.AVAILABLE_MODELS:
            print(f"Warning: Unknown Whisper model '{self.model_name}', using 'base'")
            self.model_name = 'base'


@dataclass
class SentimentConfig:
    """Sentiment analysis configuration."""
    model_name: str = field(
        default_factory=lambda: get_env(
            'SENTIMENT_MODEL', 
            'cardiffnlp/twitter-roberta-base-sentiment-latest'
        )
    )
    max_length: int = field(default_factory=lambda: get_env_int('SENTIMENT_MAX_LENGTH', 512))
    batch_size: int = field(default_factory=lambda: get_env_int('SENTIMENT_BATCH_SIZE', 1))


@dataclass
class NPUConfig:
    """AMD Ryzen AI NPU configuration."""
    enabled: bool = field(default_factory=lambda: get_env_bool('NPU_ENABLED', True))
    cache_dir: Path = field(
        default_factory=lambda: Path(get_env('NPU_CACHE_DIR', './npu_cache'))
    )
    config_file: Path = field(
        default_factory=lambda: Path(get_env('NPU_CONFIG_FILE', './vaiep_config.json'))
    )
    
    # Model paths
    whisper_encoder_onnx: Optional[Path] = field(default=None)
    whisper_decoder_onnx: Optional[Path] = field(default=None)
    sentiment_onnx: Optional[Path] = field(default=None)
    
    # Provider priority
    provider_priority: List[str] = field(default_factory=lambda: [
        'VitisAIExecutionProvider',  # AMD NPU
        'DmlExecutionProvider',       # DirectML GPU
        'CUDAExecutionProvider',      # NVIDIA GPU
        'CPUExecutionProvider'        # CPU fallback
    ])
    
    # Performance settings
    optimize_level: int = field(default_factory=lambda: get_env_int('NPU_OPTIMIZE_LEVEL', 1))
    
    def __post_init__(self):
        """Create cache directory and validate config."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model paths from environment
        encoder_path = get_env('NPU_WHISPER_ENCODER', '')
        decoder_path = get_env('NPU_WHISPER_DECODER', '')
        sentiment_path = get_env('NPU_SENTIMENT_MODEL', '')
        
        if encoder_path:
            self.whisper_encoder_onnx = Path(encoder_path)
        if decoder_path:
            self.whisper_decoder_onnx = Path(decoder_path)
        if sentiment_path:
            self.sentiment_onnx = Path(sentiment_path)


@dataclass
class OutputConfig:
    """Output file configuration."""
    recordings_dir: Path = field(
        default_factory=lambda: Path(get_env('OUTPUT_RECORDINGS_DIR', './enhanced_recordings'))
    )
    logs_dir: Path = field(
        default_factory=lambda: Path(get_env('OUTPUT_LOGS_DIR', './enhanced_logs'))
    )
    save_audio: bool = field(default_factory=lambda: get_env_bool('OUTPUT_SAVE_AUDIO', True))
    save_enhanced: bool = field(default_factory=lambda: get_env_bool('OUTPUT_SAVE_ENHANCED', True))
    log_format: Literal['jsonl', 'json'] = field(
        default_factory=lambda: get_env('OUTPUT_LOG_FORMAT', 'jsonl')  # type: ignore
    )
    
    def __post_init__(self):
        """Create output directories."""
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: get_env('LOG_LEVEL', 'INFO'))
    file_enabled: bool = field(default_factory=lambda: get_env_bool('LOG_FILE_ENABLED', True))
    file_path: Path = field(
        default_factory=lambda: Path(get_env('LOG_FILE_PATH', './recording.log'))
    )
    console_enabled: bool = field(default_factory=lambda: get_env_bool('LOG_CONSOLE_ENABLED', True))
    max_bytes: int = field(default_factory=lambda: get_env_int('LOG_MAX_BYTES', 10_000_000))
    backup_count: int = field(default_factory=lambda: get_env_int('LOG_BACKUP_COUNT', 5))
    
    VALID_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    def __post_init__(self):
        self.level = self.level.upper()
        if self.level not in self.VALID_LEVELS:
            print(f"Warning: Invalid log level '{self.level}', using 'INFO'")
            self.level = 'INFO'


@dataclass
class Config:
    """Main configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    npu: NPUConfig = field(default_factory=NPUConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Feature flags
    enable_ai: bool = field(default_factory=lambda: get_env_bool('ENABLE_AI', True))
    enable_transcription: bool = field(default_factory=lambda: get_env_bool('ENABLE_TRANSCRIPTION', True))
    enable_sentiment: bool = field(default_factory=lambda: get_env_bool('ENABLE_SENTIMENT', True))
    enable_noise_reduction: bool = field(default_factory=lambda: get_env_bool('ENABLE_NOISE_REDUCTION', True))
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls()
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'audio' in data:
            for key, value in data['audio'].items():
                if hasattr(config.audio, key):
                    setattr(config.audio, key, value)
        
        if 'vad' in data:
            for key, value in data['vad'].items():
                if hasattr(config.vad, key):
                    setattr(config.vad, key, value)
        
        if 'whisper' in data:
            for key, value in data['whisper'].items():
                if hasattr(config.whisper, key):
                    setattr(config.whisper, key, value)
        
        if 'sentiment' in data:
            for key, value in data['sentiment'].items():
                if hasattr(config.sentiment, key):
                    setattr(config.sentiment, key, value)
        
        if 'npu' in data:
            for key, value in data['npu'].items():
                if hasattr(config.npu, key):
                    setattr(config.npu, key, value)
        
        if 'output' in data:
            for key, value in data['output'].items():
                if hasattr(config.output, key):
                    setattr(config.output, key, value)
        
        return config
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'audio': {
                'sample_rate': self.audio.sample_rate,
                'chunk_duration': self.audio.chunk_duration,
                'channels': self.audio.channels,
                'dtype': self.audio.dtype,
                'blocksize_ms': self.audio.blocksize_ms
            },
            'vad': {
                'energy_threshold': self.vad.energy_threshold,
                'zcr_threshold': self.vad.zcr_threshold,
                'min_voice_percentage': self.vad.min_voice_percentage,
                'frame_length_ms': self.vad.frame_length_ms,
                'use_spectral': self.vad.use_spectral
            },
            'whisper': {
                'model_name': self.whisper.model_name,
                'language': self.whisper.language,
                'task': self.whisper.task,
                'fp16': self.whisper.fp16
            },
            'sentiment': {
                'model_name': self.sentiment.model_name,
                'max_length': self.sentiment.max_length
            },
            'npu': {
                'enabled': self.npu.enabled,
                'cache_dir': str(self.npu.cache_dir),
                'provider_priority': self.npu.provider_priority
            },
            'output': {
                'recordings_dir': str(self.output.recordings_dir),
                'logs_dir': str(self.output.logs_dir),
                'save_audio': self.output.save_audio,
                'log_format': self.output.log_format
            },
            'logging': {
                'level': self.logging.level,
                'file_enabled': self.logging.file_enabled,
                'file_path': str(self.logging.file_path)
            },
            'features': {
                'enable_ai': self.enable_ai,
                'enable_transcription': self.enable_transcription,
                'enable_sentiment': self.enable_sentiment,
                'enable_noise_reduction': self.enable_noise_reduction
            }
        }


# Global default config instance
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get the default configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = Config.from_env()
    return _default_config


def set_config(config: Config) -> None:
    """Set the default configuration instance."""
    global _default_config
    _default_config = config


# Example usage and validation
if __name__ == '__main__':
    config = get_config()
    print("=" * 60)
    print("CURRENT CONFIGURATION")
    print("=" * 60)
    
    import json
    print(json.dumps(config.to_dict(), indent=2))
    
    print("\n" + "=" * 60)
    print("DOTENV SUPPORT:", "Available" if DOTENV_AVAILABLE else "Not installed (pip install python-dotenv)")
    print("=" * 60)
