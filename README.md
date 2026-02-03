# Voice Forensics Scanner

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time voice detection and speaker identification system with far-field audio support. Designed for forensic audio analysis, surveillance monitoring, and speaker recognition applications.

## Features

- 🎤 **Real-time Voice Detection** - Ultra-sensitive voice activity detection (VAD)
- 👤 **Speaker Identification** - Automatic speaker recognition using voice embeddings
- 📡 **Far-Field Support** - Detects voices from near (0-1m) to extreme distances (6m+)
- 🔊 **SNR Classification** - Automatic signal-to-noise ratio analysis
- 📊 **Live Dashboard** - Web-based monitoring at `http://localhost:5050`
- 🗄️ **Speaker Database** - Persistent speaker profiles with voice fingerprints
- 📝 **JSONL Logging** - Structured detection logs with timestamps

## Field Detection Ranges

| Field Type | SNR Range | Distance |
|------------|-----------|----------|
| NEAR | ≥35dB | 0-1m |
| MEDIUM | 25-35dB | 1-3m |
| FAR | 15-25dB | 3-6m |
| ULTRA-FAR | 6-15dB | 6m+ |
| EXTREME | 2-6dB | Very far |
| WHISPER | <2dB | Faint/whisper |

## Installation

### Prerequisites
- Python 3.10 or higher
- Windows 10/11 (primary support) or Linux
- Audio input device (microphone, USB interface, or virtual audio cable)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/VoiceForensicsScanner.git
cd VoiceForensicsScanner

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
copy .env.example .env
# Edit .env with your settings

# List available audio devices
python src/soft_voice_scanner.py --list-devices

# Run the scanner
python src/soft_voice_scanner.py --device <DEVICE_ID> --ultra-sensitive
```

### Windows Quick Launch

```batch
# Use the provided batch file
START_SCANNER.bat
```

## Usage

### Basic Commands

```bash
# List audio devices
python src/soft_voice_scanner.py --list-devices

# Run with specific device (ultra-sensitive mode)
python src/soft_voice_scanner.py --device 12 --ultra-sensitive

# Run with minimum SNR threshold
python src/soft_voice_scanner.py --device 12 --min-snr 3.0

# Disable auto-enrollment of new speakers
python src/soft_voice_scanner.py --device 12 --no-enroll

# Enable audio recording
python src/soft_voice_scanner.py --device 12 --record
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--device`, `-d` | Audio device index (use --list-devices to find) |
| `--ultra-sensitive` | Enable maximum sensitivity mode |
| `--min-snr` | Minimum SNR threshold in dB (default: 2.0) |
| `--no-enroll` | Disable automatic speaker enrollment |
| `--no-agc` | Disable automatic gain control |
| `--record` | Save audio recordings |
| `--neural` | Enable neural enhancement (requires GPU) |
| `--output-dir` | Custom output directory for logs |
| `--list-devices` | List available audio input devices |

### Interactive Controls

While the scanner is running:
- `Enter` - Show current status
- `s` - Save current state
- `q` - Quit scanner

## Configuration

### Environment Variables (.env)

```env
# Audio Settings
AUDIO_SAMPLE_RATE=48000
AUDIO_CHANNELS=1
AUDIO_DEVICE_ID=12

# Voice Detection
VAD_ENERGY_THRESHOLD=0.01
VAD_SPEECH_PAD_MS=300

# Speaker Recognition
SPEAKER_SIMILARITY_THRESHOLD=0.75
AUTO_ENROLL_NEW_SPEAKERS=true

# NPU/GPU Acceleration (optional)
NPU_ENABLED=false
NPU_PROVIDER=DirectML
```

## Project Structure

```
VoiceForensicsScanner/
├── src/
│   ├── soft_voice_scanner.py   # Main scanner entry point
│   ├── speaker_database.py     # Speaker profile management
│   ├── field_processor.py      # Audio processing & SNR analysis
│   ├── config.py               # Configuration management
│   ├── live_matcher.py         # Real-time speaker matching
│   ├── prosody.py              # Prosodic feature extraction
│   ├── diarization.py          # Speaker diarization
│   └── live_scan/
│       ├── engine.py           # Core scanning engine
│       ├── audio_stream.py     # Audio stream handling
│       ├── voice_biometrics.py # Voice embedding extraction
│       ├── alert_manager.py    # Alert/notification system
│       └── pattern_pipeline.py # Detection pattern analysis
├── voice_library/              # Speaker voice fingerprints
├── logs/                       # Detection logs (JSONL)
├── recordings/                 # Audio recordings (optional)
├── examples/                   # Example scripts
├── requirements.txt
├── .env.example
├── START_SCANNER.bat           # Windows launcher
└── README.md
```

## Output Format

Detection logs are saved as JSONL files with the following structure:

```json
{
  "timestamp": "2026-02-03T01:24:07.123456",
  "detection_number": 1234,
  "detection_type": "soft_voice_far_field",
  "speaker_id": "VOICE_0001",
  "speaker_name": "Unknown Speaker 1",
  "confidence": 0.85,
  "snr_db": 12.5,
  "field_type": "ULTRA-FAR",
  "audio_features": {
    "mfcc": [...],
    "voice_confidence": 0.85,
    "rms_energy": 0.00012,
    "spectral_centroid": 3602.8,
    "voice_band_energy_ratio": 0.92
  },
  "sample_rate": 48000
}
```

## Hardware Recommendations

- **Microphone**: ZOOM H6 or similar professional audio interface
- **Virtual Audio**: VoiceMeeter for audio routing
- **CPU**: AMD Ryzen with AI NPU for accelerated processing
- **RAM**: 8GB minimum, 16GB recommended

## Troubleshooting

### No audio detected
1. Check device index with `--list-devices`
2. Verify microphone permissions in Windows Settings
3. Test audio input in another application

### Low detection confidence
1. Enable `--ultra-sensitive` mode
2. Lower `--min-snr` threshold
3. Check microphone placement and gain levels

### High CPU usage
1. Reduce sample rate in `.env`
2. Disable `--neural` mode if enabled
3. Enable NPU acceleration if available

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) for speaker embeddings
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) for voice activity detection
- [Librosa](https://librosa.org/) for audio analysis

---

**⚠️ Legal Notice**: This software is intended for lawful purposes only. Users are responsible for compliance with all applicable laws regarding audio recording and surveillance in their jurisdiction.
