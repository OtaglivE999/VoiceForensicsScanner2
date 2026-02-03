# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-03

### Added
- Initial production release
- Real-time voice activity detection with WebRTC VAD
- Speaker identification using Resemblyzer voice embeddings
- Far-field audio support with SNR-based distance classification
- Six field types: NEAR, MEDIUM, FAR, ULTRA-FAR, EXTREME, WHISPER
- Ultra-sensitive detection mode for faint voices
- Automatic speaker enrollment and tracking
- JSONL structured logging with timestamps
- Web dashboard for live monitoring (port 5050)
- Speaker database with persistent voice fingerprints
- Automatic gain control (AGC)
- Windows batch launchers
- Comprehensive configuration via environment variables

### Supported Hardware
- ZOOM H6 and compatible USB audio interfaces
- VoiceMeeter virtual audio routing
- AMD Ryzen AI NPU acceleration (optional)
- NVIDIA CUDA acceleration (optional)

## [Unreleased]

### Planned
- Linux systemd service support
- Docker containerization
- REST API for remote control
- Multi-channel array processing
- Real-time transcription integration
