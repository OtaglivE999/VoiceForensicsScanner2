"""
Voice Forensics Scanner - Example: Basic Usage
Demonstrates how to use the scanner programmatically.
"""

import sys
sys.path.insert(0, '../src')

from soft_voice_scanner import SoftVoiceScanner


def main():
    """Basic scanner usage example."""
    
    # Create scanner with custom settings
    scanner = SoftVoiceScanner(
        device_id=12,           # Audio device index
        ultra_sensitive=True,   # Maximum sensitivity
        min_snr=2.0,           # Minimum SNR threshold
        auto_enroll=True,       # Auto-enroll new speakers
        agc_enabled=True,       # Automatic gain control
        record=False            # Don't save recordings
    )
    
    print("Starting voice scanner...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        # Run the scanner (blocking)
        scanner.run()
    except KeyboardInterrupt:
        print("\nStopping scanner...")
        scanner.stop()
    
    print("Scanner stopped.")


if __name__ == "__main__":
    main()
