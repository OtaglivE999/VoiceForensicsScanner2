#!/usr/bin/env python3
"""
Verify and install required modules for VoiceForensicsScanner.

Usage:
    python verify_install.py           # Check and report missing modules
    python verify_install.py --install # Check and auto-install missing modules
"""

import importlib
import subprocess
import sys

# Mapping: (import_name, pip_package_name, required)
REQUIRED_MODULES = [
    ("sounddevice", "sounddevice", True),
    ("numpy", "numpy", True),
    ("scipy", "scipy", True),
    ("librosa", "librosa", True),
    ("soundfile", "soundfile", True),
    ("webrtcvad", "webrtcvad", True),
    ("resemblyzer", "resemblyzer", True),
    ("noisereduce", "noisereduce", True),
    ("torch", "torch", True),
    ("torchaudio", "torchaudio", True),
    ("dotenv", "python-dotenv", True),
]

OPTIONAL_MODULES = [
    ("whisper", "openai-whisper", False),
    ("onnxruntime", "onnxruntime", False),
    ("flask", "flask", False),
    ("flask_cors", "flask-cors", False),
    ("transformers", "transformers", False),
]


def check_python_version():
    """Verify Python version meets minimum requirement."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"[FAIL] Python 3.10+ required, found {major}.{minor}")
        return False
    print(f"[OK]   Python {major}.{minor}")
    return True


def check_module(import_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def install_package(pip_name):
    """Install a package using pip."""
    print(f"       Installing {pip_name}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"       {pip_name} installed successfully.")
            return True
        else:
            print(f"       Failed to install {pip_name}: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"       Timeout installing {pip_name}.")
        return False
    except Exception as e:
        print(f"       Error installing {pip_name}: {e}")
        return False


def verify_modules(auto_install=False):
    """Check all required and optional modules, optionally installing missing ones."""
    missing_required = []
    missing_optional = []
    installed = []
    failed = []

    print("=" * 60)
    print("  Voice Forensics Scanner - Module Verification")
    print("=" * 60)
    print()

    # Python version check
    if not check_python_version():
        print()
        print("Please install Python 3.10 or later.")
        return False
    print()

    # Required modules
    print("--- Required Modules ---")
    for import_name, pip_name, _ in REQUIRED_MODULES:
        if check_module(import_name):
            print(f"[OK]   {pip_name}")
        else:
            print(f"[MISS] {pip_name}")
            missing_required.append((import_name, pip_name))
    print()

    # Optional modules
    print("--- Optional Modules ---")
    for import_name, pip_name, _ in OPTIONAL_MODULES:
        if check_module(import_name):
            print(f"[OK]   {pip_name}")
        else:
            print(f"[----] {pip_name} (not installed)")
            missing_optional.append((import_name, pip_name))
    print()

    # Summary
    if not missing_required and not missing_optional:
        print("All modules are installed.")
        return True

    if missing_required:
        print(f"Missing required modules: {len(missing_required)}")
        for _, pip_name in missing_required:
            print(f"  - {pip_name}")

    if missing_optional:
        print(f"Missing optional modules: {len(missing_optional)}")
        for _, pip_name in missing_optional:
            print(f"  - {pip_name}")

    print()

    # Auto-install if requested
    if auto_install and missing_required:
        print("--- Installing Missing Required Modules ---")
        for import_name, pip_name in missing_required:
            if install_package(pip_name):
                installed.append(pip_name)
            else:
                failed.append(pip_name)
        print()

        if failed:
            print(f"[FAIL] Could not install: {', '.join(failed)}")
            print("       Try installing manually: pip install " + " ".join(failed))
            return False
        elif installed:
            print(f"[OK]   Installed {len(installed)} module(s) successfully.")
            return True

    elif missing_required and not auto_install:
        print("To auto-install missing required modules, run:")
        print(f"  python {sys.argv[0]} --install")
        print()
        print("Or install all at once with:")
        print("  pip install -r requirements.txt")
        return False

    return not missing_required


if __name__ == "__main__":
    auto_install = "--install" in sys.argv
    success = verify_modules(auto_install=auto_install)
    sys.exit(0 if success else 1)
