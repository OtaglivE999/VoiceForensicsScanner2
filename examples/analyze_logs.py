"""
Voice Forensics Scanner - Example: Analyze Logs
Demonstrates how to analyze JSONL detection logs.
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def analyze_logs(log_dir: str = "../logs"):
    """Analyze all JSONL log files in directory."""
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return
    
    # Find all JSONL files
    log_files = list(log_path.glob("*.jsonl"))
    if not log_files:
        print("No log files found.")
        return
    
    print(f"Found {len(log_files)} log files")
    print("=" * 60)
    
    # Aggregate statistics
    all_entries = []
    speaker_counts = defaultdict(int)
    field_counts = defaultdict(int)
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        all_entries.append(entry)
                        
                        # Count by speaker
                        speaker_id = entry.get('speaker_id', 'unknown')
                        speaker_counts[speaker_id] += 1
                        
                        # Count by field type
                        field_type = entry.get('field_type', 'unknown')
                        field_counts[field_type] += 1
                    except json.JSONDecodeError:
                        continue
    
    if not all_entries:
        print("No valid entries found.")
        return
    
    # Sort by timestamp
    all_entries.sort(key=lambda x: x.get('timestamp', ''))
    
    # Print summary
    print(f"\n📊 DETECTION SUMMARY")
    print(f"   Total Detections: {len(all_entries)}")
    
    # Time range
    first_ts = all_entries[0].get('timestamp', 'N/A')
    last_ts = all_entries[-1].get('timestamp', 'N/A')
    print(f"   Time Range: {first_ts} to {last_ts}")
    
    # Speaker breakdown
    print(f"\n👤 SPEAKERS DETECTED ({len(speaker_counts)} unique)")
    for speaker, count in sorted(speaker_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   {speaker}: {count} detections")
    
    # Field type breakdown
    print(f"\n📡 FIELD DISTRIBUTION")
    for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_entries) * 100
        bar = "█" * int(pct / 5)
        print(f"   {field:12}: {bar} {count} ({pct:.1f}%)")
    
    # Confidence statistics
    confidences = [e.get('confidence', 0) for e in all_entries if 'confidence' in e]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        print(f"\n🎯 CONFIDENCE")
        print(f"   Average: {avg_conf:.1%}")
        print(f"   High (>80%): {sum(1 for c in confidences if c > 0.8)}")
        print(f"   Medium (50-80%): {sum(1 for c in confidences if 0.5 <= c <= 0.8)}")
        print(f"   Low (<50%): {sum(1 for c in confidences if c < 0.5)}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze voice scanner logs")
    parser.add_argument("--log-dir", default="../logs", help="Log directory path")
    args = parser.parse_args()
    
    analyze_logs(args.log_dir)
