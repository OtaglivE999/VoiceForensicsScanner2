"""
Speaker Database Module
Persistent storage for speaker voice embeddings and profiles
Enables cross-recording speaker recognition and role tracking
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


@dataclass
class SpeakerProfile:
    """Profile for a known speaker."""
    speaker_id: str
    display_name: str
    role: str  # PERPETRATOR, VICTIM, WITNESS, UNKNOWN
    role_confidence: float
    first_seen: str  # ISO datetime
    last_seen: str
    recording_count: int
    total_speaking_time: float
    notes: str = ""
    embedding_file: str = ""
    
    # Pattern history
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    emotion_history: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'speaker_id': self.speaker_id,
            'display_name': self.display_name,
            'role': self.role,
            'role_confidence': self.role_confidence,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'recording_count': self.recording_count,
            'total_speaking_time': self.total_speaking_time,
            'notes': self.notes,
            'embedding_file': self.embedding_file,
            'pattern_counts': self.pattern_counts,
            'emotion_history': self.emotion_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpeakerProfile':
        return cls(
            speaker_id=data.get('speaker_id', ''),
            display_name=data.get('display_name', ''),
            role=data.get('role', 'UNKNOWN'),
            role_confidence=data.get('role_confidence', 0.0),
            first_seen=data.get('first_seen', ''),
            last_seen=data.get('last_seen', ''),
            recording_count=data.get('recording_count', 0),
            total_speaking_time=data.get('total_speaking_time', 0.0),
            notes=data.get('notes', ''),
            embedding_file=data.get('embedding_file', ''),
            pattern_counts=data.get('pattern_counts', {}),
            emotion_history=data.get('emotion_history', {})
        )


class SpeakerDatabase:
    """
    Persistent database for speaker profiles and voice embeddings.
    Enables cross-recording speaker identification.
    """
    
    def __init__(self, 
                 database_path: str = "./speakers/speaker_database.json",
                 embeddings_dir: str = "./speakers/embeddings",
                 match_threshold: float = 0.75):
        """
        Initialize the speaker database.
        
        Args:
            database_path: Path to JSON database file
            embeddings_dir: Directory for embedding numpy files
            match_threshold: Cosine similarity threshold for speaker matching
        """
        self.database_path = Path(database_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.match_threshold = match_threshold
        
        # Create directories
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize database
        self.profiles: Dict[str, SpeakerProfile] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self._load_database()
    
    def _load_database(self) -> None:
        """Load speaker profiles from disk."""
        if self.database_path.exists():
            try:
                with open(self.database_path, 'r') as f:
                    data = json.load(f)
                
                for speaker_id, profile_data in data.get('profiles', {}).items():
                    self.profiles[speaker_id] = SpeakerProfile.from_dict(profile_data)
                    
                    # Load embedding if exists
                    embedding_file = self.embeddings_dir / f"{speaker_id}.npy"
                    if embedding_file.exists():
                        self.embeddings[speaker_id] = np.load(embedding_file)
                
                print(f"   Loaded {len(self.profiles)} speaker profiles from database")
                
            except Exception as e:
                print(f"   ⚠️ Error loading speaker database: {e}")
                self.profiles = {}
                self.embeddings = {}
    
    def _save_database(self) -> None:
        """Save speaker profiles to disk."""
        try:
            data = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'profiles': {
                    speaker_id: profile.to_dict()
                    for speaker_id, profile in self.profiles.items()
                }
            }
            
            with open(self.database_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"   ⚠️ Error saving speaker database: {e}")
    
    def _save_embedding(self, speaker_id: str, embedding: np.ndarray) -> str:
        """Save embedding to disk."""
        embedding_file = self.embeddings_dir / f"{speaker_id}.npy"
        np.save(embedding_file, embedding)
        return str(embedding_file)
    
    def _generate_speaker_id(self) -> str:
        """Generate a unique speaker ID."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{timestamp}_{len(self.profiles)}"
        return f"SPK_{hashlib.md5(hash_input.encode()).hexdigest()[:8].upper()}"
    
    def find_matching_speaker(self, 
                             embedding: np.ndarray,
                             threshold: float = None) -> Optional[Tuple[str, float]]:
        """
        Find a matching speaker in the database.
        
        Args:
            embedding: Voice embedding to match
            threshold: Similarity threshold (uses default if None)
        
        Returns:
            Tuple of (speaker_id, similarity) or None if no match
        """
        if embedding is None or len(self.embeddings) == 0:
            return None
        
        threshold = threshold or self.match_threshold
        best_match = None
        best_similarity = 0.0
        
        for speaker_id, stored_embedding in self.embeddings.items():
            similarity = self._cosine_similarity(embedding, stored_embedding)
            
            if similarity > best_similarity and similarity >= threshold:
                best_match = speaker_id
                best_similarity = similarity
        
        if best_match:
            return (best_match, best_similarity)
        return None
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def add_speaker(self,
                   embedding: np.ndarray,
                   display_name: str = None,
                   role: str = "UNKNOWN",
                   role_confidence: float = 0.0,
                   notes: str = "",
                   speaking_time: float = 0.0) -> str:
        """
        Add a new speaker to the database.
        
        Args:
            embedding: Voice embedding
            display_name: Human-readable name
            role: Speaker role
            role_confidence: Confidence in role assignment
            notes: Additional notes
            speaking_time: Total speaking time in seconds
        
        Returns:
            New speaker ID
        """
        speaker_id = self._generate_speaker_id()
        now = datetime.now().isoformat()
        
        # Save embedding
        embedding_file = ""
        if embedding is not None:
            embedding_file = self._save_embedding(speaker_id, embedding)
            self.embeddings[speaker_id] = embedding
        
        # Create profile
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            display_name=display_name or f"Speaker {len(self.profiles) + 1}",
            role=role,
            role_confidence=role_confidence,
            first_seen=now,
            last_seen=now,
            recording_count=1,
            total_speaking_time=speaking_time,
            notes=notes,
            embedding_file=embedding_file
        )
        
        self.profiles[speaker_id] = profile
        self._save_database()
        
        return speaker_id
    
    def update_speaker(self,
                      speaker_id: str,
                      embedding: np.ndarray = None,
                      role: str = None,
                      role_confidence: float = None,
                      speaking_time: float = None,
                      patterns: Dict[str, int] = None,
                      emotions: Dict[str, int] = None,
                      notes: str = None) -> bool:
        """
        Update an existing speaker's profile.
        
        Args:
            speaker_id: Speaker to update
            embedding: New embedding to merge (averaged with existing)
            role: Updated role
            role_confidence: Updated role confidence
            speaking_time: Additional speaking time
            patterns: Pattern counts to add
            emotions: Emotion counts to add
            notes: Updated notes
        
        Returns:
            True if updated, False if speaker not found
        """
        if speaker_id not in self.profiles:
            return False
        
        profile = self.profiles[speaker_id]
        profile.last_seen = datetime.now().isoformat()
        profile.recording_count += 1
        
        # Update embedding (running average)
        if embedding is not None and speaker_id in self.embeddings:
            old_embedding = self.embeddings[speaker_id]
            # Weighted average favoring existing
            weight = profile.recording_count - 1
            new_embedding = (old_embedding * weight + embedding) / (weight + 1)
            self.embeddings[speaker_id] = new_embedding
            self._save_embedding(speaker_id, new_embedding)
        elif embedding is not None:
            self.embeddings[speaker_id] = embedding
            self._save_embedding(speaker_id, embedding)
        
        # Update role if higher confidence
        if role is not None and role_confidence is not None:
            if role_confidence > profile.role_confidence:
                profile.role = role
                profile.role_confidence = role_confidence
        
        # Add speaking time
        if speaking_time is not None:
            profile.total_speaking_time += speaking_time
        
        # Merge pattern counts
        if patterns:
            for pattern, count in patterns.items():
                profile.pattern_counts[pattern] = profile.pattern_counts.get(pattern, 0) + count
        
        # Merge emotion counts
        if emotions:
            for emotion, count in emotions.items():
                profile.emotion_history[emotion] = profile.emotion_history.get(emotion, 0) + count
        
        # Update notes
        if notes is not None:
            if profile.notes:
                profile.notes += f"\n{datetime.now().strftime('%Y-%m-%d')}: {notes}"
            else:
                profile.notes = notes
        
        self._save_database()
        return True
    
    def get_speaker(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Get a speaker profile by ID."""
        return self.profiles.get(speaker_id)
    
    def get_all_speakers(self) -> List[SpeakerProfile]:
        """Get all speaker profiles."""
        return list(self.profiles.values())
    
    def get_speakers_by_role(self, role: str) -> List[SpeakerProfile]:
        """Get all speakers with a specific role."""
        return [p for p in self.profiles.values() if p.role == role]
    
    def rename_speaker(self, speaker_id: str, display_name: str) -> bool:
        """Rename a speaker."""
        if speaker_id not in self.profiles:
            return False
        
        self.profiles[speaker_id].display_name = display_name
        self._save_database()
        return True
    
    def set_speaker_role(self, 
                        speaker_id: str, 
                        role: str, 
                        confidence: float = 1.0) -> bool:
        """Manually set a speaker's role."""
        if speaker_id not in self.profiles:
            return False
        
        self.profiles[speaker_id].role = role
        self.profiles[speaker_id].role_confidence = confidence
        self._save_database()
        return True
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker from the database."""
        if speaker_id not in self.profiles:
            return False
        
        # Remove embedding file
        embedding_file = self.embeddings_dir / f"{speaker_id}.npy"
        if embedding_file.exists():
            embedding_file.unlink()
        
        # Remove from memory
        del self.profiles[speaker_id]
        if speaker_id in self.embeddings:
            del self.embeddings[speaker_id]
        
        self._save_database()
        return True
    
    def merge_speakers(self, 
                      keep_id: str, 
                      merge_id: str) -> bool:
        """
        Merge two speaker profiles (when same person detected as different speakers).
        
        Args:
            keep_id: Speaker ID to keep
            merge_id: Speaker ID to merge into keep_id
        
        Returns:
            True if merged, False if either speaker not found
        """
        if keep_id not in self.profiles or merge_id not in self.profiles:
            return False
        
        keep = self.profiles[keep_id]
        merge = self.profiles[merge_id]
        
        # Merge statistics
        keep.recording_count += merge.recording_count
        keep.total_speaking_time += merge.total_speaking_time
        
        # Keep earliest first_seen
        if merge.first_seen < keep.first_seen:
            keep.first_seen = merge.first_seen
        
        # Keep latest last_seen
        if merge.last_seen > keep.last_seen:
            keep.last_seen = merge.last_seen
        
        # Merge pattern counts
        for pattern, count in merge.pattern_counts.items():
            keep.pattern_counts[pattern] = keep.pattern_counts.get(pattern, 0) + count
        
        # Merge emotion history
        for emotion, count in merge.emotion_history.items():
            keep.emotion_history[emotion] = keep.emotion_history.get(emotion, 0) + count
        
        # Average embeddings
        if keep_id in self.embeddings and merge_id in self.embeddings:
            avg_embedding = (self.embeddings[keep_id] + self.embeddings[merge_id]) / 2
            self.embeddings[keep_id] = avg_embedding
            self._save_embedding(keep_id, avg_embedding)
        
        # Merge notes
        if merge.notes:
            keep.notes += f"\n[Merged from {merge_id}]: {merge.notes}"
        
        # Delete merged speaker
        self.delete_speaker(merge_id)
        self._save_database()
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        roles = {}
        for profile in self.profiles.values():
            roles[profile.role] = roles.get(profile.role, 0) + 1
        
        return {
            'total_speakers': len(self.profiles),
            'speakers_by_role': roles,
            'total_recordings': sum(p.recording_count for p in self.profiles.values()),
            'total_speaking_time': sum(p.total_speaking_time for p in self.profiles.values())
        }
    
    def export_report(self) -> str:
        """Export a human-readable report of all speakers."""
        lines = [
            "=" * 60,
            "SPEAKER DATABASE REPORT",
            "=" * 60,
            f"Total speakers: {len(self.profiles)}",
            ""
        ]
        
        for profile in sorted(self.profiles.values(), key=lambda p: p.first_seen):
            lines.extend([
                f"📢 {profile.display_name} ({profile.speaker_id})",
                f"   Role: {profile.role} (confidence: {profile.role_confidence:.1%})",
                f"   First seen: {profile.first_seen}",
                f"   Recordings: {profile.recording_count}",
                f"   Total speaking time: {profile.total_speaking_time:.1f}s",
                ""
            ])
            
            if profile.pattern_counts:
                lines.append("   Pattern history:")
                for pattern, count in sorted(profile.pattern_counts.items(), 
                                            key=lambda x: x[1], reverse=True):
                    lines.append(f"      • {pattern}: {count}")
                lines.append("")
        
        return "\n".join(lines)


# Convenience function
def get_speaker_database() -> SpeakerDatabase:
    """Get the default speaker database instance."""
    from config_enhanced import get_enhanced_config
    config = get_enhanced_config()
    
    return SpeakerDatabase(
        database_path=str(config.speaker_database.database_path),
        embeddings_dir=str(config.speaker_database.embeddings_dir),
        match_threshold=config.diarization.speaker_match_threshold
    )


if __name__ == '__main__':
    db = get_speaker_database()
    print(db.export_report())
    print("\n📊 Statistics:")
    print(json.dumps(db.get_statistics(), indent=2))
