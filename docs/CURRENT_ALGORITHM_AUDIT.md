# Current Algorithm Audit — VoiceForensicsScanner v2

**Baseline commit:** `4f10b60b6876323ca74f85fbd294a87d4f1dbe99`
**Date:** 2026-07-13
**Application:** `src/iphone_web_audio_app.html` (2959 lines, single-file)

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  iPhone / Browser                                                │
│                                                                  │
│  getUserMedia(mono) ──► AudioContext(48 kHz typical)              │
│       │                                                          │
│       ├──► MediaRecorder(1s chunks) ──► evidenceClip (session)   │
│       │                                                          │
│       ├──► AudioWorklet (scan-processor) ──► AnalyserNode        │
│       │    (or direct connection if worklet fails)               │
│       │                                                          │
│       ├──► PCM Ring Buffer (10s, ctx.sampleRate)                 │
│       │    └── captureSpeakerClip() ──► speakerAudioClips[]      │
│       │                                                          │
│       └──► scanLoop() (requestAnimationFrame, 150ms throttle)    │
│            │                                                     │
│            ├── analyser.getFloatTimeDomainData(2048 samples)     │
│            ├── analyser.getByteFrequencyData(2048 bins)          │
│            │                                                     │
│            ├── estimateSNR(timeData)         ← VAD decision      │
│            │   └── hasVoice = rms>0.008 && snr>6dB              │
│            │                                                     │
│            ├── [if voice] extractFeatures(freqData, timeData)    │
│            │   ├── spectral centroid, spread, flatness, rolloff  │
│            │   ├── 13 "MFCC-like" mel-band averages             │
│            │   ├── autocorrelation pitch (60–500 Hz)             │
│            │   ├── jitter / shimmer (glottal-cycle tracking)     │
│            │   ├── LPC formants (F1, F2, F3, nasal, nasalance)  │
│            │   └── zero-crossing rate                            │
│            │                                                     │
│            ├── computeEmbedding256(features)                     │
│            │   └── 25 features → trigonometric projection → 256D │
│            │                                                     │
│            ├── match(features)               ← feature-only      │
│            │   └── compareFeatures() weighted score ≥ 0.65      │
│            │                                                     │
│            ├── [if no match && autoEnroll] enroll(features)      │
│            │   └── immediate single-observation enrollment       │
│            │                                                     │
│            ├── EMA update of avgFeatures and embedding_256d      │
│            │                                                     │
│            └── addDetectionEntry() + saveDetectionLog()          │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                 │
│  │  Storage (localStorage)                     │                 │
│  │  ├── vfs_detection_log (≤500 entries)       │                 │
│  │  ├── vfs_speaker_library (speakers + nextId)│                 │
│  │  ├── vfs_sessions (≤100 timestamps)         │                 │
│  │  ├── vfs_device_id                          │                 │
│  │  └── vfs_case_info                          │                 │
│  └─────────────────────────────────────────────┘                 │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                 │
│  │  Evidence ZIP export (handleDownloadAll)     │                 │
│  │  ├── detection_log.csv       (SHA-256)      │                 │
│  │  ├── speaker_library.json    (SHA-256)      │                 │
│  │  ├── session_audio.<ext>     (SHA-256)      │                 │
│  │  ├── speakers/*.wav          (SHA-256 each) │                 │
│  │  ├── session_timeline.json   (SHA-256)      │                 │
│  │  ├── session_transcript.vtt  (SHA-256)      │                 │
│  │  ├── manifest.json           (self-hash)    │                 │
│  │  └── README.txt                             │                 │
│  └─────────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Confirmed Call Flow

### Audio Capture
1. `startScanning()` (line 2098) creates `AudioContext` (browser default rate, typically 48 kHz)
2. `getUserMedia({ audio: { channelCount: 1 } })` acquires mono microphone
3. `createMediaStreamSource(stream)` wraps as Web Audio node
4. Connects through optional AudioWorklet to `AnalyserNode(fftSize=4096)`
5. `initPcmRingBuffer(ctx.sampleRate)` allocates 10s circular buffer at native rate

### Scan Loop (every 150ms)
1. `scanLoop()` (line 2405) reads `analyser.frequencyBinCount` (= 2048) samples
2. `getFloatTimeDomainData()` → 2048 Float32 time samples
3. `getByteFrequencyData()` → 2048 Uint8 frequency bins
4. `feedPcmRing(timeData)` → stores in ring buffer
5. `estimateSNR(timeData)` → binary voice decision
6. If voice: `extractFeatures()` → `computeEmbedding256()` → `match()` → maybe `enroll()`
7. EMA update of matched speaker's avgFeatures and embedding_256d
8. `addDetectionEntry()` appends to log

### Matching
1. `match(features)` (line 1598) iterates all speakers
2. Calls `compareFeatures()` — weighted feature similarity (NOT embedding cosine)
3. Returns best match if score ≥ 0.65 (minConfidence)
4. `detectionEmbedding` is computed but **not used in matching**

### Enrollment
1. If `match()` returns null and `autoEnroll` is on → `enroll(features)` (line 2460)
2. Immediate: one observation creates a permanent profile
3. Assigns `SPK_NNN` ID, stores features + embedding + metadata

### Profile Update
1. On match: `updateAverage()` EMA blends all features unconditionally
2. Embedding: EMA blend with alpha = 1/min(detections, 20), then L2-renormalize
3. No quality gate — every matched observation updates the profile

---

## Algorithmic Defects

### D1: Embedding computed but not used for live matching
- **Location:** `scanLoop()` line 2457–2459
- **Issue:** `detectionEmbedding = fingerprinter.computeEmbedding256(detectionFeatures)` is computed every cycle, but `fingerprinter.match(detectionFeatures)` uses only `compareFeatures()` (weighted feature distances). The 256-D embedding is stored but never drives matching decisions.
- **Impact:** Embedding exists only for export metadata; matching relies entirely on handcrafted feature distances.

### D2: Immediate single-observation enrollment
- **Location:** `scanLoop()` line 2460–2462
- **Issue:** `if (!speakerResult && settings.autoEnroll) { speakerResult = fingerprinter.enroll(detectionFeatures); }`. A single 42ms observation creates a permanent speaker profile.
- **Impact:** Noise bursts, environmental transients, or brief speech fragments can spawn permanent speaker IDs.

### D3: Deterministic projection labeled as embedding model
- **Location:** `computeEmbedding256()` line 1713–1759
- **Issue:** The function projects 25 handcrafted features through deterministic trigonometric mixing (prime-indexed frequencies, golden-ratio phases). This is not a trained speaker-recognition model.
- **Impact:** `PROCESSING_META.embedding_model: 'vfs-spectral-projection-v1'` is technically accurate naming, but exports could be mistaken for trained neural embeddings.

### D4: "MFCC-like" features are not MFCCs
- **Location:** `extractFeatures()` lines 1376–1385
- **Issue:** Computes energy in 13 mel-scale bands from `freqData` (Uint8 byte-scaled frequency bins). This is band-averaged energy, not DCT-based mel-frequency cepstral coefficients. No windowing, no proper FFT, no log-compression, no DCT.
- **Impact:** Feature labeled `mfcc` in code and exports does not match the standard acoustic feature.

### D5: Processing window is ~42ms, not 2 seconds
- **Location:** `scanLoop()` lines 2413–2417; `PROCESSING_META.window_sec: 2.0` line 1316
- **Issue:** `analyser.frequencyBinCount` = `fftSize / 2` = 2048 samples. At 48 kHz, 2048 samples = 42.7ms. The declared `window_sec: 2.0` in metadata is 47× too large.
- **Impact:** Evidence metadata claims 2-second processing windows; actual analysis operates on ~43ms snapshots.

### D6: Sample rate mismatch — metadata says 16 kHz, runtime is 48 kHz
- **Location:** `PROCESSING_META.sample_rate: 16000` (line 1313) vs actual `ctx.sampleRate` (typically 48000)
- **Issue:** No resampling to 16 kHz occurs anywhere. All features are computed on native-rate data (typically 48 kHz). WAV exports are written at `ctx.sampleRate`. But `PROCESSING_META` declares 16000.
- **Impact:** Exported evidence claims processing at 16 kHz when actual processing and WAV files are at 48 kHz.

### D7: SNR zones labeled as physical distances
- **Location:** `getFieldZone()` lines 1827–1834
- **Issue:** Zones like `NEAR: 0-1m`, `FAR: 3-6m`, `ULTRA-FAR: 6m+` present SNR ranges as measured physical distances. No distance measurement occurs.
- **Impact:** Potentially misleading forensic claims about source proximity.

### D8: Unconditional profile update on every match
- **Location:** `match()` line 1610, `scanLoop()` lines 2466–2483
- **Issue:** Every matched observation EMA-updates both `avgFeatures` and `embedding_256d`, regardless of signal quality, clipping, SNR, or confidence margin.
- **Impact:** Low-quality observations and noise can drift profiles over time.

### D9: No model compatibility boundary
- **Location:** `matchAgainstReference()` lines 789–816
- **Issue:** Cross-model comparison check exists (`MATCH_THRESHOLDS.cross_model_comparison: false`) but reference matching still proceeds via `compareFeatures()` even when embedding models differ. The feature comparison is not gated by model version.
- **Impact:** Feature-level comparison silently operates across incompatible processing versions.

### D10: localStorage size limits
- **Location:** `saveDetectionLog()` lines 573–575
- **Issue:** 500-entry detection log with 256-element embedding arrays per entry can exceed localStorage quotas (5–10MB typical). Fallback trims to 250 entries on quota error, losing data.
- **Impact:** Long sessions may lose detection history.

### D11: `voiceConf` does not represent a calibrated probability
- **Location:** `estimateSNR()` lines 1819–1821
- **Issue:** `voiceConf = (energyConf + snrConf) / 2` where `energyConf = min(1, rms * 15)` and `snrConf = min(1, snr / 35)`. This is a heuristic score, not a calibrated probability.
- **Impact:** UI displays this as a percentage without documenting its definition.

### D12: analyser smoothingTimeConstant affects frequency data
- **Location:** `analyser.smoothingTimeConstant = 0.7` (line 2139)
- **Issue:** This exponentially smooths successive frequency frames, making each `getByteFrequencyData()` read a blend of current and past spectra. Features extracted from smoothed frequency data inherit temporal blurring.
- **Impact:** Spectral features do not represent the current frame precisely; they leak information from previous frames.

---

## Storage Keys

| Key | Format | Max Size | Usage |
|-----|--------|----------|-------|
| `vfs_detection_log` | JSON array | ≤500 entries | Detection events |
| `vfs_speaker_library` | JSON `{speakers, nextId}` | Unbounded | Speaker profiles |
| `vfs_sessions` | JSON array | ≤100 timestamps | Session history |
| `vfs_device_id` | Plain string | ~30 chars | Device identifier |
| `vfs_case_info` | JSON `{caseId, operator, notes}` | Small | Case metadata |

---

## Export Format Versions

| Format Identifier | Version | Location |
|-------------------|---------|----------|
| `VoiceForensics-SpeakerLibrary-v2` | v2 | Speaker JSON export |
| `VoiceForensics-EvidenceBundle-v2` | v2 | Manifest JSON |
| `VoiceForensics-SessionTimeline-v1` | v1 | Timeline JSON |
| `VoiceForensics-ReferenceEnrollment-v1` | v1 | Import format |

---

## Processing Metadata (as declared)

```
vad_model:               energy-threshold-v1
diarization_model:       vfs-spectral-compare-v1
embedding_model:         vfs-spectral-projection-v1
embedding_model_version: 1.0.0
embedding_dim:           256
sample_rate:             16000   ← INCORRECT (actual: browser native, typically 48000)
window_sec:              2.0     ← INCORRECT (actual: ~0.043s at 48 kHz)
hop_sec:                 0.15    ← scan loop interval, not hop
```

---

## Match Thresholds

```
embedding_cosine:
  same_speaker_strong:    0.85
  same_speaker_candidate: 0.75
  inconclusive_low:       0.65
  no_match_below:         0.65

feature_similarity:
  same_speaker_strong:    0.85
  same_speaker_candidate: 0.70
  inconclusive_low:       0.55
  no_match_below:         0.55

cross_model_comparison: false
```

Note: Embedding cosine thresholds are declared but never used in live matching (see D1).

---

## Dependencies

- No external JS libraries (fully self-contained HTML)
- Web Audio API (AudioContext, AnalyserNode, AudioWorklet)
- MediaRecorder API
- Web Crypto API (SHA-256)
- Web Speech API (optional, for transcription)
- Geolocation API (optional, for coordinates)
- Playwright + Chromium (testing only)

---

## Compatibility Boundaries for v3

Any v3 implementation must:
1. Import v1/v2 speaker libraries as read-only legacy records
2. Mark legacy profiles as `incompatible_with_v3_embedding`
3. Never cosine-compare v2 spectral-projection embeddings with v3 trained embeddings
4. Never manufacture v3 embeddings from old feature JSON
5. Preserve existing export file names and ZIP structure for backward-compatible parsing
6. Keep `VoiceForensics-SpeakerLibrary-v2` and `VoiceForensics-EvidenceBundle-v2` parseable
7. Use a new format version (`v3`) for new exports
8. Report actual browser sample rate separately from model sample rate
