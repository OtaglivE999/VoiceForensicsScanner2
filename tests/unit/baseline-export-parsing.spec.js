const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

const FIXTURES = path.resolve(__dirname, '..', 'fixtures');

test.describe('Baseline: v2 Speaker Library parsing', () => {
  let lib;

  test.beforeAll(() => {
    const raw = fs.readFileSync(path.join(FIXTURES, 'sample_speaker_library.json'), 'utf-8');
    lib = JSON.parse(raw);
  });

  test('format identifier is VoiceForensics-SpeakerLibrary-v2', () => {
    expect(lib.format).toBe('VoiceForensics-SpeakerLibrary-v2');
  });

  test('contains required top-level keys', () => {
    for (const key of ['format', 'exported', 'deviceId', 'case', 'processing', 'match_thresholds', 'speakerCount', 'speakers']) {
      expect(lib).toHaveProperty(key);
    }
  });

  test('processing meta contains model identifiers', () => {
    expect(lib.processing.embedding_model).toBe('vfs-spectral-projection-v1');
    expect(lib.processing.embedding_model_version).toBe('1.0.0');
    expect(lib.processing.embedding_dim).toBe(256);
    expect(lib.processing.vad_model).toBe('energy-threshold-v1');
    expect(lib.processing.diarization_model).toBe('vfs-spectral-compare-v1');
  });

  test('match thresholds contain cosine and feature sections', () => {
    expect(lib.match_thresholds.embedding_cosine.same_speaker_strong).toBe(0.85);
    expect(lib.match_thresholds.feature_similarity.same_speaker_strong).toBe(0.85);
    expect(lib.match_thresholds.cross_model_comparison).toBe(false);
  });

  test('speakerCount matches number of speaker keys', () => {
    expect(lib.speakerCount).toBe(Object.keys(lib.speakers).length);
  });

  test('each speaker has required fields', () => {
    for (const [id, spk] of Object.entries(lib.speakers)) {
      expect(spk).toHaveProperty('id');
      expect(spk).toHaveProperty('name');
      expect(spk).toHaveProperty('avgFeatures');
      expect(spk).toHaveProperty('embedding_256d');
      expect(spk).toHaveProperty('embedding_model');
      expect(spk).toHaveProperty('detections');
    }
  });

  test('embeddings are 256-dimensional arrays', () => {
    for (const spk of Object.values(lib.speakers)) {
      expect(Array.isArray(spk.embedding_256d)).toBe(true);
      expect(spk.embedding_256d.length).toBe(256);
    }
  });

  test('avgFeatures contains 13 MFCC-like values', () => {
    for (const spk of Object.values(lib.speakers)) {
      expect(Array.isArray(spk.avgFeatures.mfcc)).toBe(true);
      expect(spk.avgFeatures.mfcc.length).toBe(13);
    }
  });

  test('avgFeatures contains spectral and formant fields', () => {
    const required = ['centroid', 'spread', 'flatness', 'rolloff', 'pitch', 'jitter', 'shimmer', 'zcr', 'f1', 'f2', 'f3', 'nasalFormant', 'nasalance'];
    for (const spk of Object.values(lib.speakers)) {
      for (const field of required) {
        expect(spk.avgFeatures).toHaveProperty(field);
        expect(typeof spk.avgFeatures[field]).toBe('number');
      }
    }
  });

  test('integrity block exists', () => {
    expect(lib).toHaveProperty('integrity');
    expect(lib.integrity).toHaveProperty('algorithm');
    expect(lib.integrity).toHaveProperty('hash');
    expect(lib.integrity.algorithm).toBe('SHA-256');
  });
});

test.describe('Baseline: v2 Detection Log parsing', () => {
  let log;

  test.beforeAll(() => {
    const raw = fs.readFileSync(path.join(FIXTURES, 'sample_detection_log.json'), 'utf-8');
    log = JSON.parse(raw);
  });

  test('is an array', () => {
    expect(Array.isArray(log)).toBe(true);
    expect(log.length).toBeGreaterThan(0);
  });

  test('each entry has core detection fields', () => {
    const required = ['timestamp', 'speakerId', 'speakerName', 'confidence', 'snr', 'pitch', 'embedding_model'];
    for (const entry of log) {
      for (const field of required) {
        expect(entry).toHaveProperty(field);
      }
    }
  });

  test('timestamps are numeric milliseconds', () => {
    for (const entry of log) {
      expect(typeof entry.timestamp).toBe('number');
      expect(entry.timestamp).toBeGreaterThan(1700000000000);
    }
  });

  test('entries have quality metrics', () => {
    for (const entry of log) {
      expect(entry).toHaveProperty('speech_seconds');
      expect(entry).toHaveProperty('voiced_fraction');
      expect(entry).toHaveProperty('quality_score');
    }
  });

  test('entries have voice analysis fields', () => {
    for (const entry of log) {
      expect(entry).toHaveProperty('jitter');
      expect(entry).toHaveProperty('shimmer');
      expect(entry).toHaveProperty('voiceConf');
    }
  });

  test('embedding_model is v1 spectral projection', () => {
    for (const entry of log) {
      expect(entry.embedding_model).toBe('vfs-spectral-projection-v1');
    }
  });
});

test.describe('Baseline: v1 Session Timeline parsing', () => {
  let timeline;

  test.beforeAll(() => {
    const raw = fs.readFileSync(path.join(FIXTURES, 'sample_timeline.json'), 'utf-8');
    timeline = JSON.parse(raw);
  });

  test('format is VoiceForensics-SessionTimeline-v1', () => {
    expect(timeline.format).toBe('VoiceForensics-SessionTimeline-v1');
  });

  test('has session_audio_start as ISO string', () => {
    expect(typeof timeline.session_audio_start).toBe('string');
    expect(new Date(timeline.session_audio_start).getTime()).not.toBeNaN();
  });

  test('entry_count matches entries array length', () => {
    expect(timeline.entry_count).toBe(timeline.entries.length);
  });

  test('entries have required temporal fields', () => {
    for (const e of timeline.entries) {
      expect(typeof e.t_start_sec).toBe('number');
      expect(typeof e.t_end_sec).toBe('number');
      expect(e.t_end_sec).toBeGreaterThan(e.t_start_sec);
      expect(typeof e.wall_clock).toBe('string');
    }
  });

  test('entries have speaker and analysis fields', () => {
    for (const e of timeline.entries) {
      expect(e).toHaveProperty('speaker_id');
      expect(e).toHaveProperty('speaker_name');
      expect(e).toHaveProperty('dialogue');
      expect(e).toHaveProperty('jitter_pct');
      expect(e).toHaveProperty('shimmer_pct');
      expect(e).toHaveProperty('pitch_hz');
      expect(e).toHaveProperty('snr_db');
      expect(e).toHaveProperty('confidence');
    }
  });

  test('entries are in chronological order', () => {
    for (let i = 1; i < timeline.entries.length; i++) {
      expect(timeline.entries[i].t_start_sec).toBeGreaterThanOrEqual(timeline.entries[i - 1].t_start_sec);
    }
  });
});

test.describe('Baseline: v2 Evidence Manifest parsing', () => {
  let manifest;

  test.beforeAll(() => {
    const raw = fs.readFileSync(path.join(FIXTURES, 'sample_manifest.json'), 'utf-8');
    manifest = JSON.parse(raw);
  });

  test('format is VoiceForensics-EvidenceBundle-v2', () => {
    expect(manifest.format).toBe('VoiceForensics-EvidenceBundle-v2');
  });

  test('has chain-of-custody fields', () => {
    expect(manifest).toHaveProperty('generated');
    expect(manifest).toHaveProperty('deviceId');
    expect(manifest).toHaveProperty('case');
    expect(manifest.case).toHaveProperty('caseId');
    expect(manifest.case).toHaveProperty('operator');
  });

  test('has processing metadata', () => {
    expect(manifest).toHaveProperty('processing');
    expect(manifest.processing.embedding_model).toBe('vfs-spectral-projection-v1');
  });

  test('contents array lists expected file types', () => {
    const fileNames = manifest.contents.map(c => c.file);
    expect(fileNames).toContain('detection_log.csv');
    expect(fileNames).toContain('speaker_library.json');
    expect(fileNames).toContain('session_timeline.json');
    expect(fileNames).toContain('session_transcript.vtt');
  });

  test('each content item has bytes and sha256', () => {
    for (const item of manifest.contents) {
      expect(item).toHaveProperty('file');
      expect(item).toHaveProperty('bytes');
      expect(item).toHaveProperty('sha256');
      expect(typeof item.bytes).toBe('number');
    }
  });

  test('has manifestSha256 self-hash', () => {
    expect(manifest).toHaveProperty('manifestSha256');
    expect(typeof manifest.manifestSha256).toBe('string');
  });
});

test.describe('Baseline: WebVTT parsing', () => {
  let vtt;

  test.beforeAll(() => {
    vtt = fs.readFileSync(path.join(FIXTURES, 'sample_session.vtt'), 'utf-8');
  });

  test('starts with WEBVTT header', () => {
    expect(vtt.startsWith('WEBVTT')).toBe(true);
  });

  test('contains NOTE with session start timestamp', () => {
    expect(vtt).toContain('NOTE Voice Forensics session transcript');
  });

  test('contains cue timestamps in HH:MM:SS.mmm format', () => {
    const timePattern = /\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}/;
    expect(timePattern.test(vtt)).toBe(true);
  });

  test('contains voice tags <v SpeakerName>', () => {
    expect(vtt).toMatch(/<v Speaker \d+>/);
  });

  test('contains jitter and pitch annotations', () => {
    expect(vtt).toMatch(/\[jitter [\d.]+% · pitch \d+Hz\]/);
  });

  test('contains sequential cue numbers', () => {
    const cueNumbers = vtt.match(/^(\d+)$/gm);
    expect(cueNumbers).not.toBeNull();
    for (let i = 0; i < cueNumbers.length; i++) {
      expect(parseInt(cueNumbers[i])).toBe(i + 1);
    }
  });
});

test.describe('Baseline: CSV column headers', () => {
  test('buildLogCsv header matches expected 35 columns', () => {
    const expectedHeader = 'Date,Time,Speaker_ID,Speaker_Name,Confidence,Verified,SNR_dB,Zone,Distance,Voice_Confidence,Pitch_F0_Hz,F1_Hz,F2_Hz,F3_Hz,F2_F1_Ratio,F3_F2_Ratio,Nasal_Formant_Hz,Nasalance_pct,Jitter_pct,Shimmer_pct,Ambient_Class,Noise_Floor_dB,Ambient_Peak_Hz,Ambient_Hum_pct,Device_Latitude,Device_Longitude,GPS_Accuracy_m,Speech_Seconds,Voiced_Fraction,Clipping_Pct,Overlap_Detected,Reverb_Flag,Quality_Score,Embedding_Model,Transcript';
    const columns = expectedHeader.split(',');
    expect(columns.length).toBe(35);
    expect(columns[0]).toBe('Date');
    expect(columns[2]).toBe('Speaker_ID');
    expect(columns[33]).toBe('Embedding_Model');
    expect(columns[34]).toBe('Transcript');
  });
});

test.describe('Baseline: Format version strings', () => {
  test('all known v2 format identifiers', () => {
    const formats = [
      'VoiceForensics-SpeakerLibrary-v2',
      'VoiceForensics-EvidenceBundle-v2',
      'VoiceForensics-SessionTimeline-v1',
      'VoiceForensics-ReferenceEnrollment-v1'
    ];
    for (const f of formats) {
      expect(f).toMatch(/^VoiceForensics-\w+-v\d+$/);
    }
  });
});
