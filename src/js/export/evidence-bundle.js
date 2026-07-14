/**
 * Versioned evidence bundle builder.
 *
 * Produces a structured evidence bundle with:
 * - Version-stamped manifest with chain-of-custody hashes
 * - Speaker library with processing metadata
 * - Detection log CSV
 * - Session timeline JSON
 * - WebVTT subtitles
 *
 * All content is SHA-256 hashed for integrity verification.
 */

const BUNDLE_VERSION = 'VoiceForensics-EvidenceBundle-v3';
const SPEAKER_LIB_VERSION = 'VoiceForensics-SpeakerLibrary-v3';
const TIMELINE_VERSION = 'VoiceForensics-SessionTimeline-v2';

class EvidenceBundleBuilder {
  constructor(opts = {}) {
    this._processingMeta = opts.processingMeta || {};
    this._matchThresholds = opts.matchThresholds || {};
    this._deviceId = opts.deviceId || 'unknown';
  }

  /**
   * Build the complete evidence bundle contents.
   *
   * @param {object} data
   * @param {object[]} data.speakers – speaker profiles
   * @param {object[]} data.detections – detection log entries
   * @param {object} data.session – session info
   * @param {object} [data.audioClips] – per-speaker audio (id→ArrayBuffer)
   * @returns {Promise<{ files: Array<{name:string, content:string|Uint8Array}>, manifest: object }>}
   */
  async build(data) {
    const files = [];
    const contentHashes = [];

    // Speaker library
    const speakerJson = this._buildSpeakerLibrary(data.speakers);
    const speakerStr = JSON.stringify(speakerJson, null, 2);
    files.push({ name: 'speaker_library.json', content: speakerStr });
    contentHashes.push({
      filename: 'speaker_library.json',
      bytes: speakerStr.length,
      sha256: await _sha256(speakerStr),
    });

    // Detection log CSV
    const csv = this._buildDetectionCsv(data.detections);
    files.push({ name: 'detection_log.csv', content: csv });
    contentHashes.push({
      filename: 'detection_log.csv',
      bytes: csv.length,
      sha256: await _sha256(csv),
    });

    // Session timeline
    const timeline = this._buildTimeline(data.detections, data.session);
    const timelineStr = JSON.stringify(timeline, null, 2);
    files.push({ name: 'session_timeline.json', content: timelineStr });
    contentHashes.push({
      filename: 'session_timeline.json',
      bytes: timelineStr.length,
      sha256: await _sha256(timelineStr),
    });

    // WebVTT
    const vtt = this._buildVtt(data.detections, data.session);
    files.push({ name: 'session.vtt', content: vtt });
    contentHashes.push({
      filename: 'session.vtt',
      bytes: vtt.length,
      sha256: await _sha256(vtt),
    });

    // Audio clips
    if (data.audioClips) {
      for (const [speakerId, buffer] of Object.entries(data.audioClips)) {
        const name = `audio_${speakerId}.wav`;
        const bytes = new Uint8Array(buffer);
        files.push({ name, content: bytes });
        contentHashes.push({
          filename: name,
          bytes: bytes.length,
          sha256: await _sha256Bytes(bytes),
        });
      }
    }

    // Manifest
    const manifest = {
      format: BUNDLE_VERSION,
      exportedAt: new Date().toISOString(),
      deviceId: this._deviceId,
      processingMeta: this._processingMeta,
      matchThresholds: this._matchThresholds,
      contents: contentHashes,
      speakerCount: data.speakers.length,
      detectionCount: data.detections.length,
      chainOfCustody: {
        hashAlgorithm: 'SHA-256',
        note: 'Each file hash is computed before ZIP compression.',
      },
    };

    const manifestStr = JSON.stringify(manifest, null, 2);
    manifest.manifestSha256 = await _sha256(manifestStr);

    const finalManifestStr = JSON.stringify(manifest, null, 2);
    files.push({ name: 'manifest.json', content: finalManifestStr });

    return { files, manifest };
  }

  _buildSpeakerLibrary(speakers) {
    const lib = {
      format: SPEAKER_LIB_VERSION,
      exportedAt: new Date().toISOString(),
      processingMeta: this._processingMeta,
      matchThresholds: this._matchThresholds,
      speakerCount: speakers.length,
      speakers: {},
    };

    for (const spk of speakers) {
      lib.speakers[spk.id] = {
        id: spk.id,
        name: spk.name || spk.id,
        detections: spk.detections || 0,
        enrolledAt: spk.enrolledAt || null,
        observationCount: spk.observationCount || null,
        embedding: spk.embedding ? Array.from(spk.embedding) : null,
        embeddingModelId: spk.embeddingModelId || this._processingMeta.embedding_model || null,
        embeddingModelVersion: spk.embeddingModelVersion || this._processingMeta.embedding_model_version || null,
        embeddingDim: spk.embedding ? spk.embedding.length : null,
        embeddingL2Normalized: true,
        avgFeatures: spk.avgFeatures || null,
      };
    }

    return lib;
  }

  _buildDetectionCsv(detections) {
    const headers = [
      'timestamp', 'datetime', 'speakerId', 'speakerName',
      'snr', 'voiceConf', 'rms', 'peak',
      'pitch', 'f1', 'f2', 'f3',
      'jitter', 'shimmer',
      'matchMethod', 'matchSimilarity', 'matchConfidence',
      'durationMs', 'clippingPct',
    ];
    const lines = [headers.join(',')];

    for (const det of detections) {
      const row = [
        det.timestamp || '',
        det.timestamp ? new Date(det.timestamp).toISOString() : '',
        det.speakerId || '',
        _csvEscape(det.speakerName || ''),
        _num(det.snr),
        _num(det.voiceConf),
        _num(det.rms),
        _num(det.peak),
        _num(det.pitch),
        _num(det.f1),
        _num(det.f2),
        _num(det.f3),
        _num(det.jitter),
        _num(det.shimmer),
        det.matchMethod || '',
        _num(det.matchSimilarity),
        det.matchConfidence || '',
        det.durationMs || '',
        _num(det.clippingPct),
      ];
      lines.push(row.join(','));
    }

    return lines.join('\n') + '\n';
  }

  _buildTimeline(detections, session) {
    const entries = detections.map(det => ({
      timestamp: det.timestamp,
      offsetSec: session && session.startTime
        ? (det.timestamp - session.startTime) / 1000
        : null,
      speakerId: det.speakerId || null,
      speakerName: det.speakerName || null,
      snr: det.snr,
      voiceConf: det.voiceConf,
      pitch: det.pitch,
      matchMethod: det.matchMethod || null,
      matchSimilarity: det.matchSimilarity || null,
    }));

    return {
      format: TIMELINE_VERSION,
      session_audio_start: session && session.startTime
        ? new Date(session.startTime).toISOString()
        : null,
      entry_count: entries.length,
      entries,
    };
  }

  _buildVtt(detections, session) {
    const startMs = session && session.startTime ? session.startTime : 0;
    let vtt = 'WEBVTT\n';
    vtt += `NOTE Session started at ${new Date(startMs).toISOString()}\n\n`;

    detections.forEach((det, i) => {
      const offsetMs = det.timestamp ? det.timestamp - startMs : i * 150;
      const startTime = _msToVttTime(offsetMs);
      const endTime = _msToVttTime(offsetMs + (det.durationMs || 300));
      const speaker = det.speakerName || det.speakerId || 'Unknown';

      vtt += `${i + 1}\n`;
      vtt += `${startTime} --> ${endTime}\n`;
      vtt += `<v ${speaker}>`;
      if (det.pitch) vtt += ` [pitch:${Math.round(det.pitch)}Hz]`;
      if (det.jitter) vtt += ` [jitter:${(det.jitter * 100).toFixed(1)}%]`;
      vtt += '\n\n';
    });

    return vtt;
  }
}

// ── Helpers ───────────────────────────────────────────────────

function _num(v) {
  return v != null && typeof v === 'number' ? v.toFixed(4) : '';
}

function _csvEscape(s) {
  if (s.includes(',') || s.includes('"') || s.includes('\n')) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function _msToVttTime(ms) {
  const totalSec = Math.max(0, ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = Math.floor(totalSec % 60);
  const frac = Math.floor((totalSec % 1) * 1000);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(frac).padStart(3, '0')}`;
}

async function _sha256(str) {
  const data = new TextEncoder().encode(str);
  return _sha256Bytes(data);
}

async function _sha256Bytes(bytes) {
  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const buf = await crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('');
  }
  return 'sha256-unavailable-insecure-context';
}

export {
  EvidenceBundleBuilder,
  BUNDLE_VERSION,
  SPEAKER_LIB_VERSION,
  TIMELINE_VERSION,
};
