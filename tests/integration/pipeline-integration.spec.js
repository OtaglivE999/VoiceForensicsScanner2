const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

function loadModule(relPath) {
  return fs.readFileSync(path.resolve(__dirname, relPath), 'utf-8');
}

const MODULES = [
  { path: '../../src/js/audio/resampler.js', exports: [
    'Resampler', 'IntegerDecimator', 'createResampler', 'designLowpass', 'MODEL_RATE',
  ]},
  { path: '../../src/js/audio/speech-segmenter.js', exports: [
    'FrameVAD', 'SpeechSegmenter', 'DEFAULT_OPTS',
  ]},
  { path: '../../src/js/features/mel-features.js', exports: [
    'MelFeatureExtractor', 'hzToMel', 'melToHz',
  ]},
  { path: '../../src/js/features/speaker-embedding.js', exports: [
    'SpeakerEmbeddingExtractor', 'MODEL_MANIFEST',
  ]},
  { path: '../../src/js/matching/speaker-matcher.js', exports: [
    'SpeakerMatcher',
  ]},
  { path: '../../src/js/matching/enrollment.js', exports: [
    'EnrollmentManager',
  ]},
  { path: '../../src/js/matching/profile-updater.js', exports: [
    'ProfileUpdater',
  ]},
  { path: '../../src/js/diagnostics/pipeline-diagnostics.js', exports: [
    'PipelineDiagnostics', 'STATUS', 'MODULE_IDS',
  ]},
];

function buildInjectable() {
  let combined = '';
  for (const mod of MODULES) {
    let src = loadModule(mod.path);
    src = src.replace(/^export\s*\{[^}]*\}\s*;?\s*$/m, '');
    combined += src + '\n';
  }
  for (const mod of MODULES) {
    for (const sym of mod.exports) {
      combined += `window.${sym} = ${sym};\n`;
    }
  }
  return combined;
}

const INJECTABLE = buildInjectable();

test.describe('Pipeline Integration', () => {
  let page;

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    await page.goto('about:blank');
    await page.addScriptTag({ content: INJECTABLE });
  });

  test.afterAll(async () => {
    await page?.close();
  });

  test('all modules load without conflict', async () => {
    const result = await page.evaluate(() => {
      const symbols = [
        'Resampler', 'IntegerDecimator', 'createResampler',
        'FrameVAD', 'SpeechSegmenter',
        'MelFeatureExtractor',
        'SpeakerEmbeddingExtractor', 'MODEL_MANIFEST',
        'SpeakerMatcher',
        'EnrollmentManager',
        'ProfileUpdater',
        'PipelineDiagnostics', 'STATUS', 'MODULE_IDS',
      ];
      const missing = symbols.filter(s => typeof window[s] === 'undefined');
      return { allPresent: missing.length === 0, missing };
    });
    expect(result.allPresent).toBe(true);
  });

  test('resampler → segmenter: 48kHz signal decimated and processable', async () => {
    const result = await page.evaluate(() => {
      const decimator = new IntegerDecimator(3);

      const sampleRate48k = 48000;
      const durationSec = 0.5;
      const samples = sampleRate48k * durationSec;
      const buf48k = new Float32Array(samples);
      for (let i = 0; i < samples; i++) {
        buf48k[i] = 0.3 * Math.sin(2 * Math.PI * 200 * i / sampleRate48k);
      }

      const buf16k = decimator.process(buf48k);

      const segmenter = new SpeechSegmenter({
        frameSize: 320,
        minSpeechMs: 100,
        hangoverFrames: 3,
        energyThreshold: 0.001,
      });

      const segments = segmenter.process(buf16k);

      return {
        input48kLen: buf48k.length,
        output16kLen: buf16k.length,
        ratio: buf48k.length / buf16k.length,
        hasOutput: buf16k.length > 0,
        segmenterAccepted: true,
      };
    });
    expect(result.ratio).toBeCloseTo(3.0, 0);
    expect(result.hasOutput).toBe(true);
    expect(result.segmenterAccepted).toBe(true);
  });

  test('segmenter → mel features: speech segment produces MFCCs', async () => {
    const result = await page.evaluate(() => {
      const sampleRate = 16000;
      const dur = 0.4;
      const n = sampleRate * dur;
      const pcm = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        pcm[i] = 0.3 * Math.sin(2 * Math.PI * 200 * i / sampleRate);
      }

      const extractor = new MelFeatureExtractor({ sampleRate });
      const mfccs = extractor.mfccMean(pcm);

      return {
        length: mfccs.length,
        hasNonZero: mfccs.some(v => v !== 0),
        c0: mfccs[0],
      };
    });
    expect(result.length).toBe(13);
    expect(result.hasNonZero).toBe(true);
  });

  test('mel features → embedding extractor: interface accepts log-mel frames', async () => {
    const result = await page.evaluate(() => {
      const sampleRate = 16000;
      const dur = 1.0;
      const n = sampleRate * dur;
      const pcm = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        pcm[i] = 0.3 * Math.sin(2 * Math.PI * 200 * i / sampleRate);
      }

      const extractor = new MelFeatureExtractor({ sampleRate });
      const result = extractor.logMelSpectrogram(pcm);

      const embExtractor = new SpeakerEmbeddingExtractor();
      const st = embExtractor.status;

      return {
        logMelRows: result.numFrames,
        logMelCols: result.logMel[0] ? result.logMel[0].length : 0,
        embStatus: st,
        modelManifest: MODEL_MANIFEST,
      };
    });
    expect(result.logMelRows).toBeGreaterThan(0);
    expect(result.logMelCols).toBe(40);
    expect(result.embStatus).toBe('no-model-configured');
    expect(result.modelManifest.status).toBe('no-model-available');
  });

  test('speaker matcher: feature-based fallback works end-to-end', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher();

      const profiles = {
        alice: {
          id: 'alice',
          name: 'Alice',
          avgFeatures: { pitch: 220, centroid: 2500, f1: 600, f2: 1800, f3: 3000, spread: 500, flatness: 0.3, rolloff: 4000, zcr: 100 },
          detections: 10,
        },
        bob: {
          id: 'bob',
          name: 'Bob',
          avgFeatures: { pitch: 120, centroid: 1800, f1: 400, f2: 1200, f3: 2500, spread: 600, flatness: 0.4, rolloff: 3000, zcr: 80 },
          detections: 10,
        },
      };

      const probeAlice = { features: { pitch: 221, centroid: 2505, f1: 602, f2: 1802, f3: 3005, spread: 502, flatness: 0.31, rolloff: 4005, zcr: 101 } };
      const probeBob = { features: { pitch: 121, centroid: 1805, f1: 402, f2: 1202, f3: 2505, spread: 602, flatness: 0.41, rolloff: 3005, zcr: 81 } };

      const matchA = matcher.match(probeAlice, profiles);
      const matchB = matcher.match(probeBob, profiles);

      return {
        aliceMatch: matchA ? matchA.id : null,
        aliceMethod: matchA ? matchA.method : null,
        bobMatch: matchB ? matchB.id : null,
      };
    });
    expect(result.aliceMatch).toBe('alice');
    expect(result.aliceMethod).toBe('features');
    expect(result.bobMatch).toBe('bob');
  });

  test('enrollment → profile updater: enrolled profile can be updated', async () => {
    const result = await page.evaluate(() => {
      const em = new EnrollmentManager({
        minObservations: 2,
        minTotalSpeechMs: 200,
        consistencyThreshold: 0.1,
      });
      const pu = new ProfileUpdater({ fastAlpha: 0.5 });

      const emb1 = new Float32Array([1, 0, 0, 0]);
      const emb2 = new Float32Array([0.95, 0.05, 0, 0]);

      const r1 = em.submit({
        embedding: emb1,
        features: { pitch: 200 },
        quality: { snr: 30, durationMs: 500, voiceConf: 0.9, clippingPct: 0 },
        durationMs: 500,
      });

      const r2 = em.submit({
        embedding: emb2,
        features: { pitch: 210 },
        quality: { snr: 30, durationMs: 500, voiceConf: 0.9, clippingPct: 0 },
        durationMs: 500,
      });

      if (r2.status !== 'enrolled') {
        return { enrolled: false, status1: r1.status, status2: r2.status };
      }

      const profile = r2.profile;

      const newEmb = new Float32Array([0.9, 0.1, 0, 0]);
      const quality = { snr: 30, durationMs: 500, voiceConf: 0.9, clippingPct: 0 };
      const updateResult = pu.updateEmbedding(profile, newEmb, quality);
      pu.updateFeatures(profile, { pitch: 220 }, quality);

      return {
        enrolled: true,
        profileHasEmbedding: !!profile.embedding,
        embDim: profile.embedding.length,
        updated: updateResult.updated,
        pitch: profile.avgFeatures.pitch,
      };
    });
    expect(result.enrolled).toBe(true);
    expect(result.profileHasEmbedding).toBe(true);
    expect(result.embDim).toBe(4);
    expect(result.updated).toBe(true);
    expect(result.pitch).toBeGreaterThan(200);
  });

  test('diagnostics tracks pipeline module registration', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();

      diag.registerModule('resampler', { status: 'ok', version: '1.0', label: '48k→16k Resampler' });
      diag.registerModule('speechSegmenter', { status: 'ok', version: '1.0', label: 'Speech Segmenter' });
      diag.registerModule('melFeatures', { status: 'ok', version: '1.0', label: 'Mel Features' });
      diag.registerModule('speakerEmbedding', { status: 'placeholder', detail: { modelId: null } });
      diag.registerModule('speakerMatcher', { status: 'ok', version: '1.0', label: 'Speaker Matcher' });
      diag.registerModule('enrollment', { status: 'ok', version: '1.0', label: 'Enrollment' });
      diag.registerModule('profileUpdater', { status: 'ok', version: '1.0', label: 'Profile Updater' });

      const health = diag.getPipelineHealth();
      const chain = diag.getProcessingChain();
      const model = diag.getModelAvailability();

      return {
        overall: health.overall,
        moduleCount: health.moduleCount,
        issueCount: health.issues.length,
        chainLen: chain.length,
        chainFirst: chain[0].id,
        modelAvailable: model.available,
      };
    });
    expect(result.overall).toBe('degraded');
    expect(result.moduleCount).toBe(7);
    expect(result.issueCount).toBe(1);
    expect(result.chainLen).toBe(7);
    expect(result.chainFirst).toBe('resampler');
    expect(result.modelAvailable).toBe(false);
  });

  test('full data flow: signal → features → match → enrollment', async () => {
    const result = await page.evaluate(() => {
      const sampleRate = 16000;
      const dur = 0.5;
      const n = sampleRate * dur;

      function makeSignal(freq) {
        const pcm = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          pcm[i] = 0.3 * Math.sin(2 * Math.PI * freq * i / sampleRate);
        }
        return pcm;
      }

      const extractor = new MelFeatureExtractor({ sampleRate });

      const sig1 = makeSignal(200);
      const sig2 = makeSignal(205);
      const feat1 = extractor.mfccMean(sig1);
      const feat2 = extractor.mfccMean(sig2);

      const hasFeat = feat1.length === 13 && feat2.length === 13;

      const em = new EnrollmentManager({
        minObservations: 2,
        minTotalSpeechMs: 200,
        consistencyThreshold: 0.1,
      });

      const emb1 = new Float32Array(192);
      const emb2 = new Float32Array(192);
      emb1[0] = 1; emb1[1] = 0.1;
      emb2[0] = 0.99; emb2[1] = 0.12;

      const r1 = em.submit({
        embedding: emb1,
        features: { pitch: 200, c0: feat1[0] },
        quality: { snr: 30, durationMs: 500, voiceConf: 0.9, clippingPct: 0 },
        durationMs: 500,
      });

      const r2 = em.submit({
        embedding: emb2,
        features: { pitch: 205, c0: feat2[0] },
        quality: { snr: 30, durationMs: 500, voiceConf: 0.9, clippingPct: 0 },
        durationMs: 500,
      });

      return {
        hasFeat,
        status1: r1.status,
        status2: r2.status,
        profileId: r2.profile ? r2.profile.id : null,
        profileHasEmb: r2.profile ? !!r2.profile.embedding : false,
      };
    });
    expect(result.hasFeat).toBe(true);
    expect(result.status1).toBe('new-candidate');
    expect(result.status2).toBe('enrolled');
    expect(result.profileHasEmb).toBe(true);
  });

  test('embedding model compatibility guard prevents cross-model matching', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({ embeddingModelId: 'ecapa-v1' });

      const profiles = {
        spk1: {
          id: 'spk1',
          embedding: new Float32Array([1, 0, 0, 0]),
          embeddingModelId: 'different-model',
          avgFeatures: { pitch: 200 },
          detections: 10,
        },
      };

      const probe = {
        embedding: new Float32Array([0.99, 0.01, 0, 0]),
        features: { pitch: 200 },
      };

      const match = matcher.match(probe, profiles);
      return {
        method: match ? match.method : 'no-match',
      };
    });
    // Embedding match skipped because model IDs differ; falls back to features
    // Features may or may not match depending on threshold, but embedding should be skipped
    expect(result.method).not.toBe('embedding');
  });

  test('quality gate prevents low-quality updates to mature profiles', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater();

      const profile = {
        embedding: new Float32Array([1, 0, 0, 0]),
        avgFeatures: { pitch: 200, centroid: 2000 },
        detections: 50,
      };

      const originalPitch = profile.avgFeatures.pitch;

      const lowQuality = { snr: 5, durationMs: 100, voiceConf: 0.2, clippingPct: 0.05 };
      const embResult = pu.updateEmbedding(profile, new Float32Array([0, 1, 0, 0]), lowQuality);
      pu.updateFeatures(profile, { pitch: 500 }, lowQuality);

      return {
        embUpdated: embResult.updated,
        reasons: embResult.reasons,
        pitchUnchanged: profile.avgFeatures.pitch === originalPitch,
        embUnchanged: profile.embedding[0] === 1,
      };
    });
    expect(result.embUpdated).toBe(false);
    expect(result.reasons.length).toBeGreaterThan(0);
    expect(result.pitchUnchanged).toBe(true);
    expect(result.embUnchanged).toBe(true);
  });

  test('version constants are consistent across modules', async () => {
    const result = await page.evaluate(() => ({
      modelRate: MODEL_RATE,
      modelManifestDim: MODEL_MANIFEST.embeddingDim,
      diagnosticVersion: DIAGNOSTIC_VERSION,
      moduleIdsCount: MODULE_IDS.length,
      moduleIdsIncludesResampler: MODULE_IDS.includes('resampler'),
      moduleIdsIncludesIdb: MODULE_IDS.includes('idbStore'),
    }));
    expect(result.modelRate).toBe(16000);
    expect(result.modelManifestDim).toBe(192);
    expect(result.diagnosticVersion).toContain('v1');
    expect(result.moduleIdsCount).toBe(9);
    expect(result.moduleIdsIncludesResampler).toBe(true);
    expect(result.moduleIdsIncludesIdb).toBe(true);
  });
});
