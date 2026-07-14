const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

const SRC = fs.readFileSync(
  path.resolve(__dirname, '../../src/js/export/evidence-bundle.js'), 'utf-8'
);

const INJECTABLE = SRC
  .replace(/^export\s*\{[^}]*\}\s*;?\s*$/m, '')
  + '\nwindow.EvidenceBundleBuilder = EvidenceBundleBuilder;\n'
  + 'window.BUNDLE_VERSION = BUNDLE_VERSION;\n'
  + 'window.SPEAKER_LIB_VERSION = SPEAKER_LIB_VERSION;\n'
  + 'window.TIMELINE_VERSION = TIMELINE_VERSION;\n';

// Need a proper origin for crypto.subtle
let server;
let port;

test.describe('Evidence Bundle Builder', () => {
  let page;

  test.beforeAll(async ({ browser }) => {
    const http = require('http');
    server = http.createServer((req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end('<!DOCTYPE html><html><body></body></html>');
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    port = server.address().port;

    page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${port}`);
    await page.addScriptTag({ content: INJECTABLE });
  });

  test.afterAll(async () => {
    await page?.close();
    if (server) server.close();
  });

  test('exports expected symbols', async () => {
    const result = await page.evaluate(() => ({
      has: typeof window.EvidenceBundleBuilder === 'function',
      bundleVersion: window.BUNDLE_VERSION,
      speakerVersion: window.SPEAKER_LIB_VERSION,
      timelineVersion: window.TIMELINE_VERSION,
    }));
    expect(result.has).toBe(true);
    expect(result.bundleVersion).toContain('v3');
    expect(result.speakerVersion).toContain('v3');
    expect(result.timelineVersion).toContain('v2');
  });

  test('build produces expected files', async () => {
    const result = await page.evaluate(async () => {
      const builder = new EvidenceBundleBuilder({
        processingMeta: { embedding_model: 'test' },
        deviceId: 'dev-1',
      });

      const { files, manifest } = await builder.build({
        speakers: [{ id: 'spk1', name: 'Alice', detections: 5, avgFeatures: { pitch: 150 } }],
        detections: [{ timestamp: 1000, speakerId: 'spk1', speakerName: 'Alice', snr: 30, voiceConf: 0.8 }],
        session: { startTime: 0 },
      });

      return {
        fileNames: files.map(f => f.name),
        manifestFormat: manifest.format,
        speakerCount: manifest.speakerCount,
        detectionCount: manifest.detectionCount,
        hasManifestSha: !!manifest.manifestSha256,
        contentsCount: manifest.contents.length,
      };
    });
    expect(result.fileNames).toContain('speaker_library.json');
    expect(result.fileNames).toContain('detection_log.csv');
    expect(result.fileNames).toContain('session_timeline.json');
    expect(result.fileNames).toContain('session.vtt');
    expect(result.fileNames).toContain('manifest.json');
    expect(result.manifestFormat).toBe('VoiceForensics-EvidenceBundle-v3');
    expect(result.speakerCount).toBe(1);
    expect(result.detectionCount).toBe(1);
    expect(result.hasManifestSha).toBe(true);
    expect(result.contentsCount).toBe(4);
  });

  test('speaker library has correct structure', async () => {
    const result = await page.evaluate(async () => {
      const builder = new EvidenceBundleBuilder({
        processingMeta: { embedding_model: 'ecapa-v1', embedding_model_version: '1.0' },
      });

      const { files } = await builder.build({
        speakers: [{
          id: 'spk1',
          name: 'Alice',
          detections: 10,
          embedding: new Float32Array([0.5, 0.5, 0.5, 0.5]),
          avgFeatures: { pitch: 150, f1: 500 },
        }],
        detections: [],
        session: {},
      });

      const libFile = files.find(f => f.name === 'speaker_library.json');
      const lib = JSON.parse(libFile.content);
      const spk = lib.speakers['spk1'];

      return {
        format: lib.format,
        speakerCount: lib.speakerCount,
        hasProcessingMeta: !!lib.processingMeta,
        spkName: spk.name,
        spkEmbedding: spk.embedding,
        spkEmbeddingModel: spk.embeddingModelId,
        spkDim: spk.embeddingDim,
        spkNormalized: spk.embeddingL2Normalized,
        hasAvgFeatures: !!spk.avgFeatures,
      };
    });
    expect(result.format).toBe('VoiceForensics-SpeakerLibrary-v3');
    expect(result.speakerCount).toBe(1);
    expect(result.hasProcessingMeta).toBe(true);
    expect(result.spkName).toBe('Alice');
    expect(result.spkEmbedding.length).toBe(4);
    expect(result.spkEmbeddingModel).toBe('ecapa-v1');
    expect(result.spkDim).toBe(4);
    expect(result.spkNormalized).toBe(true);
    expect(result.hasAvgFeatures).toBe(true);
  });

  test('detection CSV has correct headers', async () => {
    const result = await page.evaluate(async () => {
      const builder = new EvidenceBundleBuilder();
      const { files } = await builder.build({
        speakers: [],
        detections: [{ timestamp: 1000, snr: 25, voiceConf: 0.7 }],
        session: {},
      });

      const csvFile = files.find(f => f.name === 'detection_log.csv');
      const headerLine = csvFile.content.split('\n')[0];
      return { headers: headerLine.split(',') };
    });
    expect(result.headers).toContain('timestamp');
    expect(result.headers).toContain('speakerId');
    expect(result.headers).toContain('snr');
    expect(result.headers).toContain('matchMethod');
    expect(result.headers).toContain('durationMs');
  });

  test('timeline has correct structure', async () => {
    const result = await page.evaluate(async () => {
      const builder = new EvidenceBundleBuilder();
      const { files } = await builder.build({
        speakers: [],
        detections: [
          { timestamp: 5000, speakerId: 'spk1', snr: 30 },
          { timestamp: 6000, speakerId: 'spk2', snr: 25 },
        ],
        session: { startTime: 4000 },
      });

      const tlFile = files.find(f => f.name === 'session_timeline.json');
      const tl = JSON.parse(tlFile.content);
      return {
        format: tl.format,
        entryCount: tl.entry_count,
        firstOffset: tl.entries[0].offsetSec,
        secondOffset: tl.entries[1].offsetSec,
      };
    });
    expect(result.format).toBe('VoiceForensics-SessionTimeline-v2');
    expect(result.entryCount).toBe(2);
    expect(result.firstOffset).toBe(1.0);
    expect(result.secondOffset).toBe(2.0);
  });

  test('VTT has proper formatting', async () => {
    const result = await page.evaluate(async () => {
      const builder = new EvidenceBundleBuilder();
      const { files } = await builder.build({
        speakers: [],
        detections: [
          { timestamp: 1000, speakerName: 'Alice', pitch: 150, jitter: 0.02, durationMs: 500 },
        ],
        session: { startTime: 0 },
      });

      const vttFile = files.find(f => f.name === 'session.vtt');
      return { content: vttFile.content };
    });
    expect(result.content).toMatch(/^WEBVTT/);
    expect(result.content).toContain('NOTE Session started');
    expect(result.content).toContain('<v Alice>');
    expect(result.content).toContain('[pitch:150Hz]');
    expect(result.content).toContain('[jitter:2.0%]');
    expect(result.content).toContain('-->');
  });

  test('manifest contents include SHA-256 hashes', async () => {
    const result = await page.evaluate(async () => {
      const builder = new EvidenceBundleBuilder();
      const { manifest } = await builder.build({
        speakers: [],
        detections: [],
        session: {},
      });

      const allHaveHash = manifest.contents.every(c => c.sha256 && c.sha256.length === 64);
      const allHaveBytes = manifest.contents.every(c => typeof c.bytes === 'number');
      return { allHaveHash, allHaveBytes, hashAlg: manifest.chainOfCustody.hashAlgorithm };
    });
    expect(result.allHaveHash).toBe(true);
    expect(result.allHaveBytes).toBe(true);
    expect(result.hashAlg).toBe('SHA-256');
  });

  test('CSV escapes commas in speaker names', async () => {
    const result = await page.evaluate(async () => {
      const builder = new EvidenceBundleBuilder();
      const { files } = await builder.build({
        speakers: [],
        detections: [{ timestamp: 1000, speakerName: 'Last, First', snr: 20 }],
        session: {},
      });
      const csv = files.find(f => f.name === 'detection_log.csv').content;
      return { csv };
    });
    expect(result.csv).toContain('"Last, First"');
  });

  test('empty bundle builds without errors', async () => {
    const result = await page.evaluate(async () => {
      const builder = new EvidenceBundleBuilder();
      const { files, manifest } = await builder.build({
        speakers: [],
        detections: [],
        session: {},
      });
      return { fileCount: files.length, speakerCount: manifest.speakerCount };
    });
    expect(result.fileCount).toBe(5);
    expect(result.speakerCount).toBe(0);
  });
});
