const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

const SRC = fs.readFileSync(
  path.resolve(__dirname, '../../src/js/matching/profile-updater.js'), 'utf-8'
);

const INJECTABLE = SRC
  .replace(/^export\s*\{[^}]*\}\s*;?\s*$/m, '')
  + '\nwindow.ProfileUpdater = ProfileUpdater;\n'
  + 'window.UPDATE_DEFAULTS = UPDATE_DEFAULTS;\n';

const GOOD_QUALITY = { snr: 30, clippingPct: 0, durationMs: 500, voiceConf: 0.8 };

test.describe('Profile Updater', () => {
  let page;

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    await page.goto('about:blank');
    await page.addScriptTag({ content: INJECTABLE });
  });

  test.afterAll(async () => {
    await page?.close();
  });

  test('exports expected symbols', async () => {
    const result = await page.evaluate(() => ({
      has: typeof window.ProfileUpdater === 'function',
      hasDefaults: typeof window.UPDATE_DEFAULTS === 'object',
    }));
    expect(result.has).toBe(true);
    expect(result.hasDefaults).toBe(true);
  });

  test('checkQuality passes good signal', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater();
      return pu.checkQuality({ snr: 30, clippingPct: 0, durationMs: 500, voiceConf: 0.8 });
    });
    expect(result.pass).toBe(true);
    expect(result.reasons.length).toBe(0);
  });

  test('checkQuality rejects low SNR', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater();
      return pu.checkQuality({ snr: 5, clippingPct: 0, durationMs: 500, voiceConf: 0.8 });
    });
    expect(result.pass).toBe(false);
    expect(result.reasons[0]).toContain('SNR');
  });

  test('checkQuality rejects clipping', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater();
      return pu.checkQuality({ snr: 30, clippingPct: 0.05, durationMs: 500, voiceConf: 0.8 });
    });
    expect(result.pass).toBe(false);
    expect(result.reasons[0]).toContain('clipping');
  });

  test('checkQuality rejects short duration', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater();
      return pu.checkQuality({ snr: 30, clippingPct: 0, durationMs: 100, voiceConf: 0.8 });
    });
    expect(result.pass).toBe(false);
    expect(result.reasons[0]).toContain('duration');
  });

  test('checkQuality rejects low voice confidence', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater();
      return pu.checkQuality({ snr: 30, clippingPct: 0, durationMs: 500, voiceConf: 0.2 });
    });
    expect(result.pass).toBe(false);
    expect(result.reasons[0]).toContain('voiceConf');
  });

  test('checkQuality reports multiple failures', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater();
      return pu.checkQuality({ snr: 5, clippingPct: 0.05, durationMs: 100, voiceConf: 0.1 });
    });
    expect(result.pass).toBe(false);
    expect(result.reasons.length).toBe(4);
  });

  test('updateEmbedding sets first embedding', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater();
      const profile = { detections: 0 };
      const emb = new Float32Array([1, 0, 0, 0]);
      const quality = { snr: 30, clippingPct: 0, durationMs: 500, voiceConf: 0.8 };
      const r = pu.updateEmbedding(profile, emb, quality);
      return { updated: r.updated, dim: profile.embedding.length, detections: profile.detections };
    });
    expect(result.updated).toBe(true);
    expect(result.dim).toBe(4);
    expect(result.detections).toBe(1);
  });

  test('updateEmbedding applies EMA and re-normalizes', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater({ fastAlpha: 0.5 });
      const profile = { embedding: new Float32Array([1, 0, 0, 0]), detections: 1 };
      const newEmb = new Float32Array([0, 1, 0, 0]);
      const quality = { snr: 30, clippingPct: 0, durationMs: 500, voiceConf: 0.8 };
      pu.updateEmbedding(profile, newEmb, quality);

      let norm = 0;
      for (let i = 0; i < 4; i++) norm += profile.embedding[i] * profile.embedding[i];
      norm = Math.sqrt(norm);

      return {
        e0: profile.embedding[0],
        e1: profile.embedding[1],
        norm,
        detections: profile.detections,
      };
    });
    // With alpha=0.5: [0.5, 0.5, 0, 0] then normalized
    expect(result.e0).toBeGreaterThan(0.3);
    expect(result.e1).toBeGreaterThan(0.3);
    expect(Math.abs(result.norm - 1.0)).toBeLessThan(1e-6);
    expect(result.detections).toBe(2);
  });

  test('updateEmbedding rejects poor quality', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater();
      const profile = { embedding: new Float32Array([1, 0, 0, 0]), detections: 5 };
      const quality = { snr: 3, clippingPct: 0, durationMs: 500, voiceConf: 0.8 };
      const r = pu.updateEmbedding(profile, new Float32Array([0, 1, 0, 0]), quality);
      return { updated: r.updated, reasons: r.reasons, e0: profile.embedding[0] };
    });
    expect(result.updated).toBe(false);
    expect(result.reasons.length).toBeGreaterThan(0);
    expect(result.e0).toBe(1);
  });

  test('updateFeatures applies EMA to numeric keys', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater({ fastAlpha: 0.5 });
      const profile = { avgFeatures: { pitch: 100, centroid: 200 }, detections: 1 };
      const quality = { snr: 30, clippingPct: 0, durationMs: 500, voiceConf: 0.8 };
      pu.updateFeatures(profile, { pitch: 120, centroid: 220 }, quality);
      return { pitch: profile.avgFeatures.pitch, centroid: profile.avgFeatures.centroid };
    });
    expect(result.pitch).toBe(110);
    expect(result.centroid).toBe(210);
  });

  test('updateFeatures updates MFCC arrays', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater({ fastAlpha: 0.5 });
      const profile = { avgFeatures: { mfcc: [10, 20, 30] }, detections: 1 };
      const quality = { snr: 30, clippingPct: 0, durationMs: 500, voiceConf: 0.8 };
      pu.updateFeatures(profile, { mfcc: [20, 30, 40] }, quality);
      return profile.avgFeatures.mfcc;
    });
    expect(result[0]).toBe(15);
    expect(result[1]).toBe(25);
    expect(result[2]).toBe(35);
  });

  test('alpha decreases after maxDetectionsForFastLearn', async () => {
    const result = await page.evaluate(() => {
      const pu = new ProfileUpdater({
        maxDetectionsForFastLearn: 5,
        fastAlpha: 0.5,
        slowAlpha: 0.01,
      });

      const profile1 = { avgFeatures: { pitch: 100 }, detections: 1 };
      const profile2 = { avgFeatures: { pitch: 100 }, detections: 100 };
      const quality = { snr: 30, clippingPct: 0, durationMs: 500, voiceConf: 0.8 };

      pu.updateFeatures(profile1, { pitch: 200 }, quality);
      pu.updateFeatures(profile2, { pitch: 200 }, quality);

      return {
        youngPitch: profile1.avgFeatures.pitch,
        maturePitch: profile2.avgFeatures.pitch,
      };
    });
    // Young: 100 + 0.5*(200-100) = 150
    // Mature: 100 + 0.01*(200-100) = 101
    expect(result.youngPitch).toBe(150);
    expect(result.maturePitch).toBe(101);
  });
});
