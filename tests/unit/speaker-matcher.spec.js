const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

const SRC = fs.readFileSync(
  path.resolve(__dirname, '../../src/js/matching/speaker-matcher.js'), 'utf-8'
);

const INJECTABLE = SRC
  .replace(/^export\s*\{[^}]*\}\s*;?\s*$/m, '')
  + '\nwindow.SpeakerMatcher = SpeakerMatcher;\n'
  + 'window.MATCH_DEFAULTS = MATCH_DEFAULTS;\n'
  + 'window.DEFAULT_FEATURE_WEIGHTS = DEFAULT_FEATURE_WEIGHTS;\n';

function makeEmbedding(dim, seed) {
  const v = [];
  let s = seed;
  for (let i = 0; i < dim; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    v.push((s / 0x7fffffff) * 2 - 1);
  }
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < dim; i++) v[i] /= norm;
  return v;
}

test.describe('Speaker Matcher module', () => {
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
      hasSpeakerMatcher: typeof window.SpeakerMatcher === 'function',
      hasMatchDefaults: typeof window.MATCH_DEFAULTS === 'object',
      hasFeatureWeights: typeof window.DEFAULT_FEATURE_WEIGHTS === 'object',
    }));
    expect(result.hasSpeakerMatcher).toBe(true);
    expect(result.hasMatchDefaults).toBe(true);
    expect(result.hasFeatureWeights).toBe(true);
  });

  test('matchByEmbedding finds best match above threshold', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({ embeddingThreshold: 0.5 });

      const probe = new Float32Array([0.5, 0.5, 0.5, 0.5]);
      let n = 0;
      for (let i = 0; i < 4; i++) n += probe[i] * probe[i];
      n = Math.sqrt(n);
      for (let i = 0; i < 4; i++) probe[i] /= n;

      const profiles = {
        spk1: { embedding: new Float32Array([0.5, 0.5, 0.5, 0.5].map(v => v / Math.sqrt(1))) },
        spk2: { embedding: new Float32Array([1, 0, 0, 0]) },
      };

      const result = matcher.matchByEmbedding(probe, profiles);
      return result;
    });
    expect(result).not.toBe(null);
    expect(result.id).toBe('spk1');
    expect(result.similarity).toBeGreaterThan(0.9);
    expect(result.method).toBe('embedding');
  });

  test('matchByEmbedding returns null when no match above threshold', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({ embeddingThreshold: 0.99 });
      const probe = new Float32Array([1, 0, 0, 0]);
      const profiles = {
        spk1: { embedding: new Float32Array([0, 1, 0, 0]) },
      };
      return matcher.matchByEmbedding(probe, profiles);
    });
    expect(result).toBe(null);
  });

  test('matchByEmbedding returns null for null probe', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher();
      return matcher.matchByEmbedding(null, { spk1: { embedding: new Float32Array([1, 0]) } });
    });
    expect(result).toBe(null);
  });

  test('matchByEmbedding skips incompatible model IDs', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({
        embeddingModelId: 'model-v2',
        embeddingThreshold: 0.5,
      });

      const probe = new Float32Array([1, 0, 0, 0]);
      const profiles = {
        spk1: {
          embedding: new Float32Array([1, 0, 0, 0]),
          embeddingModelId: 'model-v1',
        },
        spk2: {
          embedding: new Float32Array([0.9, 0.1, 0, 0]),
          embeddingModelId: 'model-v2',
        },
      };
      const r = matcher.matchByEmbedding(probe, profiles);
      return r ? r.id : null;
    });
    expect(result).toBe('spk2');
  });

  test('matchByFeatures computes weighted similarity', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({ featureThreshold: 0.5 });
      const probe = { pitch: 150, centroid: 200, spread: 50, flatness: 0.3, rolloff: 100, f1: 500, f2: 1500, f3: 2500, zcr: 0.1, mfcc: [1, 2, 3] };
      const profiles = {
        spk1: { avgFeatures: { pitch: 155, centroid: 198, spread: 48, flatness: 0.28, rolloff: 102, f1: 490, f2: 1510, f3: 2480, zcr: 0.11, mfcc: [1.1, 2.1, 3.1] } },
        spk2: { avgFeatures: { pitch: 300, centroid: 400, spread: 100, flatness: 0.8, rolloff: 200, f1: 800, f2: 2000, f3: 3500, zcr: 0.5, mfcc: [-1, -2, -3] } },
      };
      return matcher.matchByFeatures(probe, profiles);
    });
    expect(result).not.toBe(null);
    expect(result.id).toBe('spk1');
    expect(result.method).toBe('features');
    expect(result.similarity).toBeGreaterThan(0.8);
  });

  test('matchByFeatures returns null for empty profiles', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher();
      return matcher.matchByFeatures({ pitch: 150 }, {});
    });
    expect(result).toBe(null);
  });

  test('match() prefers embedding over features', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({
        embeddingThreshold: 0.5,
        featureThreshold: 0.5,
      });

      const probe = {
        embedding: new Float32Array([1, 0, 0, 0]),
        features: { pitch: 150, centroid: 200, mfcc: [1, 2] },
      };
      const profiles = {
        spk1: {
          embedding: new Float32Array([1, 0, 0, 0]),
          avgFeatures: { pitch: 150, centroid: 200, mfcc: [1, 2] },
        },
      };
      const r = matcher.match(probe, profiles);
      return r ? r.method : null;
    });
    expect(result).toBe('embedding');
  });

  test('match() falls back to features when no embedding match', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({
        embeddingThreshold: 0.99,
        featureThreshold: 0.5,
      });

      const probe = {
        embedding: new Float32Array([1, 0, 0, 0]),
        features: { pitch: 150, centroid: 200, spread: 50, flatness: 0.3, rolloff: 100, f1: 500, f2: 1500, f3: 2500, zcr: 0.1, mfcc: [1, 2, 3] },
      };
      const profiles = {
        spk1: {
          embedding: new Float32Array([0, 1, 0, 0]),
          avgFeatures: { pitch: 152, centroid: 198, spread: 49, flatness: 0.31, rolloff: 101, f1: 502, f2: 1498, f3: 2505, zcr: 0.11, mfcc: [1, 2, 3] },
        },
      };
      return matcher.match(probe, profiles);
    });
    expect(result).not.toBe(null);
    expect(result.method).toBe('features');
  });

  test('match() returns null when both methods fail', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({
        embeddingThreshold: 0.99,
        featureThreshold: 0.99,
      });
      const probe = {
        embedding: new Float32Array([1, 0, 0, 0]),
        features: { pitch: 150 },
      };
      const profiles = {
        spk1: {
          embedding: new Float32Array([0, 1, 0, 0]),
          avgFeatures: { pitch: 999 },
        },
      };
      return matcher.match(probe, profiles);
    });
    expect(result).toBe(null);
  });

  test('confidence is high above highConfidence threshold', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({
        embeddingThreshold: 0.5,
        embeddingHighConfidence: 0.8,
      });
      const probe = new Float32Array([1, 0, 0, 0]);
      const profiles = {
        spk1: { embedding: new Float32Array([1, 0, 0, 0]) },
      };
      return matcher.matchByEmbedding(probe, profiles);
    });
    expect(result.confidence).toBe('high');
    expect(result.similarity).toBeGreaterThan(0.99);
  });

  test('confidence is medium between threshold and highConfidence', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({
        embeddingThreshold: 0.3,
        embeddingHighConfidence: 0.95,
      });
      // cosine([1,1,0,0], [1,0,1,0]) = 0.5 → above 0.3, below 0.95
      const a = new Float32Array([1, 1, 0, 0]);
      const b = new Float32Array([1, 0, 1, 0]);
      let na = 0, nb = 0;
      for (let i = 0; i < 4; i++) { na += a[i]*a[i]; nb += b[i]*b[i]; }
      na = Math.sqrt(na); nb = Math.sqrt(nb);
      for (let i = 0; i < 4; i++) { a[i] /= na; b[i] /= nb; }

      const profiles = { spk1: { embedding: b } };
      return matcher.matchByEmbedding(a, profiles);
    });
    expect(result.confidence).toBe('medium');
  });

  test('works with Map-based profiles', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({ embeddingThreshold: 0.5 });
      const probe = new Float32Array([1, 0, 0, 0]);
      const profiles = new Map();
      profiles.set('spk1', { embedding: new Float32Array([1, 0, 0, 0]) });
      return matcher.matchByEmbedding(probe, profiles);
    });
    expect(result).not.toBe(null);
    expect(result.id).toBe('spk1');
  });

  test('opts returns a copy of configuration', async () => {
    const result = await page.evaluate(() => {
      const matcher = new SpeakerMatcher({ embeddingThreshold: 0.7 });
      const o = matcher.opts;
      o.embeddingThreshold = 999;
      return { original: matcher.opts.embeddingThreshold, tampered: o.embeddingThreshold };
    });
    expect(result.original).toBe(0.7);
    expect(result.tampered).toBe(999);
  });
});
