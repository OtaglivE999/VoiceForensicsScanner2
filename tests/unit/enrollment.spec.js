const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

const SRC = fs.readFileSync(
  path.resolve(__dirname, '../../src/js/matching/enrollment.js'), 'utf-8'
);

const INJECTABLE = SRC
  .replace(/^export\s*\{[^}]*\}\s*;?\s*$/m, '')
  + '\nwindow.EnrollmentManager = EnrollmentManager;\n'
  + 'window.ENROLLMENT_DEFAULTS = ENROLLMENT_DEFAULTS;\n';

function makeEmb(seed) {
  const v = [];
  let s = seed;
  for (let i = 0; i < 8; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    v.push((s / 0x7fffffff) * 2 - 1);
  }
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm);
  return v.map(x => x / norm);
}

test.describe('Enrollment Manager', () => {
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
      has: typeof window.EnrollmentManager === 'function',
      hasDefaults: typeof window.ENROLLMENT_DEFAULTS === 'object',
    }));
    expect(result.has).toBe(true);
    expect(result.hasDefaults).toBe(true);
  });

  test('first observation creates a new candidate', async () => {
    const result = await page.evaluate(() => {
      const mgr = new EnrollmentManager({ minObservations: 3 });
      const r = mgr.submit({
        embedding: new Float32Array([1, 0, 0, 0]),
        durationMs: 500,
        timestamp: 1000,
      });
      return { status: r.status, hasCandidateId: !!r.candidateId, count: mgr.candidateCount };
    });
    expect(result.status).toBe('new-candidate');
    expect(result.hasCandidateId).toBe(true);
    expect(result.count).toBe(1);
  });

  test('second similar observation accumulates', async () => {
    const result = await page.evaluate(() => {
      const mgr = new EnrollmentManager({
        minObservations: 3,
        minTotalSpeechMs: 1000,
        consistencyThreshold: 0.5,
      });

      mgr.submit({ embedding: new Float32Array([1, 0, 0, 0]), durationMs: 500, timestamp: 1000 });
      const r = mgr.submit({ embedding: new Float32Array([0.99, 0.1, 0, 0]), durationMs: 500, timestamp: 2000 });
      return { status: r.status, observations: r.observations };
    });
    expect(result.status).toBe('accumulating');
    expect(result.observations).toBe(2);
  });

  test('enrolls after enough consistent observations', async () => {
    const result = await page.evaluate(() => {
      const mgr = new EnrollmentManager({
        minObservations: 3,
        minTotalSpeechMs: 1000,
        consistencyThreshold: 0.5,
      });

      const emb = new Float32Array([1, 0, 0, 0]);
      mgr.submit({ embedding: emb, durationMs: 400, timestamp: 1000 });
      mgr.submit({ embedding: new Float32Array([0.99, 0.1, 0, 0]), durationMs: 400, timestamp: 2000 });
      const r = mgr.submit({ embedding: new Float32Array([0.98, 0.15, 0, 0]), durationMs: 400, timestamp: 3000 });

      return {
        status: r.status,
        hasProfile: !!r.profile,
        obsCount: r.profile ? r.profile.observationCount : 0,
        hasEmbedding: r.profile ? !!r.profile.embedding : false,
        candidateCount: mgr.candidateCount,
      };
    });
    expect(result.status).toBe('enrolled');
    expect(result.hasProfile).toBe(true);
    expect(result.obsCount).toBe(3);
    expect(result.hasEmbedding).toBe(true);
    expect(result.candidateCount).toBe(0);
  });

  test('different speakers create separate candidates', async () => {
    const result = await page.evaluate(() => {
      const mgr = new EnrollmentManager({
        minObservations: 3,
        consistencyThreshold: 0.5,
      });

      mgr.submit({ embedding: new Float32Array([1, 0, 0, 0]), durationMs: 500, timestamp: 1000 });
      mgr.submit({ embedding: new Float32Array([0, 1, 0, 0]), durationMs: 500, timestamp: 2000 });
      return { candidateCount: mgr.candidateCount };
    });
    expect(result.candidateCount).toBe(2);
  });

  test('profile embedding is L2 normalized', async () => {
    const result = await page.evaluate(() => {
      const mgr = new EnrollmentManager({
        minObservations: 2,
        minTotalSpeechMs: 500,
        consistencyThreshold: 0.3,
      });

      mgr.submit({ embedding: new Float32Array([1, 0, 0, 0]), durationMs: 300, timestamp: 1000 });
      const r = mgr.submit({ embedding: new Float32Array([0.9, 0.2, 0, 0]), durationMs: 300, timestamp: 2000 });

      if (!r.profile || !r.profile.embedding) return { norm: -1 };
      let n = 0;
      for (let i = 0; i < r.profile.embedding.length; i++) {
        n += r.profile.embedding[i] * r.profile.embedding[i];
      }
      return { norm: Math.sqrt(n) };
    });
    expect(Math.abs(result.norm - 1.0)).toBeLessThan(1e-6);
  });

  test('profile contains averaged features', async () => {
    const result = await page.evaluate(() => {
      const mgr = new EnrollmentManager({
        minObservations: 2,
        minTotalSpeechMs: 500,
        consistencyThreshold: 0.3,
      });

      const f1 = { pitch: 100, centroid: 200, spread: 50, f1: 500, f2: 1500, f3: 2500 };
      const f2 = { pitch: 110, centroid: 210, spread: 55, f1: 510, f2: 1510, f3: 2510 };

      mgr.submit({ features: f1, durationMs: 300, timestamp: 1000 });
      const r = mgr.submit({ features: f2, durationMs: 300, timestamp: 2000 });

      return {
        status: r.status,
        avgPitch: r.profile ? r.profile.avgFeatures.pitch : null,
      };
    });
    expect(result.status).toBe('enrolled');
    expect(result.avgPitch).toBe(105);
  });

  test('expired candidates are pruned', async () => {
    const result = await page.evaluate(() => {
      const mgr = new EnrollmentManager({
        minObservations: 5,
        candidateTimeoutMs: 5000,
      });

      mgr.submit({ embedding: new Float32Array([1, 0, 0, 0]), durationMs: 500, timestamp: 1000 });
      const countBefore = mgr.candidateCount;

      // Submit far in the future
      mgr.submit({ embedding: new Float32Array([0, 0, 0, 1]), durationMs: 500, timestamp: 100000 });
      const countAfter = mgr.candidateCount;

      return { countBefore, countAfter };
    });
    expect(result.countBefore).toBe(1);
    // Old candidate pruned, new one created
    expect(result.countAfter).toBe(1);
  });

  test('reset clears all candidates', async () => {
    const result = await page.evaluate(() => {
      const mgr = new EnrollmentManager();
      mgr.submit({ embedding: new Float32Array([1, 0]), durationMs: 500, timestamp: 1000 });
      mgr.submit({ embedding: new Float32Array([0, 1]), durationMs: 500, timestamp: 2000 });
      mgr.reset();
      return mgr.candidateCount;
    });
    expect(result).toBe(0);
  });

  test('does not enroll before minTotalSpeechMs', async () => {
    const result = await page.evaluate(() => {
      const mgr = new EnrollmentManager({
        minObservations: 2,
        minTotalSpeechMs: 2000,
        consistencyThreshold: 0.3,
      });

      mgr.submit({ embedding: new Float32Array([1, 0, 0, 0]), durationMs: 100, timestamp: 1000 });
      const r = mgr.submit({ embedding: new Float32Array([0.99, 0.1, 0, 0]), durationMs: 100, timestamp: 2000 });
      return r.status;
    });
    expect(result).toBe('accumulating');
  });

  test('getCandidateIds returns active candidate IDs', async () => {
    const result = await page.evaluate(() => {
      const mgr = new EnrollmentManager();
      mgr.submit({ embedding: new Float32Array([1, 0]), durationMs: 500, timestamp: 1000 });
      mgr.submit({ embedding: new Float32Array([0, 1]), durationMs: 500, timestamp: 2000 });
      return mgr.getCandidateIds();
    });
    expect(result.length).toBe(2);
    expect(result[0]).toMatch(/^candidate-/);
  });
});
