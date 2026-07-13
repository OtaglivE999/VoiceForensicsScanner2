const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

const SRC = fs.readFileSync(
  path.resolve(__dirname, '../../src/js/features/speaker-embedding.js'), 'utf-8'
);

const INJECTABLE = SRC
  .replace(/^export\s*\{[^}]*\}\s*;?\s*$/m, '')
  + '\nwindow.SpeakerEmbeddingExtractor = SpeakerEmbeddingExtractor;\n'
  + 'window.MODEL_MANIFEST = MODEL_MANIFEST;\n'
  + 'window.EMBEDDING_DIM = EMBEDDING_DIM;\n';

test.describe('Speaker Embedding module', () => {
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
      hasSpeakerEmbeddingExtractor: typeof window.SpeakerEmbeddingExtractor === 'function',
      hasModelManifest: typeof window.MODEL_MANIFEST === 'object',
      hasEmbeddingDim: typeof window.EMBEDDING_DIM === 'number',
    }));
    expect(result.hasSpeakerEmbeddingExtractor).toBe(true);
    expect(result.hasModelManifest).toBe(true);
    expect(result.hasEmbeddingDim).toBe(true);
  });

  test('default manifest indicates no model available', async () => {
    const result = await page.evaluate(() => {
      return {
        modelId: MODEL_MANIFEST.modelId,
        status: MODEL_MANIFEST.status,
        sha256: MODEL_MANIFEST.sha256,
      };
    });
    expect(result.modelId).toBe(null);
    expect(result.status).toBe('no-model-available');
    expect(result.sha256).toBe(null);
  });

  test('isConfigured returns false with default manifest', async () => {
    const result = await page.evaluate(() => {
      const ext = new SpeakerEmbeddingExtractor();
      return ext.isConfigured;
    });
    expect(result).toBe(false);
  });

  test('isConfigured returns true when modelId and sha256 are set', async () => {
    const result = await page.evaluate(() => {
      const ext = new SpeakerEmbeddingExtractor({
        modelId: 'test-model',
        sha256: 'abc123',
      });
      return ext.isConfigured;
    });
    expect(result).toBe(true);
  });

  test('status reports no-model-configured with default manifest', async () => {
    const result = await page.evaluate(() => {
      const ext = new SpeakerEmbeddingExtractor();
      return ext.status;
    });
    expect(result).toBe('no-model-configured');
  });

  test('status reports not-loaded when configured but not loaded', async () => {
    const result = await page.evaluate(() => {
      const ext = new SpeakerEmbeddingExtractor({
        modelId: 'test-model',
        sha256: 'abc123',
      });
      return ext.status;
    });
    expect(result).toBe('not-loaded');
  });

  test('load throws when no model is configured', async () => {
    const result = await page.evaluate(async () => {
      const ext = new SpeakerEmbeddingExtractor();
      try {
        await ext.load(new ArrayBuffer(10));
        return { threw: false };
      } catch (e) {
        return { threw: true, message: e.message };
      }
    });
    expect(result.threw).toBe(true);
    expect(result.message).toContain('No model configured');
  });

  test('load throws on SHA-256 mismatch or insecure context', async () => {
    const result = await page.evaluate(async () => {
      const ext = new SpeakerEmbeddingExtractor({
        modelId: 'test-model',
        sha256: '0000000000000000000000000000000000000000000000000000000000000000',
      });
      const fakeBytes = new Uint8Array([1, 2, 3, 4]).buffer;
      try {
        await ext.load(fakeBytes);
        return { threw: false };
      } catch (e) {
        return { threw: true, message: e.message };
      }
    });
    expect(result.threw).toBe(true);
    // In insecure contexts (about:blank), crypto.subtle is unavailable;
    // in secure contexts, the SHA-256 mismatch is caught
    expect(
      result.message.includes('SHA-256 mismatch') ||
      result.message.includes('crypto.subtle unavailable')
    ).toBe(true);
  });

  test('extract throws when model is not loaded', async () => {
    const result = await page.evaluate(async () => {
      const ext = new SpeakerEmbeddingExtractor({
        modelId: 'test-model',
        sha256: 'abc123',
      });
      try {
        await ext.extract([new Float32Array(40)]);
        return { threw: false };
      } catch (e) {
        return { threw: true, message: e.message };
      }
    });
    expect(result.threw).toBe(true);
    expect(result.message).toContain('Model not loaded');
  });

  test('cosineSimilarity of identical vectors is 1', async () => {
    const result = await page.evaluate(() => {
      const v = new Float32Array([0.5, 0.3, 0.8, 0.1]);
      return SpeakerEmbeddingExtractor.cosineSimilarity(v, v);
    });
    expect(Math.abs(result - 1.0)).toBeLessThan(1e-6);
  });

  test('cosineSimilarity of orthogonal vectors is 0', async () => {
    const result = await page.evaluate(() => {
      const a = new Float32Array([1, 0, 0, 0]);
      const b = new Float32Array([0, 1, 0, 0]);
      return SpeakerEmbeddingExtractor.cosineSimilarity(a, b);
    });
    expect(Math.abs(result)).toBeLessThan(1e-6);
  });

  test('cosineSimilarity of opposite vectors is -1', async () => {
    const result = await page.evaluate(() => {
      const a = new Float32Array([1, 2, 3]);
      const b = new Float32Array([-1, -2, -3]);
      return SpeakerEmbeddingExtractor.cosineSimilarity(a, b);
    });
    expect(Math.abs(result + 1.0)).toBeLessThan(1e-6);
  });

  test('cosineSimilarity throws on dimension mismatch', async () => {
    const result = await page.evaluate(() => {
      try {
        const a = new Float32Array([1, 2, 3]);
        const b = new Float32Array([1, 2]);
        SpeakerEmbeddingExtractor.cosineSimilarity(a, b);
        return { threw: false };
      } catch (e) {
        return { threw: true, message: e.message };
      }
    });
    expect(result.threw).toBe(true);
    expect(result.message).toContain('dimensions must match');
  });

  test('manifest returns a copy of configuration', async () => {
    const result = await page.evaluate(() => {
      const ext = new SpeakerEmbeddingExtractor({
        modelId: 'ecapa-tdnn-v1',
        version: '1.0.0',
        sha256: 'deadbeef',
      });
      const m = ext.manifest;
      m.modelId = 'tampered';
      return {
        original: ext.manifest.modelId,
        tampered: m.modelId,
      };
    });
    expect(result.original).toBe('ecapa-tdnn-v1');
    expect(result.tampered).toBe('tampered');
  });

  test('EMBEDDING_DIM is 192', async () => {
    const result = await page.evaluate(() => EMBEDDING_DIM);
    expect(result).toBe(192);
  });

  test('dispose resets loaded state', async () => {
    const result = await page.evaluate(() => {
      const ext = new SpeakerEmbeddingExtractor();
      ext._loaded = true;
      ext._session = {};
      ext.dispose();
      return { loaded: ext.isLoaded, status: ext.status };
    });
    expect(result.loaded).toBe(false);
  });
});
