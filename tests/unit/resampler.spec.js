const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

const RESAMPLER_SRC = fs.readFileSync(
  path.resolve(__dirname, '../../src/js/audio/resampler.js'), 'utf-8'
);

// Strip the ES module `export` line so the code runs as a plain script
// and attaches classes to `window`.
const INJECTABLE = RESAMPLER_SRC
  .replace(/^export\s*\{[^}]*\}\s*;?\s*$/m, '')
  + '\nwindow.Resampler = Resampler;\n'
  + 'window.IntegerDecimator = IntegerDecimator;\n'
  + 'window.createResampler = createResampler;\n'
  + 'window.designLowpass = designLowpass;\n'
  + 'window.MODEL_RATE = MODEL_RATE;\n';

test.describe('Resampler module', () => {
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
      hasCreateResampler: typeof window.createResampler === 'function',
      hasResampler: typeof window.Resampler === 'function',
      hasIntegerDecimator: typeof window.IntegerDecimator === 'function',
      MODEL_RATE: window.MODEL_RATE,
    }));
    expect(result.hasCreateResampler).toBe(true);
    expect(result.hasResampler).toBe(true);
    expect(result.hasIntegerDecimator).toBe(true);
    expect(result.MODEL_RATE).toBe(16000);
  });

  test('createResampler returns IntegerDecimator for 48000→16000', async () => {
    const info = await page.evaluate(() => {
      const r = createResampler(48000, 16000);
      return { type: r.constructor.name, factor: r.factor, outputRate: r.outputRate };
    });
    expect(info.type).toBe('IntegerDecimator');
    expect(info.factor).toBe(3);
    expect(info.outputRate).toBe(16000);
  });

  test('createResampler returns passthrough for 16000→16000', async () => {
    const info = await page.evaluate(() => {
      const r = createResampler(16000, 16000);
      return r.info;
    });
    expect(info.passthrough).toBe(true);
  });

  test('createResampler returns Resampler for 44100→16000', async () => {
    const info = await page.evaluate(() => {
      const r = createResampler(44100, 16000);
      return { type: r.constructor.name, L: r.L, M: r.M };
    });
    expect(info.type).toBe('Resampler');
    // 16000/44100 = 160/441
    expect(info.L).toBe(160);
    expect(info.M).toBe(441);
  });

  test('passthrough preserves input exactly', async () => {
    const result = await page.evaluate(() => {
      const r = createResampler(16000, 16000);
      const input = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
      const output = r.process(input);
      return { length: output.length, values: Array.from(output) };
    });
    expect(result.length).toBe(5);
    for (let i = 0; i < 5; i++) {
      expect(Math.abs(result.values[i] - (i + 1) * 0.1)).toBeLessThan(1e-6);
    }
  });

  test('48→16 kHz decimation produces correct output length', async () => {
    const result = await page.evaluate(() => {
      const r = createResampler(48000, 16000);
      const input = new Float32Array(4800);
      for (let i = 0; i < 4800; i++) input[i] = Math.sin(2 * Math.PI * 440 * i / 48000);
      const output = r.process(input);
      return output.length;
    });
    expect(result).toBeGreaterThan(1500);
    expect(result).toBeLessThanOrEqual(1650);
  });

  test('48→16 kHz preserves a 440 Hz sine wave', async () => {
    const result = await page.evaluate(() => {
      const r = createResampler(48000, 16000);
      const N = 48000;
      const input = new Float32Array(N);
      for (let i = 0; i < N; i++) input[i] = Math.sin(2 * Math.PI * 440 * i / 48000);
      const output = r.process(input);

      let crossings = 0;
      const start = 200;
      for (let i = start + 1; i < output.length; i++) {
        if (output[i - 1] <= 0 && output[i] > 0) crossings++;
      }
      const duration = (output.length - start) / 16000;
      const measuredFreq = crossings / duration;

      let sumSq = 0;
      for (let i = start; i < output.length; i++) sumSq += output[i] * output[i];
      const rms = Math.sqrt(sumSq / (output.length - start));

      return { length: output.length, measuredFreq, rms };
    });

    expect(Math.abs(result.measuredFreq - 440)).toBeLessThan(5);
    expect(result.rms).toBeGreaterThan(0.6);
    expect(result.rms).toBeLessThan(0.8);
  });

  test('48→16 kHz attenuates frequencies above Nyquist (8 kHz)', async () => {
    const result = await page.evaluate(() => {
      const r = createResampler(48000, 16000);
      const N = 48000;
      const input = new Float32Array(N);
      for (let i = 0; i < N; i++) input[i] = Math.sin(2 * Math.PI * 12000 * i / 48000);
      const output = r.process(input);

      let sumSq = 0;
      const start = 300;
      for (let i = start; i < output.length; i++) sumSq += output[i] * output[i];
      const rms = Math.sqrt(sumSq / (output.length - start));
      return { rms };
    });

    expect(result.rms).toBeLessThan(0.02);
  });

  test('streaming: multiple process() calls produce continuous output', async () => {
    const result = await page.evaluate(() => {
      const r = createResampler(48000, 16000);
      const chunkSize = 2048;
      const numChunks = 10;
      let totalOut = 0;
      const allOutputs = [];

      for (let c = 0; c < numChunks; c++) {
        const input = new Float32Array(chunkSize);
        for (let i = 0; i < chunkSize; i++) {
          const t = (c * chunkSize + i) / 48000;
          input[i] = Math.sin(2 * Math.PI * 440 * t);
        }
        const out = r.process(input);
        totalOut += out.length;
        allOutputs.push(Array.from(out));
      }

      // Concatenate all chunks and measure signal quality vs reference sine
      const allSamples = [];
      for (const o of allOutputs) allSamples.push(...o);

      // Compare steady-state output against expected 440Hz sine at 16kHz
      // Skip transient at the start (filter delay)
      const skip = 200;
      let maxErr = 0;
      // Find best-fit phase by checking a few offsets
      let bestPhaseErr = Infinity;
      let bestPhase = 0;
      for (let p = 0; p < 36; p++) {
        const ph = p * Math.PI / 18;
        let err = 0;
        for (let i = skip; i < Math.min(skip + 100, allSamples.length); i++) {
          const expected = Math.sin(2 * Math.PI * 440 * i / 16000 + ph);
          err += Math.abs(allSamples[i] - expected);
        }
        if (err < bestPhaseErr) { bestPhaseErr = err; bestPhase = ph; }
      }
      for (let i = skip; i < allSamples.length; i++) {
        const expected = Math.sin(2 * Math.PI * 440 * i / 16000 + bestPhase);
        maxErr = Math.max(maxErr, Math.abs(allSamples[i] - expected));
      }

      const expectedTotal = Math.round(chunkSize * numChunks / 3);
      return { totalOut, expectedTotal, maxErr };
    });

    expect(Math.abs(result.totalOut - result.expectedTotal)).toBeLessThan(500);
    // Output should track a clean 440Hz sine with < 10% error
    expect(result.maxErr).toBeLessThan(0.1);
  });

  test('reset() clears state', async () => {
    const result = await page.evaluate(() => {
      const r = createResampler(48000, 16000);
      const input = new Float32Array(4800);
      for (let i = 0; i < 4800; i++) input[i] = 1.0;
      r.process(input);
      r.reset();

      let stateSum = 0;
      const st = r._state || r._overlap;
      if (st) for (let i = 0; i < st.length; i++) stateSum += Math.abs(st[i]);
      return { stateSum };
    });

    expect(result.stateSum).toBe(0);
  });

  test('IntegerDecimator throws on non-integer factor', async () => {
    const result = await page.evaluate(() => {
      try {
        new IntegerDecimator(2.5, 48000);
        return { threw: false };
      } catch (e) {
        return { threw: true, message: e.message };
      }
    });
    expect(result.threw).toBe(true);
  });

  test('Resampler throws on zero/negative rate', async () => {
    const result = await page.evaluate(() => {
      try {
        new Resampler(0, 16000);
        return { threw: false };
      } catch (e) {
        return { threw: true };
      }
    });
    expect(result.threw).toBe(true);
  });

  test('designLowpass creates symmetric filter with DC gain = 1', async () => {
    const result = await page.evaluate(() => {
      const h = designLowpass(0.5, 65);
      const symmetric = h.every((v, i) => Math.abs(v - h[h.length - 1 - i]) < 1e-10);
      let dcGain = 0;
      for (let i = 0; i < h.length; i++) dcGain += h[i];
      return { symmetric, dcGain: Math.abs(dcGain - 1), length: h.length };
    });

    expect(result.symmetric).toBe(true);
    expect(result.dcGain).toBeLessThan(1e-6);
    expect(result.length).toBe(65);
  });

  test('32000→16000 uses IntegerDecimator with factor 2', async () => {
    const info = await page.evaluate(() => {
      const r = createResampler(32000, 16000);
      return { type: r.constructor.name, factor: r.factor };
    });
    expect(info.type).toBe('IntegerDecimator');
    expect(info.factor).toBe(2);
  });
});
