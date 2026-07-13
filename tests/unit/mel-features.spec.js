const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

const SRC = fs.readFileSync(
  path.resolve(__dirname, '../../src/js/features/mel-features.js'), 'utf-8'
);

const INJECTABLE = SRC
  .replace(/^export\s*\{[^}]*\}\s*;?\s*$/m, '')
  + '\nwindow.MelFeatureExtractor = MelFeatureExtractor;\n'
  + 'window.hzToMel = hzToMel;\n'
  + 'window.melToHz = melToHz;\n'
  + 'window.buildMelFilterbank = buildMelFilterbank;\n'
  + 'window.dctII = dctII;\n'
  + 'window.MEL_DEFAULTS = MEL_DEFAULTS;\n';

test.describe('Mel Features module', () => {
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
      hasMelFeatureExtractor: typeof window.MelFeatureExtractor === 'function',
      hasHzToMel: typeof window.hzToMel === 'function',
      hasMelToHz: typeof window.melToHz === 'function',
      hasBuildMelFilterbank: typeof window.buildMelFilterbank === 'function',
      hasDctII: typeof window.dctII === 'function',
    }));
    expect(result.hasMelFeatureExtractor).toBe(true);
    expect(result.hasHzToMel).toBe(true);
    expect(result.hasMelToHz).toBe(true);
    expect(result.hasBuildMelFilterbank).toBe(true);
    expect(result.hasDctII).toBe(true);
  });

  test('hzToMel and melToHz are inverses', async () => {
    const result = await page.evaluate(() => {
      const freqs = [0, 300, 1000, 3000, 8000];
      const errors = freqs.map(f => Math.abs(melToHz(hzToMel(f)) - f));
      return { maxError: Math.max(...errors) };
    });
    expect(result.maxError).toBeLessThan(0.01);
  });

  test('hzToMel gives known values', async () => {
    const result = await page.evaluate(() => ({
      mel0: hzToMel(0),
      mel1000: hzToMel(1000),
    }));
    expect(result.mel0).toBe(0);
    // 2595 * log10(1 + 1000/700) ≈ 1000
    expect(Math.abs(result.mel1000 - 1000)).toBeLessThan(1);
  });

  test('mel filterbank has correct dimensions', async () => {
    const result = await page.evaluate(() => {
      const fb = buildMelFilterbank(40, 512, 16000, 64, 7600);
      return {
        numBands: fb.length,
        binLength: fb[0].length,
        expectedBins: 512 / 2 + 1,
      };
    });
    expect(result.numBands).toBe(40);
    expect(result.binLength).toBe(result.expectedBins);
  });

  test('mel filterbank filters are non-negative and triangular', async () => {
    const result = await page.evaluate(() => {
      const fb = buildMelFilterbank(40, 512, 16000, 64, 7600);
      let anyNegative = false;
      let allHavePeak = true;
      for (let m = 0; m < fb.length; m++) {
        let maxVal = 0;
        for (let k = 0; k < fb[m].length; k++) {
          if (fb[m][k] < -1e-10) anyNegative = true;
          if (fb[m][k] > maxVal) maxVal = fb[m][k];
        }
        if (maxVal < 0.5) allHavePeak = false;
      }
      return { anyNegative, allHavePeak };
    });
    expect(result.anyNegative).toBe(false);
    expect(result.allHavePeak).toBe(true);
  });

  test('DCT-II of constant input gives energy in coeff 0 only', async () => {
    const result = await page.evaluate(() => {
      const input = new Float32Array(40).fill(1.0);
      const coeffs = dctII(input, 13);
      return {
        c0: coeffs[0],
        restMax: Math.max(...Array.from(coeffs).slice(1).map(Math.abs)),
      };
    });
    expect(result.c0).toBeGreaterThan(30);
    expect(result.restMax).toBeLessThan(1e-6);
  });

  test('logMelSpectrogram returns correct shape for 1s signal', async () => {
    const result = await page.evaluate(() => {
      const ext = new MelFeatureExtractor();
      const sr = 16000;
      const pcm = new Float32Array(sr);
      for (let i = 0; i < sr; i++) {
        pcm[i] = 0.3 * Math.sin(2 * Math.PI * 440 * i / sr);
      }
      const { logMel, numFrames, numBands } = ext.logMelSpectrogram(pcm);
      return { numFrames, numBands, firstFrameLen: logMel[0].length };
    });
    // (16000 - 512) / 160 + 1 = 97.something → 97 frames
    expect(result.numFrames).toBeGreaterThan(90);
    expect(result.numFrames).toBeLessThan(110);
    expect(result.numBands).toBe(40);
    expect(result.firstFrameLen).toBe(40);
  });

  test('logMelSpectrogram concentrates energy in correct bands for 440Hz', async () => {
    const result = await page.evaluate(() => {
      const ext = new MelFeatureExtractor();
      const sr = 16000;
      const pcm = new Float32Array(sr);
      for (let i = 0; i < sr; i++) {
        pcm[i] = 0.5 * Math.sin(2 * Math.PI * 440 * i / sr);
      }
      const { logMel, numFrames } = ext.logMelSpectrogram(pcm);

      // Average log-mel energy across frames (skip first few for pre-emphasis transient)
      const avg = new Float32Array(40);
      const start = 5;
      for (let f = start; f < numFrames; f++) {
        for (let b = 0; b < 40; b++) avg[b] += logMel[f][b];
      }
      for (let b = 0; b < 40; b++) avg[b] /= (numFrames - start);

      // Find peak band
      let peakBand = 0, peakVal = -Infinity;
      for (let b = 0; b < 40; b++) {
        if (avg[b] > peakVal) { peakVal = avg[b]; peakBand = b; }
      }

      return { peakBand, peakVal };
    });
    // 440Hz in mel scale with fMin=64, fMax=7600, 40 bands
    // should peak in the lower bands (roughly band 5-12)
    expect(result.peakBand).toBeGreaterThan(2);
    expect(result.peakBand).toBeLessThan(20);
  });

  test('mfcc returns correct dimensions', async () => {
    const result = await page.evaluate(() => {
      const ext = new MelFeatureExtractor({ numMfcc: 13 });
      const sr = 16000;
      const pcm = new Float32Array(sr);
      for (let i = 0; i < sr; i++) {
        pcm[i] = 0.3 * Math.sin(2 * Math.PI * 200 * i / sr);
      }
      const { mfcc, numFrames, numCoeffs } = ext.mfcc(pcm);
      return {
        numFrames,
        numCoeffs,
        firstLen: mfcc[0].length,
      };
    });
    expect(result.numFrames).toBeGreaterThan(90);
    expect(result.numCoeffs).toBe(13);
    expect(result.firstLen).toBe(13);
  });

  test('mfccMean returns a single vector of correct size', async () => {
    const result = await page.evaluate(() => {
      const ext = new MelFeatureExtractor({ numMfcc: 13 });
      const pcm = new Float32Array(8000);
      for (let i = 0; i < 8000; i++) {
        pcm[i] = 0.3 * Math.sin(2 * Math.PI * 300 * i / 16000);
      }
      const mean = ext.mfccMean(pcm);
      return { length: mean.length, c0: mean[0] };
    });
    expect(result.length).toBe(13);
    expect(result.c0).not.toBe(0);
  });

  test('mfcc c0 (energy) distinguishes loud vs quiet signals', async () => {
    const result = await page.evaluate(() => {
      const ext = new MelFeatureExtractor({ numMfcc: 13 });
      const sr = 16000;

      const loud = new Float32Array(sr);
      const quiet = new Float32Array(sr);
      for (let i = 0; i < sr; i++) {
        loud[i] = 0.8 * Math.sin(2 * Math.PI * 300 * i / sr);
        quiet[i] = 0.01 * Math.sin(2 * Math.PI * 300 * i / sr);
      }

      const loudMean = ext.mfccMean(loud);
      const quietMean = ext.mfccMean(quiet);
      return { loudC0: loudMean[0], quietC0: quietMean[0] };
    });
    expect(result.loudC0).toBeGreaterThan(result.quietC0);
  });

  test('deltas produces correct shape and non-zero values', async () => {
    const result = await page.evaluate(() => {
      const ext = new MelFeatureExtractor({ numMfcc: 13 });
      const pcm = new Float32Array(16000);
      // Chirp: frequency changes over time
      for (let i = 0; i < 16000; i++) {
        const t = i / 16000;
        const freq = 200 + 2000 * t;
        pcm[i] = 0.3 * Math.sin(2 * Math.PI * freq * t);
      }
      const { mfcc, numFrames } = ext.mfcc(pcm);
      const d = ext.deltas(mfcc);
      const dd = ext.deltas(d);

      let nonZeroDelta = 0;
      for (let f = 2; f < d.length - 2; f++) {
        for (let c = 0; c < d[f].length; c++) {
          if (Math.abs(d[f][c]) > 1e-6) nonZeroDelta++;
        }
      }

      return {
        deltaLen: d.length,
        deltaFrameSize: d[0].length,
        ddLen: dd.length,
        nonZeroDelta,
      };
    });
    expect(result.deltaLen).toBeGreaterThan(90);
    expect(result.deltaFrameSize).toBe(13);
    expect(result.ddLen).toBe(result.deltaLen);
    expect(result.nonZeroDelta).toBeGreaterThan(100);
  });

  test('different signals produce different MFCC vectors', async () => {
    const result = await page.evaluate(() => {
      const ext = new MelFeatureExtractor({ numMfcc: 13 });
      const sr = 16000;

      // Low-frequency signal
      const low = new Float32Array(sr);
      for (let i = 0; i < sr; i++) {
        low[i] = 0.3 * Math.sin(2 * Math.PI * 200 * i / sr);
      }

      // High-frequency signal
      const high = new Float32Array(sr);
      for (let i = 0; i < sr; i++) {
        high[i] = 0.3 * Math.sin(2 * Math.PI * 3000 * i / sr);
      }

      const lowMfcc = ext.mfccMean(low);
      const highMfcc = ext.mfccMean(high);

      // Euclidean distance
      let dist = 0;
      for (let c = 0; c < 13; c++) {
        dist += (lowMfcc[c] - highMfcc[c]) ** 2;
      }
      dist = Math.sqrt(dist);

      return { dist };
    });
    expect(result.dist).toBeGreaterThan(5);
  });

  test('empty input produces empty output', async () => {
    const result = await page.evaluate(() => {
      const ext = new MelFeatureExtractor();
      const { logMel, numFrames } = ext.logMelSpectrogram(new Float32Array(0));
      const { mfcc } = ext.mfcc(new Float32Array(0));
      const mean = ext.mfccMean(new Float32Array(100)); // too short for one frame
      return { logMelFrames: numFrames, mfccLen: mfcc.length, meanAllZero: mean.every(v => v === 0) };
    });
    expect(result.logMelFrames).toBe(0);
    expect(result.mfccLen).toBe(0);
  });

  test('pre-emphasis boosts high frequencies', async () => {
    const result = await page.evaluate(() => {
      const sr = 16000;
      // Compare spectrogram with and without pre-emphasis
      const extWith = new MelFeatureExtractor({ preEmphasis: 0.97 });
      const extWithout = new MelFeatureExtractor({ preEmphasis: 0 });

      // Signal with both low and high frequency content
      const pcm = new Float32Array(sr);
      for (let i = 0; i < sr; i++) {
        pcm[i] = 0.3 * Math.sin(2 * Math.PI * 200 * i / sr)
               + 0.1 * Math.sin(2 * Math.PI * 4000 * i / sr);
      }

      const { logMel: withPre } = extWith.logMelSpectrogram(pcm);
      const { logMel: withoutPre } = extWithout.logMelSpectrogram(pcm);

      // Average the high bands (top 10) across middle frames
      let highWith = 0, highWithout = 0;
      const start = 10, end = 80;
      for (let f = start; f < end; f++) {
        for (let b = 30; b < 40; b++) {
          highWith += withPre[f][b];
          highWithout += withoutPre[f][b];
        }
      }
      highWith /= (end - start) * 10;
      highWithout /= (end - start) * 10;

      return { highWith, highWithout };
    });
    // Pre-emphasis should boost higher bands
    expect(result.highWith).toBeGreaterThan(result.highWithout);
  });
});
