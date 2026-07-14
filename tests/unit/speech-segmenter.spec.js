const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

const SRC = fs.readFileSync(
  path.resolve(__dirname, '../../src/js/audio/speech-segmenter.js'), 'utf-8'
);

const INJECTABLE = SRC
  .replace(/^export\s*\{[^}]*\}\s*;?\s*$/m, '')
  + '\nwindow.FrameVAD = FrameVAD;\n'
  + 'window.SpeechSegmenter = SpeechSegmenter;\n'
  + 'window.DEFAULT_OPTS = DEFAULT_OPTS;\n';

test.describe('Speech Segmenter module', () => {
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
      hasFrameVAD: typeof window.FrameVAD === 'function',
      hasSpeechSegmenter: typeof window.SpeechSegmenter === 'function',
      hasDefaultOpts: typeof window.DEFAULT_OPTS === 'object',
    }));
    expect(result.hasFrameVAD).toBe(true);
    expect(result.hasSpeechSegmenter).toBe(true);
    expect(result.hasDefaultOpts).toBe(true);
  });

  test('FrameVAD classifies silence as non-speech', async () => {
    const result = await page.evaluate(() => {
      const vad = new FrameVAD(16000);
      const silence = new Float32Array(512);
      const r = vad.classify(silence);
      return { speech: r.speech, energy: r.energy, zcr: r.zcr };
    });
    expect(result.speech).toBe(false);
    expect(result.energy).toBeLessThan(0.001);
  });

  test('FrameVAD classifies loud speech-like signal as speech', async () => {
    const result = await page.evaluate(() => {
      const vad = new FrameVAD(16000);
      // Warm up noise floor with some silence
      for (let i = 0; i < 30; i++) {
        vad.classify(new Float32Array(512));
      }
      // Generate a signal with speech-like characteristics:
      // voiced pitch ~150Hz with moderate ZCR
      const frame = new Float32Array(512);
      for (let i = 0; i < 512; i++) {
        const t = i / 16000;
        frame[i] = 0.3 * Math.sin(2 * Math.PI * 150 * t)
                 + 0.1 * Math.sin(2 * Math.PI * 300 * t)
                 + 0.05 * Math.sin(2 * Math.PI * 900 * t);
      }
      const r = vad.classify(frame);
      return { speech: r.speech, energy: r.energy, zcr: r.zcr, flatness: r.flatness };
    });
    expect(result.speech).toBe(true);
    expect(result.energy).toBeGreaterThan(0.01);
  });

  test('FrameVAD rejects white noise (high spectral flatness)', async () => {
    const result = await page.evaluate(() => {
      const vad = new FrameVAD(16000);
      // Warm up noise floor
      for (let i = 0; i < 30; i++) {
        vad.classify(new Float32Array(512));
      }
      // White noise has high spectral flatness
      const frame = new Float32Array(512);
      let seed = 42;
      for (let i = 0; i < 512; i++) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        frame[i] = 0.3 * ((seed / 0x7fffffff) * 2 - 1);
      }
      const r = vad.classify(frame);
      return { speech: r.speech, flatness: r.flatness };
    });
    expect(result.flatness).toBeGreaterThan(0.5);
    expect(result.speech).toBe(false);
  });

  test('SpeechSegmenter returns empty for silence', async () => {
    const result = await page.evaluate(() => {
      const seg = new SpeechSegmenter(16000);
      const silence = new Float32Array(16000); // 1 second silence
      const segments = seg.process(silence, 0);
      return segments.length;
    });
    expect(result).toBe(0);
  });

  test('SpeechSegmenter emits segment for sustained speech', async () => {
    const result = await page.evaluate(() => {
      const seg = new SpeechSegmenter(16000, {
        minSpeechMs: 200,
        minSilenceMs: 300,
        hangoverFrames: 4,
      });

      // 0.5s silence to calibrate noise floor
      const silence = new Float32Array(8000);
      seg.process(silence, 0);

      // 1s speech-like signal (150Hz + harmonics)
      const speech = new Float32Array(16000);
      for (let i = 0; i < 16000; i++) {
        const t = i / 16000;
        speech[i] = 0.3 * Math.sin(2 * Math.PI * 150 * t)
                  + 0.1 * Math.sin(2 * Math.PI * 300 * t)
                  + 0.05 * Math.sin(2 * Math.PI * 900 * t);
      }
      const s1 = seg.process(speech, 500);

      // Follow with 1s silence to trigger segment end
      const silence2 = new Float32Array(16000);
      const s2 = seg.process(silence2, 1500);

      const all = [...s1, ...s2];
      return {
        count: all.length,
        durationMs: all.length > 0 ? all[0].durationMs : 0,
        hasPcm: all.length > 0 ? all[0].pcm.length > 0 : false,
        frameCount: all.length > 0 ? all[0].frameCount : 0,
      };
    });
    expect(result.count).toBe(1);
    expect(result.durationMs).toBeGreaterThan(200);
    expect(result.hasPcm).toBe(true);
    expect(result.frameCount).toBeGreaterThan(5);
  });

  test('SpeechSegmenter rejects segments shorter than minSpeechMs', async () => {
    const result = await page.evaluate(() => {
      const seg = new SpeechSegmenter(16000, {
        minSpeechMs: 500,
        minSilenceMs: 200,
        hangoverFrames: 2,
      });

      // Calibrate noise floor
      seg.process(new Float32Array(8000), 0);

      // Very short speech burst (50ms = 800 samples at 16kHz)
      const speech = new Float32Array(800);
      for (let i = 0; i < 800; i++) {
        speech[i] = 0.3 * Math.sin(2 * Math.PI * 150 * i / 16000);
      }
      const s1 = seg.process(speech, 500);

      // Silence to end
      const s2 = seg.process(new Float32Array(16000), 550);

      return [...s1, ...s2].length;
    });
    expect(result).toBe(0);
  });

  test('SpeechSegmenter bridges brief pauses with hangover', async () => {
    const result = await page.evaluate(() => {
      const seg = new SpeechSegmenter(16000, {
        minSpeechMs: 200,
        minSilenceMs: 500,
        hangoverFrames: 15,
      });

      // Calibrate
      seg.process(new Float32Array(8000), 0);

      // 400ms speech
      const speech1 = new Float32Array(6400);
      for (let i = 0; i < 6400; i++) {
        speech1[i] = 0.3 * Math.sin(2 * Math.PI * 150 * i / 16000);
      }
      seg.process(speech1, 500);

      // Brief 100ms pause (should be bridged by hangover)
      seg.process(new Float32Array(1600), 900);

      // 400ms more speech
      const speech2 = new Float32Array(6400);
      for (let i = 0; i < 6400; i++) {
        speech2[i] = 0.3 * Math.sin(2 * Math.PI * 150 * i / 16000);
      }
      seg.process(speech2, 1000);

      // Long silence to emit
      const segments = seg.process(new Float32Array(16000), 1400);
      const flushed = seg.flush();
      const all = [...segments, ...flushed];

      return {
        count: all.length,
        totalPcmSamples: all.reduce((s, x) => s + x.pcm.length, 0),
      };
    });
    // Should be a single bridged segment, not two
    expect(result.count).toBe(1);
    expect(result.totalPcmSamples).toBeGreaterThan(10000);
  });

  test('SpeechSegmenter force-emits at maxSpeechMs', async () => {
    const result = await page.evaluate(() => {
      const seg = new SpeechSegmenter(16000, {
        minSpeechMs: 100,
        maxSpeechMs: 1000,
        minSilenceMs: 500,
        hangoverFrames: 4,
      });

      seg.process(new Float32Array(8000), 0);

      // 3 seconds of continuous speech
      const segments = [];
      for (let chunk = 0; chunk < 6; chunk++) {
        const speech = new Float32Array(8000);
        for (let i = 0; i < 8000; i++) {
          const t = (chunk * 8000 + i) / 16000;
          speech[i] = 0.3 * Math.sin(2 * Math.PI * 150 * t)
                    + 0.1 * Math.sin(2 * Math.PI * 300 * t);
        }
        const s = seg.process(speech, 500 + chunk * 500);
        segments.push(...s);
      }

      return {
        count: segments.length,
        firstDuration: segments.length > 0 ? segments[0].durationMs : 0,
      };
    });
    // Should have emitted at least 2 segments due to 1s max
    expect(result.count).toBeGreaterThanOrEqual(2);
    expect(result.firstDuration).toBeLessThanOrEqual(1200);
  });

  test('flush() emits in-progress segment', async () => {
    const result = await page.evaluate(() => {
      const seg = new SpeechSegmenter(16000, {
        minSpeechMs: 200,
        minSilenceMs: 500,
        hangoverFrames: 4,
      });

      seg.process(new Float32Array(8000), 0);

      // 500ms speech, no trailing silence
      const speech = new Float32Array(8000);
      for (let i = 0; i < 8000; i++) {
        speech[i] = 0.3 * Math.sin(2 * Math.PI * 150 * i / 16000);
      }
      seg.process(speech, 500);

      const flushed = seg.flush();
      return {
        count: flushed.length,
        durationMs: flushed.length > 0 ? flushed[0].durationMs : 0,
      };
    });
    expect(result.count).toBe(1);
    expect(result.durationMs).toBeGreaterThan(200);
  });

  test('reset() clears all state', async () => {
    const result = await page.evaluate(() => {
      const seg = new SpeechSegmenter(16000);

      // Build up some state
      const speech = new Float32Array(4800);
      for (let i = 0; i < 4800; i++) {
        speech[i] = 0.3 * Math.sin(2 * Math.PI * 150 * i / 16000);
      }
      seg.process(new Float32Array(8000), 0);
      seg.process(speech, 500);

      seg.reset();

      // After reset, flush should produce nothing
      const flushed = seg.flush();
      return { count: flushed.length };
    });
    expect(result.count).toBe(0);
  });

  test('info returns configuration', async () => {
    const result = await page.evaluate(() => {
      const seg = new SpeechSegmenter(16000, { minSpeechMs: 400 });
      return seg.info;
    });
    expect(result.sampleRate).toBe(16000);
    expect(result.minSpeechMs).toBe(400);
    expect(result.frameSize).toBeGreaterThanOrEqual(256);
  });

  test('SpeechSegmenter handles streaming chunks correctly', async () => {
    const result = await page.evaluate(() => {
      const seg = new SpeechSegmenter(16000, {
        minSpeechMs: 200,
        minSilenceMs: 400,
        hangoverFrames: 6,
      });

      // Feed in small streaming chunks (like from AudioWorklet)
      const chunkSize = 128;
      const allSegments = [];

      // 500ms silence calibration
      for (let i = 0; i < Math.ceil(8000 / chunkSize); i++) {
        allSegments.push(...seg.process(new Float32Array(chunkSize), i * 8));
      }

      // 600ms speech in small chunks
      for (let i = 0; i < Math.ceil(9600 / chunkSize); i++) {
        const chunk = new Float32Array(chunkSize);
        for (let j = 0; j < chunkSize; j++) {
          const t = (i * chunkSize + j) / 16000;
          chunk[j] = 0.3 * Math.sin(2 * Math.PI * 150 * t)
                   + 0.1 * Math.sin(2 * Math.PI * 300 * t);
        }
        allSegments.push(...seg.process(chunk, 500 + i * 8));
      }

      // 800ms silence to trigger end
      for (let i = 0; i < Math.ceil(12800 / chunkSize); i++) {
        allSegments.push(...seg.process(new Float32Array(chunkSize), 1100 + i * 8));
      }

      return {
        count: allSegments.length,
        hasPcm: allSegments.length > 0 && allSegments[0].pcm.length > 0,
      };
    });
    expect(result.count).toBe(1);
    expect(result.hasPcm).toBe(true);
  });

  test('segment PCM contains the actual speech audio', async () => {
    const result = await page.evaluate(() => {
      const seg = new SpeechSegmenter(16000, {
        minSpeechMs: 100,
        minSilenceMs: 300,
        hangoverFrames: 4,
      });

      // Calibrate
      seg.process(new Float32Array(8000), 0);

      // 500ms speech at known amplitude
      const speech = new Float32Array(8000);
      for (let i = 0; i < 8000; i++) {
        speech[i] = 0.25 * Math.sin(2 * Math.PI * 200 * i / 16000);
      }
      seg.process(speech, 500);

      // End
      const segs = seg.process(new Float32Array(16000), 1000);
      const flushed = seg.flush();
      const all = [...segs, ...flushed];

      if (all.length === 0) return { hasSegment: false };

      // Check the PCM has non-trivial content
      const pcm = all[0].pcm;
      let rms = 0;
      for (let i = 0; i < pcm.length; i++) rms += pcm[i] * pcm[i];
      rms = Math.sqrt(rms / pcm.length);

      return { hasSegment: true, rms, length: pcm.length };
    });
    expect(result.hasSegment).toBe(true);
    expect(result.rms).toBeGreaterThan(0.05);
    expect(result.length).toBeGreaterThan(1000);
  });
});
