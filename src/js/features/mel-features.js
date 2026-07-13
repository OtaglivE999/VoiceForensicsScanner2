/**
 * Log-mel spectrogram and MFCC feature extraction.
 *
 * Computes a proper mel-scale filterbank from PCM, applies log
 * compression, and optionally extracts MFCCs via DCT-II.
 * Designed to operate on 16 kHz resampled audio segments.
 */

const MEL_DEFAULTS = {
  sampleRate: 16000,
  fftSize: 512,
  hopSize: 160,         // 10ms at 16kHz
  numMelBands: 40,
  numMfcc: 13,
  fMin: 64,
  fMax: 7600,
  preEmphasis: 0.97,
  windowType: 'hann',
};

// ── Mel scale helpers ─────────────────────────────────────────

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

// ── Window functions ──────────────────────────────────────────

function hannWindow(N) {
  const w = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    w[n] = 0.5 * (1 - Math.cos(2 * Math.PI * n / (N - 1)));
  }
  return w;
}

// ── FFT ───────────────────────────────────────────────────────

function fftMagnitude(frame) {
  const N = frame.length;
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  re.set(frame);

  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let size = 2; size <= N; size *= 2) {
    const halfSize = size / 2;
    const angle = -2 * Math.PI / size;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    for (let start = 0; start < N; start += size) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < halfSize; k++) {
        const evenIdx = start + k;
        const oddIdx = start + k + halfSize;
        const tRe = curRe * re[oddIdx] - curIm * im[oddIdx];
        const tIm = curRe * im[oddIdx] + curIm * re[oddIdx];
        re[oddIdx] = re[evenIdx] - tRe;
        im[oddIdx] = im[evenIdx] - tIm;
        re[evenIdx] += tRe;
        im[evenIdx] += tIm;
        const newRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newRe;
      }
    }
  }

  const half = N / 2 + 1;
  const mag = new Float32Array(half);
  for (let k = 0; k < half; k++) {
    mag[k] = re[k] * re[k] + im[k] * im[k];
  }
  return mag;
}

// ── Mel filterbank construction ───────────────────────────────

function buildMelFilterbank(numBands, fftSize, sampleRate, fMin, fMax) {
  const numBins = fftSize / 2 + 1;
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);

  const melPoints = new Float32Array(numBands + 2);
  for (let i = 0; i < numBands + 2; i++) {
    melPoints[i] = melMin + i * (melMax - melMin) / (numBands + 1);
  }

  const hzPoints = new Float32Array(numBands + 2);
  for (let i = 0; i < numBands + 2; i++) {
    hzPoints[i] = melToHz(melPoints[i]);
  }

  const binPoints = new Float32Array(numBands + 2);
  for (let i = 0; i < numBands + 2; i++) {
    binPoints[i] = Math.floor((fftSize + 1) * hzPoints[i] / sampleRate);
  }

  const filterbank = [];
  for (let m = 0; m < numBands; m++) {
    const filter = new Float32Array(numBins);
    const start = binPoints[m];
    const center = binPoints[m + 1];
    const end = binPoints[m + 2];

    for (let k = Math.floor(start); k <= Math.floor(center); k++) {
      if (k >= 0 && k < numBins && center > start) {
        filter[k] = (k - start) / (center - start);
      }
    }
    for (let k = Math.floor(center); k <= Math.floor(end); k++) {
      if (k >= 0 && k < numBins && end > center) {
        filter[k] = (end - k) / (end - center);
      }
    }
    filterbank.push(filter);
  }
  return filterbank;
}

// ── DCT-II for MFCC ──────────────────────────────────────────

function dctII(input, numCoeffs) {
  const N = input.length;
  const output = new Float32Array(numCoeffs);
  for (let k = 0; k < numCoeffs; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += input[n] * Math.cos(Math.PI * k * (n + 0.5) / N);
    }
    output[k] = sum;
  }
  return output;
}

// ── Main class ────────────────────────────────────────────────

class MelFeatureExtractor {
  constructor(opts = {}) {
    this._opts = { ...MEL_DEFAULTS, ...opts };
    const { fftSize, sampleRate, numMelBands, fMin, fMax } = this._opts;

    this._window = hannWindow(fftSize);
    this._filterbank = buildMelFilterbank(numMelBands, fftSize, sampleRate, fMin, fMax);
  }

  /**
   * Compute log-mel spectrogram from a PCM segment.
   * @param {Float32Array} pcm – mono PCM at the configured sample rate
   * @returns {{ logMel: Float32Array[], numFrames: number, numBands: number }}
   */
  logMelSpectrogram(pcm) {
    const { fftSize, hopSize, numMelBands, preEmphasis } = this._opts;

    // Pre-emphasis
    const emphasized = new Float32Array(pcm.length);
    emphasized[0] = pcm[0];
    for (let i = 1; i < pcm.length; i++) {
      emphasized[i] = pcm[i] - preEmphasis * pcm[i - 1];
    }

    const numFrames = Math.max(0, Math.floor((emphasized.length - fftSize) / hopSize) + 1);
    const logMel = [];

    for (let f = 0; f < numFrames; f++) {
      const offset = f * hopSize;
      const windowed = new Float32Array(fftSize);
      for (let i = 0; i < fftSize; i++) {
        windowed[i] = emphasized[offset + i] * this._window[i];
      }

      const powerSpec = fftMagnitude(windowed);

      const melEnergies = new Float32Array(numMelBands);
      for (let m = 0; m < numMelBands; m++) {
        let sum = 0;
        const filter = this._filterbank[m];
        for (let k = 0; k < powerSpec.length; k++) {
          sum += powerSpec[k] * filter[k];
        }
        melEnergies[m] = Math.log(Math.max(sum, 1e-10));
      }
      logMel.push(melEnergies);
    }

    return { logMel, numFrames, numBands: numMelBands };
  }

  /**
   * Compute MFCCs from a PCM segment.
   * @param {Float32Array} pcm
   * @returns {{ mfcc: Float32Array[], numFrames: number, numCoeffs: number }}
   */
  mfcc(pcm) {
    const { numMfcc } = this._opts;
    const { logMel, numFrames } = this.logMelSpectrogram(pcm);

    const mfccFrames = [];
    for (let f = 0; f < numFrames; f++) {
      const coeffs = dctII(logMel[f], numMfcc);
      mfccFrames.push(coeffs);
    }

    return { mfcc: mfccFrames, numFrames, numCoeffs: numMfcc };
  }

  /**
   * Compute a single summary MFCC vector by averaging across frames.
   * @param {Float32Array} pcm
   * @returns {Float32Array} – averaged MFCC vector (numMfcc dimensions)
   */
  mfccMean(pcm) {
    const { numMfcc } = this._opts;
    const { mfcc, numFrames } = this.mfcc(pcm);
    if (numFrames === 0) return new Float32Array(numMfcc);

    const mean = new Float32Array(numMfcc);
    for (let f = 0; f < numFrames; f++) {
      for (let c = 0; c < numMfcc; c++) {
        mean[c] += mfcc[f][c];
      }
    }
    for (let c = 0; c < numMfcc; c++) mean[c] /= numFrames;
    return mean;
  }

  /**
   * Compute delta (first derivative) features.
   * @param {Float32Array[]} features – array of feature vectors per frame
   * @param {number} [width=2] – delta window width
   * @returns {Float32Array[]}
   */
  deltas(features, width = 2) {
    const numFrames = features.length;
    if (numFrames === 0) return [];
    const dim = features[0].length;
    const result = [];

    for (let t = 0; t < numFrames; t++) {
      const d = new Float32Array(dim);
      let norm = 0;
      for (let n = 1; n <= width; n++) {
        norm += 2 * n * n;
        const prev = Math.max(0, t - n);
        const next = Math.min(numFrames - 1, t + n);
        for (let c = 0; c < dim; c++) {
          d[c] += n * (features[next][c] - features[prev][c]);
        }
      }
      if (norm > 0) {
        for (let c = 0; c < dim; c++) d[c] /= norm;
      }
      result.push(d);
    }
    return result;
  }

  get info() {
    return { ...this._opts };
  }
}

export { MelFeatureExtractor, hzToMel, melToHz, buildMelFilterbank, dctII, MEL_DEFAULTS };
