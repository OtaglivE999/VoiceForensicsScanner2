/**
 * Band-limited resampler: native sample rate → 16 kHz.
 *
 * Uses a windowed-sinc (Blackman) low-pass FIR filter followed by
 * rational-rate decimation.  Optimised for the common 48000→16000
 * case (integer factor 3), but handles arbitrary input rates by
 * finding the smallest L/M rational approximation and polyphase
 * filtering.
 */

const MODEL_RATE = 16000;

// ── FIR design helpers ─────────────────────────────────────────

function blackmanWindow(n, N) {
  const a0 = 0.42, a1 = 0.5, a2 = 0.08;
  return a0 - a1 * Math.cos(2 * Math.PI * n / (N - 1))
            + a2 * Math.cos(4 * Math.PI * n / (N - 1));
}

function designLowpass(cutoff, numTaps) {
  const h = new Float64Array(numTaps);
  const mid = (numTaps - 1) / 2;
  for (let n = 0; n < numTaps; n++) {
    const x = n - mid;
    const sinc = x === 0 ? cutoff : Math.sin(Math.PI * cutoff * x) / (Math.PI * x);
    h[n] = sinc * blackmanWindow(n, numTaps);
  }
  let sum = 0;
  for (let i = 0; i < numTaps; i++) sum += h[i];
  if (sum !== 0) for (let i = 0; i < numTaps; i++) h[i] /= sum;
  return h;
}

// ── Rational approximation via continued fractions ─────────────

function gcd(a, b) {
  a = Math.abs(a); b = Math.abs(b);
  while (b) { [a, b] = [b, a % b]; }
  return a;
}

function rationalApprox(value, maxDenom) {
  let bestNum = Math.round(value), bestDen = 1;
  let bestErr = Math.abs(value - bestNum);
  for (let d = 2; d <= maxDenom; d++) {
    const n = Math.round(value * d);
    const err = Math.abs(value - n / d);
    if (err < bestErr) {
      bestErr = err;
      bestNum = n;
      bestDen = d;
      if (err < 1e-12) break;
    }
  }
  const g = gcd(bestNum, bestDen);
  return { L: bestNum / g, M: bestDen / g };
}

// ── Resampler class ────────────────────────────────────────────

class Resampler {
  /**
   * @param {number} inputRate  – browser's AudioContext.sampleRate
   * @param {number} [outputRate=16000] – target rate
   * @param {number} [quality=64] – filter half-length (per polyphase arm)
   */
  constructor(inputRate, outputRate = MODEL_RATE, quality = 64) {
    if (inputRate <= 0 || outputRate <= 0) {
      throw new RangeError('Sample rates must be positive');
    }

    this.inputRate = inputRate;
    this.outputRate = outputRate;

    if (inputRate === outputRate) {
      this._passthrough = true;
      this.L = 1;
      this.M = 1;
      this._filter = null;
      this._state = null;
      return;
    }

    this._passthrough = false;

    // Rational rate ratio: outputRate/inputRate = L/M
    // We upsample by L, filter, then decimate by M.
    const ratio = outputRate / inputRate;
    const { L, M } = rationalApprox(ratio, 1024);
    this.L = L;
    this.M = M;

    // Anti-alias filter: cutoff at π/max(L,M)
    const cutoff = 1 / Math.max(L, M);
    const numTaps = 2 * quality * Math.max(L, M) + 1;
    this._protoFilter = designLowpass(cutoff, numTaps);

    // Scale by L (interpolation gain)
    for (let i = 0; i < this._protoFilter.length; i++) {
      this._protoFilter[i] *= L;
    }

    // State buffer for overlap between calls
    this._stateLen = this._protoFilter.length;
    this._state = new Float32Array(this._stateLen);
    this._phase = 0; // polyphase output phase
  }

  /**
   * Resample a block of Float32 PCM.
   * @param {Float32Array} input
   * @returns {Float32Array} resampled output at outputRate
   */
  process(input) {
    if (this._passthrough) return new Float32Array(input);

    const L = this.L;
    const M = this.M;
    const filter = this._protoFilter;
    const filterLen = filter.length;

    // Concatenate state + input
    const extended = new Float32Array(this._stateLen + input.length);
    extended.set(this._state, 0);
    extended.set(input, this._stateLen);

    // Total virtual upsampled length
    const virtualLen = input.length * L;
    // Estimate output size
    const maxOut = Math.ceil(virtualLen / M) + 1;
    const output = new Float32Array(maxOut);

    let outIdx = 0;
    let phase = this._phase;

    // Direct polyphase implementation:
    // For each output sample n, the corresponding position in the
    // upsampled stream is n*M.  The input sample index is floor(n*M/L),
    // and the filter phase is (n*M) mod L.
    while (phase < virtualLen) {
      const inputIdx = Math.floor(phase / L);
      const filterPhase = phase - inputIdx * L;

      let sum = 0;
      // Walk the filter taps that align with this phase
      for (let k = filterPhase; k < filterLen; k += L) {
        const srcIdx = inputIdx - Math.floor(k / L) + Math.floor((filterLen - 1) / (2 * L));
        if (srcIdx >= 0 && srcIdx < extended.length) {
          sum += extended[srcIdx] * filter[k];
        }
      }
      output[outIdx++] = sum;
      phase += M;
    }

    // Save state (last filterLen samples of input for next call)
    const stateStart = Math.max(0, extended.length - this._stateLen);
    this._state.set(extended.subarray(stateStart, stateStart + this._stateLen));
    this._phase = phase - virtualLen;

    return output.subarray(0, outIdx);
  }

  /** Reset internal state between sessions. */
  reset() {
    if (this._state) this._state.fill(0);
    this._phase = 0;
  }

  /** @returns {{ inputRate: number, outputRate: number, L: number, M: number }} */
  get info() {
    return {
      inputRate: this.inputRate,
      outputRate: this.outputRate,
      L: this.L,
      M: this.M,
      passthrough: this._passthrough
    };
  }
}

// ── Simple (fast) integer-ratio decimator for 48→16 kHz ────────

class IntegerDecimator {
  /**
   * Fast path for the common 48000→16000 case (factor 3).
   * Uses a 127-tap Blackman-windowed sinc at 8 kHz cutoff.
   * @param {number} factor – decimation factor (must be integer)
   * @param {number} inputRate
   */
  constructor(factor, inputRate) {
    if (!Number.isInteger(factor) || factor < 1) {
      throw new RangeError('Decimation factor must be a positive integer');
    }
    this.factor = factor;
    this.inputRate = inputRate;
    this.outputRate = inputRate / factor;

    if (factor === 1) {
      this._passthrough = true;
      this._filter = null;
      this._state = null;
      return;
    }

    this._passthrough = false;
    const numTaps = Math.max(63, factor * 64 + 1) | 1; // odd
    this._filter = designLowpass(1 / factor, numTaps);
    this._overlap = new Float32Array(numTaps - 1);
    this._phase = 0; // decimation phase: samples to skip before next output
  }

  /**
   * @param {Float32Array} input
   * @returns {Float32Array}
   */
  process(input) {
    if (this._passthrough) return new Float32Array(input);

    const filter = this._filter;
    const fLen = filter.length;
    const factor = this.factor;

    const buf = new Float32Array(this._overlap.length + input.length);
    buf.set(this._overlap, 0);
    buf.set(input, this._overlap.length);

    const maxOut = Math.ceil(buf.length / factor) + 1;
    const output = new Float32Array(maxOut);
    let outIdx = 0;
    let pos = this._phase;

    while (pos + fLen <= buf.length) {
      let sum = 0;
      for (let k = 0; k < fLen; k++) {
        sum += buf[pos + k] * filter[k];
      }
      output[outIdx++] = sum;
      pos += factor;
    }

    // Keep last (fLen-1) samples as overlap for next call
    const overlapStart = buf.length - (fLen - 1);
    this._overlap = new Float32Array(buf.subarray(Math.max(0, overlapStart)));
    this._phase = pos - Math.max(0, overlapStart);

    return output.subarray(0, outIdx);
  }

  reset() {
    if (this._overlap) this._overlap.fill(0);
    this._phase = 0;
  }
}

// ── Factory ────────────────────────────────────────────────────

/**
 * Create the best resampler for a given input rate → 16 kHz.
 * Uses the fast integer decimator when the ratio is exact,
 * otherwise falls back to the general polyphase resampler.
 *
 * @param {number} inputRate
 * @param {number} [outputRate=16000]
 * @returns {Resampler|IntegerDecimator}
 */
function createResampler(inputRate, outputRate = MODEL_RATE) {
  if (inputRate === outputRate) {
    return new Resampler(inputRate, outputRate);
  }
  if (inputRate > outputRate && inputRate % outputRate === 0) {
    return new IntegerDecimator(inputRate / outputRate, inputRate);
  }
  return new Resampler(inputRate, outputRate);
}

export { Resampler, IntegerDecimator, createResampler, designLowpass, MODEL_RATE };
