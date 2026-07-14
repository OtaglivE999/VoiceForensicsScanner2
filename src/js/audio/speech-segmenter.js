/**
 * Frame-based VAD and speech segment builder.
 *
 * Classifies short frames (~20ms) as speech or silence using
 * energy, zero-crossing rate, and spectral flatness.  Accumulates
 * consecutive speech frames into segments and only emits when a
 * minimum duration threshold is met.  Includes hang-over logic
 * to bridge brief pauses within a single utterance.
 */

const DEFAULT_OPTS = {
  frameSize: 320,         // samples per frame (20ms at 16kHz)
  energyFloor: 1e-8,      // minimum energy to avoid log(0)
  energyThreshold: 0.005, // RMS threshold for speech
  zcrLow: 5,              // ZCR (per sec) below this → DC/silence
  zcrHigh: 4000,          // ZCR (per sec) above this → broadband noise
  flatnessThreshold: 0.7, // spectral flatness above → noise
  hangoverFrames: 8,      // frames to hold speech state after drop
  minSpeechMs: 300,       // minimum segment duration to emit
  maxSpeechMs: 30000,     // force-emit if segment exceeds this
  minSilenceMs: 500,      // silence gap to split segments
  adaptRate: 0.02,        // noise floor adaptation speed
};

class FrameVAD {
  constructor(sampleRate, opts = {}) {
    this._sr = sampleRate;
    this._opts = { ...DEFAULT_OPTS, ...opts };
    this._noiseFloor = this._opts.energyFloor;
    this._frameCount = 0;
  }

  /**
   * Classify a single frame.
   * @param {Float32Array} frame – PCM samples (frameSize length)
   * @returns {{ speech: boolean, energy: number, zcr: number, flatness: number }}
   */
  classify(frame) {
    const N = frame.length;

    // RMS energy
    let sumSq = 0;
    for (let i = 0; i < N; i++) sumSq += frame[i] * frame[i];
    const energy = Math.sqrt(sumSq / N);

    // Zero-crossing rate (per second)
    let crossings = 0;
    for (let i = 1; i < N; i++) {
      if ((frame[i] >= 0) !== (frame[i - 1] >= 0)) crossings++;
    }
    const frameSec = N / this._sr;
    const zcr = crossings / frameSec;

    // Spectral flatness (geometric / arithmetic mean of magnitude spectrum)
    const flatness = this._spectralFlatness(frame);

    // Adapt noise floor during silence
    this._frameCount++;
    const aboveNoise = energy > this._noiseFloor * 3;
    if (!aboveNoise && this._frameCount > 20) {
      this._noiseFloor += this._opts.adaptRate * (energy - this._noiseFloor);
      if (this._noiseFloor < this._opts.energyFloor) {
        this._noiseFloor = this._opts.energyFloor;
      }
    }

    const dynamicThreshold = Math.max(
      this._opts.energyThreshold,
      this._noiseFloor * 4
    );

    const speech =
      energy > dynamicThreshold &&
      zcr > this._opts.zcrLow &&
      zcr < this._opts.zcrHigh &&
      flatness < this._opts.flatnessThreshold;

    return { speech, energy, zcr, flatness };
  }

  _spectralFlatness(frame) {
    const N = frame.length;
    const fftSize = N;
    const re = new Float32Array(fftSize);
    const im = new Float32Array(fftSize);
    for (let i = 0; i < N; i++) re[i] = frame[i];

    _fftInPlace(re, im);

    const half = Math.floor(fftSize / 2);
    let logSum = 0;
    let arithSum = 0;
    let count = 0;
    for (let k = 1; k <= half; k++) {
      const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
      const m = Math.max(mag, 1e-20);
      logSum += Math.log(m);
      arithSum += m;
      count++;
    }
    if (count === 0 || arithSum < 1e-20) return 1.0;
    const geoMean = Math.exp(logSum / count);
    const ariMean = arithSum / count;
    return geoMean / ariMean;
  }

  reset() {
    this._noiseFloor = this._opts.energyFloor;
    this._frameCount = 0;
  }
}

// Radix-2 in-place FFT (Cooley-Tukey)
function _fftInPlace(re, im) {
  const N = re.length;
  // Bit-reversal permutation
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
  // FFT butterfly
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
}

// Pad to next power of 2 for FFT — used by FrameVAD internally
function _nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

class SpeechSegmenter {
  /**
   * @param {number} sampleRate
   * @param {object} [opts]
   */
  constructor(sampleRate, opts = {}) {
    this._sr = sampleRate;
    this._opts = { ...DEFAULT_OPTS, ...opts };
    // Snap frameSize to a power of 2 for FFT
    this._frameSize = _nextPow2(this._opts.frameSize);
    this._vad = new FrameVAD(sampleRate, { ...this._opts, frameSize: this._frameSize });

    this._hangover = 0;
    this._inSpeech = false;
    this._silenceFrames = 0;

    this._segmentPcm = [];
    this._segmentFrameCount = 0;
    this._segmentStartTime = null;

    this._pendingPcm = new Float32Array(0);
  }

  /**
   * Feed PCM samples and get back completed speech segments.
   * Call this repeatedly with new audio data.
   *
   * @param {Float32Array} pcm – raw PCM at constructor's sampleRate
   * @param {number} [timestamp] – optional wall-clock ms of first sample
   * @returns {Array<{ pcm: Float32Array, startTime: number|null, durationMs: number, frameCount: number }>}
   */
  process(pcm, timestamp) {
    const combined = new Float32Array(this._pendingPcm.length + pcm.length);
    combined.set(this._pendingPcm);
    combined.set(pcm, this._pendingPcm.length);

    const segments = [];
    let offset = 0;
    const frameSize = this._frameSize;
    const frameDurationMs = (frameSize / this._sr) * 1000;
    const hangoverMax = this._opts.hangoverFrames;
    const minSilenceFrames = Math.ceil(this._opts.minSilenceMs / frameDurationMs);
    const maxSpeechFrames = Math.ceil(this._opts.maxSpeechMs / frameDurationMs);

    while (offset + frameSize <= combined.length) {
      const frame = combined.subarray(offset, offset + frameSize);
      const result = this._vad.classify(frame);

      if (result.speech) {
        this._hangover = hangoverMax;
        this._silenceFrames = 0;

        if (!this._inSpeech) {
          this._inSpeech = true;
          this._segmentPcm = [];
          this._segmentFrameCount = 0;
          this._segmentStartTime = timestamp != null
            ? timestamp + (offset / this._sr) * 1000
            : null;
        }
      } else if (this._inSpeech) {
        this._hangover--;
        if (this._hangover <= 0) {
          this._silenceFrames++;
        }
      }

      if (this._inSpeech) {
        this._segmentPcm.push(new Float32Array(frame));
        this._segmentFrameCount++;

        const shouldEmit =
          (this._silenceFrames >= minSilenceFrames) ||
          (this._segmentFrameCount >= maxSpeechFrames);

        if (shouldEmit) {
          const seg = this._emitSegment();
          if (seg) segments.push(seg);
        }
      }

      offset += frameSize;
    }

    this._pendingPcm = combined.subarray(offset);
    return segments;
  }

  /**
   * Flush any in-progress segment (call when stopping).
   * @returns {Array<{ pcm: Float32Array, startTime: number|null, durationMs: number, frameCount: number }>}
   */
  flush() {
    const segments = [];
    if (this._inSpeech) {
      const seg = this._emitSegment();
      if (seg) segments.push(seg);
    }
    this._pendingPcm = new Float32Array(0);
    return segments;
  }

  _emitSegment() {
    const totalSamples = this._segmentFrameCount * this._frameSize;
    const durationMs = (totalSamples / this._sr) * 1000;

    this._inSpeech = false;
    this._hangover = 0;
    this._silenceFrames = 0;

    if (durationMs < this._opts.minSpeechMs) {
      this._segmentPcm = [];
      this._segmentFrameCount = 0;
      this._segmentStartTime = null;
      return null;
    }

    const pcm = new Float32Array(totalSamples);
    let pos = 0;
    for (const chunk of this._segmentPcm) {
      pcm.set(chunk, pos);
      pos += chunk.length;
    }

    const seg = {
      pcm,
      startTime: this._segmentStartTime,
      durationMs,
      frameCount: this._segmentFrameCount,
    };

    this._segmentPcm = [];
    this._segmentFrameCount = 0;
    this._segmentStartTime = null;

    return seg;
  }

  reset() {
    this._vad.reset();
    this._inSpeech = false;
    this._hangover = 0;
    this._silenceFrames = 0;
    this._segmentPcm = [];
    this._segmentFrameCount = 0;
    this._segmentStartTime = null;
    this._pendingPcm = new Float32Array(0);
  }

  get info() {
    return {
      sampleRate: this._sr,
      frameSize: this._frameSize,
      minSpeechMs: this._opts.minSpeechMs,
      maxSpeechMs: this._opts.maxSpeechMs,
      hangoverFrames: this._opts.hangoverFrames,
    };
  }
}

export { FrameVAD, SpeechSegmenter, DEFAULT_OPTS };
