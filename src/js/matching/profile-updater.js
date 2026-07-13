/**
 * Quality-gated speaker profile updates.
 *
 * Only updates a speaker profile's averaged features and embedding
 * when the new observation meets minimum quality criteria.  Applies
 * exponential moving average with an adaptive learning rate that
 * decreases as the profile matures.
 */

const UPDATE_DEFAULTS = {
  minSnr: 15,
  maxClippingPct: 0.01,
  minDurationMs: 300,
  minVoiceConf: 0.5,
  maxDetectionsForFastLearn: 20,
  fastAlpha: 0.15,
  slowAlpha: 0.02,
};

class ProfileUpdater {
  constructor(opts = {}) {
    this._opts = { ...UPDATE_DEFAULTS, ...opts };
  }

  /**
   * Determine if an observation meets quality for profile update.
   *
   * @param {object} quality – signal quality metrics
   * @param {number} quality.snr
   * @param {number} quality.clippingPct
   * @param {number} quality.durationMs
   * @param {number} quality.voiceConf
   * @returns {{ pass: boolean, reasons: string[] }}
   */
  checkQuality(quality) {
    const reasons = [];
    if (quality.snr < this._opts.minSnr) {
      reasons.push(`SNR ${quality.snr.toFixed(1)} < ${this._opts.minSnr}`);
    }
    if (quality.clippingPct > this._opts.maxClippingPct) {
      reasons.push(`clipping ${(quality.clippingPct * 100).toFixed(1)}% > ${(this._opts.maxClippingPct * 100).toFixed(1)}%`);
    }
    if (quality.durationMs < this._opts.minDurationMs) {
      reasons.push(`duration ${quality.durationMs}ms < ${this._opts.minDurationMs}ms`);
    }
    if (quality.voiceConf < this._opts.minVoiceConf) {
      reasons.push(`voiceConf ${quality.voiceConf.toFixed(2)} < ${this._opts.minVoiceConf}`);
    }
    return { pass: reasons.length === 0, reasons };
  }

  /**
   * Update a profile's embedding with EMA.
   *
   * @param {object} profile – mutable profile object
   * @param {Float32Array} newEmbedding – L2-normalized
   * @param {object} quality – quality metrics
   * @returns {{ updated: boolean, reasons?: string[] }}
   */
  updateEmbedding(profile, newEmbedding, quality) {
    const qc = this.checkQuality(quality);
    if (!qc.pass) return { updated: false, reasons: qc.reasons };
    if (!newEmbedding || newEmbedding.length === 0) {
      return { updated: false, reasons: ['no embedding'] };
    }

    const alpha = this._alpha(profile.detections || 0);

    if (!profile.embedding || profile.embedding.length === 0) {
      profile.embedding = new Float32Array(newEmbedding);
      profile.detections = (profile.detections || 0) + 1;
      return { updated: true };
    }

    const dim = profile.embedding.length;
    for (let i = 0; i < dim; i++) {
      profile.embedding[i] = (1 - alpha) * profile.embedding[i] + alpha * newEmbedding[i];
    }

    // Re-normalize
    let norm = 0;
    for (let i = 0; i < dim; i++) norm += profile.embedding[i] * profile.embedding[i];
    norm = Math.sqrt(norm);
    if (norm > 1e-12) {
      for (let i = 0; i < dim; i++) profile.embedding[i] /= norm;
    }

    profile.detections = (profile.detections || 0) + 1;
    return { updated: true };
  }

  /**
   * Update a profile's averaged features with EMA.
   *
   * @param {object} profile
   * @param {object} newFeatures
   * @param {object} quality
   * @returns {{ updated: boolean, reasons?: string[] }}
   */
  updateFeatures(profile, newFeatures, quality) {
    const qc = this.checkQuality(quality);
    if (!qc.pass) return { updated: false, reasons: qc.reasons };
    if (!newFeatures) return { updated: false, reasons: ['no features'] };

    if (!profile.avgFeatures) {
      profile.avgFeatures = { ...newFeatures };
      if (newFeatures.mfcc) profile.avgFeatures.mfcc = [...newFeatures.mfcc];
      return { updated: true };
    }

    const alpha = this._alpha(profile.detections || 0);
    const avg = profile.avgFeatures;

    const numericKeys = Object.keys(newFeatures).filter(
      k => k !== 'mfcc' && typeof newFeatures[k] === 'number'
    );

    for (const k of numericKeys) {
      if (avg[k] != null) {
        avg[k] = (1 - alpha) * avg[k] + alpha * newFeatures[k];
      } else {
        avg[k] = newFeatures[k];
      }
    }

    if (newFeatures.mfcc && avg.mfcc) {
      const len = Math.min(newFeatures.mfcc.length, avg.mfcc.length);
      for (let i = 0; i < len; i++) {
        avg.mfcc[i] = (1 - alpha) * avg.mfcc[i] + alpha * newFeatures.mfcc[i];
      }
    } else if (newFeatures.mfcc && !avg.mfcc) {
      avg.mfcc = [...newFeatures.mfcc];
    }

    return { updated: true };
  }

  _alpha(detections) {
    if (detections < this._opts.maxDetectionsForFastLearn) {
      return this._opts.fastAlpha;
    }
    return this._opts.slowAlpha;
  }

  get opts() {
    return { ...this._opts };
  }
}

export { ProfileUpdater, UPDATE_DEFAULTS };
