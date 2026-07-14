/**
 * Speaker matching engine — embedding-based with feature fallback.
 *
 * Uses cosine similarity on L2-normalized speaker embeddings as the
 * primary matching signal.  Falls back to weighted feature-distance
 * comparison when embeddings are unavailable (placeholder model).
 *
 * Never compares embeddings from different model IDs or versions.
 */

const MATCH_DEFAULTS = {
  embeddingThreshold: 0.65,
  embeddingHighConfidence: 0.80,
  featureThreshold: 0.70,
  featureHighConfidence: 0.85,
  maxCandidates: 10,
};

class SpeakerMatcher {
  /**
   * @param {object} [opts]
   * @param {string} [opts.embeddingModelId] – current model ID for compatibility checks
   */
  constructor(opts = {}) {
    this._opts = { ...MATCH_DEFAULTS, ...opts };
    this._embeddingModelId = opts.embeddingModelId || null;
  }

  /**
   * Match a probe against enrolled profiles using embeddings.
   *
   * @param {Float32Array} probeEmbedding – L2-normalized probe vector
   * @param {Map|Object} profiles – enrolled profiles, each with:
   *   { embedding: Float32Array, embeddingModelId: string, ... }
   * @returns {{ id: string, similarity: number, confidence: string }|null}
   */
  matchByEmbedding(probeEmbedding, profiles) {
    if (!probeEmbedding || probeEmbedding.length === 0) return null;

    const candidates = [];
    const entries = profiles instanceof Map
      ? profiles.entries()
      : Object.entries(profiles);

    for (const [id, profile] of entries) {
      if (!profile.embedding || profile.embedding.length === 0) continue;
      if (!this._isCompatibleModel(profile)) continue;

      const sim = _cosineSim(probeEmbedding, profile.embedding);
      if (sim >= this._opts.embeddingThreshold) {
        candidates.push({ id, similarity: sim });
      }
    }

    if (candidates.length === 0) return null;

    candidates.sort((a, b) => b.similarity - a.similarity);
    const best = candidates[0];

    return {
      id: best.id,
      similarity: best.similarity,
      confidence: best.similarity >= this._opts.embeddingHighConfidence
        ? 'high' : 'medium',
      method: 'embedding',
      candidateCount: Math.min(candidates.length, this._opts.maxCandidates),
    };
  }

  /**
   * Match a probe against enrolled profiles using handcrafted features.
   * Fallback for when embeddings are not available.
   *
   * @param {object} probeFeatures – feature object from extractFeatures()
   * @param {Map|Object} profiles – enrolled profiles, each with:
   *   { avgFeatures: object, ... }
   * @param {object} [weights] – per-feature weights
   * @returns {{ id: string, similarity: number, confidence: string }|null}
   */
  matchByFeatures(probeFeatures, profiles, weights) {
    if (!probeFeatures) return null;

    const w = weights || DEFAULT_FEATURE_WEIGHTS;
    const candidates = [];
    const entries = profiles instanceof Map
      ? profiles.entries()
      : Object.entries(profiles);

    for (const [id, profile] of entries) {
      if (!profile.avgFeatures) continue;
      const sim = _featureSimilarity(probeFeatures, profile.avgFeatures, w);
      if (sim >= this._opts.featureThreshold) {
        candidates.push({ id, similarity: sim });
      }
    }

    if (candidates.length === 0) return null;

    candidates.sort((a, b) => b.similarity - a.similarity);
    const best = candidates[0];

    return {
      id: best.id,
      similarity: best.similarity,
      confidence: best.similarity >= this._opts.featureHighConfidence
        ? 'high' : 'medium',
      method: 'features',
      candidateCount: Math.min(candidates.length, this._opts.maxCandidates),
    };
  }

  /**
   * Combined match: tries embedding first, falls back to features.
   *
   * @param {{ embedding?: Float32Array, features?: object }} probe
   * @param {Map|Object} profiles
   * @returns {{ id: string, similarity: number, confidence: string, method: string }|null}
   */
  match(probe, profiles) {
    if (probe.embedding) {
      const result = this.matchByEmbedding(probe.embedding, profiles);
      if (result) return result;
    }
    if (probe.features) {
      return this.matchByFeatures(probe.features, profiles);
    }
    return null;
  }

  _isCompatibleModel(profile) {
    if (!this._embeddingModelId) return true;
    if (!profile.embeddingModelId) return true;
    return profile.embeddingModelId === this._embeddingModelId;
  }

  get opts() {
    return { ...this._opts };
  }
}

// ── Feature comparison ────────────────────────────────────────

const DEFAULT_FEATURE_WEIGHTS = {
  pitch: 0.20,
  centroid: 0.15,
  spread: 0.10,
  flatness: 0.05,
  rolloff: 0.10,
  f1: 0.10,
  f2: 0.10,
  f3: 0.05,
  zcr: 0.05,
  mfcc: 0.10,
};

function _featureSimilarity(a, b, weights) {
  let totalWeight = 0;
  let weightedSim = 0;

  for (const [key, weight] of Object.entries(weights)) {
    if (key === 'mfcc') {
      if (a.mfcc && b.mfcc) {
        const mfccSim = _mfccSimilarity(a.mfcc, b.mfcc);
        weightedSim += weight * mfccSim;
        totalWeight += weight;
      }
      continue;
    }

    const va = a[key], vb = b[key];
    if (va == null || vb == null) continue;
    if (typeof va !== 'number' || typeof vb !== 'number') continue;

    const maxVal = Math.max(Math.abs(va), Math.abs(vb), 1e-10);
    const sim = 1 - Math.abs(va - vb) / maxVal;
    weightedSim += weight * Math.max(0, sim);
    totalWeight += weight;
  }

  return totalWeight > 0 ? weightedSim / totalWeight : 0;
}

function _mfccSimilarity(a, b) {
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 1e-12 ? (dot / denom + 1) / 2 : 0;
}

function _cosineSim(a, b) {
  const len = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 1e-12 ? dot / denom : 0;
}

export {
  SpeakerMatcher,
  MATCH_DEFAULTS,
  DEFAULT_FEATURE_WEIGHTS,
};
