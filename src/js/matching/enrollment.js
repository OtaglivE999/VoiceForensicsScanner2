/**
 * Multi-observation speaker enrollment.
 *
 * Requires multiple speech segments before creating a speaker profile,
 * preventing noise bursts or brief transients from spawning permanent
 * speaker IDs.  Tracks enrollment candidates and promotes them to
 * full profiles only after consistency checks pass.
 */

const ENROLLMENT_DEFAULTS = {
  minObservations: 3,
  maxObservations: 10,
  minTotalSpeechMs: 1500,
  consistencyThreshold: 0.60,
  embeddingDim: 192,
  candidateTimeoutMs: 60000,
};

let _nextCandidateId = 1;

class EnrollmentManager {
  /**
   * @param {object} [opts]
   */
  constructor(opts = {}) {
    this._opts = { ...ENROLLMENT_DEFAULTS, ...opts };
    this._candidates = new Map();
  }

  /**
   * Submit an observation for potential enrollment.
   * Returns a candidate status or a completed profile ready for storage.
   *
   * @param {object} observation
   * @param {Float32Array} [observation.embedding] – L2-normalized
   * @param {object} [observation.features] – extracted features
   * @param {number} observation.durationMs – segment duration
   * @param {number} [observation.timestamp] – wall-clock ms
   * @returns {{ status: string, candidateId?: string, profile?: object }}
   *   status: 'new-candidate' | 'accumulating' | 'enrolled' | 'rejected'
   */
  submit(observation) {
    const now = observation.timestamp || Date.now();
    this._pruneExpired(now);

    // Try to match against existing candidates
    const match = this._findCandidate(observation);

    if (match) {
      match.observations.push(observation);
      match.totalSpeechMs += observation.durationMs || 0;
      match.lastSeen = now;

      if (this._isReadyToEnroll(match)) {
        const profile = this._buildProfile(match);
        this._candidates.delete(match.id);
        return { status: 'enrolled', candidateId: match.id, profile };
      }

      if (match.observations.length >= this._opts.maxObservations) {
        if (this._checkConsistency(match)) {
          const profile = this._buildProfile(match);
          this._candidates.delete(match.id);
          return { status: 'enrolled', candidateId: match.id, profile };
        }
        this._candidates.delete(match.id);
        return { status: 'rejected', candidateId: match.id };
      }

      return {
        status: 'accumulating',
        candidateId: match.id,
        observations: match.observations.length,
        totalSpeechMs: match.totalSpeechMs,
      };
    }

    // New candidate
    const candidateId = `candidate-${_nextCandidateId++}`;
    this._candidates.set(candidateId, {
      id: candidateId,
      observations: [observation],
      totalSpeechMs: observation.durationMs || 0,
      createdAt: now,
      lastSeen: now,
    });

    return { status: 'new-candidate', candidateId };
  }

  _findCandidate(observation) {
    let bestMatch = null;
    let bestSim = -1;

    for (const [, candidate] of this._candidates) {
      const sim = this._similarityToCandidate(observation, candidate);
      if (sim > this._opts.consistencyThreshold && sim > bestSim) {
        bestSim = sim;
        bestMatch = candidate;
      }
    }

    return bestMatch;
  }

  _similarityToCandidate(observation, candidate) {
    const lastObs = candidate.observations[candidate.observations.length - 1];

    if (observation.embedding && lastObs.embedding) {
      return _cosineSim(observation.embedding, lastObs.embedding);
    }

    if (observation.features && lastObs.features) {
      return _featureOverlap(observation.features, lastObs.features);
    }

    return 0;
  }

  _isReadyToEnroll(candidate) {
    return (
      candidate.observations.length >= this._opts.minObservations &&
      candidate.totalSpeechMs >= this._opts.minTotalSpeechMs &&
      this._checkConsistency(candidate)
    );
  }

  _checkConsistency(candidate) {
    const obs = candidate.observations;
    if (obs.length < 2) return true;

    // Check pairwise consistency of available signals
    let totalSim = 0;
    let pairs = 0;

    for (let i = 0; i < obs.length; i++) {
      for (let j = i + 1; j < obs.length; j++) {
        if (obs[i].embedding && obs[j].embedding) {
          totalSim += _cosineSim(obs[i].embedding, obs[j].embedding);
          pairs++;
        } else if (obs[i].features && obs[j].features) {
          totalSim += _featureOverlap(obs[i].features, obs[j].features);
          pairs++;
        }
      }
    }

    if (pairs === 0) return true;
    return (totalSim / pairs) >= this._opts.consistencyThreshold;
  }

  _buildProfile(candidate) {
    const obs = candidate.observations;
    const profile = {
      id: null, // assigned by caller
      enrolledAt: new Date().toISOString(),
      observationCount: obs.length,
      totalSpeechMs: candidate.totalSpeechMs,
      detections: obs.length,
    };

    // Average embedding
    const withEmbeddings = obs.filter(o => o.embedding);
    if (withEmbeddings.length > 0) {
      const dim = withEmbeddings[0].embedding.length;
      const avg = new Float32Array(dim);
      for (const o of withEmbeddings) {
        for (let i = 0; i < dim; i++) avg[i] += o.embedding[i];
      }
      for (let i = 0; i < dim; i++) avg[i] /= withEmbeddings.length;

      // L2 normalize
      let norm = 0;
      for (let i = 0; i < dim; i++) norm += avg[i] * avg[i];
      norm = Math.sqrt(norm);
      if (norm > 1e-12) for (let i = 0; i < dim; i++) avg[i] /= norm;

      profile.embedding = avg;
    }

    // Average features
    const withFeatures = obs.filter(o => o.features);
    if (withFeatures.length > 0) {
      profile.avgFeatures = _averageFeatures(withFeatures.map(o => o.features));
    }

    return profile;
  }

  _pruneExpired(now) {
    const timeout = this._opts.candidateTimeoutMs;
    for (const [id, candidate] of this._candidates) {
      if (now - candidate.lastSeen > timeout) {
        this._candidates.delete(id);
      }
    }
  }

  get candidateCount() {
    return this._candidates.size;
  }

  getCandidateIds() {
    return Array.from(this._candidates.keys());
  }

  reset() {
    this._candidates.clear();
  }
}

// ── Helpers ───────────────────────────────────────────────────

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

function _featureOverlap(a, b) {
  const keys = ['pitch', 'centroid', 'spread', 'f1', 'f2', 'f3'];
  let total = 0, count = 0;
  for (const k of keys) {
    if (a[k] != null && b[k] != null) {
      const max = Math.max(Math.abs(a[k]), Math.abs(b[k]), 1e-10);
      total += 1 - Math.abs(a[k] - b[k]) / max;
      count++;
    }
  }
  return count > 0 ? Math.max(0, total / count) : 0;
}

function _averageFeatures(featureList) {
  const keys = ['pitch', 'centroid', 'spread', 'flatness', 'rolloff',
                'f1', 'f2', 'f3', 'f2f1', 'f3f2', 'zcr',
                'nasalFormant', 'nasalance', 'jitter', 'shimmer'];
  const avg = {};
  for (const k of keys) {
    let sum = 0, count = 0;
    for (const f of featureList) {
      if (f[k] != null && typeof f[k] === 'number') {
        sum += f[k];
        count++;
      }
    }
    if (count > 0) avg[k] = sum / count;
  }

  // Average MFCCs
  const mfccLists = featureList.filter(f => f.mfcc && f.mfcc.length > 0);
  if (mfccLists.length > 0) {
    const len = mfccLists[0].mfcc.length;
    avg.mfcc = new Array(len).fill(0);
    for (const f of mfccLists) {
      for (let i = 0; i < len; i++) avg.mfcc[i] += f.mfcc[i];
    }
    for (let i = 0; i < len; i++) avg.mfcc[i] /= mfccLists.length;
  }

  return avg;
}

export { EnrollmentManager, ENROLLMENT_DEFAULTS };
