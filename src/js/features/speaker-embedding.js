/**
 * Speaker embedding interface — ONNX Runtime Web loader + inference.
 *
 * Defines the embedding pipeline contract: load a pinned ONNX model,
 * validate its SHA-256, run inference on mel features, and return
 * L2-normalized embedding vectors.
 *
 * PLACEHOLDER STATUS:
 * No trained model asset is bundled.  The interface, loader, validation,
 * and error handling are fully implemented.  When a real model is
 * provided (with version + SHA-256 pin), set MODEL_MANIFEST and this
 * module becomes operational.
 */

const EMBEDDING_DIM = 192;

const MODEL_MANIFEST = {
  modelId: null,
  version: null,
  sha256: null,
  url: null,
  inputName: 'input',
  outputName: 'output',
  embeddingDim: EMBEDDING_DIM,
  sampleRate: 16000,
  status: 'no-model-available',
};

class SpeakerEmbeddingExtractor {
  /**
   * @param {object} [manifest] – override MODEL_MANIFEST fields
   */
  constructor(manifest = {}) {
    this._manifest = { ...MODEL_MANIFEST, ...manifest };
    this._session = null;
    this._loaded = false;
    this._loadError = null;
  }

  /**
   * Whether a real model is configured (not a placeholder).
   */
  get isConfigured() {
    return this._manifest.modelId !== null &&
           this._manifest.sha256 !== null;
  }

  get isLoaded() {
    return this._loaded;
  }

  get status() {
    if (!this.isConfigured) return 'no-model-configured';
    if (this._loadError) return 'load-error';
    if (this._loaded) return 'ready';
    return 'not-loaded';
  }

  get manifest() {
    return { ...this._manifest };
  }

  /**
   * Load the ONNX model.  Validates SHA-256 of the model bytes.
   * @param {ArrayBuffer} modelBytes – raw ONNX file bytes
   * @returns {Promise<void>}
   * @throws if SHA-256 doesn't match manifest or ONNX Runtime fails
   */
  async load(modelBytes) {
    if (!this.isConfigured) {
      throw new Error(
        'No model configured. Set modelId and sha256 in the manifest.'
      );
    }

    const hash = await _sha256Hex(modelBytes);
    if (hash !== this._manifest.sha256) {
      this._loadError = `SHA-256 mismatch: expected ${this._manifest.sha256}, got ${hash}`;
      throw new Error(this._loadError);
    }

    if (typeof globalThis.ort === 'undefined') {
      this._loadError = 'ONNX Runtime Web (ort) is not available';
      throw new Error(this._loadError);
    }

    try {
      this._session = await globalThis.ort.InferenceSession.create(
        modelBytes,
        { executionProviders: ['wasm'] }
      );
      this._loaded = true;
      this._loadError = null;
    } catch (e) {
      this._loadError = `ONNX session creation failed: ${e.message}`;
      throw new Error(this._loadError);
    }
  }

  /**
   * Extract a speaker embedding from mel features.
   *
   * @param {Float32Array[]} logMelFrames – array of log-mel vectors
   *   (each Float32Array of length numMelBands), as produced by
   *   MelFeatureExtractor.logMelSpectrogram().logMel
   * @returns {Promise<Float32Array>} – L2-normalized embedding vector
   * @throws if model is not loaded
   */
  async extract(logMelFrames) {
    if (!this._loaded || !this._session) {
      throw new Error(
        `Model not loaded (status: ${this.status}). ` +
        'Call load() with valid model bytes first.'
      );
    }

    const numFrames = logMelFrames.length;
    const numBands = logMelFrames[0].length;
    const inputData = new Float32Array(numFrames * numBands);
    for (let f = 0; f < numFrames; f++) {
      inputData.set(logMelFrames[f], f * numBands);
    }

    const inputTensor = new globalThis.ort.Tensor(
      'float32', inputData, [1, numFrames, numBands]
    );

    const feeds = {};
    feeds[this._manifest.inputName] = inputTensor;
    const results = await this._session.run(feeds);
    const output = results[this._manifest.outputName];
    const raw = new Float32Array(output.data);

    return _l2Normalize(raw);
  }

  /**
   * Compute cosine similarity between two embeddings.
   * @param {Float32Array} a
   * @param {Float32Array} b
   * @returns {number} cosine similarity in [-1, 1]
   */
  static cosineSimilarity(a, b) {
    if (a.length !== b.length) {
      throw new RangeError('Embedding dimensions must match');
    }
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom > 1e-12 ? dot / denom : 0;
  }

  dispose() {
    if (this._session) {
      this._session.release?.();
      this._session = null;
    }
    this._loaded = false;
  }
}

// ── Helpers ───────────────────────────────────────────────────

function _l2Normalize(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 1e-12) {
    for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  }
  return vec;
}

async function _sha256Hex(buffer) {
  if (typeof crypto === 'undefined' || !crypto.subtle) {
    throw new Error('crypto.subtle unavailable (requires secure context)');
  }
  const hashBuf = await crypto.subtle.digest('SHA-256', buffer);
  const bytes = new Uint8Array(hashBuf);
  let hex = '';
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i].toString(16).padStart(2, '0');
  }
  return hex;
}

export {
  SpeakerEmbeddingExtractor,
  MODEL_MANIFEST,
  EMBEDDING_DIM,
};
