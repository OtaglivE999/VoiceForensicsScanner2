/**
 * Pipeline diagnostics — exposes module health, processing chain state,
 * and model availability for the UI layer.
 */

const DIAGNOSTIC_VERSION = 'VoiceForensics-Diagnostics-v1';

const MODULE_IDS = [
  'resampler',
  'speechSegmenter',
  'melFeatures',
  'speakerEmbedding',
  'speakerMatcher',
  'enrollment',
  'profileUpdater',
  'idbStore',
  'evidenceBundle',
];

const STATUS = {
  OK: 'ok',
  DEGRADED: 'degraded',
  ERROR: 'error',
  NOT_LOADED: 'not-loaded',
  PLACEHOLDER: 'placeholder',
};

class PipelineDiagnostics {
  constructor() {
    this._modules = new Map();
    this._events = [];
    this._maxEvents = 200;
    this._startTime = Date.now();
    this._listeners = [];
  }

  registerModule(id, info) {
    if (!id || typeof id !== 'string') {
      throw new Error('Module id must be a non-empty string');
    }
    this._modules.set(id, {
      id,
      status: info.status || STATUS.NOT_LOADED,
      version: info.version || null,
      label: info.label || id,
      detail: info.detail || null,
      lastUpdated: Date.now(),
    });
    this._emit('module-registered', { id });
  }

  updateModuleStatus(id, status, detail) {
    const mod = this._modules.get(id);
    if (!mod) {
      throw new Error(`Unknown module: ${id}`);
    }
    mod.status = status;
    if (detail !== undefined) mod.detail = detail;
    mod.lastUpdated = Date.now();
    this._emit('module-status-changed', { id, status, detail });
  }

  getModuleStatus(id) {
    const mod = this._modules.get(id);
    return mod ? { ...mod } : null;
  }

  getAllModuleStatuses() {
    const result = {};
    for (const [id, mod] of this._modules) {
      result[id] = { ...mod };
    }
    return result;
  }

  getPipelineHealth() {
    const modules = Array.from(this._modules.values());
    if (modules.length === 0) {
      return { overall: STATUS.NOT_LOADED, moduleCount: 0, issues: [] };
    }

    const issues = [];
    let hasError = false;
    let hasDegraded = false;
    let hasPlaceholder = false;

    for (const mod of modules) {
      if (mod.status === STATUS.ERROR) {
        hasError = true;
        issues.push({ id: mod.id, status: mod.status, detail: mod.detail });
      } else if (mod.status === STATUS.DEGRADED) {
        hasDegraded = true;
        issues.push({ id: mod.id, status: mod.status, detail: mod.detail });
      } else if (mod.status === STATUS.PLACEHOLDER) {
        hasPlaceholder = true;
        issues.push({ id: mod.id, status: mod.status, detail: mod.detail });
      }
    }

    let overall;
    if (hasError) {
      overall = STATUS.ERROR;
    } else if (hasDegraded || hasPlaceholder) {
      overall = STATUS.DEGRADED;
    } else {
      overall = STATUS.OK;
    }

    return { overall, moduleCount: modules.length, issues };
  }

  getProcessingChain() {
    const chain = [];
    const orderedIds = [
      'resampler',
      'speechSegmenter',
      'melFeatures',
      'speakerEmbedding',
      'speakerMatcher',
      'enrollment',
      'profileUpdater',
      'idbStore',
      'evidenceBundle',
    ];

    for (const id of orderedIds) {
      const mod = this._modules.get(id);
      if (mod) {
        chain.push({
          id: mod.id,
          label: mod.label,
          status: mod.status,
          version: mod.version,
        });
      }
    }

    for (const [id, mod] of this._modules) {
      if (!orderedIds.includes(id)) {
        chain.push({
          id: mod.id,
          label: mod.label,
          status: mod.status,
          version: mod.version,
        });
      }
    }

    return chain;
  }

  getModelAvailability() {
    const embMod = this._modules.get('speakerEmbedding');
    if (!embMod) {
      return {
        available: false,
        status: STATUS.NOT_LOADED,
        modelId: null,
        detail: 'Embedding module not registered',
      };
    }

    const available = embMod.status === STATUS.OK;
    return {
      available,
      status: embMod.status,
      modelId: embMod.detail && embMod.detail.modelId ? embMod.detail.modelId : null,
      detail: embMod.detail,
    };
  }

  logEvent(type, data) {
    const event = {
      type,
      timestamp: Date.now(),
      uptimeMs: Date.now() - this._startTime,
      data: data || null,
    };
    this._events.push(event);
    if (this._events.length > this._maxEvents) {
      this._events.splice(0, this._events.length - this._maxEvents);
    }
    this._emit('event-logged', event);
  }

  getEvents(limit) {
    const n = limit || this._maxEvents;
    return this._events.slice(-n).map(e => ({ ...e }));
  }

  getSnapshot() {
    return {
      format: DIAGNOSTIC_VERSION,
      timestamp: new Date().toISOString(),
      uptimeMs: Date.now() - this._startTime,
      pipeline: this.getPipelineHealth(),
      chain: this.getProcessingChain(),
      model: this.getModelAvailability(),
      modules: this.getAllModuleStatuses(),
      recentEvents: this.getEvents(20),
    };
  }

  onStatusChange(callback) {
    this._listeners.push(callback);
    return () => {
      this._listeners = this._listeners.filter(cb => cb !== callback);
    };
  }

  _emit(type, data) {
    for (const cb of this._listeners) {
      try {
        cb(type, data);
      } catch (_) {
        // listener errors must not break the pipeline
      }
    }
  }

  reset() {
    this._modules.clear();
    this._events = [];
    this._listeners = [];
    this._startTime = Date.now();
  }
}

export {
  PipelineDiagnostics,
  DIAGNOSTIC_VERSION,
  MODULE_IDS,
  STATUS,
};
