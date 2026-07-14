/**
 * IndexedDB persistence layer for speaker profiles and detection logs.
 *
 * Replaces localStorage with IndexedDB for:
 * - Larger storage quotas (typically 50-100MB+ vs 5MB)
 * - Structured data with indexed queries
 * - Binary-friendly (embeddings as typed arrays)
 *
 * Schema versioning ensures safe upgrades without silent data loss.
 */

const DB_NAME = 'VoiceForensicsScanner';
const DB_VERSION = 1;

const STORES = {
  speakers: 'speakers',
  detections: 'detections',
  sessions: 'sessions',
  meta: 'meta',
};

class IDBStore {
  constructor(dbName = DB_NAME, version = DB_VERSION) {
    this._dbName = dbName;
    this._version = version;
    this._db = null;
  }

  /**
   * Open the database, creating/upgrading stores as needed.
   * @returns {Promise<void>}
   */
  async open() {
    if (this._db) return;

    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this._dbName, this._version);

      req.onupgradeneeded = (event) => {
        const db = event.target.result;

        if (!db.objectStoreNames.contains(STORES.speakers)) {
          const store = db.createObjectStore(STORES.speakers, { keyPath: 'id' });
          store.createIndex('name', 'name', { unique: false });
          store.createIndex('enrolledAt', 'enrolledAt', { unique: false });
        }

        if (!db.objectStoreNames.contains(STORES.detections)) {
          const store = db.createObjectStore(STORES.detections, { keyPath: 'id', autoIncrement: true });
          store.createIndex('timestamp', 'timestamp', { unique: false });
          store.createIndex('speakerId', 'speakerId', { unique: false });
        }

        if (!db.objectStoreNames.contains(STORES.sessions)) {
          const store = db.createObjectStore(STORES.sessions, { keyPath: 'id' });
          store.createIndex('startTime', 'startTime', { unique: false });
        }

        if (!db.objectStoreNames.contains(STORES.meta)) {
          db.createObjectStore(STORES.meta, { keyPath: 'key' });
        }
      };

      req.onsuccess = (event) => {
        this._db = event.target.result;
        resolve();
      };

      req.onerror = (event) => {
        reject(new Error(`IndexedDB open failed: ${event.target.error}`));
      };
    });
  }

  // ── Speaker profiles ──────────────────────────────────────

  async putSpeaker(profile) {
    return this._put(STORES.speakers, profile);
  }

  async getSpeaker(id) {
    return this._get(STORES.speakers, id);
  }

  async getAllSpeakers() {
    return this._getAll(STORES.speakers);
  }

  async deleteSpeaker(id) {
    return this._delete(STORES.speakers, id);
  }

  async getSpeakerCount() {
    return this._count(STORES.speakers);
  }

  // ── Detections ────────────────────────────────────────────

  async addDetection(entry) {
    return this._put(STORES.detections, entry);
  }

  async getDetectionsByTimeRange(startMs, endMs) {
    return this._getByIndex(STORES.detections, 'timestamp', IDBKeyRange.bound(startMs, endMs));
  }

  async getDetectionsBySpeaker(speakerId) {
    return this._getByIndex(STORES.detections, 'speakerId', IDBKeyRange.only(speakerId));
  }

  async getDetectionCount() {
    return this._count(STORES.detections);
  }

  async getRecentDetections(limit = 100) {
    return new Promise((resolve, reject) => {
      const tx = this._db.transaction(STORES.detections, 'readonly');
      const store = tx.objectStore(STORES.detections);
      const index = store.index('timestamp');
      const results = [];

      const req = index.openCursor(null, 'prev');
      req.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor && results.length < limit) {
          results.push(cursor.value);
          cursor.continue();
        } else {
          resolve(results);
        }
      };
      req.onerror = () => reject(new Error('Failed to read detections'));
    });
  }

  async clearDetections() {
    return this._clear(STORES.detections);
  }

  // ── Sessions ──────────────────────────────────────────────

  async putSession(session) {
    return this._put(STORES.sessions, session);
  }

  async getSession(id) {
    return this._get(STORES.sessions, id);
  }

  async getAllSessions() {
    return this._getAll(STORES.sessions);
  }

  // ── Meta ──────────────────────────────────────────────────

  async getMeta(key) {
    const record = await this._get(STORES.meta, key);
    return record ? record.value : undefined;
  }

  async setMeta(key, value) {
    return this._put(STORES.meta, { key, value });
  }

  // ── Migration helper ──────────────────────────────────────

  /**
   * Import data from localStorage (v2 format) without overwriting
   * existing IndexedDB data.
   *
   * @param {object} lsData – { speakers: {...}, detections: [...] }
   * @returns {Promise<{ speakersImported: number, detectionsImported: number }>}
   */
  async importFromLocalStorage(lsData) {
    let speakersImported = 0;
    let detectionsImported = 0;

    if (lsData.speakers) {
      const entries = typeof lsData.speakers === 'object'
        ? Object.entries(lsData.speakers)
        : [];
      for (const [id, speaker] of entries) {
        const existing = await this.getSpeaker(id);
        if (!existing) {
          await this.putSpeaker({ id, ...speaker, migratedFrom: 'localStorage' });
          speakersImported++;
        }
      }
    }

    if (Array.isArray(lsData.detections)) {
      for (const det of lsData.detections) {
        await this.addDetection({ ...det, migratedFrom: 'localStorage' });
        detectionsImported++;
      }
    }

    await this.setMeta('lastMigration', {
      from: 'localStorage',
      at: new Date().toISOString(),
      speakersImported,
      detectionsImported,
    });

    return { speakersImported, detectionsImported };
  }

  // ── Generic helpers ───────────────────────────────────────

  _put(storeName, data) {
    return new Promise((resolve, reject) => {
      const tx = this._db.transaction(storeName, 'readwrite');
      const req = tx.objectStore(storeName).put(data);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(new Error(`put failed: ${req.error}`));
    });
  }

  _get(storeName, key) {
    return new Promise((resolve, reject) => {
      const tx = this._db.transaction(storeName, 'readonly');
      const req = tx.objectStore(storeName).get(key);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(new Error(`get failed: ${req.error}`));
    });
  }

  _getAll(storeName) {
    return new Promise((resolve, reject) => {
      const tx = this._db.transaction(storeName, 'readonly');
      const req = tx.objectStore(storeName).getAll();
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(new Error(`getAll failed: ${req.error}`));
    });
  }

  _getByIndex(storeName, indexName, range) {
    return new Promise((resolve, reject) => {
      const tx = this._db.transaction(storeName, 'readonly');
      const index = tx.objectStore(storeName).index(indexName);
      const req = index.getAll(range);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(new Error(`getByIndex failed: ${req.error}`));
    });
  }

  _count(storeName) {
    return new Promise((resolve, reject) => {
      const tx = this._db.transaction(storeName, 'readonly');
      const req = tx.objectStore(storeName).count();
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(new Error(`count failed: ${req.error}`));
    });
  }

  _delete(storeName, key) {
    return new Promise((resolve, reject) => {
      const tx = this._db.transaction(storeName, 'readwrite');
      const req = tx.objectStore(storeName).delete(key);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(new Error(`delete failed: ${req.error}`));
    });
  }

  _clear(storeName) {
    return new Promise((resolve, reject) => {
      const tx = this._db.transaction(storeName, 'readwrite');
      const req = tx.objectStore(storeName).clear();
      req.onsuccess = () => resolve();
      req.onerror = () => reject(new Error(`clear failed: ${req.error}`));
    });
  }

  close() {
    if (this._db) {
      this._db.close();
      this._db = null;
    }
  }
}

export { IDBStore, DB_NAME, DB_VERSION, STORES };
