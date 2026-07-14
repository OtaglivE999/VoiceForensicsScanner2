const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

const SRC = fs.readFileSync(
  path.resolve(__dirname, '../../src/js/storage/idb-store.js'), 'utf-8'
);

const INJECTABLE = SRC
  .replace(/^export\s*\{[^}]*\}\s*;?\s*$/m, '')
  + '\nwindow.IDBStore = IDBStore;\n'
  + 'window.DB_NAME = DB_NAME;\n'
  + 'window.DB_VERSION = DB_VERSION;\n'
  + 'window.STORES = STORES;\n';

test.describe('IDB Store', () => {
  let page;
  let server;

  test.beforeAll(async ({ browser }) => {
    // IndexedDB requires a proper origin — serve a minimal page
    const http = require('http');
    server = http.createServer((req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end('<!DOCTYPE html><html><body></body></html>');
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const port = server.address().port;

    page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${port}`);
    await page.addScriptTag({ content: INJECTABLE });
  });

  test.afterAll(async () => {
    await page?.close();
    if (server) server.close();
  });

  test('exports expected symbols', async () => {
    const result = await page.evaluate(() => ({
      has: typeof window.IDBStore === 'function',
      hasDbName: typeof window.DB_NAME === 'string',
      hasVersion: typeof window.DB_VERSION === 'number',
      hasStores: typeof window.STORES === 'object',
    }));
    expect(result.has).toBe(true);
    expect(result.hasDbName).toBe(true);
    expect(result.hasVersion).toBe(true);
    expect(result.hasStores).toBe(true);
  });

  test('opens database successfully', async () => {
    const result = await page.evaluate(async () => {
      const store = new IDBStore('test-open-' + Date.now());
      await store.open();
      const opened = !!store._db;
      store.close();
      return opened;
    });
    expect(result).toBe(true);
  });

  test('CRUD for speakers', async () => {
    const result = await page.evaluate(async () => {
      const store = new IDBStore('test-speakers-' + Date.now());
      await store.open();

      await store.putSpeaker({ id: 'spk1', name: 'Alice', detections: 5 });
      await store.putSpeaker({ id: 'spk2', name: 'Bob', detections: 3 });

      const spk1 = await store.getSpeaker('spk1');
      const all = await store.getAllSpeakers();
      const count = await store.getSpeakerCount();

      await store.deleteSpeaker('spk1');
      const afterDelete = await store.getSpeakerCount();

      store.close();
      return {
        name: spk1.name,
        allCount: all.length,
        count,
        afterDelete,
      };
    });
    expect(result.name).toBe('Alice');
    expect(result.allCount).toBe(2);
    expect(result.count).toBe(2);
    expect(result.afterDelete).toBe(1);
  });

  test('speaker upsert replaces existing', async () => {
    const result = await page.evaluate(async () => {
      const store = new IDBStore('test-upsert-' + Date.now());
      await store.open();

      await store.putSpeaker({ id: 'spk1', name: 'Alice', detections: 5 });
      await store.putSpeaker({ id: 'spk1', name: 'Alice Updated', detections: 10 });

      const spk = await store.getSpeaker('spk1');
      const count = await store.getSpeakerCount();
      store.close();
      return { name: spk.name, detections: spk.detections, count };
    });
    expect(result.name).toBe('Alice Updated');
    expect(result.detections).toBe(10);
    expect(result.count).toBe(1);
  });

  test('detections CRUD and indexed queries', async () => {
    const result = await page.evaluate(async () => {
      const store = new IDBStore('test-detections-' + Date.now());
      await store.open();

      await store.addDetection({ timestamp: 1000, speakerId: 'spk1', snr: 30 });
      await store.addDetection({ timestamp: 2000, speakerId: 'spk1', snr: 25 });
      await store.addDetection({ timestamp: 3000, speakerId: 'spk2', snr: 20 });

      const count = await store.getDetectionCount();
      const byTime = await store.getDetectionsByTimeRange(1500, 3500);
      const bySpeaker = await store.getDetectionsBySpeaker('spk1');
      const recent = await store.getRecentDetections(2);

      await store.clearDetections();
      const afterClear = await store.getDetectionCount();

      store.close();
      return {
        count,
        byTimeCount: byTime.length,
        bySpeakerCount: bySpeaker.length,
        recentCount: recent.length,
        recentFirst: recent.length > 0 ? recent[0].timestamp : null,
        afterClear,
      };
    });
    expect(result.count).toBe(3);
    expect(result.byTimeCount).toBe(2);
    expect(result.bySpeakerCount).toBe(2);
    expect(result.recentCount).toBe(2);
    expect(result.recentFirst).toBe(3000);
    expect(result.afterClear).toBe(0);
  });

  test('sessions CRUD', async () => {
    const result = await page.evaluate(async () => {
      const store = new IDBStore('test-sessions-' + Date.now());
      await store.open();

      await store.putSession({ id: 'sess1', startTime: 1000, deviceId: 'dev1' });
      await store.putSession({ id: 'sess2', startTime: 2000, deviceId: 'dev1' });

      const sess = await store.getSession('sess1');
      const all = await store.getAllSessions();

      store.close();
      return { startTime: sess.startTime, allCount: all.length };
    });
    expect(result.startTime).toBe(1000);
    expect(result.allCount).toBe(2);
  });

  test('meta key-value store', async () => {
    const result = await page.evaluate(async () => {
      const store = new IDBStore('test-meta-' + Date.now());
      await store.open();

      await store.setMeta('schemaVersion', 3);
      await store.setMeta('lastExport', '2024-01-01');

      const v = await store.getMeta('schemaVersion');
      const e = await store.getMeta('lastExport');
      const missing = await store.getMeta('nonexistent');

      store.close();
      return { v, e, missing };
    });
    expect(result.v).toBe(3);
    expect(result.e).toBe('2024-01-01');
    expect(result.missing).toBe(undefined);
  });

  test('importFromLocalStorage migrates data without overwriting', async () => {
    const result = await page.evaluate(async () => {
      const store = new IDBStore('test-import-' + Date.now());
      await store.open();

      // Pre-existing speaker
      await store.putSpeaker({ id: 'existing', name: 'Original', detections: 10 });

      const lsData = {
        speakers: {
          existing: { name: 'Should Not Overwrite', detections: 1 },
          newSpk: { name: 'New Speaker', detections: 5 },
        },
        detections: [
          { timestamp: 1000, speakerId: 'newSpk', snr: 20 },
          { timestamp: 2000, speakerId: 'newSpk', snr: 25 },
        ],
      };

      const result = await store.importFromLocalStorage(lsData);

      const existing = await store.getSpeaker('existing');
      const newSpk = await store.getSpeaker('newSpk');
      const detCount = await store.getDetectionCount();
      const migration = await store.getMeta('lastMigration');

      store.close();
      return {
        speakersImported: result.speakersImported,
        detectionsImported: result.detectionsImported,
        existingName: existing.name,
        newSpkName: newSpk.name,
        newSpkMigrated: newSpk.migratedFrom,
        detCount,
        hasMigration: !!migration,
      };
    });
    expect(result.speakersImported).toBe(1);
    expect(result.detectionsImported).toBe(2);
    expect(result.existingName).toBe('Original');
    expect(result.newSpkName).toBe('New Speaker');
    expect(result.newSpkMigrated).toBe('localStorage');
    expect(result.detCount).toBe(2);
    expect(result.hasMigration).toBe(true);
  });

  test('close and reopen preserves data', async () => {
    const dbName = 'test-persist-' + Date.now();
    const result = await page.evaluate(async (name) => {
      const store1 = new IDBStore(name);
      await store1.open();
      await store1.putSpeaker({ id: 'spk1', name: 'Persistent' });
      store1.close();

      const store2 = new IDBStore(name);
      await store2.open();
      const spk = await store2.getSpeaker('spk1');
      store2.close();

      return { name: spk ? spk.name : null };
    }, dbName);
    expect(result.name).toBe('Persistent');
  });

  test('getSpeaker returns undefined for missing key', async () => {
    const result = await page.evaluate(async () => {
      const store = new IDBStore('test-missing-' + Date.now());
      await store.open();
      const spk = await store.getSpeaker('nonexistent');
      store.close();
      return { found: spk !== undefined };
    });
    expect(result.found).toBe(false);
  });
});
