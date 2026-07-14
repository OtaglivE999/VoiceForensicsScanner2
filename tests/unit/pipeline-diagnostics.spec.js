const { test, expect } = require('playwright/test');
const fs = require('fs');
const path = require('path');

const SRC = fs.readFileSync(
  path.resolve(__dirname, '../../src/js/diagnostics/pipeline-diagnostics.js'), 'utf-8'
);

const INJECTABLE = SRC
  .replace(/^export\s*\{[^}]*\}\s*;?\s*$/m, '')
  + '\nwindow.PipelineDiagnostics = PipelineDiagnostics;\n'
  + 'window.DIAGNOSTIC_VERSION = DIAGNOSTIC_VERSION;\n'
  + 'window.MODULE_IDS = MODULE_IDS;\n'
  + 'window.STATUS = STATUS;\n';

test.describe('Pipeline Diagnostics', () => {
  let page;

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage();
    await page.goto('about:blank');
    await page.addScriptTag({ content: INJECTABLE });
  });

  test.afterAll(async () => {
    await page?.close();
  });

  test('exports expected symbols', async () => {
    const result = await page.evaluate(() => ({
      hasDiag: typeof window.PipelineDiagnostics === 'function',
      hasVersion: typeof window.DIAGNOSTIC_VERSION === 'string',
      hasModuleIds: Array.isArray(window.MODULE_IDS),
      hasStatus: typeof window.STATUS === 'object',
    }));
    expect(result.hasDiag).toBe(true);
    expect(result.hasVersion).toBe(true);
    expect(result.hasModuleIds).toBe(true);
    expect(result.hasStatus).toBe(true);
  });

  test('registers and retrieves module status', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      diag.registerModule('resampler', { status: 'ok', version: '1.0', label: 'Resampler' });
      const mod = diag.getModuleStatus('resampler');
      return { id: mod.id, status: mod.status, version: mod.version, label: mod.label };
    });
    expect(result.id).toBe('resampler');
    expect(result.status).toBe('ok');
    expect(result.version).toBe('1.0');
    expect(result.label).toBe('Resampler');
  });

  test('rejects empty module id', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      try {
        diag.registerModule('', {});
        return { threw: false };
      } catch (e) {
        return { threw: true, msg: e.message };
      }
    });
    expect(result.threw).toBe(true);
    expect(result.msg).toContain('non-empty');
  });

  test('updates module status', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      diag.registerModule('melFeatures', { status: 'not-loaded' });
      diag.updateModuleStatus('melFeatures', 'ok', 'loaded successfully');
      const mod = diag.getModuleStatus('melFeatures');
      return { status: mod.status, detail: mod.detail };
    });
    expect(result.status).toBe('ok');
    expect(result.detail).toBe('loaded successfully');
  });

  test('update rejects unknown module', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      try {
        diag.updateModuleStatus('nonexistent', 'ok');
        return { threw: false };
      } catch (e) {
        return { threw: true, msg: e.message };
      }
    });
    expect(result.threw).toBe(true);
    expect(result.msg).toContain('Unknown module');
  });

  test('pipeline health reports overall status', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      diag.registerModule('resampler', { status: 'ok' });
      diag.registerModule('segmenter', { status: 'ok' });
      const allOk = diag.getPipelineHealth();

      diag.registerModule('embedding', { status: 'placeholder', detail: 'no model' });
      const degraded = diag.getPipelineHealth();

      diag.updateModuleStatus('segmenter', 'error', 'init failed');
      const errored = diag.getPipelineHealth();

      return {
        allOk: allOk.overall,
        degraded: degraded.overall,
        degradedIssues: degraded.issues.length,
        errored: errored.overall,
        erroredIssues: errored.issues.length,
      };
    });
    expect(result.allOk).toBe('ok');
    expect(result.degraded).toBe('degraded');
    expect(result.degradedIssues).toBe(1);
    expect(result.errored).toBe('error');
    expect(result.erroredIssues).toBe(2);
  });

  test('empty pipeline returns not-loaded health', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      return diag.getPipelineHealth();
    });
    expect(result.overall).toBe('not-loaded');
    expect(result.moduleCount).toBe(0);
  });

  test('processing chain returns modules in pipeline order', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      diag.registerModule('speakerMatcher', { status: 'ok', label: 'Matcher' });
      diag.registerModule('resampler', { status: 'ok', label: 'Resampler' });
      diag.registerModule('melFeatures', { status: 'ok', label: 'Mel' });
      const chain = diag.getProcessingChain();
      return chain.map(c => c.id);
    });
    expect(result[0]).toBe('resampler');
    expect(result[1]).toBe('melFeatures');
    expect(result[2]).toBe('speakerMatcher');
  });

  test('model availability with no embedding module', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      return diag.getModelAvailability();
    });
    expect(result.available).toBe(false);
    expect(result.status).toBe('not-loaded');
  });

  test('model availability with placeholder embedding', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      diag.registerModule('speakerEmbedding', {
        status: 'placeholder',
        detail: { modelId: null, reason: 'no trained model' },
      });
      return diag.getModelAvailability();
    });
    expect(result.available).toBe(false);
    expect(result.status).toBe('placeholder');
  });

  test('model availability with loaded model', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      diag.registerModule('speakerEmbedding', {
        status: 'ok',
        detail: { modelId: 'ecapa-tdnn-v1' },
      });
      return diag.getModelAvailability();
    });
    expect(result.available).toBe(true);
    expect(result.modelId).toBe('ecapa-tdnn-v1');
  });

  test('event logging with capacity limit', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      for (let i = 0; i < 210; i++) {
        diag.logEvent('test', { i });
      }
      const events = diag.getEvents();
      return {
        count: events.length,
        firstI: events[0].data.i,
        lastI: events[events.length - 1].data.i,
      };
    });
    expect(result.count).toBe(200);
    expect(result.firstI).toBe(10);
    expect(result.lastI).toBe(209);
  });

  test('getEvents with limit', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      diag.logEvent('a', { n: 1 });
      diag.logEvent('b', { n: 2 });
      diag.logEvent('c', { n: 3 });
      const events = diag.getEvents(2);
      return { count: events.length, types: events.map(e => e.type) };
    });
    expect(result.count).toBe(2);
    expect(result.types).toEqual(['b', 'c']);
  });

  test('onStatusChange fires for registration and updates', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      const received = [];
      diag.onStatusChange((type, data) => {
        received.push({ type, data });
      });
      diag.registerModule('resampler', { status: 'ok' });
      diag.updateModuleStatus('resampler', 'degraded', 'high latency');
      diag.logEvent('test-event', {});
      return {
        count: received.length,
        types: received.map(r => r.type),
      };
    });
    expect(result.count).toBe(3);
    expect(result.types).toContain('module-registered');
    expect(result.types).toContain('module-status-changed');
    expect(result.types).toContain('event-logged');
  });

  test('unsubscribe stops notifications', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      let count = 0;
      const unsub = diag.onStatusChange(() => { count++; });
      diag.registerModule('a', { status: 'ok' });
      unsub();
      diag.registerModule('b', { status: 'ok' });
      return { count };
    });
    expect(result.count).toBe(1);
  });

  test('getSnapshot returns complete diagnostic state', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      diag.registerModule('resampler', { status: 'ok', version: '1.0', label: 'Resampler' });
      diag.registerModule('speakerEmbedding', { status: 'placeholder' });
      diag.logEvent('scan-started', {});
      const snap = diag.getSnapshot();
      return {
        format: snap.format,
        hasTimestamp: typeof snap.timestamp === 'string',
        hasUptime: typeof snap.uptimeMs === 'number',
        overall: snap.pipeline.overall,
        chainLen: snap.chain.length,
        modelAvail: snap.model.available,
        moduleCount: Object.keys(snap.modules).length,
        eventCount: snap.recentEvents.length,
      };
    });
    expect(result.format).toContain('Diagnostics-v1');
    expect(result.hasTimestamp).toBe(true);
    expect(result.hasUptime).toBe(true);
    expect(result.overall).toBe('degraded');
    expect(result.chainLen).toBe(2);
    expect(result.modelAvail).toBe(false);
    expect(result.moduleCount).toBe(2);
    expect(result.eventCount).toBe(1);
  });

  test('reset clears all state', async () => {
    const result = await page.evaluate(() => {
      const diag = new PipelineDiagnostics();
      diag.registerModule('resampler', { status: 'ok' });
      diag.logEvent('test', {});
      let notified = false;
      diag.onStatusChange(() => { notified = true; });
      diag.reset();
      const health = diag.getPipelineHealth();
      const events = diag.getEvents();
      diag.registerModule('x', { status: 'ok' });
      return {
        moduleCount: health.moduleCount,
        eventCount: events.length,
        notifiedAfterReset: notified,
      };
    });
    expect(result.moduleCount).toBe(0);
    expect(result.eventCount).toBe(0);
    expect(result.notifiedAfterReset).toBe(false);
  });
});
