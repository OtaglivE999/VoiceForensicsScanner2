# VoiceForensicsScanner v3 Upgrade Summary

## Module Architecture

All major algorithms extracted from the monolithic HTML file into separate ES modules:

```
src/js/
├── audio/
│   ├── resampler.js          Phase 1 – Band-limited 48kHz→16kHz decimation
│   └── speech-segmenter.js   Phase 2 – Frame-based VAD + speech segment builder
├── features/
│   ├── mel-features.js       Phase 3 – Log-mel spectrogram, MFCC, delta features
│   └── speaker-embedding.js  Phase 4 – ONNX embedding interface (PLACEHOLDER)
├── matching/
│   ├── speaker-matcher.js    Phase 5 – Embedding-based matching with feature fallback
│   ├── enrollment.js         Phase 6 – Multi-observation enrollment
│   └── profile-updater.js    Phase 7 – Quality-gated EMA profile updates
├── storage/
│   └── idb-store.js          Phase 8 – IndexedDB persistence with schema versioning
├── export/
│   └── evidence-bundle.js    Phase 9 – Versioned evidence bundle builder (v3)
└── diagnostics/
    └── pipeline-diagnostics.js  Phase 10 – Pipeline health and module status
```

## Defects Addressed (D1–D12)

| Defect | Description | Status |
|--------|-------------|--------|
| D1 | Hardcoded sample rate assumption | Fixed – Resampler handles arbitrary capture rates |
| D2 | Single-buffer processing misses speech boundaries | Fixed – SpeechSegmenter with hangover bridging |
| D3 | No real spectral features | Fixed – Mel filterbank, log-mel, MFCC, deltas |
| D4 | No trained embedding model | Interface ready – ONNX loader with SHA-256 validation (PLACEHOLDER, no model bundled) |
| D5 | Naive cosine-only matching | Fixed – Embedding primary + weighted feature fallback |
| D6 | Single-shot enrollment | Fixed – Multi-observation with consistency checking |
| D7 | Unconditional profile updates | Fixed – Quality-gated EMA with maturity-based learning rate |
| D8 | localStorage limits | Fixed – IndexedDB with indexed queries and migration helper |
| D9 | Unversioned exports | Fixed – v3 bundles with SHA-256 chain-of-custody |
| D10 | No pipeline visibility | Fixed – Diagnostics module with health/chain/model status |
| D11 | No cross-module validation | Fixed – Integration tests verify all module contracts |
| D12 | No final review | This document |

## Remaining Placeholders

1. **Speaker Embedding Model (Phase 4)**: The `SpeakerEmbeddingExtractor` interface, loader, SHA-256 validation, and tests are implemented, but no actual ONNX model is bundled. The `MODEL_MANIFEST` has `status: 'no-model-available'`. When a trained model becomes available:
   - Set `MODEL_MANIFEST.modelId` and `MODEL_MANIFEST.sha256`
   - The loader validates the model hash before loading into ONNX Runtime Web
   - All downstream modules (matcher, enrollment, updater) are ready to consume real embeddings

2. **HTML Integration**: The modules are standalone ES modules not yet wired into `src/iphone_web_audio_app.html`. Integration requires replacing the inline processing with module imports and connecting the pipeline.

## Test Coverage

| Suite | Tests | Phase |
|-------|-------|-------|
| Baseline export parsing | 36 | 0 |
| Resampler | 14 | 1 |
| Speech segmenter | 14 | 2 |
| Mel features | 15 | 3 |
| Speaker embedding | 16 | 4 |
| Speaker matcher | 14 | 5 |
| Enrollment | 11 | 6 |
| Profile updater | 13 | 7 |
| IDB store | 10 | 8 |
| Evidence bundle | 9 | 9 |
| Pipeline diagnostics | 17 | 10 |
| Integration | 11 | 11 |
| **Total** | **180** | |

All tests run via Playwright with Chromium, including browser API tests (IndexedDB, crypto.subtle) that spin up a local HTTP server for proper origin access.

## Key Design Decisions

- **Model safety**: Embeddings from different model IDs are never compared. The matcher checks `embeddingModelId` compatibility before cosine similarity.
- **Quality gates**: Profile updates require minimum SNR (15dB), low clipping (<1%), sufficient duration (>300ms), and voice confidence (>0.5).
- **Adaptive learning**: Young profiles (< maxDetectionsForFastLearn) use fast EMA (α=0.15); mature profiles use slow EMA (α=0.02). Embeddings are L2-renormalized after each update.
- **No silent migration**: `importFromLocalStorage` never overwrites existing IndexedDB data and marks imported records with `migratedFrom: 'localStorage'`.
- **Forensic integrity**: Evidence bundles include per-file SHA-256 hashes and a manifest self-hash. Profile IDs represent model-generated voice clusters, not legally verified people.
